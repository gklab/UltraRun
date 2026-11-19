"""Benchmark a native MLX ReaDy drafter with the exact BF16 UltraRun verifier."""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from math import ceil, floor, log, pi, sqrt
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn

from ultrarun.memory import GIB, metal_job
from ultrarun.model import Model
from ultrarun.packed_projection import pack_activations
from ultrarun.profiles import OPTIMIZED_MODEL
from ultrarun.server import warm_verification_graphs
from ultrarun.spec_tree import VerificationTree, accepted_path
from scripts.bench_dflash2 import observe_layers, pages, prefill, read_thermal_state


ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "weights" / "Qwen3.8-27B-ReaDy"
TARGET = ROOT / "weights" / "Qwen3.8-27B"
WEIGHT_SHA256 = {
    "model.safetensors": "746f7d94f77d1d032614d2c0af805bf08cbe22b4a019b7ba84e103d36139c3ac",
    "reader/reader.safetensors": "fd3e2baeac1ff491699f0b2001aaec638c528b149bd3d285088f2cf7a08af295",
    "config.json": "dd65fb1b01c2adea69512ff2990a79d58eb7fe2c7ea97375aa66f657a29a5bfd",
    "reader/reader_config.json": "f237d538269325ee1aad7073e2a3c86a8729069a0330173a515217dc5a6c9db2",
}


def verified_checkpoint():
    for name, digest in WEIGHT_SHA256.items():
        with (DRAFT / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"ReaDy checkpoint checksum mismatch for {name}")


def linear(x, weight):
    return x @ weight.T


def norm(x, weight, eps):
    normalized = mx.fast.rms_norm(x.astype(mx.float32), None, eps).astype(x.dtype)
    return normalized * weight if weight is not None else normalized


class StateReader:
    def __init__(self, weights):
        self.weights = weights
        shapes = {
            "query.weight": (768, 5120),
            "query_up": (2, 48, 8, 128),
            "value_down": (2, 48, 128, 8),
            "down.weight": (128, 864),
            "up.weight": (5120, 128),
        }
        if {key: value.shape for key, value in weights.items()} != shapes:
            raise ValueError("Unrecognized ReaDy state reader weights")

    def prepare(self, states):
        state = mx.stack(states)[None]
        assert state.shape == (1, 2, 48, 128, 128) and state.dtype == mx.float32
        rms = mx.sqrt(mx.mean(mx.square(state), axis=(-2, -1), keepdims=True) + 1e-6)
        values = (state / rms).astype(mx.bfloat16)
        memory = self.weights["value_down"].swapaxes(-1, -2) @ values
        scale = mx.log(rms[..., 0, 0]).astype(mx.bfloat16)
        return memory, scale

    def __call__(self, hidden, prepared):
        memory, scale = prepared
        batch, tokens, width = hidden.shape
        eps = 2**-23
        queries = linear(norm(hidden, None, eps), self.weights["query.weight"])
        queries = queries.reshape(batch, tokens, 2, 48, 8).transpose(0, 2, 3, 1, 4)
        keys = queries @ self.weights["query_up"]
        read = (keys @ memory.swapaxes(-1, -2)).transpose(0, 3, 1, 2, 4)
        read = read / sqrt(128)
        scales = mx.broadcast_to(scale[:, None, :, :, None], (batch, tokens, 2, 48, 1))
        features = mx.concatenate((read, scales), axis=-1).reshape(batch, tokens, -1)
        small = linear(norm(features, None, eps), self.weights["down.weight"])
        return hidden + linear(nn.silu(small), self.weights["up.weight"])


def frequencies(config):
    dimension = config.head_dim
    scaling = config.rope_parameters
    base = scaling["rope_theta"]
    length = scaling["original_max_position_embeddings"]
    factor = scaling["factor"]

    def boundary(rotations):
        return dimension * log(length / (2 * pi * rotations)) / (2 * log(base))

    low = max(0, floor(boundary(scaling["beta_fast"])))
    high = min(dimension - 1, ceil(boundary(scaling["beta_slow"])))
    axis = mx.arange(dimension // 2, dtype=mx.float32)
    blend = mx.clip((axis - low) / (high - low), 0.0, 1.0)
    return ((1.0 - blend) + blend / factor) * (base ** (-2.0 * axis / dimension)), 1 + 0.1 * log(factor)


class DraftContext:
    """Project confirmed target features to persistent per-layer draft K/V."""

    normalize = staticmethod(norm)

    def layer(self, index):
        prefix = f"layers.{index}."
        return lambda name: self.weights[prefix + name + ".weight"]

    def kv(self, x, layer, offset):
        key = linear(x, layer("self_attn.k_proj"))
        value = linear(x, layer("self_attn.v_proj"))
        return self.format_kv(key, value, layer, offset)

    def format_kv(self, key, value, layer, offset):
        c = self.config
        shape = (1, key.shape[1], c.num_key_value_heads, c.head_dim)
        key = self.normalize(key.reshape(shape), layer("self_attn.k_norm"), c.rms_norm_eps).transpose(0, 2, 1, 3)
        return self.rotate(key, offset), value.reshape(shape).transpose(0, 2, 1, 3)

    def context_features(self, features):
        return self.normalize(linear(features, self.weights["fc.weight"]),
                              self.weights["hidden_norm.weight"], self.config.rms_norm_eps)

    def context_kv(self, x, index, offset):
        return self.kv(x, self.layer(index), offset)

    def context(self, features, start, budget, *, layer_count=None):
        c = self.config
        layers = c.num_hidden_layers if layer_count is None else layer_count
        blocks = [[] for _ in range(layers)]
        for pos in range(0, features.shape[1], 512):
            segment = features[:, pos:pos + 512, :]
            x = self.context_features(segment)
            chunk = [self.context_kv(x, i, start + pos) for i in range(layers)]
            mx.eval(chunk)
            for layer, pair in zip(blocks, chunk, strict=True):
                layer.append(pair)
            budget.check()
        return [
            tuple(mx.concatenate(parts, axis=2) for parts in zip(*layer, strict=True))
            for layer in blocks
        ]


class ReadyDraft(DraftContext):
    def __init__(self, directory):
        self.config = SimpleNamespace(**json.loads((directory / "config.json").read_text()))
        self.reader_config = json.loads((directory / "reader/reader_config.json").read_text())
        self.weights = mx.load(str(directory / "model.safetensors"))
        self.reader = StateReader(mx.load(str(directory / "reader/reader.safetensors")))
        assert len(self.weights) == 62
        assert all(a.dtype == mx.bfloat16 for a in (*self.weights.values(), *self.reader.weights.values()))
        self.frequencies, self.attention_factor = frequencies(self.config)

    def rotate(self, words, offset):
        angle = mx.arange(offset, offset + words.shape[2], dtype=mx.float32)[:, None] * self.frequencies
        cos = (mx.cos(angle) * self.attention_factor).astype(words.dtype)
        sin = (mx.sin(angle) * self.attention_factor).astype(words.dtype)
        a, b = mx.split(words, 2, axis=-1)
        return mx.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1)

    def score(self, model, pending, prefix, state, proposals, reader_enabled):
        c = self.config
        token_ids = mx.array([pending, *([c.mask_token_id] * (proposals - 1))], dtype=mx.int32)
        x = model.embedding[token_ids][None]
        offset = prefix[0][0].shape[2]
        prepared = self.reader.prepare(state) if reader_enabled else None
        for i, (known_k, known_v) in enumerate(prefix):
            w = self.layer(i)
            h = norm(x, w("input_layernorm"), c.rms_norm_eps)
            q = linear(h, w("self_attn.q_proj")).reshape(1, proposals, c.num_attention_heads, c.head_dim)
            q = self.rotate(norm(q, w("self_attn.q_norm"), c.rms_norm_eps).transpose(0, 2, 1, 3), offset)
            key, value = self.kv(h, w, offset)
            attn = mx.fast.scaled_dot_product_attention(
                q, mx.concatenate((known_k, key), axis=2), mx.concatenate((known_v, value), axis=2),
                scale=c.head_dim**-0.5,
            )
            x = x + linear(attn.transpose(0, 2, 1, 3).reshape(1, proposals, -1), w("self_attn.o_proj"))
            h = norm(x, w("post_attention_layernorm"), c.rms_norm_eps)
            if prepared is not None and i == self.reader_config["draft_layer"]:
                h = self.reader(h, prepared)
            gate = nn.silu(linear(h, w("mlp.gate_proj")))
            x = x + linear(gate * linear(h, w("mlp.up_proj")), w("mlp.down_proj"))
        hidden = norm(x, self.weights["norm.weight"], c.rms_norm_eps)[0]
        logits = model.packed_head.project(pack_activations(hidden), proposals)
        logits = model.vocabulary_order.token_order(logits)
        return logits

    def propose(self, model, pending, prefix, cache, proposals, reader_enabled):
        state = [cache[i].delta for i in self.reader_config["state_layers"]] if reader_enabled else None
        logits = self.score(model, pending, prefix, state, proposals, reader_enabled)
        markov_in = self.weights["markov_head.markov_w1.weight"]
        markov_out = self.weights["markov_head.markov_w2.weight"]
        last = mx.array(pending, dtype=mx.int32)
        picks = []
        for i in range(proposals):
            bias = linear(markov_in[last][None], markov_out)[0]
            last = mx.argmax(logits[i] + bias)
            picks.append(last)
        return mx.stack(picks).astype(mx.int32)


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def prepare_prefix(model, draft, trace, budget):
    expected, ids = trace["output"], trace["input"]
    prefill_begin = perf_counter()
    cache, first, features = prefill(model, draft, ids)
    assert first == expected[0]
    prefix = draft.context(features, 0, budget)
    mx.eval(prefix)
    del features
    budget.check(force=True)
    print("prefill", json.dumps({
        "seconds": perf_counter() - prefill_begin,
        "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics(),
    }), flush=True)
    return cache, first, prefix


def decode(model, draft, trace, prepared, width, reader_enabled, budget, *, tree_proposals=False):
    for _ in decode_steps(model, draft, trace, prepared, width, reader_enabled, budget,
                          tree_proposals=tree_proposals):
        pass


def decode_steps(model, draft, trace, prepared, width, reader_enabled, budget, *, tree_proposals=False,
                 label=None):
    expected, ids = trace["output"], trace["input"]
    initial_cache, first, prefix = prepared
    cache = [replace(state) for state in initial_cache]
    observe_verified = getattr(draft, "observe_verified", None)
    if label is None:
        mx.reset_peak_memory()
    branch = {}
    observe_layers(model, draft.config.target_layer_ids, lambda i, offset, output: branch.__setitem__(i, output))
    stats = Counter()
    batch_widths = Counter()
    phases = Counter()
    tokens = [first]
    offset = len(ids)
    conv_taps = model.config["linear_conv_kernel_dim"] - 1
    first_diff = None
    side_commits = 0
    before = pages()
    thermal = read_thermal_state()
    seconds = 0.0
    while len(tokens) < len(expected):
        remaining = len(expected) - len(tokens)
        round_start = started = perf_counter()
        if tree_proposals:
            batch, parents, spine = draft.propose_tree(model, tokens[-1], prefix, min(width, remaining + 1))
        else:
            proposed = draft.propose(model, tokens[-1], prefix, cache, width - 1, reader_enabled)
            batch = [tokens[-1], *proposed[:remaining].tolist()]
            parents = [-1, *range(len(batch) - 1)]
            spine = len(batch) - 1
        batch_widths[len(batch)] += 1
        phases["draft"] += perf_counter() - started

        started = perf_counter()
        tree = VerificationTree(parents, offset, conv_taps, spine=spine)
        hidden = model.hidden(mx.array(batch, dtype=mx.int32), cache, offset, keep_replay=True, tree=tree)
        pred = model.predictions(hidden)
        model.evaluate([pred, *branch.values()], cache)
        next_tokens = pred.tolist()
        phases["target"] += perf_counter() - started

        started = perf_counter()
        path = accepted_path(parents, batch, next_tokens)[:remaining]
        new = [batch[i] for i in path[1:]] + [next_tokens[path[-1]]]
        if observe_verified is not None:
            observe_verified(new)
        if first_diff is None:
            first_diff = next((len(tokens) + i for i, token in enumerate(new)
                               if token != expected[len(tokens) + i]), None)
        keep = len(new)
        side_commits += model.accept_tree(cache, tree, offset, path, defer_kv=True)
        if keep < remaining:
            indices = mx.array(path, dtype=mx.int32) if tree_proposals else slice(keep)
            known = mx.concatenate([branch[i][indices] for i in draft.config.target_layer_ids], axis=-1)[None]
            extra = draft.context(known, offset, budget)
            prefix = [tuple(mx.concatenate((a, b), axis=2) for a, b in zip(old, new_kv, strict=True))
                      for old, new_kv in zip(prefix, extra, strict=True)]
            model.evaluate(prefix, cache)
        offset += keep
        stats[keep] += 1
        tokens.extend(new)
        branch.clear()
        budget.check()
        phases["commit"] += perf_counter() - started
        if len(tokens) // 256 != (len(tokens) - keep) // 256:
            print("progress", width, len(tokens), sum(stats.values()), label or "", flush=True)
        seconds += perf_counter() - round_start
        yield len(tokens)

    after = pages()
    result = {
        "width": width, "reader": reader_enabled, "tree_proposals": tree_proposals, "tokens": len(tokens),
        "rounds": sum(stats.values()), "mean_advance": (len(tokens)-1) / sum(stats.values()),
        "acceptance": stats, "batch_widths": batch_widths, "side_commits": side_commits,
        "decode_tok_s": (len(tokens)-1) / seconds,
        "phases_s": phases, "exact_output": tokens == expected,
        "first_divergence": first_diff, "token_sha256": digest(tokens),
        "thermal": [thermal, read_thermal_state()], "peak_gib": mx.get_peak_memory() / GIB,
        "faults": {key: after[key] - before[key] for key in before},
        **budget.metrics(),
    }
    if label is not None:
        result["draft_backend"] = label
    print("result", json.dumps(result), flush=True)
    if first_diff is not None:
        raise ValueError(f"BF16 verification first diverged at output token {first_diff}")


def benchmark_target(budget):
    model = Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
    warm_verification_graphs(model, budget)
    del model.mtp_layer, model.mtp_fc, model.mtp_norms
    mx.clear_cache()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("code", "chinese_technical_report", "english_technical_tutorial"), default="code")
    parser.add_argument("--width", action="append", type=int, choices=(8, 16))
    parser.add_argument("--disable-reader", action="store_true")
    args = parser.parse_args()
    verified_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[args.workload]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    with metal_job(80, 2) as budget:
        model = benchmark_target(budget)
        budget.check(force=True, required_bytes=5*GIB)
        draft = ReadyDraft(DRAFT)
        mx.eval(draft.weights, draft.reader.weights)
        budget.admit(model, 5120, 512, False, extra_bytes=GIB)
        print("load", json.dumps({"weights_bf16": len(draft.weights) + len(draft.reader.weights),
              "peak_gib": mx.get_peak_memory()/GIB, **budget.metrics()}), flush=True)
        prefix = prepare_prefix(model, draft, trace, budget)
        for width in args.width or (8,):
            decode(model, draft, trace, prefix, width, not args.disable_reader, budget)
            mx.clear_cache()
            budget.check(force=True)


if __name__ == "__main__":
    main()

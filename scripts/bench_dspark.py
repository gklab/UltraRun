"""Measure the published Red Hat BF16 DSpark head on the original BF16 verifier."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import file_digest
import json
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.packed_projection import pack_activations
from scripts.audit_dspine_complement import capture_mtp
from scripts.bench_dflash2 import pages, prefill
from scripts.bench_ready import DraftContext, ROOT, benchmark_target, decode, linear, norm


DRAFT = ROOT / "weights/Qwen3.8-27B-RedHat-DSpark"
CHECKSUMS = {
    "config.json": "6b2b39729c0afcac855b4b9808a85d8d323f2e4ca754749861c922dcc3c50a40",
    "model.safetensors": "6cf6f33fbb2dfd7e74f8009132f0a2a2b4c71cd6cce9360a7d4145c9218f1781",
}


def verify_checkpoint():
    for name, expected in CHECKSUMS.items():
        with (DRAFT / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Unexpected DSpark checkpoint digest for {name}")


class DSparkDraft(DraftContext):
    def __init__(self, prompt_length):
        saved = json.loads((DRAFT / "config.json").read_text())
        config = self.config = SimpleNamespace(**saved["transformer_layer_config"])
        assert saved["speculators_config"]["verifier"]["name_or_path"] == "Qwen/Qwen3.8-27B"
        assert saved["sample_from_anchor"] and saved["markov_head_type"] == "vanilla"
        assert config.sliding_window and set(config.layer_types) == {"sliding_attention"}
        assert not saved["sliding_window_non_causal"]
        assert saved["draft_vocab_size"] == config.vocab_size
        assert config.rope_parameters["rope_type"] == "default"
        assert config.rope_parameters["partial_rotary_factor"] == 1.0
        # Speculators counts completed target layers. UltraRun observes their zero-based outputs.
        config.target_layer_ids = [i - 1 for i in saved["aux_hidden_state_layer_ids"]]
        config.mask_token_id = saved["mask_token_id"]
        config.block_size = saved["block_size"]
        self.origin = max(0, prompt_length - config.sliding_window + 1)
        self.weights = mx.load(str(DRAFT / "model.safetensors"))
        assert len(self.weights) == 62 and all(w.dtype == mx.bfloat16 for w in self.weights.values())
        self.markov_in = self.weights["markov_head.markov_w1.weight"]
        self.markov_out = self.weights["markov_head.markov_w2.weight"]
        assert self.markov_in.shape == self.markov_out.shape == (config.vocab_size, saved["markov_rank"])

    def rotate(self, words, offset):
        return mx.fast.rope(words, self.config.head_dim, traditional=False,
                            base=self.config.rope_parameters["rope_theta"], scale=1.0, offset=offset)

    def logits(self, model, pending, prefix, proposals):
        config = self.config
        ids = mx.array([[pending, *([config.mask_token_id] * (proposals - 1))]], dtype=mx.int32)
        hidden = model.embedding[ids]
        available = prefix[0][0].shape[2]
        offset = self.origin + available
        cached = min(available, config.sliding_window - 1)
        columns = mx.arange(offset - cached, offset + proposals)
        query_positions = mx.arange(offset, offset + proposals)
        distance = query_positions[:, None] - columns[None]
        mask = (distance >= 0) & (distance < config.sliding_window)
        for i, (known_k, known_v) in enumerate(prefix):
            weight = self.layer(i)
            h = norm(hidden, weight("input_layernorm"), config.rms_norm_eps)
            query = linear(h, weight("self_attn.q_proj"))
            query = query.reshape(1, proposals, config.num_attention_heads, config.head_dim)
            query = norm(query, weight("self_attn.q_norm"), config.rms_norm_eps).transpose(0, 2, 1, 3)
            query = self.rotate(query, offset)
            key, value = self.kv(h, weight, offset)
            keys = mx.concatenate((known_k[:, :, available - cached:], key), axis=2)
            values = mx.concatenate((known_v[:, :, available - cached:], value), axis=2)
            result = mx.fast.scaled_dot_product_attention(
                query, keys, values, scale=config.head_dim**-0.5, mask=mask,
            )
            result = result.transpose(0, 2, 1, 3).reshape(1, proposals, -1)
            hidden = hidden + linear(result, weight("self_attn.o_proj"))
            h = norm(hidden, weight("post_attention_layernorm"), config.rms_norm_eps)
            gate = nn.silu(linear(h, weight("mlp.gate_proj")))
            hidden = hidden + linear(gate * linear(h, weight("mlp.up_proj")), weight("mlp.down_proj"))
        h = norm(hidden, self.weights["norm.weight"], config.rms_norm_eps)[0]
        logits = model.packed_head.project(pack_activations(h), proposals)
        return model.vocabulary_order.token_order(logits).astype(mx.bfloat16)

    def propose(self, model, pending, prefix, _cache, proposals, _reader_enabled):
        logits = self.logits(model, pending, prefix, proposals)
        previous = mx.array([pending], dtype=mx.int32)
        picks = []
        for row in logits:
            scores = row[None] + linear(self.markov_in[previous], self.markov_out)
            previous = mx.argmax(scores, axis=-1).astype(mx.int32)
            picks.append(previous)
        return mx.concatenate(picks)

    def conditional_ranks(self, model, output, prefix):
        # Use the already verified predecessor at each row to measure the best
        # coverage possible if a search could always find the right branch.
        logits = self.logits(model, output[0], prefix, self.config.block_size)[:len(output) - 1]
        previous = mx.array(output[:-1], dtype=mx.int32)
        truth = mx.array(output[1:], dtype=mx.int32)[:, None]
        conditional = logits + linear(self.markov_in[previous], self.markov_out)
        chosen = mx.take_along_axis(conditional, truth, axis=-1)
        tokens = mx.arange(self.config.vocab_size)
        wins = (conditional > chosen) | ((conditional == chosen) & (tokens < truth))
        return (1 + mx.sum(wins, axis=-1)).tolist()


def prepare(model, draft, trace, budget):
    started = perf_counter()
    cache, first, features = prefill(model, draft, trace["input"])
    assert first == trace["output"][0]
    assert features.shape[1] + draft.origin == len(trace["input"])
    prefix = draft.context(features, draft.origin, budget)
    mx.eval(prefix)
    budget.check(force=True)
    print("prefill", json.dumps({
        "seconds": perf_counter() - started, "prefix_rows": features.shape[1], "origin": draft.origin,
        "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics(),
    }), flush=True)
    return cache, first, prefix


def advance(ranks, top):
    accepted = next((i for i, rank in enumerate(ranks) if rank > top), len(ranks))
    return min(len(ranks), accepted + 1)


def replay(rows, cap, top):
    offset = 0
    kept = Counter()
    while offset < len(rows):
        count = min(cap, advance(rows[offset], top))
        kept[count] += 1
        offset += count
    return {"rounds": sum(kept.values()), "mean_advance": len(rows) / sum(kept.values()), "acceptance": kept}


def ranked_branch_advance(ranks, width, max_rank):
    # The oracle expands only the correct predecessor and pays for its
    # higher-scoring siblings. An actual verifier tree must spend nodes
    # on prefixes whose correctness is still unknown.
    remaining = width - 1
    kept = 1
    for rank in ranks:
        if rank > max_rank or rank > remaining:
            break
        remaining -= rank
        kept += 1
    return min(kept, len(ranks))


def minimum_rounds(rows, native, draft_advance):
    count = len(rows)
    rounds = [0] * (count + 1)
    for i in range(count - 1, -1, -1):
        choices = [draft_advance(rows[i])]
        if i in native:
            choices.append(native[i]["advance"])
        rounds[i] = 1 + min(rounds[i + kept] for kept in choices)
    return rounds[0]


def audit(model, draft, trace, prepared, budget):
    cache, first, prompt = prepared
    teacher = [replace(state) for state in cache], first, prompt
    native, prefix = capture_mtp(model, draft, trace, teacher, budget)
    output = trace["output"]
    rows, timings = [], []
    print("audit_native_rounds", len(native), flush=True)
    for pos in range(len(output) - 1):
        end = len(trace["input"]) + pos - draft.origin
        local = [tuple(x[:, :, :end] for x in pair) for pair in prefix]
        started = perf_counter()
        rows.append(draft.conditional_ranks(model, output[pos:pos + draft.config.block_size + 1], local))
        timings.append((perf_counter() - started) * 1000)
        if (pos + 1) % 256 == 0:
            print("audit_anchors", pos + 1, flush=True)
        budget.check()
    deltas = [min(8, advance(rows[i], 1)) - record["advance"] for i, record in native.items()]
    print("audit", json.dumps({
        "anchors": len(rows), "native_rounds": len(native),
        "rank_probe_median_ms": median(timings),
        "observed_rank1": {width: replay(rows, width, 1) for width in (8, 9)},
        "oracle_conditional_rank2": replay(rows, 9, 2), "oracle_conditional_rank4": replay(rows, 9, 4),
        "oracle_minimum_rounds_with_saved_mtp": {
            "dspark_chain8": minimum_rounds(rows, native, lambda ranks: min(8, advance(ranks, 1))),
            "branch8_rank4": minimum_rounds(rows, native, lambda ranks: ranked_branch_advance(ranks, 8, 4)),
            "branch16_rank4": minimum_rounds(rows, native, lambda ranks: ranked_branch_advance(ranks, 16, 4)),
        },
        "longer_prefix_at_native_anchor": {"dspark": sum(n > 0 for n in deltas),
                                            "mtp": sum(n < 0 for n in deltas)},
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("code", "chinese_technical_report", "english_technical_tutorial"),
                        default="code")
    parser.add_argument("--draft-tokens", action="append", type=int, choices=(7, 8))
    parser.add_argument("--audit", action="store_true", help="check conditional candidate ranks at all true prefixes")
    args = parser.parse_args()
    verify_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[args.workload]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    before, thermal = pages(), thermal_state()
    with metal_job(80, 2) as budget:
        model = benchmark_target(budget)
        budget.check(force=True, required_bytes=5 * GIB)
        draft = DSparkDraft(len(trace["input"]))
        assert draft.config.vocab_size == model.vocab_size
        mx.eval(draft.weights)
        budget.admit(model, 5120, 512, False, extra_bytes=2 * GIB)
        print("draft", json.dumps({"workload": args.workload, "taps": draft.config.target_layer_ids,
              "weights_bf16": len(draft.weights), "peak_gib": mx.get_peak_memory() / GIB,
              **budget.metrics()}), flush=True)
        prepared = prepare(model, draft, trace, budget)
        for tokens in args.draft_tokens or (7,):
            decode(model, draft, trace, prepared, tokens + 1, False, budget)
            mx.clear_cache()
            budget.check(force=True)
        if args.audit:
            mx.clear_cache()
            budget.check(force=True)
            audit(model, draft, trace, prepared, budget)
        after = pages()
        print("host", json.dumps({"thermal": [thermal, thermal_state()],
              "faults": {key: after[key] - before[key] for key in before}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

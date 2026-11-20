"""Benchmark the pinned upstream DFlash2 drafter against the local BF16 target."""

import argparse
import ast
from collections import Counter
from copy import copy
import ctypes
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from struct import pack
import subprocess
import sys
from time import perf_counter
from types import ModuleType
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from ultrarun.memory import GIB, metal_job
from ultrarun.model import Model
from ultrarun.packed_projection import pack_activations
from ultrarun.profiles import OPTIMIZED_MODEL
from ultrarun.server import warm_verification_graphs
from ultrarun.spec_tree import VerificationTree
from scripts.dflash2_top16 import bounded_top16, probe_top16


ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHA256 = "2f8598eaca4cb814e63ea69e791c1bdf55ba82280bfd299f9141906356b7cb87"
DRAFT_WEIGHT_BYTES = 3_848_817_896
SOURCE_DEFINITIONS = {
    "DFlashConfig", "_build_rope", "DFlashAttention", "DFlashDecoderLayer",
    "_grouped_dynamic_convolve", "GroupedDynamicCausalConv", "DFlash2DecoderLayer",
    "CandidateSelector", "DFlashDraftModel", "DFlash2DraftModel", "load_draft",
}


class MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SlidingCache:
    """Keep committed BF16 context in chronological order for the sliding mask."""

    def __init__(self, max_size, keep):
        assert keep == 0
        self.offset = 0
        self.max_size = max_size
        self.keys = self.values = None

    def update_and_fetch(self, keys, values):
        self.offset += keys.shape[2]
        if self.keys is not None:
            keep = self.max_size - keys.shape[2]
            start = max(0, self.keys.shape[2] - keep)
            keys = mx.concatenate((self.keys[:, :, start:], keys), axis=2)
            values = mx.concatenate((self.values[:, :, start:], values), axis=2)
        self.keys, self.values = keys, values
        return keys, values


def initialize_rope(*, dims, base, traditional, scaling_config, max_position_embeddings):
    assert scaling_config["rope_type"] == "default"
    return nn.RoPE(dims, traditional=traditional, base=base)


def upstream_module(source_path, draft_path):
    # Execute only the pinned model definitions. Their backend dependencies are
    # provided here so the benchmark can share this repository's MLX environment.
    source = source_path.read_bytes()
    if sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected z-lab/dflash model_mlx.py from commit 07ebd93d")
    tree = ast.parse(source, filename=str(source_path))
    tree.body = [node for node in tree.body if getattr(node, "name", None) in SOURCE_DEFINITIONS]
    selector = next(node for node in tree.body if node.name == "CandidateSelector")
    select = next(node for node in selector.body if getattr(node, "name", None) == "select")
    first = select.body[0]
    assert ast.unparse(first).startswith("candidates = mx.argpartition(")
    first.value = ast.parse("rank_candidates(logits, self.top_k)", mode="eval").body
    ast.fix_missing_locations(tree)
    upstream = ModuleType("_pinned_dflash2_mlx")
    sys.modules[upstream.__name__] = upstream
    vars(upstream).update(
        mx=mx, nn=nn, json=json, dataclass=dataclass, field=field, Any=Any, Path=Path,
        MLP=MLP, initialize_rope=initialize_rope, RotatingKVCache=SlidingCache,
        snapshot_download=lambda *_args, **_kwargs: draft_path,
        rank_candidates=lambda logits, top_k: mx.argpartition(logits, -top_k, axis=-1)[..., -top_k:],
    )
    exec(compile(tree, str(source_path), "exec"), vars(upstream))
    return upstream


def load_draft(source_path, draft_path):
    checkpoint = draft_path / "model.safetensors"
    if list(draft_path.glob("*.safetensors")) != [checkpoint] or checkpoint.stat().st_size != DRAFT_WEIGHT_BYTES:
        raise ValueError("Expected the single Qwen3.8-27B DFlash2 BF16 checkpoint")
    upstream = upstream_module(source_path, draft_path)
    model = upstream.load_draft(str(draft_path))
    assert isinstance(model, upstream.DFlash2DraftModel)
    assert model.config.is_causal is False and set(model.config.layer_types) == {"sliding_attention"}
    assert all(weight.dtype == mx.bfloat16 for _, weight in tree_flatten(model.parameters()))
    return model, upstream


class ObservedLayer:
    def __init__(self, layer, index, observer):
        self.layer = layer
        self.index = index
        self.observer = observer
        self.joined_offset = None

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def prefill_copy(self, *, fast_prefill=False, group_ffn=False):
        prepared = self.layer.prefill_copy(fast_prefill=fast_prefill, group_ffn=group_ffn)
        return ObservedLayer(prepared, self.index, self.observer)

    def forward(self, hidden, cache, offset, *args):
        output = self.layer.forward(hidden, cache, offset, *args)
        self.observer(self.index, offset, output)
        return output

    def attention_result(self, hidden, cache, offset):
        if self.joined_offset is None:
            self.joined_offset = offset
        return self.layer.attention_result(hidden, cache, offset)

    def delta(self, hidden, cache, keep_replay, tree, *, projected_ab=None):
        if projected_ab is not None:
            # The combined DeltaNet prefill runs only at offset zero.
            self.joined_offset = 0
        return self.layer.delta(hidden, cache, keep_replay, tree, projected_ab=projected_ab)

    def ffn_residual(self, hidden):
        output = self.layer.ffn_residual(hidden)
        self.observer(self.index, self.joined_offset, output)
        self.joined_offset = None
        return output


def observe_layers(target, tap_ids, observer):
    for i in tap_ids:
        layer = target.layers[i]
        target.layers[i] = ObservedLayer(layer.layer if isinstance(layer, ObservedLayer) else layer, i, observer)


def bind_draft(target, draft, sample_logits):
    assert draft.config.hidden_size == target.embedding.shape[1]
    assert draft.config.vocab_size == target.vocab_size
    assert draft.config.num_target_layers == len(target.layers)
    assert draft.config.input_embedding_scale == 1.0

    def head(hidden):
        assert hidden.shape[0] == 1 and hidden.dtype == mx.bfloat16
        rows = hidden.shape[1]
        logits = target.packed_head.project(pack_activations(hidden[0]), rows)
        logits = target.vocabulary_order.token_order(logits)
        if rows in (7, 15) and rows not in sample_logits:
            sample_logits[rows] = logits
        return logits[None]

    draft.embed_tokens = lambda ids: target.embedding[ids]
    draft.lm_head = head


def prefill(target, draft, ids):
    tap_ids = draft.config.target_layer_ids
    blocks = {i: [] for i in tap_ids}
    window = draft.config.sliding_window
    start = max(0, len(ids) - window + 1) if window is not None else 0

    def observe(layer, offset, hidden):
        if offset + hidden.shape[0] > start:
            blocks[layer].append((offset, hidden[max(0, start - offset):]))

    observe_layers(target, tap_ids, observe)
    cache = target.cache()
    for _, hidden in target.prefill(mx.array(ids, dtype=mx.int32), cache, 512):
        target.evaluate(hidden, cache)
    first = mx.argmax(target.logits(hidden[-1:])[0]).item()
    features = mx.concatenate([mx.concatenate([array for _, array in sorted(blocks[i])], axis=0)
                               for i in tap_ids], axis=-1)[None]
    mx.eval(features)
    assert features.shape[1] == len(ids) - start and features.dtype == mx.bfloat16
    for layer, state in zip(target.layers, cache):
        state.logical_attention = layer.fused_attention_io
        state.compile_delta_tree = layer.gdn
        if state.logical_attention and layer.reserve_kv:
            state.prompt_kv = state.keys[:, :, :len(ids)], state.values[:, :, :len(ids)]
            state.keys = state.values = None
    return cache, first, features


def pages():
    stats = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=2)
    return {key: int(value.strip().rstrip(".")) for line in stats.splitlines()
            if ":" in line for key, value in [line.split(":", 1)]
            if key in ("Pageouts", "Swapins", "Swapouts")}


def read_thermal_state():
    foundation = ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    for lookup in (objc.objc_getClass, objc.sel_registerName):
        lookup.argtypes = (ctypes.c_char_p,)
        lookup.restype = ctypes.c_void_p
    message = objc.objc_msgSend
    message.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    message.restype = ctypes.c_void_p
    info = message(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
    state = message(info, objc.sel_registerName(b"thermalState")) or 0
    return ("nominal", "fair", "serious", "critical")[state]


def first_rejection_coverage(rejected):
    rejects = Counter(depth for depth, _ in rejected)
    recovered = Counter()
    if rejected:
        hits = mx.stack([covered for _, covered in rejected]).tolist()
        for (depth, _), hit in zip(rejected, hits):
            recovered[depth] += hit
    return {depth: {"rejected": rejects[depth], "target_in_top16": recovered[depth]} for depth in sorted(rejects)}


def saved_verifier_route(baseline, output_index):
    for offset, round_info in baseline.get("rounds", {}).items():
        first = int(offset) - len(baseline["input"]) + 1
        row = output_index - first
        if 0 <= row < len(round_info["path"]):
            node = round_info["path"][row]
            index = round_info["order"].index(node)
            assert round_info["prediction"][index] == baseline["output"][output_index]
            return {"width": len(round_info["order"]), "row": index,
                    "margin_between_tile_winners": round_info["tile_margin"][index]}
    return None


def describe_divergence(target, baseline, output, observation):
    if observation is None:
        return None
    index, row, hidden = observation
    tokens = (output[index], baseline["output"][index])
    packed = pack_activations(mx.fast.rms_norm(hidden, target.norm, target.config["rms_norm_eps"]).astype(hidden.dtype))
    scores = target.packed_head.project(packed, hidden.shape[0])
    chosen, expected = scores[row, target.vocabulary_order.to_storage(mx.array(tokens, dtype=mx.int32))].tolist()
    return {
        "output_index": index, "chosen_token": tokens[0], "expected_token": tokens[1],
        "chosen_logit": chosen, "expected_logit": expected, "logit_gap": chosen - expected,
        "verification_width": hidden.shape[0], "verification_row": row,
        "saved_mtp_route": saved_verifier_route(baseline, index),
    }


def decode(target, draft, baseline, width, cache, first, features, budget):
    taps = draft.config.target_layer_ids
    captured = dict.fromkeys(taps)

    def observe(layer, _offset, hidden):
        captured[layer] = hidden

    observe_layers(target, taps, observe)
    output = [first]
    expected = baseline["output"]
    offset = len(baseline["input"])
    conv_taps = target.config["linear_conv_kernel_dim"] - 1
    draft_cache = draft.make_cache()
    for state in draft_cache:
        state.offset = offset - features.shape[1]
    rounds = Counter()
    rejected = []
    draft_digest = sha256()
    divergence = None
    draft_s = verify_s = commit_s = 0.0
    before = pages()
    thermal_before = read_thermal_state()
    wall_start = perf_counter()
    while len(output) < len(expected):
        remaining = len(expected) - len(output)
        block = min(width, remaining + 1)
        anchors = mx.array([[output[-1]] + [draft.config.mask_token_id] * (block - 1)], dtype=mx.int32)
        started = perf_counter()
        proposals, candidates, _ = draft.propose(anchors, features, draft_cache, 0, logits_start=1)
        proposed = proposals[0].tolist()
        draft_s += perf_counter() - started
        started = perf_counter()
        inputs = mx.concatenate((anchors[0, :1], proposals[0]))
        # Use the production verifier's compact kernels with a single causal path.
        tree = VerificationTree([-1, *range(block - 1)], offset, conv_taps, spine=block - 1)
        hidden = target.hidden(inputs, cache, offset, keep_replay=True, tree=tree)
        predicted = target.predictions(hidden)
        target.evaluate([predicted, *captured.values()], cache)
        verified = predicted.tolist()
        verify_s += perf_counter() - started
        started = perf_counter()
        matched = next((i for i, pair in enumerate(zip(proposed, verified)) if pair[0] != pair[1]), len(proposed))
        draft_digest.update(pack(f"<{block}I", len(proposed), *proposed))
        confirmed = (proposed[:matched] + [verified[matched]])[:remaining]
        keep = len(confirmed)
        if divergence is None:
            row = next((i for i, token in enumerate(confirmed) if token != expected[len(output) + i]), None)
            if row is not None:
                divergence = len(output) + row, row, hidden
        output.extend(confirmed)
        rounds[keep] += 1
        target.accept_tree(cache, tree, offset, list(range(keep)))
        offset += keep
        features = mx.concatenate([captured[i][:keep] for i in taps], axis=-1)[None]
        commit = [features]
        if candidates is not None and matched < len(proposed):
            # Carry one scalar forward instead of holding a view of the full vocabulary sort.
            covered = mx.any(candidates[0, matched] == verified[matched])
            rejected.append((matched, covered))
            commit.append(covered)
        target.evaluate(commit, cache)
        budget.check()
        commit_s += perf_counter() - started
        if len(output) // 256 != (len(output) - keep) // 256:
            print("progress", width, len(output), sum(rounds.values()), flush=True)
    seconds = perf_counter() - wall_start
    thermal_after = read_thermal_state()
    after = pages()
    count = sum(rounds.values())
    decode_peak = mx.get_peak_memory() / GIB
    drift = describe_divergence(target, baseline, output, divergence)
    return {
        "width": width, "input_tokens": len(baseline["input"]), "output_tokens": len(output),
        "decode_s": seconds, "decode_tok_s": (len(output) - 1) / seconds,
        "proposal_sha256": draft_digest.hexdigest(),
        "rounds": count, "confirmed_per_round": (len(output) - 1) / count,
        "confirmed_per_round_required_at_100": 100 * seconds / count,
        "confirmed_histogram": rounds,
        "first_rejections_by_proposal_index": first_rejection_coverage(rejected),
        "draft_ms": 1000 * draft_s / count, "verify_ms": 1000 * verify_s / count,
        "commit_ms": 1000 * commit_s / count,
        "matches_baseline": output == expected,
        "first_divergence": drift,
        "token_sha256": sha256(json.dumps(output, separators=(",", ":")).encode()).hexdigest(),
        "decode_peak_gib": decode_peak,
        "thermal_before": thermal_before, "thermal_after": thermal_after,
        "host_delta": {key: after[key] - before[key] for key in before},
        **budget.metrics(),
    }


def select_top16_retest(results, timings):
    projected = []
    for width, result in results.items():
        timing = timings[width - 1]
        saved_ms = timing["full_sort"] - timing["tiled"]
        round_ms = 1000 * result["decode_s"] / result["rounds"]
        if 0.01 * round_ms <= saved_ms < round_ms:
            projected.append((result["confirmed_per_round"] * 1000 / (round_ms - saved_ms), width))
    return max(projected)[1] if projected else None


def tiled_candidates(logits, top_k):
    assert logits.shape[0] == 1 and top_k == 16
    return bounded_top16(logits[0])[None]


def load_target(budget):
    # Initialize the deployed vocabulary layout, then release the unused native drafter.
    options = {**OPTIMIZED_MODEL, "lossless_mtp_ffn": False}
    target = Model(ROOT / "weights/Qwen3.8-27B", mtp=True, budget=budget, **options)
    active_with_mtp = mx.get_active_memory()
    del target.mtp_layer, target.mtp_fc, target.mtp_norms
    mx.clear_cache()
    active_without_mtp = mx.get_active_memory()
    print("target_load", json.dumps({
        "peak_gib": mx.get_peak_memory() / GIB,
        "active_gib": active_without_mtp / GIB,
        "mtp_released_gib": (active_with_mtp - active_without_mtp) / GIB,
    }), flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=ROOT / "runs/frontier-target-margin-audit.json")
    parser.add_argument("--draft", type=Path, default=ROOT / "weights/Qwen3.8-27B-DFlash2")
    parser.add_argument("--case", default="chinese_technical_report")
    parser.add_argument("--widths", type=int, nargs="+", choices=(8, 16), default=[8, 16])
    args = parser.parse_args()
    trace = json.loads(args.trace.read_bytes())[args.case]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    with metal_job(80, 2) as budget:
        target = load_target(budget)
        budget.admit(target, 5120, 512, False, extra_bytes=6 * GIB)
        draft, upstream = load_draft(args.upstream, args.draft)
        sample_logits = {}
        bind_draft(target, draft, sample_logits)
        budget.admit(target, 5120, 512, False, extra_bytes=2 * GIB)
        warm_verification_graphs(target, budget)
        print("loaded", json.dumps({"load_peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
        mx.reset_peak_memory()
        started = perf_counter()
        cache, first, features = prefill(target, draft, trace["input"])
        prefill_peak = mx.get_peak_memory() / GIB
        prefill_s = perf_counter() - started
        assert first == trace["output"][0]
        print("prefill", args.case, first, features.shape, prefill_s, prefill_peak, flush=True)
        results = {}
        for width in dict.fromkeys(args.widths):
            mx.clear_cache()
            mx.reset_peak_memory()
            result = decode(target, draft, trace, width, [copy(state) for state in cache], first, features, budget)
            result.update(prefill_s=prefill_s, prefill_peak_gib=prefill_peak)
            print("result", args.case, json.dumps(result), flush=True)
            results[width] = result
        try:
            timings = probe_top16(sample_logits)
        except (AssertionError, RuntimeError) as error:
            print("top16_probe_error", str(error), flush=True)
        else:
            print("top16_probe_ms", json.dumps(timings), flush=True)
            width = select_top16_retest(results, timings)
            if width is not None:
                upstream.rank_candidates = tiled_candidates
                mx.clear_cache()
                mx.reset_peak_memory()
                result = decode(target, draft, trace, width, [copy(state) for state in cache], first, features, budget)
                baseline = results[width]
                fields = ("proposal_sha256", "token_sha256", "confirmed_histogram", "rounds")
                result["matches_full_sort"] = all(result[key] == baseline[key] for key in fields)
                print("top16_result", args.case, json.dumps(result), flush=True)
        budget.check(force=True)


if __name__ == "__main__":
    main()

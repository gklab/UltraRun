"""Measure narrowing Tandem's draft queries with full BF16 target verification."""

import argparse
from collections import Counter
from copy import copy
from hashlib import sha256
import json
from pathlib import Path

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.server import warm_verification_graphs
from scripts.bench_dflash2 import ROOT, bind_draft, load_draft, load_target, pages, prefill, tiled_candidates
from scripts.bench_dflash2_packed import fork_site_allocator, fork_site_runner, pack_variants, trimmed_output
from scripts.bench_dflash2_tandem import draft_runner, exercise, streams_at_prefix, verified_directories
from scripts.dflash2_ranked_head import lattice_for
from scripts.tandem_qkv_prefix import kernel_factory, project_prefix
from scripts.tandem_overlap import Overlap
from scripts.tandem_rank_calibration import RankCalibration


CAP = 8


class PrefixQueryLayer:
    """Compute only the used queries and keep all proposal keys and values."""

    def __init__(self, layer, *, retain_tail, projection=None):
        self.layer = layer
        self.retain_tail = retain_tail
        self.projection = projection
        assert layer.attention_conv.kernel_size == layer.mlp_conv.kernel_size == 2

    def __call__(self, x, ctx, rope, cache):
        layer = self.layer
        normalized = layer.input_layernorm(x)
        values, kernel = layer.attention_conv.prepare(normalized)
        mixed = layer.self_attn(values, ctx, rope, cache, query_rows=CAP, prefix_projection=self.projection)
        prefix = x[:, :CAP] + layer.attention_conv.finish(mixed, kernel[:, :CAP])
        values, kernel = layer.mlp_conv.prepare(layer.post_attention_layernorm(prefix))
        result = prefix + layer.mlp_conv.finish(layer.mlp(values), kernel)
        if self.retain_tail:
            # These proposal rows are not scored. Their residual is still read
            # as K/V by the next bidirectional attention layer.
            return mx.concatenate((result, x[:, CAP:]), axis=1)
        return result


def variants(draft, *, compare_fused_qkv, compare_draft_widths, compare_calibration):
    if compare_calibration:
        return {"uncalibrated": draft, "calibrated": draft}
    final = copy(draft)
    final.layers = [*draft.layers[:-1], PrefixQueryLayer(draft.layers[-1], retain_tail=False)]
    if compare_draft_widths:
        return {"block16": final, "block8": draft}
    if compare_fused_qkv:
        fused = copy(final)
        last = PrefixQueryLayer(draft.layers[-1], retain_tail=False, projection=project_prefix)
        fused.layers = [*final.layers[:-1], last]
        return {"split_qkv": final, "fused_qkv": fused}
    early = copy(final)
    early.layers = [*final.layers[:-2], PrefixQueryLayer(final.layers[-2], retain_tail=True), final.layers[-1]]
    return {"prefix_output": trimmed_output(draft, CAP), "prefix_query": final, "prefix_penultimate": early}


def check_initial_logits(drafts, blocks, prefilled, context, *, require_exact=False, rows=CAP - 1):
    _, first, features = prefilled
    reference = None
    for label, draft in drafts.items():
        caches = draft.make_cache()
        for state in caches:
            state.offset = context - features.shape[1]
        block = blocks[label]
        tokens = mx.array([[first] + [draft.config.mask_token_id] * (block - 1)], dtype=mx.int32)
        hidden = draft.hidden_states(tokens, features, caches, logits_start=1)
        logits = draft.compute_logits(hidden)
        mx.eval(logits)
        assert logits.shape[1] == rows and all(state.offset == context for state in caches)
        if reference is None:
            reference = logits
            continue
        bitwise = mx.array_equal(reference.view(mx.uint16), logits.view(mx.uint16)).item()
        same = mx.sum(mx.argmax(reference, axis=-1) == mx.argmax(logits, axis=-1)).item()
        print("draft_prefix", json.dumps({"arm": label, "bitwise": bitwise,
                                           "matching_top1": same, "rows": rows}), flush=True)
        if require_exact:
            assert bitwise, "The fused projection changed the first draft logits"


def check_fused_projection(attention):
    features = attention.original.q_proj.weight.shape[-1]
    for width in (9, 11, 16):
        x = mx.random.normal((1, width, features)).astype(mx.bfloat16)
        split = attention.split_prefix_projection(x, CAP)
        fused = project_prefix(attention, x, CAP)
        for a, b in zip(split, fused, strict=True):
            assert mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item(), width
        print("qkv_projection", json.dumps({"width": width, "bitwise": True}), flush=True)


def overlap_runner(block, lattice, overlap, route):
    allocate = fork_site_allocator(skip_unavailable=True)

    def merged(draft, upstream, token, features, cache, checked):
        choices, scores = lattice(draft, upstream, token, features, cache, checked)
        return overlap.merge(token, choices, scores)

    return draft_runner(block, route_digest=route, lattice=merged, allocate=allocate, observe=overlap.observe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/private/tmp/localllm-ref/dflash2/model_mlx.py"))
    parser.add_argument("--workload", choices=("code", "english_technical_tutorial", "chinese_technical_report"),
                        default="code")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare-fused-qkv", action="store_true",
                            help="compare two prefix query paths with the same attention shapes")
    comparison.add_argument("--compare-draft-widths", action="store_true",
                            help="run the original BF16 draft with eight versus sixteen input positions")
    comparison.add_argument("--compare-calibration", action="store_true",
                            help="estimate rank reliability online for the eight-position draft")
    comparison.add_argument("--compare-overlap", action="store_true",
                            help="merge aligned scores from the previous and fresh 16-position drafts")
    args = parser.parse_args()
    if args.compare_fused_qkv:
        kernel_factory()
    path = verified_directories(("tandem_b16",))["tandem_b16"]
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[args.workload]
    assert (len(trace["input"]), len(trace["output"])) == (4096, 1024)
    print("workload", args.workload, flush=True)
    before = pages()
    with metal_job(64, 2) as budget:
        model = load_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=8 * GIB)
        stock, upstream = load_draft(args.upstream, path)
        assert stock.config.block_size == 16
        assert stock.config.output_multiplier == 1.0 and not stock.config.final_logit_softcapping
        upstream.rank_candidates = tiled_candidates
        bind_draft(model, stock, {})
        packed = pack_variants(stock, budget)[1]
        if args.compare_overlap:
            drafts = dict.fromkeys(("fresh", "overlap"), packed)
        else:
            drafts = variants(packed, compare_fused_qkv=args.compare_fused_qkv,
                              compare_draft_widths=args.compare_draft_widths, compare_calibration=args.compare_calibration)
        cap = stock.config.block_size if args.compare_overlap else CAP
        blocks = {name: CAP if name == "block8" or args.compare_calibration else stock.config.block_size
                  for name in drafts}
        if args.compare_fused_qkv:
            check_fused_projection(packed.layers[-1].self_attn)
        warm_verification_graphs(model, budget)
        layout = model.fused_attention_io
        model.fused_attention_io = False
        try:
            prefilled = prefill(model, stock, trace["input"])
        finally:
            model.fused_attention_io = layout
        assert prefilled[1] == trace["output"][0]
        check_initial_logits(drafts, blocks, prefilled, len(trace["input"]),
                             require_exact=args.compare_fused_qkv or args.compare_calibration or args.compare_overlap,
                             rows=cap - 1)
        print("prepare", json.dumps({"peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
        lattices = {size: lattice_for(model, size) for size in set(blocks.values())}

        def branches():
            routes, counts = {name: sha256() for name in drafts}, {name: Counter() for name in drafts}
            calibration = RankCalibration() if args.compare_calibration else None
            overlap = Overlap() if args.compare_overlap else None
            arms = {}
            for name, draft in drafts.items():
                block = blocks[name]
                if name == "overlap":
                    run = overlap_runner(block, lattices[block], overlap, routes[name])
                else:
                    run = fork_site_runner(block, counts[name], route_digest=routes[name], lattice=lattices[block],
                                           calibration=calibration if name == "calibrated" else None)
                arms[name] = draft, upstream, [False], run
            return arms, routes, counts, calibration, overlap

        caps = dict.fromkeys(drafts, cap)
        warm = streams_at_prefix(model, branches()[0], trace, prefilled, caps, budget, 65)
        ready, _ = exercise(warm, 65, report_progress=False)
        assert all(result["matches_reference"] for result in ready.values())
        mx.clear_cache()
        budget.check(force=True)
        mx.reset_peak_memory()
        thermal, live_before = thermal_state(), pages()
        arms, routes, counts, calibration, overlap = branches()
        streams = streams_at_prefix(model, arms, trace, prefilled, caps, budget, 1024)
        results, elapsed = exercise(streams, 1024, report_progress=True)
        for name, row in results.items():
            calls = overlap.stats["rounds"] if name == "overlap" else counts[name]["calls"]
            assert row["matches_reference"] and calls == row["rounds"]
            if name == "overlap":
                row["oracle_union_histogram_at_live_anchors"] = row.pop("oracle_top16_histogram_at_live_anchors")
                row["first_missing_outside_union"] = row["first_missing_conditional_rank"].pop("outside_top16", 0)
            print("result", json.dumps({"arm": name, "decode_s": elapsed[name],
                                        "decode_tok_s": row["tokens"] / elapsed[name],
                                        "draft_ms_per_round": 1000 * row["phase_s"]["draft"] / row["rounds"],
                                        "route_digest": routes[name].hexdigest(), **row}), flush=True)
        if args.compare_fused_qkv:
            assert len({route.hexdigest() for route in routes.values()}) == 1
        if calibration is not None:
            print("calibration", json.dumps(calibration.metrics()), flush=True)
        if overlap is not None:
            print("overlap", json.dumps(overlap.stats), flush=True)
        after = pages()
        print("host", json.dumps({"peak_gib": mx.get_peak_memory() / GIB,
              "thermal": [thermal, thermal_state()], "faults": {k: after[k] - n for k, n in before.items()},
              "decode_faults": {k: after[k] - n for k, n in live_before.items()}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

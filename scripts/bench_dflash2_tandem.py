"""Compare two BF16 DFlash2 heads using the same original target and tree cap."""

import argparse
from copy import copy
from hashlib import file_digest
import inspect
import json
from pathlib import Path
from time import perf_counter
from types import FunctionType

import mlx.core as mx

from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from localllm.server import warm_verification_graphs
from scripts.bench_dflash2 import ROOT, bind_draft, load_draft, load_target, pages, prefill, tiled_candidates

# Use the same lattice, two-fork allocator and exact verifier as the b32 experiment.
import run_dflash2_b32 as reference


DRAFTS = {
    "dflash2_b32": (
        "Qwen3.8-27B-DFlash2-b32", 32,
        "6d211c63896c3d5b777a86e7386840af6d982d5b4ee3887a773ab7f34b48420d",
        "01e8c919665f563339ff41ac788497f483da5418c860c051b5fee17b801b7423",
    ),
    "tandem_b16": (
        "Qwen3.8-27B-Tandem-DFlash2-b16", 16,
        "ee3e1e6d297a2f0cdd270d7f5ea150e67ae87bf1d90d314078345eb2bdf32202",
        "843d7698e043118863847a560b0375e93db253f00352282d20ff0d6b6816356a",
    ),
}


def verified_directories(names):
    paths = {}
    for name in names:
        directory, _, config_hash, weights_hash = DRAFTS[name]
        path = ROOT / "weights" / directory
        for filename, digest in (("config.json", config_hash), ("model.safetensors", weights_hash)):
            with (path / filename).open("rb") as stream:
                if file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Unexpected artifact digest for {name}/{filename}")
        paths[name] = path
    return paths


def draft_runner(block, *, route_digest=None, allocate=None, lattice=None, observe=None, tree_factory=None):
    # Give each generator its own training width while preserving the scorer
    # and traversal used by the published baseline run in this workspace.
    scope = {**vars(reference), "DRAFT_WIDTH": block}
    if tree_factory is not None:
        scope["VerificationTree"] = tree_factory
    scope["lattice"] = FunctionType(reference.lattice.__code__, scope) if lattice is None else lattice
    if allocate is None:
        allocate = reference.candidate_tree
    if route_digest is not None:
        def tracked_tree(*args):
            tree = allocate(*args)
            route_digest.update(json.dumps(tree, separators=(",", ":")).encode())
            return tree

        scope["candidate_tree"] = tracked_tree
    else:
        scope["candidate_tree"] = allocate
    if observe is None:
        return FunctionType(reference.play.__code__, scope)
    source = inspect.getsource(reference.play)
    boundary = "        depth=keep-1"
    assert source.count(boundary) == 1
    source = source.replace(boundary, "        observe_acceptance(options, logp, tokens, path, predictions)\n" + boundary)
    scope["observe_acceptance"] = observe
    exec(compile(source, __file__, "exec"), scope)
    return scope["play"]


def copy_verifier(model):
    view = copy(model)
    view.layers = list(model.layers)
    return view


def streams_at_prefix(model, arms, trace, prefilled, caps, budget, limit):
    cache, first, features = prefilled
    return {
        name: run(copy_verifier(model), draft, trace, cache, first, features, caps[name], budget,
                  upstream, checked, limit)
        for name, (draft, upstream, checked, run) in arms.items()
    }


def exercise(streams, completion, *, report_progress):
    labels = tuple(streams)
    elapsed = dict.fromkeys(labels, 0.0)
    cursors = dict.fromkeys(labels, 1)
    completed = {}
    mark = 256
    first = 0
    try:
        while streams:
            arm = min(streams, key=lambda name: (cursors[name], (labels.index(name) - first) % len(labels)))
            first = (labels.index(arm) + 1) % len(labels)
            started = perf_counter()
            try:
                cursors[arm] = next(streams[arm])
            except StopIteration as done:
                completed[arm] = done.value
                cursors[arm] = completion
                del streams[arm]
            elapsed[arm] += perf_counter() - started
            if report_progress and min(cursors.values()) >= mark:
                print("progress", json.dumps({"tokens": cursors, "seconds": elapsed,
                                              "thermal": thermal_state()}), flush=True)
                mark += 256
    finally:
        for stream in streams.values():
            stream.close()
    return completed, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("code", "chinese_technical_report", "english_technical_tutorial"),
                        default="code")
    parser.add_argument("--cap", type=int, choices=(8, 16), default=16)
    parser.add_argument("--compare-widths", action="store_true", help="run only Tandem with verifier caps 8 and 16")
    parser.add_argument("--upstream", type=Path, default=Path("/private/tmp/localllm-ref/dflash2/model_mlx.py"))
    args = parser.parse_args()
    paths = verified_directories(("tandem_b16",) if args.compare_widths else DRAFTS)
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[args.workload]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    before = pages()
    with metal_job(80, 2) as budget:
        base = load_target(budget)
        budget.admit(base, 5120, 512, False, extra_bytes=(6 if args.compare_widths else 10) * GIB)
        # The b32 draft's 31-row vocabulary projection needs the same exact
        # 16+15 panel path as the existing b32 experiment.
        model = reference.expanded(base)
        arms = {}
        for name, path in paths.items():
            draft, upstream = load_draft(args.upstream, path)
            assert draft.config.block_size == DRAFTS[name][1]
            upstream.rank_candidates = tiled_candidates
            bind_draft(model, draft, {})
            arms[name] = draft, upstream, [False], draft_runner(draft.config.block_size)
            budget.check(force=True, required_bytes=2 * GIB)
        first_draft = next(iter(arms.values()))[0]
        for draft, _, _, _ in arms.values():
            assert draft.config.target_layer_ids == first_draft.config.target_layer_ids
            assert draft.config.sliding_window == first_draft.config.sliding_window
        caps = dict.fromkeys(arms, args.cap)
        if args.compare_widths:
            tandem = arms["tandem_b16"]
            arms = {"tandem_cap8": tandem, "tandem_cap16": tandem}
            caps = {"tandem_cap8": 8, "tandem_cap16": 16}
        warm_verification_graphs(base, budget)
        layout = base.fused_attention_io
        base.fused_attention_io = False
        try:
            prefilled = prefill(base, first_draft, trace["input"])
        finally:
            base.fused_attention_io = layout
        assert prefilled[1] == trace["output"][0]
        print("prefill", json.dumps({"peak_gib": mx.get_peak_memory() / GIB,
                                     "features_shape": prefilled[2].shape, **budget.metrics()}), flush=True)
        warm = streams_at_prefix(model, arms, trace, prefilled, caps, budget, 65)
        checked, _ = exercise(warm, 65, report_progress=False)
        assert all(result["matches_reference"] for result in checked.values())
        mx.clear_cache()
        budget.check(force=True)
        mx.reset_peak_memory()
        thermal, start_faults = thermal_state(), pages()
        streams = streams_at_prefix(model, arms, trace, prefilled, caps, budget, 1024)
        results, elapsed = exercise(streams, 1024, report_progress=True)
        for name, stats in results.items():
            seconds = elapsed[name]
            print("result", json.dumps({
                "draft": name, "workload": args.workload, **stats, "decode_s": seconds,
                "decode_tok_s": stats["tokens"] / seconds,
                "advance_required_at_100": 100 * seconds / stats["rounds"],
            }), flush=True)
        assert all(result["matches_reference"] for result in results.values())
        after = pages()
        print("memory", json.dumps({
            "peak_gib": mx.get_peak_memory() / GIB, "thermal": [thermal, thermal_state()],
            "decode_faults": {key: val - start_faults[key] for key, val in after.items()},
            "whole_faults": {key: val - before[key] for key, val in after.items()}, **budget.metrics(),
        }), flush=True)


if __name__ == "__main__":
    main()

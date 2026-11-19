"""Compare 32-position FFN walks on one BF16 model and one DFlash2 request."""

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.spec_tree import VerificationTree
from scripts.bench_dflash2 import bind_draft, load_draft, load_target, pages, prefill, tiled_candidates
from scripts.bench_ready import ROOT
from scripts.bench_wide_projection_walk import walking_projection

# These two helpers provide the existing wide verifier and pinned DFlash2-b32 runner.
import run_dflash2_b32 as dflash
import verify_bf16_wide_panels as wide


class WalkingPaired(wide.PanelPaired):
    def project(self, x, tokens, *, swiglu=False):
        if tokens == 32 and swiglu:
            return walking_projection(self, x, tokens, 16, paired=True)
        return super().project(x, tokens, swiglu=swiglu)


class WalkingDown(wide.PanelDown):
    def partials(self, x, tokens):
        if tokens == 32:
            return walking_projection(self, x, tokens, 16, paired=False)
        return super().partials(x, tokens)


def walking_model(base):
    # The wide experiment installs these classes in an isolated copy of model.py.
    # Reuse all arrays and let the copied compiled layer graphs see the test kernels.
    original = wide.PanelPaired, wide.PanelDown
    wide.PanelPaired, wide.PanelDown = WalkingPaired, WalkingDown
    try:
        return wide.expanded(base)
    finally:
        wide.PanelPaired, wide.PanelDown = original


def fixed_tree(models, prefix, trace, budget):
    offset = len(trace["input"])
    tokens = mx.array(trace["output"][:32], dtype=mx.int32)
    parents = [-1, *range(23), 3, 24, 25, 26, 11, 28, 29, 30]
    tree = VerificationTree(parents, offset, 3, spine=24)
    path = [*range(12), 28, 29, 30, 31]
    follow = mx.array(trace["output"][32:40], dtype=mx.int32)
    through = offset + len(path)
    for deferred in (False, True):
        left, right = [wide.snapshot_after(model, prefix, tokens, offset, single=True,
                                          tree=tree, keep=path, defer=deferred) for model in models]
        assert all(wide.same(a, b) for a, b in zip(left[:2], right[:2], strict=True))
        if not deferred:
            assert not any(wide.state_parity(left[2], right[2], through=through).values())
        next_left, next_right = [wide.snapshot_after(models[0], accepted[2], follow, through)
                                 for accepted in (left, right)]
        assert all(wide.same(a, b) for a, b in zip(next_left[:2], next_right[:2], strict=True))
        assert not any(wide.state_parity(next_left[2], next_right[2], through=through + len(follow)).values())
    elapsed = [[], []]
    for i in range(12):
        for arm in (i % 2, 1 - i % 2):
            started = perf_counter()
            wide.snapshot_after(models[arm], prefix, tokens, offset, single=True, tree=tree)
            elapsed[arm].append(1000 * (perf_counter() - started))
        budget.check()
    before, after = elapsed
    print("fixed", json.dumps({
        "split_ms": median(before), "walking_ms": median(after),
        "paired_delta_ms": median(b - a for a, b in zip(before, after, strict=True)),
        "hidden_and_predictions_bitwise": True, "immediate_and_deferred_commit_bitwise": True,
        "thermal": thermal_state(),
    }), flush=True)


def live_pair(models, draft, upstream, trace, prompt, budget):
    cache, first, features = prompt
    checked = [False]
    streams = [dflash.play(model, draft, trace, cache, first, features, 32, budget, upstream, checked, 1024)
               for model in models]
    measurements = [0.0, 0.0]
    completed = [None, None]
    rounds = 0
    before, thermal = pages(), thermal_state()
    mx.reset_peak_memory()
    try:
        while not all(result is not None for result in completed):
            positions = [None, None]
            for i in (rounds % 2, 1 - rounds % 2):
                if completed[i] is not None:
                    continue
                started = perf_counter()
                try:
                    positions[i] = next(streams[i])
                except StopIteration as stop:
                    completed[i] = stop.value
                measurements[i] += perf_counter() - started
            assert positions[0] == positions[1], ("Verifier advanced by a different number of tokens", positions)
            rounds += 1
            if rounds % 64 == 0:
                print("live_progress", rounds, positions[0], thermal_state(), flush=True)
    finally:
        for stream in streams:
            stream.close()
    for label, result, seconds in zip(("split", "walk16"), completed, measurements, strict=True):
        assert result["matches_reference"]
        print("live", json.dumps({"arm": label, **result, "decode_s": seconds,
              "decode_tok_s": result["tokens"] / seconds}), flush=True)
    assert completed[0]["sha256"] == completed[1]["sha256"]
    after = pages()
    print("live_memory", json.dumps({"thermal": [thermal, thermal_state()],
          "peak_gib": mx.get_peak_memory() / GIB, "faults": {key: after[key] - val for key, val in before.items()},
          **budget.metrics()}), flush=True)


def main():
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    before = pages()
    with metal_job(80, 2) as budget:
        base = load_target(budget)
        budget.admit(base, 5120, 512, False, extra_bytes=7 * GIB)
        split, walk = wide.expanded(base), walking_model(base)
        for a, b in zip(split.layers, walk.layers, strict=True):
            assert a.w["ffn_in"].payload is b.w["ffn_in"].payload
            assert a.w["mlp.down_proj.weight"].payload is b.w["mlp.down_proj.weight"].payload
        draft, upstream = load_draft(Path("/private/tmp/localllm-ref/dflash2/model_mlx.py"),
                                     ROOT / "weights/Qwen3.8-27B-DFlash2-b32")
        assert draft.config.block_size == 32
        upstream.rank_candidates = tiled_candidates
        bind_draft(split, draft, {})
        layout = base.fused_attention_io
        base.fused_attention_io = False
        try:
            prompt = prefill(base, draft, trace["input"])
        finally:
            base.fused_attention_io = layout
        assert prompt[1] == trace["output"][0]
        fixed_tree((split, walk), prompt[0], trace, budget)
        mx.clear_cache()
        budget.check(force=True)
        print("preparation_peak_gib", mx.get_peak_memory() / GIB, flush=True)
        live_pair((split, walk), draft, upstream, trace, prompt, budget)
        after = pages()
        print("host", json.dumps({key: after[key] - val for key, val in before.items()}), flush=True)


if __name__ == "__main__":
    main()

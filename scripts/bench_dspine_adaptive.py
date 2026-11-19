"""Compare fixed and confidence-based DSpine verification widths on code."""

import argparse
from collections import Counter
from copy import copy
from dataclasses import replace
import json

import mlx.core as mx

from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from scripts.audit_dspine_complement import capture_mtp
from scripts.bench_dflash2 import pages
from scripts.bench_dspine import (
    DSpineDraft, ROOT, confirmation_ratio, verify_checkpoint, with_packed_ffn,
    with_packed_projections, with_prompt_vocabulary,
)
from scripts.bench_dspine_ffn import paired_decode
from scripts.bench_ready import benchmark_target, prepare_prefix


def phase_costs():
    def record(file, label=None):
        results = (json.loads(line.removeprefix("result ")) for line in (ROOT / "runs" / file).read_text().splitlines()
                   if line.startswith("result "))
        return next(row for row in results if label is None or row["draft_backend"] == label)

    narrow = record("dspine-lossless-ffn-chain8.log")
    wide = record("dspine-short-vocab-ab.log", "short_draft_vocab")
    fixed_overhead = (wide["phases_s"]["draft"] + wide["phases_s"]["commit"]) * 1000 / wide["rounds"]
    return {row["width"]: fixed_overhead + row["phases_s"]["target"] * 1000 / row["rounds"]
            for row in (narrow, wide)}


def advance(nodes, parents, truth, cap):
    children = {(parent, token): i for i, (parent, token) in enumerate(zip(parents[:cap], nodes[:cap], strict=True)) if i}
    root, matched = 0, 0
    for token in truth:
        if (root := children.get((root, token))) is None:
            break
        matched += 1
    return min(len(truth), matched + 1)


def measure_options(model, draft, trace, prefix, budget):
    ids = trace["output"]
    prompt_length = len(trace["input"])
    rows = []
    for anchor, pending in enumerate(ids[:-1]):
        remaining = len(ids) - 1 - anchor
        offset = prompt_length + anchor
        local = [tuple(x[:, :, :offset] for x in pair) for pair in prefix]
        nodes, parents, costs = draft.ranked_tree(model, pending, local, min(16, remaining + 1))
        truth = ids[anchor + 1:anchor + 17]
        narrow = advance(nodes, parents, truth, 8)
        wide = advance(nodes, parents, truth, 16)
        assert 1 <= narrow <= min(8, remaining) and narrow <= wide <= min(16, remaining)
        rows.append((narrow, wide, confirmation_ratio(costs)))
        if len(rows) % 256 == 0:
            print("option_anchors", len(rows), flush=True)
        budget.check()
    return rows


def replay(options, costs):
    cutoff = costs[16] / costs[8]
    count = len(options)

    def play(select):
        i, spent = 0, 0.0
        widths = Counter()
        while i < count:
            narrow, wide, ratio = options[i]
            width = select(i, ratio) if count - i >= 8 else 8
            spent += costs[width]
            widths[width] += 1
            i += wide if width == 16 else narrow
        return {"ms": spent, "tok_s": count * 1000 / spent, "widths": widths}

    # This bound can see future acceptance. It cannot shorten a committed prefix.
    future = [0.0] * (count + 1)
    choice = [8] * count
    for i in reversed(range(count)):
        narrow, wide, _ = options[i]
        take_narrow = (costs[8] + future[i + narrow], 8)
        take_wide = (costs[16] + future[i + wide], 16)
        future[i], choice[i] = min(take_narrow, take_wide)
    return {"cost_ms": costs, "cost_ratio": cutoff,
            "fixed_8": play(lambda _i, _ratio: 8), "fixed_16": play(lambda _i, _ratio: 16),
            "confidence": play(lambda _i, ratio: 16 if ratio >= cutoff else 8),
            "future_known": play(lambda i, _ratio: choice[i]),
            "widen_anchors": sum(ratio >= cutoff for _, _, ratio in options),
            "widen_anchors_without_gain": sum(ratio >= cutoff and a == b for a, b, ratio in options),
            "narrow_anchors_with_spare_gain": sum(ratio < cutoff and a < b for a, b, ratio in options)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide-cost-ratio", type=float, help="use an explicit cutoff without repeating the replay")
    args = parser.parse_args()
    cutoff = args.wide_cost_ratio
    if cutoff is not None and not 1.0 <= cutoff <= 2.0:
        parser.error("--wide-cost-ratio must be in [1, 2]")
    verify_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    before, thermal = pages(), thermal_state()
    with metal_job(80, 2) as budget:
        draft = DSpineDraft()
        mx.eval(draft.weights, draft.code_projection, draft.sender)
        draft = with_packed_ffn(draft, budget)
        draft = with_packed_projections(draft, budget)
        mx.clear_cache()
        model = benchmark_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=2 * GIB)
        draft = with_prompt_vocabulary(draft, model, trace["input"])
        prepared = prepare_prefix(model, draft, trace, budget)
        if cutoff is None:
            cache, first, prompt_kv = prepared
            teacher = [replace(state) for state in cache], first, prompt_kv
            native, prefix = capture_mtp(model, draft, trace, teacher, budget)
            print("mtp_replayed", len(native), "exact_tokens", len(trace["output"]), flush=True)
            del teacher, native
            options = measure_options(model, draft, trace, prefix, budget)
            assert len(options) == 1023
            del prefix
            mx.clear_cache()
            budget.check(force=True)
            outcome = replay(options, phase_costs())
            print("width_replay", json.dumps(outcome), flush=True)
            (ROOT / "runs/dspine-adaptive-width-replay.json").write_text(
                json.dumps({"result": outcome, "options": options}) + "\n")
            cutoff = outcome["cost_ratio"]
        print("adaptive_cutoff", cutoff, flush=True)
        candidate = copy(draft)
        candidate.checked_widths = set()
        candidate.wide_cost_ratio = cutoff
        paired_decode(model, (draft, candidate), trace, prepared, (8, 16), True, budget,
                      labels=("fixed_8", "adaptive_8_16"))
        after = pages()
        print("host", json.dumps({"thermal": [thermal, thermal_state()],
              "faults": {k: after[k] - before[k] for k in before}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

"""Measure DSpine context selection with the full BF16 target still verifying each token."""

import argparse
from collections import Counter
from copy import copy
from dataclasses import replace
import json
from statistics import median
from time import perf_counter

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from scripts.audit_dspine_complement import capture_mtp
from scripts.bench_dflash2 import pages
from scripts.bench_dspine import (
    DSpineDraft, ROOT, verify_checkpoint, with_packed_ffn, with_packed_projections,
    with_prompt_vocabulary,
)
from scripts.bench_dspine_adaptive import advance, phase_costs
from scripts.bench_dspine_ffn import paired_decode
from scripts.bench_dspine_reuse import WIDTH_RATIO, fresh_plan
from scripts.bench_ready import benchmark_target, prepare_prefix


WINDOWS = {
    "full": None,
    "recent_1024": (0, 1024),
    "leading_128_recent_1024": (128, 1024),
    "leading_128_recent_2048": (128, 2048),
}


def miss_type(lattice, truth, confirmed, width):
    if confirmed == width:
        return "width"
    if confirmed >= len(truth):
        return "end"
    choices, logp = lattice
    terminal = confirmed - 1
    token = truth[terminal]
    if token not in choices[terminal]:
        return "outside_top16"
    previous = choices[terminal - 1].index(truth[terminal - 1]) if terminal else 0
    scores = logp[terminal][previous]
    chosen = scores[choices[terminal].index(token)]
    wins = sum(score > chosen or (score == chosen and other < token)
               for other, score in zip(choices[terminal], scores, strict=True))
    return "top4" if wins < 4 else "rank5_to_16"


def replay(rows, costs, draft_ms):
    i, elapsed = 0, 0.0
    widths, misses = Counter(), Counter()
    while i < len(rows):
        width, confirmed, miss, _ = rows[i]
        widths[width] += 1
        misses[miss] += 1
        elapsed += costs[16 if width > 8 else 8] + draft_ms
        i += min(confirmed, len(rows) - i)
    return {"tok_s": len(rows) * 1000 / elapsed, "rounds": sum(widths.values()),
            "widths": widths, "first_miss": misses, "draft_ms": draft_ms}


def measure(model, drafts, trace, full_prefix, budget):
    ids = trace["output"]
    prompt_len = len(trace["input"])
    rows = {label: [] for label in drafts}
    order = tuple(drafts)
    for i, pending in enumerate(ids[:-1]):
        offset = prompt_len + i
        prefix = [tuple(t[:, :, :offset] for t in pair) for pair in full_prefix]
        truth = ids[i + 1:i + 17]
        limit = min(16, len(truth) + 1)
        turn = i % len(order)
        for label in (*order[turn:], *order[:turn]):
            started = perf_counter()
            lattice = drafts[label].candidate_lattice(model, pending, prefix)
            nodes, parents = fresh_plan(pending, lattice, limit)
            draft_ms = (perf_counter() - started) * 1000
            confirmed = advance(nodes, parents, truth, len(nodes))
            rows[label].append((len(nodes), confirmed, miss_type(lattice, truth, confirmed, len(nodes)), draft_ms))
        if (i + 1) % 256 == 0:
            print("context_anchors", i + 1, flush=True)
        budget.check()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("code", "chinese_technical_report", "english_technical_tutorial"),
                        default="code")
    parser.add_argument("--window", choices=tuple(WINDOWS)[1:], help="repeat a selected window on a complete request")
    args = parser.parse_args()
    verify_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[args.workload]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    before, thermal = pages(), thermal_state()
    with metal_job(80, 2) as budget:
        base = DSpineDraft()
        mx.eval(base.weights, base.code_projection, base.sender)
        base = with_packed_ffn(base, budget)
        base = with_packed_projections(base, budget)
        mx.clear_cache()
        model = benchmark_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=2 * GIB)
        base = with_prompt_vocabulary(base, model, trace["input"])
        base.wide_cost_ratio = WIDTH_RATIO
        names = ("full", args.window) if args.window else tuple(WINDOWS)
        drafts = {label: copy(base) for label in names}
        for label, draft in drafts.items():
            draft.context_window = WINDOWS[label]
            draft.checked_widths = set()
        prepared = prepare_prefix(model, base, trace, budget)
        selected = args.window
        if selected is None:
            cache, first, prompt = prepared
            teacher = [replace(state) for state in cache], first, prompt
            native, prefix = capture_mtp(model, base, trace, teacher, budget)
            print("mtp_replayed", len(native), "exact_tokens", len(trace["output"]), flush=True)
            del native, teacher
            rows = measure(model, drafts, trace, prefix, budget)
            del prefix
            mx.clear_cache()
            budget.check(force=True)

            # Use matched historical target/commit costs and the measured draft time for each window.
            target_costs = {width: ms - 16.94 for width, ms in phase_costs().items()}
            half = len(rows["full"]) // 2
            report = {}
            for name, values in rows.items():
                draft_ms = median(row[3] for row in values[16:half])
                report[name] = {"first_half": replay(values[:half], target_costs, draft_ms),
                                "second_half": replay(values[half:], target_costs, draft_ms),
                                "whole": replay(values, target_costs, draft_ms)}
            print("context_replay", json.dumps({"target_ms": target_costs, "windows": report}), flush=True)
            selected = max(names[1:], key=lambda name: report[name]["first_half"]["tok_s"])
        print("workload", args.workload, "tested_window", selected, flush=True)
        paired_decode(model, (drafts["full"], drafts[selected]), trace, prepared, 16, True, budget,
                      labels=("full", selected))
        after = pages()
        print("host", json.dumps({"thermal": [thermal, thermal_state()],
              "faults": {key: after[key] - before[key] for key in before}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

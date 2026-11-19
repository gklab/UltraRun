"""Compare row-major and losslessly packed BF16 DSpine feed-forward weights."""

import argparse
from copy import copy
import json
from statistics import median
from time import perf_counter

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from scripts.bench_dflash2 import pages
from scripts.bench_dspine import DSpineDraft, ROOT, verify_checkpoint, with_packed_ffn
from scripts.bench_ready import benchmark_target, decode_steps, prepare_prefix


def paired_time(run, count=12):
    # Warm both routes, then reverse their order on alternating trials.
    run(0)
    run(1)
    timings = []
    for i in range(count):
        pair = [0.0, 0.0]
        for arm in ((1, 0) if i % 2 else (0, 1)):
            start = perf_counter()
            run(arm)
            pair[arm] = 1000 * (perf_counter() - start)
        timings.append(pair)
    return {"row_major_ms": median(t[0] for t in timings), "packed_ms": median(t[1] for t in timings),
            "paired_delta_ms": median(a - b for a, b in timings)}


def micro(drafts, budget):
    config = drafts[0].config
    for width in (8, 16):
        values = [mx.random.normal((1, width, config.hidden_size)).astype(mx.bfloat16)
                  for _ in range(config.num_hidden_layers)]
        mx.eval(values)

        def work(arm):
            output = [drafts[arm].feed_forward(h, i) for i, h in enumerate(values)]
            mx.eval(output)
            return output

        first, second = work(0), work(1)
        matched, maximum = 0, 0.0
        for a, b in zip(first, second, strict=True):
            matched += int(mx.sum(a.view(mx.uint16) == b.view(mx.uint16)).item())
            maximum = max(maximum, float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()))
        entries = sum(a.size for a in first)
        print("ffn", json.dumps({"width": width, "elements": entries, "fraction_bit_equal": matched / entries,
              "maximum_abs_diff": maximum, **paired_time(work)}), flush=True)
        budget.check(force=True)


def paired_decode(target, drafts, trace, prepared, width, use_tree, budget, *,
                  labels=("row_major", "packed_ffn"), prepared_by_arm=None, align_tokens=False):
    # The two candidate paths share every BF16 target weight. Each gets independent layer observers and cache state.
    streams = []
    widths = (width, width) if isinstance(width, int) else width
    snapshots = (prepared, prepared) if prepared_by_arm is None else prepared_by_arm
    for label, draft, cap, snapshot in zip(labels, drafts, widths, snapshots, strict=True):
        verifier = copy(target)
        verifier.layers = list(target.layers)
        streams.append(decode_steps(verifier, draft, trace, snapshot, cap, False, budget,
                                    tree_proposals=use_tree, label=label))
    mx.reset_peak_memory()
    if align_tokens:
        positions = dict.fromkeys(streams, 1)
        turn = 0
        while positions:
            stream = min(positions, key=lambda gen: (positions[gen], (streams.index(gen) - turn) % len(streams)))
            turn = (streams.index(stream) + 1) % len(streams)
            progress = next(stream, None)
            if progress is None:
                del positions[stream]
            else:
                positions[stream] = progress
        return
    trial = 0
    while streams:
        order = tuple(reversed(streams)) if trial % 2 else tuple(streams)
        for stream in order:
            if next(stream, None) is None:
                streams.remove(stream)
        trial += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("chain8", "tree16"), default="tree16")
    parser.add_argument("--micro-only", action="store_true")
    args = parser.parse_args()
    verify_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    mx.random.seed(39)
    thermal, before = thermal_state(), pages()
    with metal_job(10 if args.micro_only else 80, 2) as budget:
        base = DSpineDraft()
        mx.eval(base.weights, base.code_projection, base.sender)
        optimized = with_packed_ffn(base, budget)
        drafts = base, optimized
        micro(drafts, budget)
        if not args.micro_only:
            target = benchmark_target(budget)
            budget.admit(target, 5120, 512, False, extra_bytes=2 * GIB)
            prepared = prepare_prefix(target, base, trace, budget)
            _, pending, prefix = prepared
            use_tree = args.layout == "tree16"
            width = 16 if use_tree else 8

            def proposal(arm):
                draft = drafts[arm]
                if use_tree:
                    return draft.propose_tree(target, pending, prefix, width)
                return draft.propose(target, pending, prefix, None, width - 1, False).tolist()

            same = proposal(0) == proposal(1)
            print("draft_4k_prefix", json.dumps({"layout": args.layout, "same_proposals": same,
                  **paired_time(proposal)}), flush=True)
            paired_decode(target, drafts, trace, prepared, width, use_tree, budget)
        after = pages()
        print("host", json.dumps({"thermal": [thermal, thermal_state()],
              "faults": {k: after[k] - before[k] for k in before}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

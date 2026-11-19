"""Measure request-local retrieval from committed BF16 target hidden states."""

import argparse
from collections import Counter, defaultdict
import json
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from scripts.audit_dspine_complement import capture_target_suffix
from scripts.bench_dflash2 import pages, prefill
from scripts.bench_ready import ROOT, benchmark_target


TAPS = (31, 63)
NEIGHBORS = 4
WIDTHS = (8, 16)
CONTEXT = 4096
OUTPUT = 1024
METHODS = ("last", "joint")


def normalize_prompt(features, context, budget):
    # Center using only data already available at the first speculative round.
    keys = []
    for h in mx.split(features[0], len(TAPS), axis=1):
        center = mx.mean(h[:context].astype(mx.float32), axis=0)
        pieces = []
        for i in range(0, h.shape[0], 512):
            centered = h[i:i + 512].astype(mx.float32) - center
            scale = mx.rsqrt(mx.maximum(mx.sum(mx.square(centered), axis=-1, keepdims=True), 1e-12))
            piece = (centered * scale).astype(mx.bfloat16)
            mx.eval(piece)
            pieces.append(piece)
            budget.check()
        key = mx.concatenate(pieces)
        mx.eval(key)
        keys.append(key)
    return keys


def top_sources(keys, labels, indices):
    mid, last = keys
    allowed = (labels[indices, None] == labels[None, :]) & (mx.arange(labels.size)[None, :] < indices[:, None])
    scores = (last[indices] @ last.T).astype(mx.float32)
    mixed = scores * 0.5 + (mid[indices] @ mid.T).astype(mx.float32) * 0.5
    output = {}
    for name, logits in zip(METHODS, (scores, mixed), strict=True):
        logits = mx.where(allowed, logits, -mx.inf)
        top = mx.argpartition(-logits, NEIGHBORS - 1, axis=-1)[:, :NEIGHBORS]
        values = mx.take_along_axis(logits, top, axis=-1)
        order = mx.argsort(-values, axis=-1)
        output[name] = mx.take_along_axis(top, order, axis=-1), mx.take_along_axis(values, order, axis=-1)
    return output


def lookup_sweep(keys, labels, budget):
    candidates = {name: [] for name in METHODS}
    cost = []
    for i in range(0, OUTPUT - 1, 64):
        indices = mx.arange(CONTEXT - 1 + i, CONTEXT - 1 + min(i + 64, OUTPUT - 1), dtype=mx.int32)
        output = top_sources(keys, labels, indices)
        for name, (sources, similarity) in output.items():
            candidates[name].extend(zip(sources.tolist(), similarity.tolist(), strict=True))
        budget.check()
    # Approximate single-round cost without amortizing across offline queries.
    # The mask excludes all rows not yet committed. Padding to the final
    # request capacity keeps the score kernel shape fixed during the sweep.
    for i in range(0, OUTPUT - 1, 32):
        index = mx.array([CONTEXT - 1 + i], dtype=mx.int32)
        t = perf_counter()
        output = top_sources(keys, labels, index)
        mx.eval(list(output.values()))
        for sources, scores in output.values():
            sources.tolist(), scores.tolist()
        cost.append((perf_counter() - t) * 1000)
        budget.check()
    return candidates, median(cost[1:])


def exact_neighbors(sequence):
    """Four recent suffix matches, ranked by the length of the token match."""
    occurrences = defaultdict(list)
    for i, token in enumerate(sequence[:CONTEXT]):
        occurrences[token].append(i)
    rows = []
    for output_index in range(OUTPUT - 1):
        current = CONTEXT + output_index
        pending = sequence[current]
        sources = []
        # Bound the token baseline's search over repeated punctuation.
        for prior in occurrences[pending][-128:]:
            if prior <= 0:
                continue
            depth = 1
            while depth < 16 and depth <= prior and sequence[prior - depth] == sequence[current - depth]:
                depth += 1
            sources.append((depth, prior))
        ranked = sorted(sources, reverse=True)[:NEIGHBORS]
        rows.append(([pos - 1 for _, pos in ranked], [float(depth) for depth, _ in ranked]))
        occurrences[pending].append(current)
    return rows


def continuation(sequence, anchor, source, width):
    pending = CONTEXT + anchor
    first = source + 2
    assert source >= 0 and first <= pending and sequence[source + 1] == sequence[pending]
    # The source text is already known when this round begins. Comparing its
    # proposal to the saved future is only for scoring, after construction.
    remaining = len(sequence) - pending - 1
    length = min(width - 1, pending + 1 - first, remaining)
    proposal = sequence[first:first + length]
    actual = sequence[pending + 1:pending + 1 + len(proposal)]
    matched = next((i for i, (a, b) in enumerate(zip(proposal, actual, strict=True)) if a != b), len(proposal))
    return len(proposal) + 1, min(matched + 1, remaining)


def decisions(sequence, candidates):
    rows = []
    for i, (sources, scores) in enumerate(candidates):
        options = []
        for source, score in zip(sources, scores, strict=True):
            if score == -float("inf"):
                continue
            options.append([continuation(sequence, i, source, width) for width in WIDTHS])
        rows.append(options)
    return rows


def target_ms(width):
    return 55.86 if width == 1 else 74.77 if width <= 8 else 94.16


def native_cost(record):
    return target_ms(record["width"]) + (13.5 if record["width"] > 1 else 0)


def oracle(retrieval, native, query_ms):
    count = len(retrieval)
    future = [0.0] * (count + 1)
    parent = [None] * count
    # Include singleton BF16 target passes at the bare weight-read floor.
    # This lets the simulated policy find its way back to a saved MTP origin.
    for i in range(count - 1, -1, -1):
        options = [(55.86, 1, "single")]
        if i in native:
            options.append((native_cost(native[i]), native[i]["advance"], "mtp"))
        for source in retrieval[i]:
            options.extend((target_ms(width) + query_ms, advance, "retrieval") for width, advance in source)
        cost, advanced, kind = min(options, key=lambda r: r[0] + future[i + r[1]])
        future[i] = cost + future[i + advanced]
        parent[i] = advanced, kind
    steps, i = Counter(), 0
    while i < count:
        advance, label = parent[i]
        steps[label] += 1
        i += advance
    return {"milliseconds": future[0], "tok_s": count * 1000 / future[0], "rounds_by_source": steps}


def report(rows, sequence, native, query_ms):
    selected = decisions(sequence, rows)
    short_hit, long_hit = [], []
    longer, recovery = 0, 0
    for i, choices in enumerate(selected):
        if not choices:
            continue
        short_hit.append(choices[0][0][1])
        maximum = max(source[-1][1] for source in choices)
        long_hit.append(maximum)
        if i in native:
            current = native[i]["advance"]
            longer += maximum > current
            recovery += max(0, maximum - current)
    return {"covered_anchors": len(long_hit), "top1_width8": Counter(short_hit),
            "top4_oracle_width16": Counter(long_hit), "known_mtp_anchors_with_longer_retrieval": longer,
            "extra_confirmed_in_fixed_mtp_rounds": recovery,
            "oracle_mtp_plus_retrieval": oracle(selected, native, query_ms)}


def run(model, trace, budget):
    ids, outputs = trace["input"], trace["output"]
    assert len(ids) == CONTEXT and len(outputs) == OUTPUT
    collector = SimpleNamespace(config=SimpleNamespace(target_layer_ids=TAPS, sliding_window=None))
    cache, first, prompt = prefill(model, collector, ids)
    assert first == outputs[0]
    native, suffix = capture_target_suffix(model, TAPS, trace, cache, first, budget)
    del cache
    assert suffix.shape[1] == OUTPUT - 1
    features = mx.concatenate((prompt, suffix), axis=1)
    mx.eval(features)
    del prompt, suffix
    keys = normalize_prompt(features, len(ids), budget)
    del features
    mx.clear_cache()
    budget.check(force=True)
    sequence = ids + outputs
    labels = mx.array(sequence[1:], dtype=mx.int32)
    sources, query_ms = lookup_sweep(keys, labels, budget)
    comparison = {"tokens": report(exact_neighbors(sequence), sequence, native, 0.0)}
    comparison.update({method: report(rows, sequence, native, query_ms) for method, rows in sources.items()})
    return {
        "exact_output_tokens": len(outputs), "native_rounds": len(native),
        "native_cost_model_tok_s": (len(outputs) - 1) * 1000 / sum(map(native_cost, native.values())),
        "bare_oracle_without_retrieval": oracle([()] * (OUTPUT - 1), native, 0),
        "single_query_two_features_ms": query_ms, "normalized_key_bytes": sum(k.nbytes for k in keys),
        "methods": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("chinese_technical_report", "code", "english_technical_tutorial"))
    args = parser.parse_args()
    traces = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())
    before = pages()
    with metal_job(80, 2) as budget:
        model = benchmark_target(budget)
        budget.admit(model, CONTEXT + OUTPUT, 512, False, extra_bytes=2 * GIB)
        for name, trace in traces.items():
            if args.only and args.only != name:
                continue
            mx.reset_peak_memory()
            faults = pages()
            print("start", name, flush=True)
            outcome = run(model, trace, budget)
            print("retrieval", json.dumps({
                "workload": name, **outcome, "thermal": thermal_state(), "peak_gib": mx.get_peak_memory() / GIB,
                "faults": {key: val - faults[key] for key, val in pages().items()}, **budget.metrics(),
            }), flush=True)
            mx.clear_cache()
            budget.check(force=True)
    print("host", json.dumps({key: val - before[key] for key, val in pages().items()}), flush=True)


if __name__ == "__main__":
    main()

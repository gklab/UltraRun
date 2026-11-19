"""Audit sparse target readouts for native MTP candidates on held-out workloads."""

import argparse
from collections import Counter
import json
from math import exp, log, sqrt
from statistics import median
from time import perf_counter

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.spec_tree import VerificationTree, accepted_path
from scripts.bench_dflash2 import observe_layers, pages
from scripts.bench_midlayer_tree import batch_layout, ROOT
from scripts.bench_ready import benchmark_target
from scripts.bench_verify_buffer_limits import partition_prompt


DEPTHS = (8, 16, 24, 32, 48, 56)
STRENGTHS = (0.125, 0.25, 0.5, 1.0, 2.0)
TAIL_LIMITS = (0.25, 0.5, 0.75, 1.0)
WIDTH = 8


def normalized(scores):
    mean = sum(scores) / len(scores)
    centered = [s - mean for s in scores]
    spread = sqrt(sum(s * s for s in centered) / len(scores))
    return [s / max(1e-6, spread) for s in centered]


def mix(row, depth, strength):
    native = row["native"]
    if strength == 0:
        return native
    weighted = [a + strength * b for a, b in zip(native, row["lens"][depth], strict=True)]
    # Keep the same total probability on the evaluated tokens. The remainder
    # belongs to unseen candidates, including cases where the true token is absent.
    delta = log(sum(exp(a) for a in native)) - log(sum(exp(a) for a in weighted))
    return [value + delta for value in weighted]


def captured_rows(model, record, captures, budget):
    location = {old: new for new, old in enumerate(record["order"])}
    seen = {location[int(node)]: scores for node, scores in record["seen"].items()}
    groups = sorted(seen)
    if not groups:
        return {}, 0.0
    assert all(len(seen[i]) == WIDTH for i in groups)
    token_ids = [[t for t, _ in seen[i]] for i in groups]
    index = mx.array(groups, dtype=mx.int32)
    vocabulary = mx.array(token_ids, dtype=mx.int32)
    started = perf_counter()
    weight = model.vocabulary_rows(vocabulary.reshape(-1)).reshape(len(groups), WIDTH, -1).astype(mx.float32)
    hidden = mx.stack([captures[depth - 1][index] for depth in DEPTHS])
    hidden = mx.fast.rms_norm(hidden, model.norm, model.config["rms_norm_eps"]).astype(mx.float32)
    logits = mx.sum(hidden[:, :, None, :] * weight[None, :, :, :], axis=-1)
    curves = logits.tolist()
    elapsed = perf_counter() - started
    result = {}
    for j, parent in enumerate(groups):
        tokens = token_ids[j]
        correct = record["prediction"][parent]
        result[parent] = {
            "tokens": tokens, "native": [s for _, s in seen[parent]],
            "truth": tokens.index(correct) if correct in tokens else None,
            "lens": {depth: normalized(scores[j]) for depth, scores in zip(DEPTHS, curves, strict=True)},
        }
    budget.check()
    return result, elapsed * 1000


def collect(model, trace, budget):
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    watched = {}
    observe_layers(model, [depth - 1 for depth in DEPTHS], lambda i, _, h: watched.__setitem__(i, h))
    cache = model.cache()
    for _, h in model.prefill(mx.array(trace["input"], dtype=mx.int32), cache, 512):
        model.evaluate(h, cache)
        budget.check()
    first = mx.argmax(model.logits(h[-1:]), axis=-1).item()
    assert first == trace["output"][0]
    partition_prompt(model, cache, len(trace["input"]))
    watched.clear()
    taps = model.config["linear_conv_kernel_dim"] - 1
    decoded, tracked, times = [first], [], []
    for offset_str, record in sorted(trace["rounds"].items(), key=lambda pair: int(pair[0])):
        offset = int(offset_str)
        assert offset == len(trace["input"]) + len(decoded) - 1
        batch, parents, path = batch_layout(record)
        fork = next((i for i in range(1, len(batch)) if parents[i] != i - 1), None)
        tree = VerificationTree(parents, offset, taps, spine=fork) if fork is not None else None
        hidden = model.hidden(mx.array(batch, dtype=mx.int32), cache, offset, keep_replay=True, tree=tree)
        prediction = model.predictions(hidden)
        model.evaluate(prediction, cache)
        predicted = prediction.tolist()
        assert predicted == record["prediction"], (offset, predicted, record["prediction"])
        assert path == accepted_path(parents, batch, predicted)
        fresh = [batch[i] for i in path[1:]] + [predicted[path[-1]]]
        assert fresh == trace["output"][len(decoded):len(decoded) + len(fresh)]
        decoded.extend(fresh)
        rows, elapsed = captured_rows(model, record, watched, budget)
        masses = [record["nodes"][i][3] for i in record["order"]]
        tracked.append({"rows": rows, "parents": parents, "path": path, "batch": batch, "masses": masses,
                        "anchor": record["anchor"], "prediction": predicted})
        if rows:
            times.append(elapsed)
        if tree is not None:
            model.accept_tree(cache, tree, offset, path, defer_kv=len(decoded) < len(trace["output"]))
        elif len(batch) > 1:
            model.accept(cache, offset, len(batch), len(path))
        model.evaluate(prediction, cache)
        watched.clear()
        budget.check()
    # The English audit did not record its final two single-token target passes.
    tail = len(trace["output"]) - len(decoded)
    for _ in range(tail):
        offset = len(trace["input"]) + len(decoded) - 1
        hidden = model.hidden(mx.array(decoded[-1:], dtype=mx.int32), cache, offset)
        prediction = model.predictions(hidden)
        model.evaluate(prediction, cache)
        decoded.append(prediction.item())
        watched.clear()
        budget.check()
    assert decoded == trace["output"]
    return tracked, median(times), tail


def rank(row, depth, strength):
    blend = mix(row, depth, strength)
    pos = row["truth"]
    return 1 + sum(score > blend[pos] or (score == blend[pos] and i < pos)
                   for i, score in enumerate(blend))


def rank_summary(rounds, depth, strength):
    result = Counter()
    for entry in rounds:
        for parent in entry["path"]:
            row = entry["rows"].get(parent)
            boundary = parent == entry["path"][-1]
            if boundary:
                result["boundaries"] += 1
            if row is None or row["truth"] is None:
                continue
            baseline = row["truth"] + 1
            scored = rank(row, depth, strength)
            result.update(in_top8=1, top1=int(scored == 1), top4=int(scored <= 4))
            if boundary:
                result.update(boundary_in_top8=1, boundary_top1=int(scored == 1), boundary_top4=int(scored <= 4),
                              boundary_promoted_to_top4=int(baseline > 4 and scored <= 4),
                              boundary_demoted_from_top4=int(baseline <= 4 and scored > 4))
    return dict(result)


def select_subtree(record, depth, strength):
    rows, batch = record["rows"], record["batch"]
    conditional = {parent: dict(zip(row["tokens"], mix(row, depth, strength), strict=True))
                   for parent, row in rows.items()}
    reach = [0.0]
    for i, parent in enumerate(record["parents"][1:], 1):
        fallback = min(0, record["masses"][i] - record["masses"][parent])
        p = conditional.get(parent, {}).get(batch[i], fallback)
        reach.append(reach[parent] + min(0, p))
    order = sorted(range(len(reach)), key=lambda i: (-reach[i], i))
    kept = set(order[:WIDTH])
    path = record["path"]
    advance = next((i for i, node in enumerate(path) if node not in kept), len(path))
    tail_mass = sum(exp(reach[i]) for i in order[WIDTH:])
    return len(path) - advance, tail_mass


def trim_summary(rounds, depth, strength, limit, token_ms):
    # Price savings generously using a full 16-to-8 gap at all wider batches.
    saving_ms = (94.16 - 74.77) * (64 - depth) / 64 if strength else (94.16 - 74.77)
    trimmed, lost = 0, 0
    for r in rounds:
        if len(r["batch"]) <= WIDTH or r["anchor"]:
            continue
        deficit, tail = select_subtree(r, depth, strength)
        if tail <= limit:
            trimmed += 1
            lost += deficit
    return {"trimmed": trimmed, "lost_from_fixed_rounds": lost,
            "optimistic_net_ms": saving_ms * trimmed - token_ms * lost}


def holdout(results, traces):
    options = [(0, 0.0, limit) for limit in TAIL_LIMITS]
    options += [(depth, w, limit) for depth in DEPTHS for w in STRENGTHS for limit in TAIL_LIMITS]
    token_prices = {name: sum(r["target_s"] for r in t["rounds"].values()) * 1000 / (len(t["output"]) - 1)
                    for name, t in traces.items()}
    for workload, replay in results.items():
        train = {key: other for key, other in results.items() if key != workload}
        best = max(options, key=lambda opt: sum(trim_summary(v, *opt, token_prices[k])["optimistic_net_ms"]
                                                for k, v in train.items()))
        native = rank_summary(replay, 0, 0.0)
        summaries = {}
        for depth in DEPTHS:
            # Optimize candidate ranking using only the other workloads' matched labels.
            strength = max((0.0, *STRENGTHS), key=lambda w: sum(
                rank_summary(v, depth, w).get("boundary_top4", 0) for v in train.values()
            ))
            summaries[depth] = {"selected_strength": strength, "held_out": rank_summary(replay, depth, strength),
                                "lens_heavy": rank_summary(replay, depth, 2.0)}
        top32 = summaries[32]["held_out"]["boundary_top4"]
        print("holdout", json.dumps({
            "workload": workload, "native": native, "depths": summaries,
            "layer32_net_boundary_top4": top32 - native["boundary_top4"],
            "policy_from_other_workloads": {"depth": best[0], "strength": best[1], "limit": best[2],
                                             **trim_summary(replay, *best, token_prices[workload])},
        }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("code", "chinese_technical_report", "english_technical_tutorial"))
    args = parser.parse_args()
    traces = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())
    if args.only:
        traces = {args.only: traces[args.only]}
    results = {}
    before = pages()
    with metal_job(80, 2) as budget:
        model = benchmark_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=2 * GIB)
        for name, trace in traces.items():
            mx.reset_peak_memory()
            fault_start = pages()
            current, median_ms, tail = collect(model, trace, budget)
            results[name] = current
            faults = {k: v - fault_start[k] for k, v in pages().items()}
            print("capture", json.dumps({
                "workload": name, "rounds": len(current), "exact_output_tokens": len(trace["output"]),
                "tail_autoregressive_rounds": tail,
                "six_depth_readouts_median_ms": median_ms, "thermal": thermal_state(), "faults": faults,
                "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics(),
            }), flush=True)
            mx.clear_cache()
            budget.check(force=True)
        after = pages()
        print("host", json.dumps({k: after[k] - v for k, v in before.items()}), flush=True)
    if len(results) > 1:
        holdout(results, traces)
    else:
        for key, cases in results.items():
            print("ranks", key, json.dumps({"native": rank_summary(cases, 0, 0),
                  **{str(depth): rank_summary(cases, depth, 1.0) for depth in DEPTHS}}), flush=True)


if __name__ == "__main__":
    main()

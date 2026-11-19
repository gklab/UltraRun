"""Replay BF16 MTP and bound the value of combining it with DSpine."""

import argparse
from collections import Counter
import json
from math import exp
from statistics import median
from time import perf_counter
from typing import NamedTuple

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.spec_tree import VerificationTree, accepted_path
from scripts.bench_dflash2 import observe_layers, pages
from scripts.bench_dspine import DSpineDraft, ROOT, candidate_tree, verify_checkpoint
from scripts.bench_ready import benchmark_target, prepare_prefix


WIDTHS = (8, 16)
WORKLOAD = "code"


class RoundOption(NamedTuple):
    source: str
    advance: int
    total_ms: float
    verifier_ms: float
    weight_read_ms: float


def capture_mtp(model, draft, trace, prepared, budget):
    """Recompute the saved MTP targets on the original branch topology."""
    cache, first, prompt = prepared
    native, features = capture_target_suffix(model, draft.config.target_layer_ids, trace, cache, first, budget)
    extra = draft.context(features, len(trace["input"]), budget)
    prefix = [tuple(mx.concatenate((a, b), axis=2) for a, b in zip(p, e, strict=True))
              for p, e in zip(prompt, extra, strict=True)]
    mx.eval(prefix)
    return native, prefix


def capture_target_suffix(model, taps, trace, cache, first, budget, *, extra_observer=None):
    """Return ordered post-layer BF16 states from the exact committed target path."""
    expected = trace["output"]
    context = len(trace["input"])
    conv = model.config["linear_conv_kernel_dim"] - 1
    branch = {}

    def observe(i, offset, out):
        branch[i] = out
        if extra_observer is not None:
            extra_observer(i, offset, out)

    observe_layers(model, taps, observe)
    native = {}
    blocks = []
    output = [first]
    for offset, record in sorted(trace["rounds"].items(), key=lambda item: int(item[0])):
        start = len(output) - 1
        assert int(offset) == context + start
        nodes, order = record["nodes"], record["order"]
        location = {node: i for i, node in enumerate(order)}
        parents = [-1, *(location[nodes[i][0]] for i in order[1:])]
        batch = [nodes[i][1] for i in order]
        assert batch[0] == output[-1]
        width = len(batch)
        jumps = [i for i in range(1, width) if parents[i] != i - 1]
        tree = VerificationTree(parents, int(offset), conv, spine=jumps[0]) if jumps else None
        hidden = model.hidden(mx.array(batch, dtype=mx.int32), cache, int(offset),
                              keep_replay=width > 1, tree=tree)
        predictions = model.predictions(hidden)
        model.evaluate([predictions, *branch.values()], cache)
        predicted = predictions.tolist()
        path = accepted_path(parents, batch, predicted)
        new = [batch[i] for i in path[1:]] + [predicted[path[-1]]]
        assert path == [location[i] for i in record["path"]], (start, path, record["path"])
        assert new == expected[len(output):len(output) + len(new)], (start, new)
        output.extend(new)

        last = path[-1]
        forks = {parents[i] for i in jumps}
        graft_allowed = last not in parents or last in forks or len(forks) < 2
        native[start] = {"width": width, "advance": len(path), "graft_allowed": graft_allowed}
        if tree is not None:
            model.accept_tree(cache, tree, int(offset), path, defer_kv=len(output) < len(expected))
        elif width > 1:
            model.accept(cache, int(offset), width, len(path))
        indices = mx.array(path, dtype=mx.int32)
        features = mx.concatenate([branch[i][indices] for i in taps], axis=-1)[None]
        model.evaluate(features, cache)
        blocks.append(features)
        branch.clear()
        budget.check()

    # The saved English trace ends with two passes outside the speculative loop.
    while len(output) < len(expected):
        start = len(output) - 1
        hidden = model.hidden(mx.array(output[-1:], dtype=mx.int32), cache, context + start)
        predicted = model.predictions(hidden)
        features = mx.concatenate([branch[i] for i in taps], axis=-1)[None]
        model.evaluate([predicted, features], cache)
        output.append(predicted.item())
        native[start] = {"width": 1, "advance": 1, "graft_allowed": True}
        blocks.append(features)
        branch.clear()
        budget.check()
    assert output == expected
    features = mx.concatenate(blocks, axis=1)
    mx.eval(features)
    return native, features


def conditional_ranks(choices, scores, truth):
    present = choices == truth[:, None]
    index = mx.argmax(present, axis=-1, keepdims=True)
    selected = mx.take_along_axis(scores, index, axis=-1)
    wins = (scores > selected) | ((scores == selected) & (choices < truth[:, None]))
    return mx.where(mx.any(present, axis=-1), 1 + mx.sum(wins, axis=-1), 17)


def tree_advance(pending, choices, logp, truth, width):
    batch, parents, _ = candidate_tree(pending, choices, logp, width)
    children = {(p, t): i for i, (p, t) in enumerate(zip(parents, batch, strict=True)) if i}
    parent, depth = 0, 0
    for token in truth:
        if (parent := children.get((parent, token))) is None:
            break
        depth += 1
    return min(depth + 1, len(truth))


def capture_dspine(model, draft, trace, full_prefix, budget):
    expected = trace["output"]
    context = len(trace["input"])
    spans = []
    for anchor, pending in enumerate(expected[:-1]):
        width = min(draft.config.block_size, len(expected) - anchor)
        preceding = mx.array(expected[anchor:anchor + width - 1], dtype=mx.int32)[:, None]
        truth = expected[anchor + 1:anchor + width]
        offset = context + anchor
        prefix = [tuple(x[:, :, :offset] for x in pair) for pair in full_prefix]
        choices, scores, rms = draft.lattice(model, pending, prefix, width, extra_senders=preceding)
        ranks = conditional_ranks(choices, scores[:, -1, :], mx.array(truth, dtype=mx.int32))
        logits = scores[:, :16, :] / mx.maximum(rms[:, None, :], draft.eps)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(choices, ranks, logp)
        choices_list, logp_list = choices.tolist(), logp.tolist()
        trees = {size: tree_advance(pending, choices_list, logp_list, truth, size) for size in WIDTHS}
        spans.append({"ranks": ranks.tolist(), "tree": trees})
        if len(spans) % 256 == 0:
            print("draft_anchors", len(spans), flush=True)
        budget.check()
    return spans


def ranked_suffix(ranks, start=0, *, top=16):
    return next((i for i, rank in enumerate(ranks[start:]) if rank > top), len(ranks) - start)


def shortest(actions, *, charge="total_ms"):
    """Minimize total cost on the fixed trace, allowing a shorter verified prefix."""
    count = len(actions)
    cost = [0.0] + [float("inf")] * count
    previous = [None] * (count + 1)
    for anchor, options in enumerate(actions):
        for option in options:
            elapsed = cost[anchor] + getattr(option, charge)
            for end in range(anchor + 1, min(count, anchor + option.advance) + 1):
                if elapsed < cost[end]:
                    cost[end] = elapsed
                    previous[end] = anchor, option.source
    rounds = Counter()
    end = count
    while end:
        end, source = previous[end]
        rounds[source] += 1
    return {"milliseconds": cost[-1], "tok_s": count * 1000 / cost[-1], "rounds_by_source": rounds}


def costs():
    measured = ROOT / "runs/dspine-bf16-code.log"
    rows = [json.loads(line.removeprefix("result ")) for line in measured.read_text().splitlines()
            if line.startswith("result ")]
    target = {w: median(row["phases_s"]["target"] * 1000 / row["rounds"]
                        for row in rows if row["width"] == w) for w in WIDTHS}
    wide = next(row for row in rows if row["tree_proposals"] and row["width"] == 16)
    draft, commit = (1000 * wide["phases_s"][phase] / wide["rounds"] for phase in ("draft", "commit"))
    fresh = json.loads((ROOT / "runs/bf16-100-tps-budget-after-output-pack.json").read_text())
    return target, draft, commit, fresh["workloads"][WORKLOAD]["draft_and_commit_ms"], fresh["weights"]["stream_ms"]


def analyze(native, spans):
    target, draft, commit, native_overhead, read = costs()
    modes = {mode: [] for mode in ("mtp", "trees", "greedy_graft", "spare_slots", "ideal_top16")}
    graft_wins = Counter()
    for i, row in enumerate(spans):
        ranks = row["ranks"]
        for actions in modes.values():
            actions.append([])
        if i in native:
            record = native[i]
            width = 8 if record["width"] <= 8 else 16
            confirmed = record["advance"]
            for actions in modes.values():
                actions[-1].append(RoundOption("mtp", confirmed, target[width] + native_overhead, target[width], read))
            eligible = confirmed < len(spans) - i
            extension = ranked_suffix(ranks, confirmed - 1) if eligible else 0
            greedy = ranked_suffix(ranks, confirmed - 1, top=1) if eligible else 0
            if extension:
                graft_wins["top16_can_extend"] += 1
                first_rank = ranks[confirmed - 1]
                graft_wins["first_extension_top1"] += first_rank == 1
                graft_wins["same_band_spare"] += bool(record["graft_allowed"] and record["width"] < width)
            for size in WIDTHS:
                if record["graft_allowed"] and extension and size > record["width"]:
                    advance = confirmed + min(extension, size - record["width"])
                    ms = target[size] + draft + native_overhead
                    modes["spare_slots"][-1].append(RoundOption("graft", advance, ms, target[size], read))
                    if greedy:
                        advance = confirmed + min(greedy, size - record["width"])
                        modes["greedy_graft"][-1].append(RoundOption("graft", advance, ms, target[size], read))
                if extension and size > confirmed:
                    modes["ideal_top16"][-1].append(RoundOption("graft", min(size, confirmed + extension),
                                                              target[size] + draft + native_overhead, target[size], read))
        for size in WIDTHS:
            observed = RoundOption("dspine_tree", row["tree"][size], target[size] + draft + commit, target[size], read)
            modes["trees"][-1].append(observed)
            modes["greedy_graft"][-1].append(observed)
            modes["spare_slots"][-1].append(observed)
            top16 = min(size, ranked_suffix(ranks) + 1)
            modes["ideal_top16"][-1].append(RoundOption("dspine_top16", top16, target[size] + draft + commit,
                                                      target[size], read))
    return {"cost_ms": {"target": target, "dspine_draft": draft, "dspine_commit": commit,
                        "mtp_draft_and_commit": native_overhead, "target_read": read},
            "mtp_anchors": len(native), "graft_wins": graft_wins,
            "root_ranks": Counter(row["ranks"][0] for row in spans),
            "nominal_cost_oracle": {mode: shortest(actions) for mode, actions in modes.items()},
            "verifier_only_oracle": shortest(modes["ideal_top16"], charge="verifier_ms"),
            "target_read_only_oracle": shortest(modes["ideal_top16"], charge="weight_read_ms")}


def probe_gate(trace, native, spans, cost_ms, price):
    """Price a causal parent guess on fixed starts, granting free extra verification slots."""
    gates = {p: Counter() for p in (0.0, 0.35, 0.5, 0.6, 0.7, 0.8)}
    context = len(trace["input"])
    for offset, saved in trace["rounds"].items():
        i = int(offset) - context
        record = native[i]
        width, advance = record["width"], record["advance"]
        if width >= 16 or i + advance >= len(spans):
            continue
        masses = [exp(node[3]) for node in saved["nodes"]]
        stop = masses.copy()
        for mass, (parent, *_) in zip(masses[1:], saved["nodes"][1:], strict=True):
            stop[parent] -= mass
        proposed_parent = max(range(len(stop)), key=stop.__getitem__)
        greedy = ranked_suffix(spans[i]["ranks"], advance - 1, top=1) if record["graft_allowed"] else 0
        for threshold, row in gates.items():
            if stop[proposed_parent] >= threshold:
                correct = proposed_parent == saved["path"][-1]
                row["draft_calls"] += 1
                row["correct_parent"] += correct
                row["extra_tokens"] += min(16 - width, greedy) if correct else 0
    for row in gates.values():
        row["estimated_ms_saved_before_extra_verification"] = (
            price * row["extra_tokens"] - cost_ms * row["draft_calls"])
    return gates


def analyze_report(trace, native, spans):
    result = analyze(native, spans)
    mtp_ms = result["nominal_cost_oracle"]["mtp"]["milliseconds"]
    result["causal_graft_gate"] = probe_gate(trace, native, spans, result["cost_ms"]["dspine_draft"],
                                            mtp_ms / len(spans))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-report", action="store_true", help="reanalyze saved ranks without loading weights")
    args = parser.parse_args()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[WORKLOAD]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    report_path = ROOT / "runs/dspine-mtp-complement.json"
    if args.reuse_report:
        report = json.loads(report_path.read_text())
        native = {int(i): row for i, row in report["native"].items()}
        spans = [{**row, "tree": {int(w): value for w, value in row["tree"].items()}}
                 for row in report["anchors"]]
        print("reanalysis", json.dumps(analyze_report(trace, native, spans)), flush=True)
        return
    verify_checkpoint()
    before = pages()
    thermal = thermal_state()
    with metal_job(80, 2) as budget:
        model = benchmark_target(budget)
        budget.check(force=True, required_bytes=6 * GIB)
        draft = DSpineDraft()
        mx.eval(draft.weights, draft.code_projection, draft.sender)
        budget.admit(model, 5120, 512, False, extra_bytes=2 * GIB)
        started = perf_counter()
        native, prefix = capture_mtp(model, draft, trace, prepare_prefix(model, draft, trace, budget), budget)
        mx.clear_cache()
        print("mtp_replayed", len(native), "exact_tokens", len(trace["output"]),
              "seconds", perf_counter() - started, flush=True)
        spans = capture_dspine(model, draft, trace, prefix, budget)
        assert len(spans) == 1023
        budget.check(force=True)
        after = pages()
        result = {"workload": WORKLOAD, "anchors": len(spans), **analyze_report(trace, native, spans),
                  "thermal": [thermal, thermal_state()], "peak_gib": mx.get_peak_memory() / GIB,
                  "faults": {key: after[key] - before[key] for key in before}, **budget.metrics()}
        print("result", json.dumps(result), flush=True)
        report = {"result": result, "native": native, "anchors": spans}
        report_path.write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()

"""Test causal candidate pruning halfway through exact BF16 verification."""

from collections import Counter
from copy import copy
from dataclasses import dataclass, replace
import inspect
from itertools import groupby
import json
from math import exp, log, sqrt
from statistics import mean, median
from time import perf_counter
from unittest.mock import patch

import mlx.core as mx

from localllm import cli, frontier
from localllm.draft_short_head import prepare_short_head
from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from localllm.model import Model, VERIFY_SUBMISSION_LAYERS
from localllm.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from localllm.server import warm_verification_graphs
from localllm.spec_tree import VerificationTree, accepted_path
from scripts.audit_dspine_complement import capture_target_suffix
from scripts.bench_dflash2 import pages
from scripts.bench_dflash2_tandem import exercise
from scripts.bench_midlayer_tree import batch_layout
from scripts.bench_ready import ROOT, TARGET, digest
from scripts.bench_verbatim_ffn import replace_once
from scripts.bench_verify_buffer_limits import partition_prompt


CUT = 32
WIDTH = 16
KEEP = 8
HOLDOUT = "code"
SAVING_IN_TOKENS = 0.35
PRODUCTION_SPECULATIVE = cli.speculative


def features(model, hidden, tokens, parents, masses):
    index = mx.array(parents[1:], dtype=mx.int32)
    vocab = model.vocabulary_rows(mx.array(tokens[1:], dtype=mx.int32)).astype(mx.float32)
    h = mx.fast.rms_norm(hidden[index], model.norm, model.config["rms_norm_eps"]).astype(mx.float32)
    logits = [0.0, *mx.sum(h * vocab, axis=1).tolist()]
    best = {}
    for parent, score in zip(parents[1:], logits[1:], strict=True):
        best[parent] = max(best.get(parent, -float("inf")), score)
    depths = [0]
    rows = []
    for i, parent in enumerate(parents[1:], 1):
        depth = depths[parent] + 1
        depths.append(depth)
        prior = exp(min(0.0, masses[i] - masses[parent]))
        rows.append((prior, exp(min(0.0, masses[parent])), depth / WIDTH, logits[i], logits[i] - best[parent]))
    return rows


def sigmoid(x):
    return 1 / (1 + exp(-max(-32, min(32, x))))


@dataclass(frozen=True)
class EdgeModel:
    offset: tuple
    scale: tuple
    coefficients: tuple

    def probability(self, row):
        value = self.coefficients[0] + sum(
            w * (v - m) / s for w, v, m, s in zip(self.coefficients[1:], row, self.offset, self.scale, strict=True))
        return sigmoid(value)


def fit(examples):
    columns = list(zip(*(row for row, _, _ in examples), strict=True))
    offset = tuple(mean(column) for column in columns)
    scale = tuple(max(0.01, sqrt(mean((v - m) ** 2 for v in column)))
                  for column, m in zip(columns, offset, strict=True))
    data = [((1, *((v - m) / s for v, m, s in zip(row, offset, scale, strict=True))), yes, weight)
            for row, yes, weight in examples]
    total = sum(weight for _, _, weight in data)
    coefficients = [0.0] * (len(columns) + 1)
    for _ in range(100):
        gradient = [0.0] * len(coefficients)
        for row, yes, weight in data:
            error = weight * (sigmoid(sum(a * x for a, x in zip(coefficients, row, strict=True))) - yes)
            for i, value in enumerate(row):
                gradient[i] += error * value
        coefficients = [w - 0.35 * (g / total + (0.025 * w if i else 0.0))
                        for i, (w, g) in enumerate(zip(coefficients, gradient, strict=True))]
    return EdgeModel(offset, scale, tuple(coefficients))


def shortlist(rows, parents, edge_model):
    reach = [0.0]
    for row, parent in zip(rows, parents[1:], strict=True):
        reach.append(reach[parent] + log(edge_model.probability(row)))
    order = sorted(range(len(reach)), key=lambda i: (-reach[i], i))
    return sorted(order[:KEEP]), sum(exp(reach[i]) for i in order[KEEP:])


def calibrate(records, edge_model):
    """Price actual lost tokens on training workloads for each predicted-loss threshold."""
    outcomes = []
    for rows, parents, path in records:
        chosen, risk = shortlist(rows, parents, edge_model)
        kept = set(chosen)
        advance = next((i for i, row in enumerate(path) if row not in kept), len(path))
        outcomes.append((risk, len(path) - advance))
    outcomes.sort()
    threshold, best = 0.0, 0.0
    selected = 0
    count, skipped = 0, 0
    for risk, same_risk in groupby(outcomes, key=lambda row: row[0]):
        losses = [loss for _, loss in same_risk]
        count += len(losses)
        skipped += sum(losses)
        value = count * SAVING_IN_TOKENS - skipped
        if value > best:
            threshold, best, selected = risk, value, count
    return threshold, selected, best


def fit_examples(model, trace, budget):
    """Use existing BF16 oracle traces from workloads excluded from the live trial."""
    cache = model.cache()
    for _, hidden in model.prefill(mx.array(trace["input"], dtype=mx.int32), cache, 512):
        model.evaluate(hidden, cache)
        budget.check()
    first = mx.argmax(model.logits(hidden[-1:]), axis=-1).item()
    partition_prompt(model, cache, len(trace["input"]))
    examples, records, score_times = [], [], []
    original = model.layers
    model.layers = list(original)
    lookup = {int(i): row for i, row in trace["rounds"].items() if len(row["order"]) == WIDTH}

    def observe(_layer, offset, hidden):
        if offset not in lookup:
            return
        record = lookup[offset]
        tokens, parents, path = batch_layout(record)
        if parents == [-1, *range(WIDTH - 1)]:
            return
        started = perf_counter()
        masses = [record["nodes"][i][3] for i in record["order"]]
        rows = features(model, hidden, tokens, parents, masses)
        score_times.append((perf_counter() - started) * 1000)
        records.append((rows, parents, path))
        reachable = set(path)
        for i, (row, parent) in enumerate(zip(rows, parents[1:], strict=True), 1):
            matched = tokens[i] == record["prediction"][parent]
            examples.append((row, int(matched), 1.0 if parent in reachable else 0.25))

    try:
        # This replayer also checks every archived prediction and accepted path.
        native, captured = capture_target_suffix(model, (CUT - 1,), trace, cache, first, budget,
                                                 extra_observer=observe)
        del captured
    finally:
        model.layers = original
    return examples, records, len(native), median(score_times)


def forward(model, cache, x, offset, tree, start, stop):
    for i in range(start, stop):
        x = model.layers[i].forward(x, cache[i], offset, True, tree)
        end = i + 1
        if model.async_verify and end % VERIFY_SUBMISSION_LAYERS == 0 and end < stop:
            model.evaluate(x, cache[end - VERIFY_SUBMISSION_LAYERS:end], asynchronous=True)
    return x


class Midpoint:
    def __init__(self, edge_model, threshold):
        self.edge_model = edge_model
        self.threshold = threshold
        self.masses = ()
        self.counts = Counter()
        self.scores_ms = []
        self.risks = []
        self.last_rows = ()

    def verify(self, model, batch, tokens, parents, tree, offset, cache, proposals, calibration):
        assert len(self.masses) == len(tokens) == WIDTH and tree.spine is not None
        self.counts["examined"] += 1
        x = forward(model, cache, model.embedding[batch], offset, tree, 0, CUT)
        started = perf_counter()
        rows = features(model, x, tokens, parents, self.masses)
        chosen, loss = shortlist(rows, parents, self.edge_model)
        self.scores_ms.append((perf_counter() - started) * 1000)
        self.risks.append(loss)
        self.last_rows = ()
        if loss > self.threshold:
            hidden = forward(model, cache, x, offset, tree, CUT, len(model.layers))
            return hidden, batch, tokens, parents, tree, proposals, model.accept_tree

        self.counts["pruned"] += 1
        self.last_rows = chosen
        location = {old: i for i, old in enumerate(chosen)}
        compact = [-1, *(location[parents[i]] for i in chosen[1:])]
        first_jump = next((i for i in range(1, KEEP) if compact[i] != i - 1), KEEP - 1)
        thin_tree = VerificationTree(compact, offset, model.config["linear_conv_kernel_dim"] - 1, spine=first_jump)
        if calibration is not None:
            calibration.pending = {location[i]: value for i, value in calibration.pending.items() if i in location}
        indices = mx.array(chosen, dtype=mx.int32)
        thin_tokens = [tokens[i] for i in chosen]
        hidden = forward(model, cache, x[indices], offset, thin_tree, CUT, len(model.layers))

        def accept(states, thin, at, path, *, defer_kv=False):
            early = copy(model)
            early.layers = model.layers[:CUT]
            late = copy(model)
            late.layers = model.layers[CUT:]
            a = early.accept_tree(states[:CUT], tree, at, [chosen[i] for i in path], defer_kv=defer_kv)
            b = late.accept_tree(states[CUT:], thin, at, path, defer_kv=defer_kv)
            return a or b

        return hidden, batch[indices], thin_tokens, compact, thin_tree, thin_tokens[1:first_jump], accept


def iterator(midpoint):
    builder = frontier.build_frontier
    if midpoint is not None:
        building = inspect.getsource(builder)
        building = replace_once(building, "    return proposals, alternatives, parents, score_rows",
                                "    midpoint.masses = tuple(nodes[i].mass for i in ordered)\n"
                                "    return proposals, alternatives, parents, score_rows")
        scope = {**vars(frontier), "midpoint": midpoint}
        exec(compile(building, __file__, "exec"), scope)
        builder = scope["build_frontier"]
    source = inspect.getsource(PRODUCTION_SPECULATIVE)
    if midpoint is not None:
        old = "        verified = model.hidden(batch, cache, offset, keep_replay=bool(proposals), tree=tree)"
        new = """        commit_tree = model.accept_tree
        if tree is not None and tree.spine is not None and len(tokens) == WIDTH:
            verified, batch, tokens, parents, tree, proposals, commit_tree = midpoint.verify(
                model, batch, tokens, parents, tree, offset, cache, proposals, calibration)
            alternatives = tokens[len(proposals) + 1:]
        else:
            verified = model.hidden(batch, cache, offset, keep_replay=bool(proposals), tree=tree)"""
        source = replace_once(source, old, new)
        assert source.count("model.accept_tree(cache, tree, offset, path") == 2
        source = source.replace("model.accept_tree(cache, tree, offset, path", "commit_tree(cache, tree, offset, path")
    tag = '    stats["accepted_per_pass_histogram"] = acceptance'
    source = replace_once(source, tag, "        yield len(generated)\n" + tag)
    scope = {**vars(cli), "build_frontier": builder, "midpoint": midpoint, "WIDTH": WIDTH}
    exec(compile(source, __file__, "exec"), scope)
    return scope["speculative"]


def exact_midpoint(model, cache, trace):
    """Check retained outputs and both commit modes by forcing a fixed batch to narrow."""
    root, continuation, after = trace["output"][:3]
    wrong = (continuation + 1) % model.vocab_size
    tokens = [root, *([wrong] * (WIDTH - 3)), continuation, after]
    offset = len(trace["input"])
    parents = [-1, *range(13), 0, 14]
    tree = VerificationTree(parents, offset, model.config["linear_conv_kernel_dim"] - 1, spine=14)
    batch = mx.array(tokens, dtype=mx.int32)
    prefer_test_branch = EdgeModel((0,) * 5, (1,) * 5, (-4, 10, 0, 0, 0, 0))
    for defer in (False, True):
        verifier = Midpoint(prefer_test_branch, threshold=float("inf"))
        verifier.masses = (0.0, *(-4.0 * i for i in range(1, 14)), -0.1, -0.2)
        a, b = ([replace(s) for s in cache] for _ in range(2))
        full = model.hidden(batch, a, offset, keep_replay=True, tree=tree)
        thin, _, thin_tokens, remapped, narrow, _, commit = verifier.verify(
            model, batch, tokens, parents, tree, offset, b, tokens[1:14], None)
        predictions = model.predictions(full)
        predicted = model.predictions(thin)
        model.evaluate([full, predictions], a)
        model.evaluate([thin, predicted], b)
        indices = mx.array(verifier.last_rows, dtype=mx.int32)
        assert mx.array_equal(full[indices].view(mx.uint16), thin.view(mx.uint16)).item()
        assert mx.array_equal(predictions[indices], predicted).item()
        path = accepted_path(remapped, thin_tokens, predicted.tolist())
        original_path = [verifier.last_rows[i] for i in path]
        assert original_path == [0, 14, 15]
        original_deferred = model.accept_tree(a, tree, offset, original_path, defer_kv=defer)
        pruned_deferred = commit(b, narrow, offset, path, defer_kv=defer)
        if defer:
            assert original_deferred and pruned_deferred
        follow = predicted[path[-1]:path[-1] + 1]
        next_a = model.hidden(follow, a, offset + len(path))
        next_b = model.hidden(follow, b, offset + len(path))
        model.evaluate(next_a, a)
        model.evaluate(next_b, b)
        assert mx.array_equal(next_a.view(mx.uint16), next_b.view(mx.uint16)).item()
    print("parity", json.dumps({"widths": [WIDTH, KEEP], "bitwise_retained": True,
                                "bitwise_followup_after_immediate_or_deferred_commit": True}), flush=True)


def compare(trace, edge_model, threshold, budget):
    def run(model, cache, mtp, *args, **kwargs):
        exact_midpoint(model, cache, trace)
        midpoint = Midpoint(edge_model, threshold)

        def start(variant):
            values = kwargs.copy()
            if (head := values.get("short_head")) is not None:
                current = copy(head)
                current.retained = head.retained.copy()
                current.complement_flags = []
                values["short_head"] = current
            return iterator(variant)(model, [replace(s) for s in cache], replace(mtp), *args, **values)

        before, thermal = pages(), thermal_state()
        results, elapsed = exercise({"full": start(None), "midpoint": start(midpoint)}, 1024, report_progress=True)
        for name, (tokens, stats) in results.items():
            assert tokens == trace["output"], (name, digest(tokens))
            print("live", json.dumps({"arm": name, "tok_s": 1023 / elapsed[name], "seconds": elapsed[name],
                                      "verified_tokens": len(tokens), "sha256": digest(tokens),
                                      "rounds": stats["target_decode_passes"],
                                      "target_s": stats["target_s"], "draft_s": stats["draft_s"],
                                      "accept_s": stats["accept_s"]}), flush=True)
        print("pruning", json.dumps({**midpoint.counts, "score_and_sync_ms": median(midpoint.scores_ms),
                                     "estimated_loss": [min(midpoint.risks), median(midpoint.risks), max(midpoint.risks)],
                                     "thermal": [thermal, thermal_state()], "peak_gib": mx.get_peak_memory() / GIB,
                                     "faults": {key: val - before[key] for key, val in pages().items()},
                                     **budget.metrics()}), flush=True)
        return results["full"]
    return run


def main():
    traces = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())
    stop = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stop = (stop,) if isinstance(stop, int) else tuple(stop)
    before = pages()
    with metal_job(64, 2) as budget:
        model = Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        budget.admit(model, 5120, 512, True, extra_bytes=GIB)
        examples, records = [], []
        for name, trace in traces.items():
            if name == HOLDOUT:
                continue
            current, training_rounds, rounds, score_ms = fit_examples(model, trace, budget)
            examples.extend(current)
            records.extend(training_rounds)
            print("training", json.dumps({"workload": name, "rounds": rounds, "examples": len(current),
                                          "row_probe_ms": score_ms, "thermal": thermal_state()}), flush=True)
            mx.clear_cache()
            budget.check(force=True)
        edge_model = fit(examples)
        threshold, selected, net = calibrate(records, edge_model)
        print("policy", json.dumps({"layer": CUT, "loss_limit": threshold, "weights": edge_model.coefficients,
                                    "offset": edge_model.offset, "scale": edge_model.scale, "training_shortened": selected,
                                    "training_wide": len(records), "training_net_in_token_time": net,
                                    "training_peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
        trace = traces[HOLDOUT]
        assert (len(trace["input"]), len(trace["output"])) == (4096, 1024)
        with patch.object(cli, "speculative", compare(trace, edge_model, threshold, budget)):
            actual, timing = cli.generate(model, trace["input"], 1024, 512, stop=stop, budget=budget,
                                          draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
        assert actual == trace["output"] and timing["draft_schedule"] == "frontier_prefetch"
        print("host", json.dumps({"prefill_peak_gib": timing["prefill_peak_gib"],
                                  "faults": {key: val - before[key] for key, val in pages().items()},
                                  **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

"""Interleave the production MTP schedule with narrower copied-phrase trees."""

from collections import Counter
from copy import copy
from dataclasses import replace
import inspect
import json
from statistics import median
from time import perf_counter
from unittest.mock import patch

import mlx.core as mx

from localllm import cli, frontier
from localllm.draft_short_head import prepare_short_head
from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from localllm.model import Model
from localllm.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from localllm.server import warm_verification_graphs
from scripts.bench_dflash2 import pages
from scripts.bench_frontier_phrase_width import (
    HOT_PRICE, NARROW, RecentWidthPrice, candidates, fit_phrase_table, phrase_features, propose_prune,
)
from scripts.bench_ready import TARGET, digest


def build_with_price(table, counters, observed, price_for_width):
    def shrink(nodes, children, selection_mass, samples, history):
        native = len(selection_mass)
        if len(nodes) <= max(native, NARROW):
            return
        counters["eligible"] += 1
        price = price_for_width(len(nodes))
        if price is None:
            counters["no_price"] += 1
            return
        observed.append(price)
        shape = [(node.parent, node.token, node.depth, node.mass) for node in nodes]
        cell = phrase_features(shape, native, history, {node: sample[1] for node, sample in samples.items()})
        keep = propose_prune(shape, selection_mass, cell, table, price, repack=False)
        if keep is None:
            return
        counters["pruned"] += 1
        counters["removed_slots"] += len(nodes) - NARROW
        history.added_nodes -= len(nodes) - native
        history.rounds -= 1
        del nodes[NARROW:]
        del children[NARROW:]
        for edges in children:
            edges[:] = [node for node in edges if node < NARROW]
        for node in list(samples):
            if node >= NARROW:
                del samples[node]

    text = inspect.getsource(frontier.build_frontier)
    anchor = "        append_history_phrase(phrases, nodes, children, remaining, width_limit)"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "\n        shrink(nodes, children, selection_mass, samples, phrases)")
    scope = {**vars(frontier), "shrink": shrink}
    exec(compile(text, __file__, "exec"), scope)
    return scope["build_frontier"]


def round_iterator(builder, samples, record=None):
    source = inspect.getsource(cli.speculative)
    boundary = '    stats["accepted_per_pass_histogram"] = acceptance'
    accept = "        accept_start = perf_counter()"
    end = '        stats["accept_s"] += perf_counter() - accept_start'
    assert source.count(boundary) == 1
    assert source.count(accept) == 1
    assert source.count(end) == 1
    source = source.replace(accept, accept + "\n        samples.append((len(tokens), accept_start - target_start))")
    if record is not None:
        source = source.replace(end, end + "\n        record(len(tokens), accept_start - target_start, perf_counter() - step_start)")
    namespace = {**vars(cli), "build_frontier": builder, "samples": samples, "record": record}
    exec(compile(source.replace(boundary, "        yield len(generated)\n" + boundary), __file__, "exec"), namespace)
    return namespace["speculative"]


def compare(trace, table, warm_before, recent_price, include_fixed):
    samples = {"full": []}
    controls = {"full": round_iterator(frontier.build_frontier, samples["full"])}
    counters, prices = {}, {}

    def add_control(name, price_for_width, record=None):
        counters[name], prices[name], samples[name] = Counter(), [], []
        builder = build_with_price(table, counters[name], prices[name], price_for_width)
        controls[name] = round_iterator(builder, samples[name], record)

    if not recent_price or include_fixed:
        add_control("narrow_phrase", lambda _width: HOT_PRICE)
    if recent_price:
        estimator = RecentWidthPrice()
        add_control("recent_phrase", estimator, estimator.record)

    indices = {name: i for i, name in enumerate(controls)}

    def run(model, cache, mtp, *args, **kwargs):
        assert sum(s.prompt_kv is not None for s in cache) == 16

        def stream(name):
            options = kwargs.copy()
            if (head := options.get("short_head")) is not None:
                short_head = copy(head)
                short_head.retained = head.retained.copy()
                short_head.complement_flags = []
                options["short_head"] = short_head
            return controls[name](model, [replace(s) for s in cache], replace(mtp), *args, **options)

        for warm_run in range(warm_before):
            warm = stream("full")
            started, initial_thermal = perf_counter(), thermal_state()
            try:
                while True:
                    next(warm)
            except StopIteration as end:
                assert end.value[0] == trace["output"]
            print("warm_before", json.dumps({"run": warm_run + 1, "seconds": perf_counter() - started,
                  "thermal": [initial_thermal, thermal_state()]}), flush=True)
            samples["full"].clear()

        streams = {name: stream(name) for name in controls}
        results, elapsed = {}, dict.fromkeys(controls, 0.0)
        progress = dict.fromkeys(controls, 1)
        before, thermal = pages(), thermal_state()
        threshold, alternate = 256, 0
        try:
            while streams:
                name = min(streams, key=lambda key: (progress[key], (indices[key] - alternate) % len(controls)))
                alternate = (indices[name] + 1) % len(controls)
                started = perf_counter()
                try:
                    progress[name] = next(streams[name])
                except StopIteration as end:
                    results[name] = end.value
                    progress[name] = 1024
                    del streams[name]
                elapsed[name] += perf_counter() - started
                if min(progress.values()) >= threshold:
                    print("progress", json.dumps({"tokens": progress, "seconds": elapsed,
                          "thermal": thermal_state()}), flush=True)
                    threshold += 256
        finally:
            for stream in streams.values():
                stream.close()
        after = pages()
        for name, (tokens, stats) in results.items():
            truth = trace["output"]
            divergence = next((i for i, (a, b) in enumerate(zip(tokens, truth)) if a != b), None)
            if divergence is None and len(tokens) != len(truth):
                divergence = min(len(tokens), len(truth))
            kept = ("target_decode_passes", "draft_s", "target_s", "accept_s", "draft_phrase_nodes")
            target_ms = {width: median(t * 1000 for size, t in samples[name] if size == width)
                         for width in (8, 11, 16) if any(size == width for size, _ in samples[name])}
            price_sample = prices.get(name, ())
            policy = {**counters.get(name, {}), "median_price": median(price_sample) if price_sample else None}
            print("result", name, json.dumps({"tok_s": (len(tokens) - 1) / elapsed[name],
                  "tokens": len(tokens), "exact_output": tokens == truth, "first_divergence": divergence,
                  "sha256": digest(tokens), "wide_rounds": sum(stats["drafts_per_pass_histogram"][NARROW:]),
                  "target_ms_by_width": target_ms, "policy": policy,
                  **{key: stats[key] for key in kept}}), flush=True)
        print("paired", json.dumps({"peak_gib": mx.get_peak_memory() / GIB, "thermal": [thermal, thermal_state()],
              "faults": {key: after[key] - before[key] for key in before}}), flush=True)
        assert results["full"][0] == trace["output"]
        return results["full"]
    return run


def live(traces, workload, warm_before=0, recent_price=False, include_fixed=False):
    trace = traces[workload]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    table = fit_phrase_table({name: list(candidates(tr)) for name, tr in traces.items()}, workload)
    stops = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stops = (stops,) if isinstance(stops, int) else tuple(stops)
    before = pages()
    with metal_job(80, 2) as budget:
        model = Model(TARGET, budget=budget, mtp=True, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        print("workload", workload, "price", "recent" if recent_price else HOT_PRICE,
              "load_peak_gib", mx.get_peak_memory() / GIB, flush=True)
        with patch.object(cli, "speculative", compare(trace, table, warm_before, recent_price, include_fixed)):
            _, stats = cli.generate(model, trace["input"], 1024, 512, stop=stops, budget=budget,
                                    draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
        assert stats["draft_schedule"] == "frontier_prefetch"
        after = pages()
        print("host", json.dumps({"prefill_peak_gib": stats["prefill_peak_gib"],
              "faults": {key: after[key] - before[key] for key in before}, **budget.metrics()}), flush=True)

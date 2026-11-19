"""Try direct BF16 packet loads only on 16-position verification batches."""

import argparse
from collections import Counter
from copy import copy
from dataclasses import replace
import inspect
import json
from statistics import median
from time import perf_counter
from types import MethodType
from unittest.mock import patch

import mlx.core as mx

from ultrarun import cli
from ultrarun.draft_short_head import prepare_short_head
from ultrarun.host import thermal_state
from ultrarun.lossless_gate_up import LosslessPaired
from ultrarun.lossless_projection import LosslessDown
from ultrarun.memory import GIB, metal_job
from ultrarun.model import Model
from ultrarun.packed_projection import pack_activations
from ultrarun.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from ultrarun.server import warm_verification_graphs
from scripts.bench_dflash2 import pages
from scripts.bench_ready import ROOT, TARGET, digest
from scripts.bench_verbatim_ffn import (
    FFN, LAYERS as SAMPLE_LAYERS, direct_down, direct_gate, replace_once, verbatim_down, verbatim_gate,
)


LAYERS = tuple(range(0, 64, 5)) + (16, 33, 62)
PRODUCTION_SPECULATIVE = cli.speculative
PACKERS = dict(zip(FFN, (verbatim_gate, verbatim_down), strict=True))
PROJECTIONS = {"gate": FFN[:1], "down": FFN[1:], "both": FFN}


def alternate_model(model, budget, projection, *, layers=LAYERS):
    view = copy(model)
    view.layers = list(model.layers)
    scratch = 2 * GIB
    resident_extra_bytes = 0
    for index in layers:
        budget.check(force=True, required_bytes=scratch)
        layer = copy(model.layers[index])
        layer.w = dict(layer.w)
        names = PROJECTIONS[projection]
        raw = [PACKERS[name](layer.w[name]) for name in names]
        # Rank four marks these diagnostic packets without changing their contents.
        chosen = [type(w)((w.payload[0][None], *w.payload[1:])) for w in raw]
        layer.w.update(zip(names, chosen, strict=True))
        mx.eval([w.payload for w in chosen])
        resident_extra_bytes += sum(t.nbytes for w in chosen for t in w.payload)
        view.layers[index] = layer
        mx.clear_cache()
    print("raw_layout", json.dumps({"layers": sorted(layers), "projection": projection,
          "resident_extra_bytes": resident_extra_bytes,
          "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
    return view


def projection_probe(model, budget):
    """Measure both 16-position projections on identical frozen activations."""
    width = 16
    packed = pack_activations(mx.random.normal((width, model.config["hidden_size"])).astype(mx.bfloat16))
    packed_mids = []
    weights = [tuple(model.layers[i].w[name] for name in FFN) for i in SAMPLE_LAYERS]
    raw = []
    for g, d in weights:
        budget.check(force=True, required_bytes=2 * GIB)
        mid = g.project(packed, width, swiglu=True)
        mx.eval(mid)
        packed_mids.append(mid)
        raw.append((verbatim_gate(g), verbatim_down(d)))

    def gate(compressed):
        if compressed:
            return [g.project(packed, width, swiglu=True) for g, _ in weights]
        return [direct_gate(g, packed, width) for g, _ in raw]

    def down(compressed):
        if compressed:
            return [d.partials(mid, width) for (_, d), mid in zip(weights, packed_mids, strict=True)]
        return [direct_down(d, mid, width) for (_, d), mid in zip(raw, packed_mids, strict=True)]

    probes = {"gate": (gate, mx.uint16), "down": (down, mx.uint32)}
    timing = {name: [] for name in probes}
    for operation, dtype in probes.values():
        for a, b in zip(operation(True), operation(False), strict=True):
            assert mx.array_equal(a.view(dtype), b.view(dtype)).item()
    names = tuple(probes)
    for trial in range(20):
        order = names if trial % 2 == 0 else names[::-1]
        for name in order:
            operation, _ = probes[name]
            elapsed = {}
            flags = (True, False) if trial % 4 < 2 else (False, True)
            for flag in flags:
                start = perf_counter()
                mx.eval(operation(flag))
                elapsed[flag] = (perf_counter() - start) * 1000
            timing[name].append((elapsed[True], elapsed[False]))
        budget.check()
    result = {}
    for i, (name, trials) in enumerate(timing.items()):
        extra = sum(t.nbytes for w in raw for t in w[i].payload)
        original, direct = zip(*trials, strict=True)
        saved = [a - b for a, b in trials]
        result[name] = {"compressed_ms": median(original), "direct_ms": median(direct),
                        "paired_saved_ms": median(saved), "faster_pairs": sum(x > 0 for x in saved),
                        "extra_gib": extra / GIB, "saved_ms_per_gib": median(saved) * GIB / extra}
    print("projection_probe", json.dumps({"layers": SAMPLE_LAYERS, "width": width, "thermal": thermal_state(),
          "timing": result, "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
    return max(result, key=lambda name: result[name]["saved_ms_per_gib"])


def routing(model, alternate, counts):
    view = copy(model)

    def hidden(_self, tokens, cache, offset, keep_replay=False, tree=None):
        active = keep_replay and tokens.shape[0] == 16
        if active:
            counts["direct_batches"] += 1
        engine = alternate if active else model
        return engine.hidden(tokens, cache, offset, keep_replay, tree)

    view.hidden = MethodType(hidden, view)
    return view


def iterator(samples):
    source = inspect.getsource(PRODUCTION_SPECULATIVE)
    end = '    stats["accepted_per_pass_histogram"] = acceptance'
    boundary = "        accept_start = perf_counter()"
    source = replace_once(source, boundary, boundary + '\n        samples.append((offset, batch.size, accept_start - target_start))')
    source = replace_once(source, end, "        yield len(generated)\n" + end)
    scope = {**vars(cli), "samples": samples}
    exec(compile(source, __file__, "exec"), scope)
    return scope["speculative"]


def compare(trace, counts, budget, projection, chosen):
    def run(model, cache, mtp, *args, **kwargs):
        def make(engine, samples):
            options = kwargs.copy()
            if (head := options.get("short_head")) is not None:
                new = copy(head)
                new.retained = head.retained.copy()
                new.complement_flags = []
                options["short_head"] = new
            return iterator(samples)(engine, [replace(s) for s in cache], replace(mtp), *args, **options)

        started = perf_counter()
        therm = thermal_state()
        warm = make(model, [])
        try:
            while True:
                next(warm)
        except StopIteration as end:
            assert end.value[0] == trace["output"]
        print("warm", json.dumps({"seconds": perf_counter() - started,
              "thermal": [therm, thermal_state()]}), flush=True)

        chosen.append(projection_probe(model, budget) if projection == "auto" else projection)
        mx.clear_cache()
        view = routing(model, alternate_model(model, budget, chosen[0]), counts)
        # Warm the alternate compiled graphs before the first timed round of either arm.
        warm = make(view, [])
        try:
            while counts["direct_batches"] < 3:
                next(warm)
        finally:
            warm.close()
        counts.clear()
        mx.reset_peak_memory()
        budget.check(force=True)

        names = ("compressed", "direct16")
        samples = {name: [] for name in names}
        streams = {name: make(engine, samples[name]) for name, engine in zip(names, (model, view), strict=True)}
        elapsed = dict.fromkeys(names, 0.0)
        progress = dict.fromkeys(names, 1)
        results = {}
        thermal, before = thermal_state(), pages()
        threshold, turn = 256, 0
        try:
            while streams:
                name = min(streams, key=lambda key: (progress[key], (names.index(key) - turn) % 2))
                turn = 1 - names.index(name)
                start = perf_counter()
                try:
                    progress[name] = next(streams[name])
                except StopIteration as end:
                    results[name] = end.value
                    progress[name] = 1024
                    del streams[name]
                elapsed[name] += perf_counter() - start
                if min(progress.values()) >= threshold:
                    print("progress", json.dumps({"tokens": progress, "seconds": elapsed,
                          "thermal": thermal_state()}), flush=True)
                    threshold += 256
        finally:
            for stream in streams.values():
                stream.close()

        for name, (tokens, record) in results.items():
            print("result", name, json.dumps({"tok_s": (len(tokens) - 1) / elapsed[name],
                  "tokens": len(tokens), "exact": tokens == trace["output"], "sha256": digest(tokens),
                  "rounds": record["target_decode_passes"], "target_s": record["target_s"],
                  "draft_s": record["draft_s"], "accept_s": record["accept_s"]}), flush=True)
        assert all(tokens == trace["output"] for tokens, _ in results.values())
        assert counts["direct_batches"] > 0
        difference = {"width16": [], "other": []}
        for a, b in zip(samples["compressed"], samples["direct16"], strict=True):
            assert a[:2] == b[:2]
            difference["width16" if a[1] == 16 else "other"].append((b[2] - a[2]) * 1000)
        after = pages()
        print("paired", json.dumps({"by_width": {name: {"rounds": len(values), "paired_median_ms": median(values),
              "total_ms": sum(values)} for name, values in difference.items()}, "peak_gib": mx.get_peak_memory() / GIB,
              "thermal": [thermal, thermal_state()], "faults": {key: after[key] - before[key] for key in before},
              **counts}), flush=True)
        return results["compressed"]
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", choices=("both", "gate", "down", "auto"), default="both",
                        help="choose which projection uses direct packet reads")
    args = parser.parse_args()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    stop = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stops = (stop,) if isinstance(stop, int) else tuple(stop)
    dispatch, partials = LosslessPaired.dispatch, LosslessDown.partials
    traced = Counter()

    def gate(w, x, width, mode, shapes, dtypes, *extra):
        if w.payload[0].ndim == 4:
            assert width == 16 and mode == "swiglu"
            traced["gate"] += 1
            return [direct_gate(w, x, width)]
        return dispatch(w, x, width, mode, shapes, dtypes, *extra)

    def down(w, x, width):
        if w.payload[0].ndim == 4:
            assert width == 16
            traced["down"] += 1
            return direct_down(w, x, width)
        return partials(w, x, width)

    before = pages()
    mx.random.seed(184)
    with metal_job(80, 2) as budget:
        model = Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        counts = Counter()
        chosen = []
        with (patch.object(LosslessPaired, "dispatch", gate), patch.object(LosslessDown, "partials", down),
              patch.object(cli, "speculative", compare(trace, counts, budget, args.projection, chosen))):
            _, stats = cli.generate(model, trace["input"], 1024, 512, stop=stops, budget=budget,
                                    draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
        assert stats["draft_schedule"] == "frontier_prefetch"
        assert set(traced) == (set(("gate", "down")) if chosen[0] == "both" else {chosen[0]})
        print("direct_dispatch_traces", json.dumps(traced), flush=True)
        after = pages()
        print("host", json.dumps({"prefill_peak_gib": stats["prefill_peak_gib"],
              "faults": {key: after[key] - before[key] for key in before}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

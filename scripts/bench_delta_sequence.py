"""Compare direct DeltaNet history writes with the deployed BF16 projection."""

import argparse
from collections import defaultdict
from copy import copy
from dataclasses import replace
from hashlib import sha256
import inspect
import json
from statistics import median
from unittest.mock import patch

import mlx.core as mx

from localllm import cli, model as modeling
from localllm.draft_short_head import prepare_short_head
from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from localllm.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from localllm.server import warm_verification_graphs
from scripts.bench_dflash2 import pages
from scripts.bench_dflash2_tandem import exercise
from scripts.bench_ready import ROOT, TARGET, digest
from scripts.bench_verbatim_ffn import replace_once
from scripts.bench_verify_stage_drift import Fixture, same_bits


BASE_SPECULATIVE = cli.speculative
WORKLOADS = ("code", "chinese_technical_report", "english_technical_tutorial")


def alternate_model(model):
    candidate = copy(model)
    candidate.direct_delta_sequence = True
    candidate.layers = list(model.layers)
    compiled = {}
    for i, base in enumerate(model.layers):
        if not base.gdn:
            continue
        layer = copy(base)
        layer.direct_delta_sequence = True
        if layer.compiled_delta is not None:
            signature = tuple((name, type(weight)) for name, weight in layer.w.items())
            if signature not in compiled:
                compiled[signature] = modeling.compile_delta_layer(layer)
            layer.compiled_delta = compiled[signature]
        candidate.layers[i] = layer
    return candidate


def check_projection(model, cache, width):
    x = model.embedding[mx.arange(width, dtype=mx.int32)]
    for layer, state in zip(model.layers, cache, strict=True):
        if not layer.gdn:
            continue
        norm = mx.fast.rms_norm(x, layer.w["input_layernorm.weight"], layer.eps).astype(x.dtype)
        qkv, z = mx.split(layer.project(norm, "delta_qkvz"), [state.conv.shape[1]], axis=-1)
        sequence, gate = layer.w["delta_qkvz"].project_sequence(modeling.pack_activations(norm), width, state.conv)
        assert same_bits(sequence, mx.concatenate((state.conv, qkv))), width
        assert same_bits(gate, z), width


def rounds(route, samples):
    source = inspect.getsource(BASE_SPECULATIVE)
    start = "        verified = model.hidden(batch, cache, offset, keep_replay=bool(proposals), tree=tree)"
    source = replace_once(source, start,
                          "        route.update(json.dumps((offset, tokens, parents if tree is not None else None)).encode())\n"
                          + start)
    end = "        accept_start = perf_counter()"
    source = replace_once(source, end, end + "\n        samples.append((offset, batch.size, accept_start - target_start))")
    boundary = '    stats["accepted_per_pass_histogram"] = acceptance'
    source = replace_once(source, boundary, "        yield len(generated)\n" + boundary)
    scope = {**vars(cli), "route": route, "samples": samples, "json": json}
    exec(compile(source, __file__, "exec"), scope)
    return scope["speculative"]


def hook(trace, model_view, budget, *, verify_fixtures):
    def run(model, cache, mtp, *args, **kwargs):
        engines = {"base": model, "direct": model_view}
        if verify_fixtures:
            for width in (1, 6, 8, 11, 16):
                check_projection(model, cache, width)
            for width in (8, 11, 16):
                a = Fixture(model, cache, trace, width, None)
                b = Fixture(model_view, cache, trace, width, None)
                predicted = a.parity(a.full()[0], b.full()[0])
                print("fixed", json.dumps({"width": width, "prediction_digest": predicted,
                                           "retained_cache_bitwise": True}), flush=True)

        def start(engine, route, samples):
            options = kwargs.copy()
            if (head := options.get("short_head")) is not None:
                current = copy(head)
                current.retained = head.retained.copy()
                current.complement_flags = []
                options["short_head"] = current
            return rounds(route, samples)(engine, [replace(s) for s in cache], replace(mtp), *args, **options)

        for engine in engines.values():
            warm = start(engine, sha256(), [])
            try:
                while next(warm) < 65:
                    pass
            finally:
                warm.close()
        mx.clear_cache()
        budget.check(force=True)
        mx.reset_peak_memory()
        routes = {name: sha256() for name in engines}
        samples = {name: [] for name in engines}
        before, thermal = pages(), thermal_state()
        results, elapsed = exercise({name: start(engine, routes[name], samples[name])
                                     for name, engine in engines.items()}, 1024, report_progress=True)
        assert routes["base"].digest() == routes["direct"].digest()
        for name, (tokens, stats) in results.items():
            assert tokens == trace["output"]
            print("decode", json.dumps({"arm": name, "tokens": len(tokens), "sha256": digest(tokens),
                                        "rounds": stats["target_decode_passes"], "seconds": elapsed[name],
                                        "target_s": stats["target_s"], "draft_s": stats["draft_s"],
                                        "tok_s": (len(tokens) - 1) / elapsed[name]}), flush=True)
        widths = defaultdict(list)
        for x, y in zip(samples["base"], samples["direct"], strict=True):
            assert x[:2] == y[:2]
            widths[x[1]].append((x[-1] - y[-1]) * 1000)
        print("paired", json.dumps({"identical_routes": True, "widths": {
              w: {"count": len(v), "first_saved_ms": v[0], "saved_median_ms": median(v), "saved_total_ms": sum(v)}
              for w, v in widths.items()},
              "peak_gib": mx.get_peak_memory() / GIB, "thermal": [thermal, thermal_state()],
              "faults": {key: after - before[key] for key, after in pages().items()}, **budget.metrics()}), flush=True)
        return results["base"]
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, action="append")
    names = parser.parse_args().workload or WORKLOADS[:2]
    traces = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())
    eos = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stop = (eos,) if isinstance(eos, int) else tuple(eos)
    before = pages()
    with metal_job(64, 2) as budget:
        model = modeling.Model(TARGET, mtp=True, budget=budget,
                               **{**OPTIMIZED_MODEL, "direct_delta_sequence": False})
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        view = alternate_model(model)
        warm_verification_graphs(view, budget)
        print("direct_widths", sorted(modeling.DIRECT_DELTA_WIDTHS), flush=True)
        for i, name in enumerate(names):
            trace = traces[name]
            assert (len(trace["input"]), len(trace["output"])) == (4096, 1024)
            print("workload", name, flush=True)
            with patch.object(cli, "speculative", hook(trace, view, budget, verify_fixtures=i == 0)):
                actual, timing = cli.generate(model, trace["input"], 1024, 512, stop=stop, budget=budget,
                                              draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
            assert actual == trace["output"] and timing["draft_schedule"] == "frontier_prefetch"
            print("prefill", json.dumps({"workload": name, "peak_gib": timing["prefill_peak_gib"]}), flush=True)
            mx.clear_cache()
            budget.check(force=True)
        after = pages()
        print("host", json.dumps({
            "thermal": thermal_state(),
            "faults": {key: after[key] - total for key, total in before.items()}, **budget.metrics(),
        }), flush=True)


if __name__ == "__main__":
    main()

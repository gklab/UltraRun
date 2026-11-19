"""Scale resident BF16 FFN packets for exact wide verifier batches."""

from collections import Counter
from copy import copy
from dataclasses import replace
import json
from statistics import median
from time import perf_counter
from types import MethodType
from unittest.mock import patch

import mlx.core as mx

from localllm import cli
from localllm.draft_short_head import prepare_short_head
from localllm.host import thermal_state
from localllm.lossless_gate_up import LosslessPaired
from localllm.lossless_projection import LosslessDown
from localllm.memory import GIB, metal_job
from localllm.model import Model
from localllm.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from localllm.server import warm_verification_graphs
from scripts.bench_dflash2 import pages
from scripts.bench_dflash2_tandem import exercise
from scripts.bench_ready import ROOT, TARGET, digest
from scripts.bench_verbatim_ffn import FFN, direct_down, direct_gate
from scripts.bench_verbatim_ffn_live import alternate_model, iterator
from scripts.bench_verify_stage_drift import Fixture


FIRST = tuple(range(0, 64, 4))
SECOND = tuple(range(2, 64, 4))
WIDTHS = (8, 11, 16)


def resident_bytes(model):
    weights = (layer.w[name] for layer in model.layers for name in FFN)
    return sum(a.nbytes for w in weights if w.payload[0].ndim == 4 for a in w.payload)


def projection_dispatch():
    dispatch, partials = LosslessPaired.dispatch, LosslessDown.partials

    def gate(weight, x, width, mode, shapes, dtypes, *extra):
        if weight.payload[0].ndim == 4:
            assert width in WIDTHS and mode == "swiglu"
            return [direct_gate(weight, x, width)]
        return dispatch(weight, x, width, mode, shapes, dtypes, *extra)

    def down(weight, x, width):
        if weight.payload[0].ndim == 4:
            assert width in WIDTHS
            return direct_down(weight, x, width)
        return partials(weight, x, width)

    return gate, down


def fixed_snapshots(models, cache, trace, budget):
    profiling = {}
    for width in WIDTHS:
        fixtures = {name: Fixture(model, cache, trace, width, None) for name, model in models.items()}
        base = fixtures["compact"]
        prediction = {name: base.parity(base.full()[0], candidate.full()[0])
                      for name, candidate in fixtures.items() if name != "compact"}
        times = {name: [] for name in models}
        names = tuple(models)
        for i in range(12):
            shift = i % len(names)
            for name in names[shift:] + names[:shift]:
                _, elapsed = fixtures[name].full()
                times[name].append(elapsed)
            budget.check()
        profiling[width] = {}
        for name, samples in times.items():
            saved = [a - b for a, b in zip(times["compact"], samples, strict=True)]
            extra_gib = resident_bytes(models[name]) / GIB
            profiling[width][name] = {
                "median_ms": median(samples), "paired_saved_ms": median(saved), "wins": sum(x > 0 for x in saved),
                "resident_gib": extra_gib, "saved_ms_per_gib": median(saved) / extra_gib if extra_gib else None,
                "bitwise_prediction": prediction.get(name),
            }
        print("fixed", json.dumps({"width": width, "models": profiling[width], "pairs": len(times["compact"]),
                                  "thermal": thermal_state(), "peak_gib": mx.get_peak_memory() / GIB,
                                  **budget.metrics()}), flush=True)
    return profiling


def routed(base, alternative, enabled, counts):
    view = copy(base)

    def hidden(_self, tokens, cache, offset, keep_replay=False, tree=None):
        width = tokens.shape[0]
        if keep_replay and width in enabled:
            counts[width] += 1
            return alternative.hidden(tokens, cache, offset, keep_replay, tree)
        return base.hidden(tokens, cache, offset, keep_replay, tree)

    view.hidden = MethodType(hidden, view)
    return view


def native_pair(trace, models, cache, mtp, args, options, budget):
    def start(name):
        kwargs = options.copy()
        if (head := kwargs.get("short_head")) is not None:
            head_copy = copy(head)
            head_copy.retained = head.retained.copy()
            head_copy.complement_flags = []
            kwargs["short_head"] = head_copy
        return iterator(samples[name])(models[name], [replace(s) for s in cache], replace(mtp), *args, **kwargs)

    samples = {name: [] for name in models}
    before, thermal = pages(), thermal_state()
    results, seconds = exercise({name: start(name) for name in models}, 1024, report_progress=True)
    assert len(results) == 2 and all(tokens == trace["output"] for tokens, _ in results.values())
    for name, (tokens, stats) in results.items():
        print("live", json.dumps({"arm": name, "decode_tok_s": (len(tokens) - 1) / seconds[name],
                                  "tokens": len(tokens), "sha256": digest(tokens), "seconds": seconds[name],
                                  "rounds": stats["target_decode_passes"], "exact": True,
                                  "phases_s": {key: stats[key] for key in ("draft_s", "target_s", "accept_s")}}), flush=True)
    groups = {w: [] for w in set(a[1] for a in samples["compact"])}
    for a, b in zip(samples["compact"], samples["raw"], strict=True):
        assert a[:2] == b[:2]
        groups[a[1]].append((a[2] - b[2]) * 1000)
    after = pages()
    print("paired", json.dumps({"widths": {width: {"rounds": len(values), "saved_median_ms": median(values),
                                                  "saved_total_ms": sum(values)} for width, values in groups.items()},
                                "faults": {k: after[k] - v for k, v in before.items()},
                                "thermal": [thermal, thermal_state()],
                                "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
    return results["compact"]


def hook(trace, budget):
    def run(model, cache, mtp, *args, **kwargs):
        # Both views share every tensor except the predecoded FFN packets in the
        # selected layers. Build them after the request's shared prefill.
        budget.admit(model, 5120, 512, True, extra_bytes=19 * GIB)
        first = alternate_model(model, budget, "both", layers=FIRST)
        measured = fixed_snapshots({"compact": model, "raw16": first}, cache, trace, budget)
        if not any(measured[w]["raw16"]["paired_saved_ms"] > 0 and measured[w]["raw16"]["wins"] >= 9
                   for w in (11, 16)):
            print("scale", json.dumps({"decision": "stop", "resident_gib": resident_bytes(first) / GIB}), flush=True)
            del first
            mx.clear_cache()
            budget.check(force=True)
            return native(model, cache, mtp, *args, **kwargs)
        second = alternate_model(first, budget, "both", layers=SECOND)
        models = {"compact": model, "raw16": first, "raw32": second}
        measured = fixed_snapshots(models, cache, trace, budget)
        # Price only widths exercised by the saved MTP schedule. The branch
        # structure and numbers of verified tokens are identical in both arms.
        coverage = Counter(len(row["order"]) for row in trace["rounds"].values())
        expected = {}
        for name in ("raw16", "raw32"):
            benefit = {w: measured[w][name]["paired_saved_ms"] * coverage[w] for w in (11, 16)}
            expected[name] = sum(max(0, value) for value in benefit.values())
        best = max(expected, key=expected.__getitem__)
        enabled = tuple(w for w in (11, 16) if measured[w][best]["paired_saved_ms"] > 0
                        and measured[w][best]["wins"] >= 9)
        print("scale", json.dumps({"decision": best, "estimated_request_saved_ms": expected,
                                   "enabled_widths": enabled, "resident_gib": resident_bytes(models[best]) / GIB,
                                   "preparation_peak_gib": mx.get_peak_memory() / GIB}), flush=True)
        if not enabled:
            del first, second, models
            mx.clear_cache()
            budget.check(force=True)
            return native(model, cache, mtp, *args, **kwargs)
        alternative = models[best]
        del first, second, models
        mx.clear_cache()
        budget.check(force=True)
        mx.reset_peak_memory()
        counts = Counter()
        view = routed(model, alternative, enabled, counts)
        result = native_pair(trace, {"compact": model, "raw": view}, cache, mtp, args, kwargs, budget)
        assert all(counts[w] > 0 for w in enabled)
        print("routed_batches", json.dumps(counts), flush=True)
        return result
    native = cli.speculative
    return run


def main():
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert (len(trace["input"]), len(trace["output"])) == (4096, 1024)
    eos = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stop = (eos,) if isinstance(eos, int) else tuple(eos)
    before = pages()
    with metal_job(66, 2) as budget:
        model = Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        gate, down = projection_dispatch()
        with (patch.object(LosslessPaired, "dispatch", gate), patch.object(LosslessDown, "partials", down),
              patch.object(cli, "speculative", hook(trace, budget))):
            tokens, stats = cli.generate(model, trace["input"], 1024, 512, stop=stop, budget=budget,
                                         draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
        assert tokens == trace["output"] and stats["draft_schedule"] == "frontier_prefetch"
        after = pages()
        print("host", json.dumps({"prefill_peak_gib": stats["prefill_peak_gib"],
                                  "decode_and_probe_peak_gib": stats["decode_peak_gib"],
                                  "faults": {k: after[k] - v for k, v in before.items()},
                                  **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

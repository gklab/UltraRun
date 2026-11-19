"""Measure whether a charged mid-request pause improves sustained BF16 decoding."""

from collections import defaultdict
from copy import copy
from dataclasses import replace
from hashlib import sha256
import json
from statistics import median
from time import perf_counter, sleep
from unittest.mock import patch

import mlx.core as mx

from localllm import cli
from localllm.draft_short_head import prepare_short_head
from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from localllm.model import Model
from localllm.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from localllm.server import warm_verification_graphs
from scripts.bench_delta_sequence import rounds
from scripts.bench_dflash2 import pages
from scripts.bench_ready import ROOT, TARGET, digest


ARMS = ("continuous_before", "pause", "continuous_after")
PAUSE_SECONDS = 5
REST_SECONDS = 20


def report(tag, data):
    print(tag, json.dumps(data), flush=True)


def generation_hook(trace, budget):
    def run(model, cache, mtp, *args, **kwargs):
        def stream(route, samples):
            options = kwargs.copy()
            if (head := options.get("short_head")) is not None:
                head = copy(head)
                head.retained = head.retained.copy()
                head.complement_flags = []
                options["short_head"] = head
            return rounds(route, samples)(model, [replace(s) for s in cache], replace(mtp), *args, **options)

        warm = stream(sha256(), [])
        try:
            while next(warm) < 65:
                pass
        finally:
            warm.close()
        mx.synchronize()
        mx.clear_cache()
        budget.check(force=True)
        report("setup", {"peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()})

        def measure(arm):
            sleep(REST_SECONDS)
            budget.check(force=True)
            before = pages()
            mx.reset_peak_memory()
            route = sha256()
            samples = []
            runner = stream(route, samples)
            milestone = 256
            window_start, prior_round = 0.0, 0
            paused = 0.0
            start_thermal = thermal_state()
            began = perf_counter()
            try:
                while True:
                    try:
                        cursor = next(runner)
                    except StopIteration as terminal:
                        seconds = perf_counter() - began
                        result = terminal.value
                        break
                    if cursor < milestone:
                        continue
                    elapsed = perf_counter() - began
                    per_width = defaultdict(list)
                    for _, width, duration in samples[prior_round:]:
                        per_width[width].append(1000 * duration)
                    report("window", {
                        "arm": arm, "tokens": cursor, "elapsed_s": elapsed,
                        "since_last_window_s": elapsed - window_start,
                        "verify_by_width_ms": {w: {"count": len(values), "median": median(values)}
                                               for w, values in per_width.items()},
                        "thermal": thermal_state(),
                    })
                    window_start, prior_round = elapsed, len(samples)
                    if arm == "pause" and milestone == 512:
                        mx.synchronize()
                        pause_start = perf_counter()
                        sleep(PAUSE_SECONDS)
                        paused = perf_counter() - pause_start
                        report("pause", {"after_tokens": cursor, "seconds": paused, "thermal": thermal_state()})
                    milestone += 256
            finally:
                runner.close()
            generated, stats = result
            assert generated == trace["output"]
            assert stats["target_decode_passes"] == len(samples) == 213
            after = pages()
            signature = route.hexdigest()
            report("run", {
                "arm": arm, "tokens": len(generated), "route_sha256": signature, "token_sha256": digest(generated),
                "rounds": len(samples), "seconds": seconds, "pause_s": paused,
                "decode_tok_s": (len(generated) - 1) / seconds,
                "phase_s": {key: stats[key] for key in ("draft_s", "target_s", "accept_s")},
                "thermal": [start_thermal, thermal_state()], "peak_gib": mx.get_peak_memory() / GIB,
                "faults": {key: after[key] - value for key, value in before.items()}, **budget.metrics(),
            })
            return result, signature

        reference, signature = measure(ARMS[0])
        for arm in ARMS[1:]:
            _, actual = measure(arm)
            assert actual == signature
        return reference
    return run


def main():
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["chinese_technical_report"]
    assert (len(trace["input"]), len(trace["output"])) == (4096, 1024)
    eos = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stop = (eos,) if isinstance(eos, int) else tuple(eos)
    before = pages()
    with metal_job(64, 2) as budget:
        model = Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        load_peak = mx.get_peak_memory() / GIB
        with patch.object(cli, "speculative", generation_hook(trace, budget)):
            tokens, stats = cli.generate(model, trace["input"], 1024, 512, stop=stop, budget=budget,
                                         draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
        assert tokens == trace["output"] and stats["draft_schedule"] == "frontier_prefetch"
        after = pages()
        report("host", {"load_peak_gib": load_peak, "prefill_peak_gib": stats["prefill_peak_gib"],
                        "faults": {key: after[key] - value for key, value in before.items()}, **budget.metrics()})


if __name__ == "__main__":
    main()

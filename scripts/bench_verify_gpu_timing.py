"""Attribute BF16 verification time to Metal execution and host-visible gaps."""

import argparse
import ctypes as ct
from dataclasses import replace
import inspect
import json
from pathlib import Path
from time import perf_counter, sleep
from unittest.mock import patch

import mlx.core as mx

from ultrarun import cli
from ultrarun.draft_short_head import prepare_short_head
from ultrarun.draft_vocab_prefetch import DraftVocabPrefetch
from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.model import Model
from ultrarun.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from ultrarun.server import warm_verification_graphs
from ultrarun.spec_tree import VerificationTree
from scripts.bench_dflash2 import pages
from scripts.bench_ready import ROOT, TARGET, digest


class BufferTime(ct.Structure):
    _fields_ = [("submitted", ct.c_double), ("gpu_start", ct.c_double), ("gpu_end", ct.c_double)]


class MetalTiming:
    def __init__(self, library):
        self.lib = ct.CDLL(str(library))
        self.lib.local_timing_install.argtypes = []
        self.lib.local_timing_install.restype = ct.c_size_t
        self.lib.local_timing_begin.argtypes = []
        self.lib.local_timing_begin.restype = ct.c_double
        self.lib.local_timing_end.argtypes = []
        self.lib.local_timing_end.restype = ct.c_double
        self.lib.local_timing_read.argtypes = [ct.POINTER(BufferTime), ct.c_size_t]
        self.lib.local_timing_read.restype = ct.c_int
        if self.lib.local_timing_install() != 1:
            raise RuntimeError("Unable to instrument the local Metal command buffer class")
        self.storage = (BufferTime * 1024)()

    def begin(self):
        start = self.lib.local_timing_begin()
        if not start:
            raise RuntimeError("A prior Metal trace is still active")
        return start

    def finish(self, start):
        stop = self.lib.local_timing_end()
        count = self.lib.local_timing_read(self.storage, len(self.storage))
        deadline = perf_counter() + 0.5
        while count == -1 and perf_counter() < deadline:
            sleep(0.0002)
            count = self.lib.local_timing_read(self.storage, len(self.storage))
        if count <= 0:
            raise RuntimeError(f"No usable Metal command buffer trace ({count})")
        records = self.storage[:count]
        if any(not start <= r.submitted <= stop for r in records):
            raise RuntimeError("Metal command submission fell outside the capture window")
        intervals = [(r.gpu_start, r.gpu_end) for r in records if r.gpu_end > r.gpu_start > 0]
        if not intervals or any(a < start - 0.001 or b > stop + 0.001 for a, b in intervals):
            raise RuntimeError("Metal device timestamps are missing or outside the capture window")
        return summarize_intervals(start, stop, intervals, count)


def summarize_intervals(start, stop, intervals, count):
    ordered = sorted((max(a, start), min(b, stop)) for a, b in intervals)
    first_gpu = right = ordered[0][0]
    active = 0.0
    for a, b in ordered:
        active += max(0.0, b - max(right, a))
        right = max(right, b)
    return {"wall_ms": 1000 * (stop - start), "gpu_ms": 1000 * active,
            "inter_buffer_ms": 1000 * (right - first_gpu - active),
            "before_gpu_ms": 1000 * (first_gpu - start), "after_gpu_ms": 1000 * (stop - right),
            "command_buffers": count, "timed_buffers": len(intervals)}


def live_decoder(timing):
    source = inspect.getsource(cli.speculative)
    start, finish = "        target_start = perf_counter()", "        accept_start = perf_counter()"
    assert source.count(start) == source.count(finish) == 1
    source = source.replace(start, start + '''
        probe = timing.begin() if stats["target_decode_passes"] % 12 == 0 else None''')
    source = source.replace(finish, finish + '''
        if probe is not None:
            measure = timing.finish(probe)
            print("live", json.dumps({"round": stats["target_decode_passes"], "width": batch.size,
                  "generated": len(generated), "thermal": thermal_state(), **measure}), flush=True)''')
    namespace = {**vars(cli), "timing": timing, "thermal_state": thermal_state}
    exec(compile(source, __file__, "exec"), namespace)
    return namespace["speculative"]


def fixed_verification(timing, model, prefix, trace, before):
    ids, output = trace["input"], trace["output"]
    for width in (8, 16):
        spine = width - (1 if width == 8 else 2)
        parents = [-1, *range(spine - 1), *([1] if width == 8 else [1, spine])]
        tree = VerificationTree(parents, len(ids), 3, spine=spine)
        batch = mx.array(output[:width], dtype=mx.int32)
        mx.eval(batch)
        for i in range(3):
            state = [replace(s) for s in prefix]
            hints = DraftVocabPrefetch(model, ids, (), 128, {})
            started = timing.begin()
            hidden = model.hidden(batch, state, len(ids), keep_replay=True, tree=tree)
            pred = model.predictions(hidden, observe_vocabulary_tiles=hints.capture)
            model.evaluate([pred, *hints.hints], state)
            data = timing.finish(started)
            print("fixed", json.dumps({"stage": "before" if before else "after", "width": width,
                  "trial": i, "predictions": digest(pred.tolist()), "thermal": thermal_state(), **data}), flush=True)


def run_request(timing, trace, decode):
    def run(model, cache, mtp, *args, **kwargs):
        assert sum(s.prompt_kv is not None for s in cache) == 16
        prefix = [replace(s) for s in cache]
        fixed_verification(timing, model, prefix, trace, True)
        out = decode(model, cache, mtp, *args, **kwargs)
        fixed_verification(timing, model, prefix, trace, False)
        return out
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path, help="local dylib built from scripts/metal_verify_timing.m")
    parser.add_argument("--workload", action="append", choices=("code", "english_technical_tutorial"))
    args = parser.parse_args()
    traces = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())
    stops = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stops = (stops,) if isinstance(stops, int) else tuple(stops)
    before = pages()
    with metal_job(80, 2) as budget:
        timing = MetalTiming(args.library)
        wake = mx.sum(mx.arange(16384))
        start = timing.begin()
        mx.eval(wake)
        print("timing_smoke", json.dumps(timing.finish(start)), flush=True)
        decode = live_decoder(timing)
        model = Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        print("load_peak_gib", mx.get_peak_memory() / GIB, flush=True)
        for case in args.workload or ("english_technical_tutorial",):
            trace = traces[case]
            assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
            print("workload", case, flush=True)
            with patch.object(cli, "speculative", run_request(timing, trace, decode)):
                out, stats = cli.generate(model, trace["input"], 1024, 512, stop=stops, budget=budget,
                                          draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
            assert out == trace["output"] and stats["draft_schedule"] == "frontier_prefetch"
            kept = ("target_s", "draft_s", "target_decode_passes", "prefill_peak_gib", "decode_peak_gib")
            print("result", json.dumps({"tokens": len(out), "exact_output": True, "sha256": digest(out),
                  **{key: stats[key] for key in kept}, **budget.metrics()}), flush=True)
            mx.clear_cache()
            budget.check(force=True)
        after = pages()
        print("faults", json.dumps({key: after[key] - before[key] for key in before}), flush=True)


if __name__ == "__main__":
    main()

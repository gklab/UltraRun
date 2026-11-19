"""Compare bounded MLX command buffers on repeatable original BF16 verification."""

import argparse
import ctypes as ct
from dataclasses import replace
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter

import mlx.core as mx

from ultrarun import cli
from ultrarun.draft_short_head import prepare_short_head
from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.model import Model
from ultrarun.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from ultrarun.server import warm_verification_graphs
from ultrarun.spec_tree import VerificationTree
from scripts.bench_dflash2 import pages
from scripts.bench_ready import ROOT, TARGET, digest
from scripts.bench_verify_gpu_timing import MetalTiming


LIMIT_KEYS = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")


class BufferLimits:
    def __init__(self, library):
        if mx.__version__ != "0.32.4" or mx.device_info()["device_name"] != "Apple M2 Ultra":
            raise ValueError("The in-process buffer comparison requires the local MLX 0.32.4 M2 Ultra ABI")
        self.lib = ct.CDLL(str(library))
        self.lib.local_buffer_limits.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
        self.lib.local_buffer_limits.restype = None
        self.lib.local_set_buffer_limits.argtypes = [ct.c_int, ct.c_int]
        self.lib.local_set_buffer_limits.restype = ct.c_int
        self.choices = {"default": self.read(), "large": (128, 1024)}
        if self.choices["default"] != (50, 50):
            raise ValueError(f"Expected native 50/50 caps, got {self.choices['default']}")

    def read(self):
        ops, mb = ct.c_int(), ct.c_int()
        self.lib.local_buffer_limits(ct.byref(ops), ct.byref(mb))
        return ops.value, mb.value

    def select(self, label):
        values = self.choices[label]
        mx.synchronize()
        if self.lib.local_set_buffer_limits(*values) != 1 or self.read() != values:
            raise RuntimeError(f"Could not select benchmark buffer caps for {label}")


def partition_prompt(model, cache, context):
    for layer, state in zip(model.layers, cache, strict=True):
        state.logical_attention = layer.fused_attention_io
        state.compile_delta_tree = layer.gdn
        if state.logical_attention and state.keys is not None:
            state.prompt_kv = state.keys[:, :, :context], state.values[:, :, :context]
            state.keys = state.values = None


def fixed(model, budget, timing, trace, limits=None):
    ids, output = trace["input"], trace["output"]
    context = len(ids)
    prefix = model.cache()
    budget.admit(model, context + len(output), 512, False)
    started = perf_counter()
    for _, hidden in model.prefill(mx.array(ids, dtype=mx.int32), prefix, 512):
        model.evaluate(hidden, prefix)
        budget.check()
    assert mx.argmax(model.logits(hidden[-1:]), axis=-1).item() == output[0]
    partition_prompt(model, prefix, context)
    print("prefill", json.dumps({"ms": 1000 * (perf_counter() - started),
          "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)

    trees = {}
    for width, spine in ((8, 7), (16, 14)):
        parents = [-1, *range(spine - 1), 1]
        if width == 16:
            parents.append(spine)
        tokens = mx.array(output[:width], dtype=mx.int32)
        mx.eval(tokens)
        trees[width] = tokens, VerificationTree(parents, context, 3, spine=spine)

    def verify(width):
        states = [replace(s) for s in prefix]
        tokens, tree = trees[width]
        hidden = model.hidden(tokens, states, context, keep_replay=True, tree=tree)
        pred = model.predictions(hidden)
        model.evaluate([pred, hidden], states)
        return hidden, pred

    gold = {width: verify(width) for width in trees}
    signatures = {width: [digest(h.view(mx.uint16).tolist()), digest(p.tolist())]
                  for width, (h, p) in gold.items()}
    print("signatures", json.dumps(signatures), flush=True)

    def parity(width, hidden, predictions):
        expected_h, expected_p = gold[width]
        assert mx.array_equal(hidden.view(mx.uint16), expected_h.view(mx.uint16)).item()
        assert mx.array_equal(predictions, expected_p).item()

    labels = ("default", "large") if limits is not None else ("configured",)
    if limits:
        limits.select("large")
        for width in trees:
            parity(width, *verify(width))
        limits.select("default")

    def sample(stage):
        grouped = {width: {label: [] for label in labels} for width in trees}
        for trial in range(6 if limits else 4):
            for width in ((8, 16) if trial % 2 == 0 else (16, 8)):
                for label in (labels if trial % 2 == 0 else labels[::-1]):
                    if limits:
                        limits.select(label)
                    start = timing.begin()
                    hidden, predictions = verify(width)
                    grouped[width][label].append(timing.finish(start))
                    parity(width, hidden, predictions)
            budget.check()
        totals = {width: {label: {key: median(r[key] for r in rows) for key in rows[0]}
                          for label, rows in by_limit.items()} for width, by_limit in grouped.items()}
        if limits:
            limits.select("default")
            for width, data in grouped.items():
                delta = [large["wall_ms"] - orig["wall_ms"]
                         for orig, large in zip(data["default"], data["large"], strict=True)]
                totals[width]["paired_large_minus_default_ms"] = {"median": median(delta), "samples": delta}
        else:
            totals = {width: items["configured"] for width, items in totals.items()}
        print("sample", stage, json.dumps({"widths": totals, "thermal": thermal_state(),
              "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)

    sample("early")
    blocks = 8 if limits else 4
    for block in range(blocks):
        times = []
        for _ in range(24):
            start = perf_counter()
            verify(8)
            times.append((perf_counter() - start) * 1000)
            budget.check()
        print("sustained", json.dumps({"rounds": (block + 1) * 24, "median_ms": median(times),
              "thermal": thermal_state(), **budget.metrics()}), flush=True)
        if limits and block == 3:
            sample("midpoint")
    sample("late")


def worker(args):
    fixtures = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())
    trace = fixtures[args.workload]
    before = pages()
    print("arm", args.arm, json.dumps({"limits": {key: os.environ.get(key) for key in LIMIT_KEYS},
          "thermal": thermal_state(), "workload": args.workload, "live": args.live}), flush=True)
    with metal_job(80, 2) as budget:
        timing = None if args.live else MetalTiming(args.library)
        model = Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        if args.live:
            model.short_draft_head = prepare_short_head(model, budget)
            stop = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
            stops = (stop,) if isinstance(stop, int) else tuple(stop)
            tokens, stats = cli.generate(model, trace["input"], 1024, 512, stop=stops, budget=budget,
                                         draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
            assert tokens == trace["output"] and stats["draft_schedule"] == "frontier_prefetch"
            print("live", json.dumps({**stats, "exact_output": True, "sha256": digest(tokens),
                  "thermal": thermal_state(), **budget.metrics()}), flush=True)
        else:
            limits = BufferLimits(args.limits_library) if args.limits_library else None
            try:
                fixed(model, budget, timing, trace, limits)
            finally:
                if limits:
                    limits.select("default")
        after = pages()
        print("faults", json.dumps({key: after[key] - before[key] for key in before}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path, help="local dylib built from scripts/metal_verify_timing.m")
    parser.add_argument("--live", action="store_true", help="compare complete 4096/1024 requests")
    parser.add_argument("--limits-library", type=Path, help="alternate caps within one fixed replay process")
    parser.add_argument("--workload", default="code", choices=("code", "english_technical_tutorial"))
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.limits_library:
        if args.live or args.arm:
            parser.error("--limits-library is for an in-process fixed replay")
        args.arm = "within_process"
    if args.arm:
        worker(args)
        return
    command = [sys.executable, "-m", "scripts.bench_verify_buffer_limits", str(args.library),
               "--workload", args.workload]
    if args.live:
        command.append("--live")
    for arm in ("default_before", "128_ops_1024_mb", "default_after"):
        env = {key: value for key, value in os.environ.items() if key not in LIMIT_KEYS}
        if arm == "128_ops_1024_mb":
            env.update(zip(LIMIT_KEYS, ("128", "1024"), strict=True))
        subprocess.run([*command, "--arm", arm], env=env, check=True)


if __name__ == "__main__":
    main()

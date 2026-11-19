"""Coordinate this workspace's Metal jobs and bound their memory working set."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import re
import subprocess
from time import monotonic

import mlx.core as mx

GIB = 2**30
DEFAULT_MEMORY_GIB = 80
DEFAULT_CACHE_GIB = 2
MINIMUM_FREE_PERCENT = 12
LOCK = Path(__file__).resolve().parent.parent / "runs" / "metal.lock"


class MemoryBudgetError(Exception):
    pass


def free_percent():
    try:
        result = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"], text=True, capture_output=True, timeout=2, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"System-wide memory free percentage: (\d+)%", result.stdout)
    return int(match[1]) if match else None


@contextmanager
def metal_job(limit_gib, cache_gib):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MemoryBudgetError("An UltraRun service or benchmark already owns the Metal workspace.") from error
        budget = MemoryBudget(limit_gib, cache_gib)
        budget.check(force=True)
        try:
            yield budget
        finally:
            mx.clear_cache()


class MemoryBudget:
    def __init__(self, limit_gib, cache_gib):
        device = mx.device_info()
        self.system_size = device["memory_size"]
        self.limit = min(
            limit_gib * GIB,
            device["max_recommended_working_set_size"] * 3 // 4,
            device["memory_size"] * 3 // 5,
        )
        self.cache_limit = min(cache_gib * GIB, self.limit // 8)
        self.next_system_check = 0
        self.minimum_free = None
        mx.set_memory_limit(self.limit)
        mx.set_cache_limit(self.cache_limit)

    def checkpoint(self, size):
        if size and size + 4 * GIB > self.limit:
            raise MemoryBudgetError("The BF16 checkpoint exceeds the local Metal working-set budget.")
        if size and self.minimum_free is not None:
            available = self.system_size * self.minimum_free // 100
            if size + 4 * GIB > available:
                raise MemoryBudgetError("macOS reports insufficient available memory to load the BF16 checkpoint.")

    def admit(self, model, context, block, mtp, *, extra_bytes=0):
        config = model.config
        heads, width = config["num_key_value_heads"], config["head_dim"]
        attention_layers = config["layer_types"].count("full_attention") + bool(mtp)
        # BF16 K and V, with room for both old and new buffers during growth.
        kv = 2 * 4 * heads * width * attention_layers * (context + 511)
        # Allow working buffers for one prefill block and for DeltaNet rollback.
        scratch = max(4 * GIB, 32 * block * (config["hidden_size"] + config["intermediate_size"]))
        required_bytes = kv + scratch + extra_bytes
        if mx.get_active_memory() + required_bytes > self.limit:
            raise MemoryBudgetError("The requested context exceeds the local Metal working-set budget.")
        self.minimum_free = None
        self.check(force=True, required_bytes=required_bytes)

    def check(self, force=False, *, required_bytes=0):
        if max(mx.get_active_memory(), mx.get_peak_memory()) > self.limit:
            raise MemoryBudgetError("MLX memory exceeded the working-set budget.")
        now = monotonic()
        if force or now >= self.next_system_check:
            self.next_system_check = now + 5
            pct = free_percent()
            if pct is not None:
                self.minimum_free = pct if self.minimum_free is None else min(self.minimum_free, pct)
                # Round the useful headroom up because macOS reports a whole percentage.
                minimum = MINIMUM_FREE_PERCENT + (100 * required_bytes + self.system_size - 1) // self.system_size
                if pct < minimum:
                    mx.clear_cache()
                    after_clear = free_percent()
                    if after_clear is not None:
                        pct = after_clear
                        self.minimum_free = min(self.minimum_free, pct)
                    if pct < MINIMUM_FREE_PERCENT:
                        raise MemoryBudgetError("macOS reports low available unified memory.")
                    if pct < minimum:
                        raise MemoryBudgetError("The requested context exceeds available unified memory.")

    def metrics(self):
        return {
            "memory_budget_gib": self.limit / GIB,
            "mlx_active_before_release_gib": mx.get_active_memory() / GIB,
            "mlx_cache_before_release_gib": mx.get_cache_memory() / GIB,
            "system_min_memory_free_pct": self.minimum_free,
        }

"""Compare exact FFN packet loops with the production BF16 projections."""

from collections import defaultdict
from functools import lru_cache
import inspect
import json
from statistics import median
from time import perf_counter

import mlx.core as mx

from localllm import lossless_gate_up as gate
from localllm import lossless_projection as down
from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from localllm.packed_projection import pack_activations
from scripts.bench_dflash2 import pages
from scripts.bench_verbatim_ffn import replace_once
from scripts.bench_wide_projection_walk import LAYERS, layer_weights


WIDTHS = (6, 8, 11, 16)


@lru_cache
def fixed_gate():
    source = inspect.getsource(gate.paired_kernel)
    source = replace_once(source, 'kernel_name = f"local_lossless_paired_{mode}"',
                          'kernel_name = f"local_lossless_paired_{mode}_fixed_packets"')
    source = replace_once(
        source,
        "for (uint packet = k_split * 16; packet < min((k_split + 1) * 16, packets); ++packet) {",
        r"""#pragma clang loop unroll(disable)
        for (uint within = 0; within < 16u; ++within) {
            uint packet = k_split * 16u + within;""",
    )
    scope = vars(gate).copy()
    exec(compile(source, __file__, "exec"), scope)
    return scope["paired_kernel"]


@lru_cache
def fixed_down(prefetch):
    source = inspect.getsource(down.project_kernel)
    name = "local_lossless_packet4_mma_fixed_prefetch" if prefetch else "local_lossless_packet4_mma_fixed"
    source = replace_once(source, '"local_lossless_packet4_mma"', repr(name))
    loop = """        uint end = min((split + 1) * 16, packets);
        for (uint packet = split * 16; packet < end; ++packet) {
            uint4 pairs = decode_packet(m, e, fallback, descriptor[output_tile * packets + packet],
                                        lane, TOKENS > 8 && TOKENS <= 16);"""
    replacement = r"""
        uint base = output_tile * packets + split * 16u;
        #pragma clang loop unroll(disable)
        for (uint within = 0; within < 16u; ++within) {
            uint packet = split * 16u + within;
            uint4 pairs = decode_packet(m, e, fallback, descriptor[base + within],
                                        lane, TOKENS > 8 && TOKENS <= 16);"""
    if prefetch:
        # Each pair has two independent weight loads. Keep the MMA operations
        # in their existing K order, including across consecutive pairs.
        replacement = r"""
        uint base = output_tile * packets + split * 16u;
        #pragma clang loop unroll(disable)
        for (uint within = 0; within < 16u; within += 2u) {
            uint2 descriptors(descriptor[base + within], descriptor[base + within + 1u]);
            uint4 first = decode_packet(m, e, fallback, descriptors.x, lane, TOKENS > 8 && TOKENS <= 16);
            uint4 next = decode_packet(m, e, fallback, descriptors.y, lane, TOKENS > 8 && TOKENS <= 16);
            #pragma unroll
            for (uint slot = 0; slot < 2u; ++slot) {
                uint packet = split * 16u + within + slot;
                uint4 pairs = slot == 0 ? first : next;"""
        source = replace_once(source, "        if (col < TOKENS)", "        }\n        if (col < TOKENS)")
    source = replace_once(source, loop, replacement)
    scope = vars(down).copy()
    exec(compile(source, __file__, "exec"), scope)
    return scope["project_kernel"]()


def gate_project(weight, packed, width):
    blocks, packets, words = weight.payload[-1].shape
    assert packets % 16 == 0
    splits = packets // 16
    threads = splits * 32
    output, = fixed_gate()("swiglu", width > 8, words == 1)(
        inputs=[packed, *weight.payload], template=[("TOKENS", width), ("SPLITS", splits)],
        grid=(threads * blocks, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(((width + 7) // 8) * blocks, 64)], output_dtypes=[mx.bfloat16],
    )
    return output


def down_project(weight, packed, width, *, prefetch=False):
    blocks, packets = weight.payload[-1].shape
    assert packets % 16 == 0
    splits = packets // 16
    output, = fixed_down(prefetch)(
        inputs=[packed, *weight.payload], template=[("TOKENS", width)],
        grid=(((blocks + 7) // 8) * 256, splits, 1), threadgroup=(256, 1, 1),
        output_shapes=[(splits, width, blocks * 8)], output_dtypes=[mx.float32],
    )
    return output


def compare(weights, width, budget):
    inputs = pack_activations(mx.random.normal((width, 5120)).astype(mx.bfloat16))
    mids = [g.project(inputs, width, swiglu=True) for g, _ in weights]
    mx.eval(inputs, mids)
    gate_paths = {
        "current": lambda: [g.project(inputs, width, swiglu=True) for g, _ in weights],
        "fixed": lambda: [gate_project(g, inputs, width) for g, _ in weights],
    }
    down_paths = {
        "current": lambda: [d.partials(mid, width) for (_, d), mid in zip(weights, mids, strict=True)],
        "fixed": lambda: [down_project(d, mid, width) for (_, d), mid in zip(weights, mids, strict=True)],
        "prefetch": lambda: [down_project(d, mid, width, prefetch=True)
                             for (_, d), mid in zip(weights, mids, strict=True)],
    }
    start_thermal = thermal_state()
    for stage, paths, bits in (("gate_up", gate_paths, mx.uint16), ("down", down_paths, mx.uint32)):
        expected = paths["current"]()
        for name, run in paths.items():
            assert all(mx.array_equal(a.view(bits), b.view(bits)).item()
                       for a, b in zip(expected, run(), strict=True)), (stage, width, name)
        del expected
        samples = defaultdict(list)
        names = tuple(paths)
        for pair in range(18):
            begin = pair % len(names)
            for name in names[begin:] + names[:begin]:
                started = perf_counter()
                for _ in range(4):
                    mx.eval(paths[name]())
                samples[name].append((perf_counter() - started) * 250)
            budget.check()
        print("packet_loop", json.dumps({
            "width": width, "stage": stage, "layers": LAYERS, "bitwise": True,
            "median_ms_four_layers": {k: median(v) for k, v in samples.items()},
            "paired_saved_ms": {k: median(a - b for a, b in zip(samples["current"], v, strict=True))
                                for k, v in samples.items() if k != "current"},
            "wins": {k: sum(a > b for a, b in zip(samples["current"], v, strict=True))
                     for k, v in samples.items() if k != "current"},
            "thermal": [start_thermal, thermal_state()], "peak_gib": mx.get_peak_memory() / GIB,
            **budget.metrics(),
        }), flush=True)


def main():
    mx.random.seed(711)
    before = pages()
    with metal_job(12, 1) as budget:
        weights = layer_weights(LAYERS, budget)
        for width in WIDTHS:
            compare(weights, width, budget)
    after = pages()
    print("host", json.dumps({k: after[k] - v for k, v in before.items()}), flush=True)


if __name__ == "__main__":
    main()

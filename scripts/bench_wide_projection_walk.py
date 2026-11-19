"""Interleave wide verifier projections that read the same BF16 weight tiles."""

import argparse
from collections import defaultdict
from functools import lru_cache
import inspect
import json
from statistics import median
from time import perf_counter

import mlx.core as mx

from ultrarun import lossless_gate_up as gate
from ultrarun import lossless_projection as down
from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.packed_projection import pack_activations, pack_weight
from scripts.bench_dflash2 import pages
from scripts.bench_ready import TARGET
from scripts.bench_verbatim_ffn import replace_once


LAYERS = (0, 9, 23, 48)
WALKS = (1, 4, 16)


GATE_WALK = r"""
        uint group = threadgroup_position_in_grid.x;
        uint output_tile = group / (2u * WALK) * WALK + group % WALK;
        uint token_base = ((group / WALK) % 2u) * PANEL;
        uint panel_rows = min(uint(PANEL), uint(TOKENS) - token_base);
"""


DOWN_WALK = r"""
        uint packed_group = threadgroup_position_in_grid.x;
        uint groups = (descriptor_shape[0] + SIMD_GROUPS - 1u) / SIMD_GROUPS;
        uint split = packed_group / (2u * groups);
        uint group = packed_group % (2u * groups);
        uint output_tile = (group / (2u * WALK) * WALK + group % WALK) * SIMD_GROUPS
                           + thread_index_in_threadgroup / 32u;
        uint token_base = ((group / WALK) % 2u) * PANEL;
        uint panel_rows = min(uint(PANEL), uint(TOKENS) - token_base);
"""


@lru_cache
def compiled_walks():
    # Keep the original packet format and all FP32 MMA and summation steps.
    # Change only the coordinates of the two independent 8- or 16-row panels.
    source = inspect.getsource(gate.paired_kernel)
    source = replace_once(source, "uint output_tile = threadgroup_position_in_grid.x;",
                          GATE_WALK)
    source = source.replace("if (TOKENS > 8)", "if (panel_rows > 8)")
    source = replace_once(source, "reinterpret_cast<device const ushort2*>(x);",
                          "reinterpret_cast<device const ushort2*>(x) + token_base / 8u * packets * 128u;")
    source = replace_once(source, 'kernel_name = f"local_lossless_paired_{mode}"',
                          'kernel_name = f"local_lossless_paired_{mode}_wide_walk"')
    gate_scope = vars(gate).copy()
    gate_scope["LOCAL_REDUCTION"] = gate.LOCAL_REDUCTION.replace(
        "TOKENS > 8 ? 128 : 64", "PANEL * 8u").replace("TOKENS", "panel_rows")
    gate_scope["SWIGLU_OUTPUT"] = gate.SWIGLU_OUTPUT.replace("TOKENS", "panel_rows").replace(
        "k_split * half_blocks + output_tile", "(token_base / 8u + k_split) * half_blocks + output_tile")
    exec(compile(source, __file__, "exec"), gate_scope)

    source = inspect.getsource(down.project_kernel)
    source = replace_once(source, """uint output_tile = thread_position_in_grid.x / 32;
        uint split = thread_position_in_grid.y;""", DOWN_WALK)
    source = replace_once(source, "reinterpret_cast<device const ushort2*>(x);",
                          "reinterpret_cast<device const ushort2*>(x) + token_base / 8u * packets * 128u;")
    source = source.replace("TOKENS > 8 && TOKENS <= 16", "panel_rows > 8")
    source = source.replace("TOKENS > 8", "panel_rows > 8")
    source = source.replace("< TOKENS", "< panel_rows")
    source = source.replace("* TOKENS + col", "* TOKENS + token_base + col")
    source = replace_once(source, 'name="local_lossless_packet4_mma"',
                          'name="local_lossless_packet4_mma_wide_walk"')
    down_scope = vars(down).copy()
    exec(compile(source, __file__, "exec"), down_scope)
    return gate_scope["paired_kernel"], down_scope["project_kernel"]


def walking_projection(weight, packed, width, walk, *, paired, simd_groups=8):
    if not 8 < width <= 32:
        raise ValueError("Interleaved projection requires 9–32 positions")
    panel = 8 if width <= 16 else 16
    gate_kernel, down_kernel = compiled_walks()
    blocks, packets = weight.payload[-1].shape[:2]
    splits = (packets + 15) // 16
    template = [("TOKENS", width), ("WALK", walk), ("PANEL", panel)]
    if paired:
        assert blocks % walk == 0
        threads = splits * 32
        kernel = gate_kernel("swiglu", panel == 16, weight.payload[-1].shape[-1] == 1)
        shape = (((width + 7) // 8) * blocks, 64)
        dtype = mx.bfloat16
        groups = blocks * 2
        template.append(("SPLITS", splits))
    else:
        output_groups = (blocks + simd_groups - 1) // simd_groups
        assert output_groups % walk == 0
        threads = simd_groups * 32
        kernel = down_kernel()
        shape = (splits, width, blocks * 8)
        dtype = mx.float32
        groups = output_groups * 2 * splits
        template.append(("SIMD_GROUPS", simd_groups))
    output, = kernel(
        inputs=[packed, *weight.payload], template=template,
        grid=(groups * threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[shape], output_dtypes=[dtype],
    )
    return output


def layer_weights(index, budget):
    names = json.loads((TARGET / "model.safetensors.index.json").read_text())["weight_map"]
    weights = []
    for layer in index:
        suffixes = ("gate_proj", "up_proj", "down_proj")
        keys = [f"model.language_model.layers.{layer}.mlp.{name}.weight" for name in suffixes]
        arrays = []
        for key in keys:
            shard = mx.load(str(TARGET / names[key]))
            arrays.append(shard[key])
            del shard
        g, u, d = arrays
        assert all(w.dtype == mx.bfloat16 for w in arrays)
        packed = gate.encode_paired(pack_weight(mx.concatenate((g, u), axis=0)), compact_descriptors=True)
        down_packed = down.encode_down(pack_weight(d))
        mx.eval(packed.payload, down_packed.payload)
        weights.append((packed, down_packed))
        del arrays, g, u, d, packed, down_packed
        mx.clear_cache()
        budget.check(force=True)
    return weights


def project_chain(weights, packed, width, walk, *, stages="both", simd_groups=8):
    g, d = weights
    if walk and stages != "down":
        mid = walking_projection(g, packed, width, walk, paired=True)
    elif width <= 16:
        mid = g.project(packed, width, swiglu=True)
    else:
        gate_end = packed.shape[0] * 2 // ((width + 7) // 8)
        mid = mx.concatenate((g.project(packed[:gate_end], 16, swiglu=True),
                              g.project(packed[gate_end:], width - 16, swiglu=True)))
    if walk and stages != "gate":
        partial = walking_projection(d, mid, width, walk, paired=False, simd_groups=simd_groups)
    elif width <= 16:
        partial = d.partials(mid, width)
    else:
        down_end = mid.shape[0] * 2 // ((width + 7) // 8)
        partial = mx.concatenate((d.partials(mid[:down_end], 16),
                                  d.partials(mid[down_end:], width - 16)), axis=1)
    return mid, partial


def timings(weights, width, budget):
    packed = pack_activations(mx.random.normal((width, 5120)).astype(mx.bfloat16))
    mx.eval(packed)
    baseline = "current" if width <= 16 else "split"
    choices = ((baseline, 0, "both", 8), *((f"walk{n}", n, "both", 8) for n in WALKS),
               ("gate4", 4, "gate", 8), ("down4", 4, "down", 8))
    if width <= 16:
        choices += tuple((f"down4x{groups}", 4, "down", groups) for groups in (2, 4, 16))
    print("checking", width, flush=True)
    for pair in weights:
        want = project_chain(pair, packed, width, 0)
        for label, walk, stages, groups in choices[1:]:
            got = project_chain(pair, packed, width, walk, stages=stages, simd_groups=groups)
            for ref, actual, bits in zip(want, got, (mx.uint16, mx.uint32), strict=True):
                assert mx.array_equal(ref.view(bits), actual.view(bits)).item(), (width, label)
    values = defaultdict(list)
    for i in range(18):
        shift = i % len(choices)
        for label, walk, stages, groups in choices[shift:] + choices[:shift]:
            started = perf_counter()
            mx.eval([project_chain(pair, packed, width, walk, stages=stages, simd_groups=groups)
                     for pair in weights])
            values[label].append(1000 * (perf_counter() - started))
        budget.check()
    return {
        "width": width, "layers": list(LAYERS), "bitwise_swiglu_and_down_partials": True,
        "median_ms": {label: median(samples) for label, samples in values.items()},
        "paired_delta_ms": {label: median(a - b for a, b in zip(samples, values[baseline], strict=True))
                            for label, samples in values.items() if label != baseline},
        "thermal": thermal_state(), "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--narrow", action="store_true", help="benchmark the production 9–16 position FFN widths")
    args = parser.parse_args()
    mx.random.seed(64)
    before = pages()
    with metal_job(12, 1) as budget:
        weights = layer_weights(LAYERS, budget)
        for width in ((9, 11, 13, 16) if args.narrow else (17, 24, 32)):
            print("projection_walk", json.dumps(timings(weights, width, budget)), flush=True)
        after = pages()
        print("host", json.dumps({k: after[k] - v for k, v in before.items()}), flush=True)


if __name__ == "__main__":
    main()

"""Lossless operand recoding for the M2 Ultra's prefill projections."""

from functools import lru_cache
from pathlib import Path
import re

import mlx.core as mx

from .lossless_gate_up import DECODE, FRAGMENT_COORDS, JOINT_HEADER, LOAD_COMPACT_JOINT, LOAD_JOINT

_METAL_HEADERS = Path(mx.__file__).parent / "include"
_KERNEL_PATH = "mlx/backend/metal/kernels"
_STEEL_GEMM = f"{_KERNEL_PATH}/steel/gemm/gemm.h"
# Products of finite half operands are multiples of 2^-48. A 17408-term partial
# sum is below 2^47. Undoing two shifts in this range stays within FP32 normals.
_MAX_SHIFT = 38


def supported():
    return (mx.__version__ == "0.32.4" and mx.device_info()["device_name"] == "Apple M2 Ultra"
            and (_METAL_HEADERS / _STEEL_GEMM).is_file())


@mx.compile
def _weight_loses_bits(weight, scale, inverse):
    half = (weight.astype(mx.float32) * scale[:, None]).astype(mx.float16)
    restored = (half.astype(mx.float32) * inverse[:, None]).astype(mx.bfloat16)
    return mx.any((weight.view(mx.uint16) != restored.view(mx.uint16)) | ~mx.isfinite(weight))


def weight_scales(weight, *, interleaved=True):
    peak = mx.maximum(mx.max(mx.abs(weight.astype(mx.float32)), axis=1), 2.0**-30)
    shift = mx.clip(mx.floor(mx.log2(65504.0 / peak)), -_MAX_SHIFT, _MAX_SHIFT).astype(mx.int32)
    forward = ((127 + shift) << 23).view(mx.float32)
    inverse = ((127 - shift) << 23).view(mx.float32)
    if _weight_loses_bits(weight, forward, inverse).item():
        return None
    if interleaved:
        inverse = mx.contiguous(inverse.reshape(2, -1).T.reshape(-1))
    mx.eval(inverse, forward)
    return inverse, forward


@lru_cache
def _encode_rows():
    return mx.fast.metal_kernel(
        name="ultrarun_prefill_encode_rows",
        input_names=["words"], output_names=["encoded", "inverse", "fallback"],
        source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        uint width = words_shape[1], begin = row * width;
        int max_exp = -200, required_shift = -200;
        for (uint j = lane; j < width; j += 128) {
            uint bits = uint(words[begin + j]) & 0x7fffu;
            if (bits == 0) continue;
            int exponent = int(bits >> 7);
            if (exponent == 0 || exponent == 255) {
                required_shift = 1000;
                continue;
            }
            uint significand = (bits & 127u) + 128;
            int trailing_zeros = 31 - int(clz(significand & (0u - significand)));
            // The lowest set bit of the scaled value must be at least 2^-24.
            required_shift = max(required_shift, 110 - exponent - trailing_zeros);
            max_exp = max(max_exp, exponent - 127);
        }
        threadgroup int maxima[4], minima[4];
        max_exp = simd_max(max_exp);
        required_shift = simd_max(required_shift);
        if ((lane & 31u) == 0) {
            maxima[lane / 32] = max_exp;
            minima[lane / 32] = required_shift;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_exp = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
        required_shift = max(max(minima[0], minima[1]), max(minima[2], minima[3]));
        int shift = clamp(15 - max_exp, -MAX_SHIFT, MAX_SHIFT);
        bool unsafe = shift < required_shift || shift > 15 - max_exp;
        if (max_exp == -200) shift = 0;
        if (lane == 0) {
            inverse[row] = as_type<float>(uint(127 - shift) << 23);
            fallback[row] = unsafe;
        }
        float factor = as_type<float>(uint(127 + shift) << 23);
        for (uint j = lane; j < width; j += 128) {
            uint word = uint(words[begin + j]);
            encoded[begin + j] = half(as_type<float>(word << 16) * factor);
        }
        """,
    )


@lru_cache
def _unpack_scaled(compact):
    load = LOAD_COMPACT_JOINT if compact else LOAD_JOINT
    return mx.fast.metal_kernel(
        name="ultrarun_prefill_unpack_scaled_" + ("compact" if compact else "full"),
        input_names=["digits", "exponents", "escaped", "descriptor", "forward"],
        output_names=["raw"], header=DECODE + JOINT_HEADER, source=r"""
        uint i = thread_position_in_grid.x;
        uint lane = i % 32;
        uint packets = descriptor_shape[1];
        uint output_tile = i / 32 / packets;
        uint packet = i / 32 % packets;
        if (output_tile >= descriptor_shape[0]) return;
        auto fallback = reinterpret_cast<device const uint4*>(escaped);
        """ + FRAGMENT_COORDS + load + r"""
        uint feature = output_tile * 8 + row;
        uint half_width = descriptor_shape[0] * 8;
        uint width = half_width * 2;
        float gate_factor = forward[feature], up_factor = forward[half_width + feature];
        for (uint j = 0; j < 4; ++j) {
            uint position = packet * 32 + j * 8 + col;
            uint dst = position * width + (INTERLEAVED ? feature * 2 : feature);
            uint offset = INTERLEAVED ? 1 : half_width;
            uint gate = gate_pairs[j], up = up_pairs[j];
            raw[dst] = half(as_type<float>(gate << 16) * gate_factor);
            raw[dst + offset] = half(as_type<float>(up << 16) * up_factor);
            raw[dst + width] = half(as_type<float>(gate & 0xffff0000u) * gate_factor);
            raw[dst + width + offset] = half(as_type<float>(up & 0xffff0000u) * up_factor);
        }
        """,
    )


@lru_cache(maxsize=1)
def _gemm_headers():
    seen = set()

    def expand(path):
        if path in seen:
            return ""
        seen.add(path)
        return re.sub(r'^#include "([^"]+)"$', lambda match: expand(match[1]),
                      (_METAL_HEADERS / path).read_text(), flags=re.M)

    # CustomKernel already supplies the MLX Metal utility definitions.
    expand(f"{_KERNEL_PATH}/utils.h")
    return expand(_STEEL_GEMM) + "\nusing namespace mlx::steel;\n"


@lru_cache(maxsize=2)
def _project(activated):
    names = ["a", "w", "left_scale", "right_scale"]
    if activated:
        names.append("silu")
        store = r"""
                bfloat16_t swish = silu[as_type<ushort>(first)];
                out[size_t(rr) * (N / 2) + cc / 2] = bfloat16_t(float(swish) * float(second));
        """
    else:
        store = r"""
                out[size_t(rr) * N + cc] = first;
                out[size_t(rr) * N + cc + 1] = second;
        """
    return mx.fast.metal_kernel(
        name="ultrarun_prefill_scaled_gate_up" if activated else "ultrarun_prefill_scaled_projection",
        input_names=names, output_names=["out"],
        header=_gemm_headers(), compile_options={"math_mode": "safe"}, source=r"""
        using G = GEMMKernel<half, bfloat16_t, 32, 64, 16, 1, 2, false, false, true, true>;
        auto grid = BlockSwizzle::swizzle(threadgroup_position_in_grid, 4);
        if (grid.x >= N / 64 || grid.y >= M / 32) return;
        threadgroup half xs[G::tgp_mem_size_a], ws[G::tgp_mem_size_b];
        threadgroup_barrier(mem_flags::mem_none);
        int row = grid.y * 32, col = grid.x * 64;
        ushort group = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
        typename G::loader_a_t load_a(a + size_t(row) * K, K, xs, group, lane);
        typename G::loader_b_t load_b(w + col, N, ws, group, lane);
        typename G::mma_t accum(group, lane);
        for (int k = 0; k < K / 16; ++k) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            load_a.load_unsafe();
            load_b.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            accum.mma(xs, ws);
            load_a.next();
            load_b.next();
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < decltype(accum)::TM; ++i) {
            int rr = row + accum.sm + i * 8;
            float inverse = left_scale[rr];
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < decltype(accum)::TN; ++j) {
                int cc = col + accum.sn + j * 16;
                auto pair = accum.Ctile.frag_at(i, j);
                // Restore the original BF16 GEMM boundary.
                bfloat16_t first = bfloat16_t(pair[0] * (inverse * right_scale[cc]));
                bfloat16_t second = bfloat16_t(pair[1] * (inverse * right_scale[cc + 1]));
        """ + store + r"""
            }
        }
        """,
    )


def encode_operand(source):
    rows = source.shape[0]
    return _encode_rows()(
        inputs=[source.view(mx.uint16)], grid=(rows * 128, 1, 1), threadgroup=(128, 1, 1),
        template=[("MAX_SHIFT", _MAX_SHIFT)],
        output_shapes=[source.shape, (rows,), (rows,)],
        output_dtypes=[mx.float16, mx.float32, mx.bool_],
    )


def _encode_source(source):
    half, left, fallback = encode_operand(source)
    if mx.any(fallback).item():
        return None
    return half, left


def unpack_operand(payload, forward, *, interleaved):
    blocks, chunks, descriptor_words = payload[-1].shape
    return _unpack_scaled(descriptor_words == 1)(
        inputs=[*payload, forward], grid=(blocks * chunks * 32, 1, 1), threadgroup=(256, 1, 1),
        template=[("INTERLEAVED", interleaved)], output_shapes=[(chunks * 32, blocks * 16)],
        output_dtypes=[mx.float16],
    )[0]


def _matmul(half, left, weight, right, silu=None):
    rows, width = half.shape
    columns = weight.shape[1]
    activated = silu is not None
    inputs = [half, weight, left, right]
    if activated:
        inputs.append(silu)
    return _project(activated)(
        inputs=inputs, output_shapes=[(rows, columns // 2 if activated else columns)],
        output_dtypes=[mx.bfloat16], template=[("M", rows), ("N", columns), ("K", width)],
        grid=(columns // 64 * 16 * 32, (rows // 32 + 15) // 16 * 2, 1), threadgroup=(32, 2, 1),
    )[0]


def project(source, payload, scales, silu):
    encoded = _encode_source(source)
    if encoded is None:
        return None
    right, forward = scales
    weight = unpack_operand(payload, forward, interleaved=True)
    return _matmul(*encoded, weight, right, silu)


def project_qkvz(source, weight, right):
    encoded = _encode_source(source)
    if encoded is None:
        return None
    return _matmul(*encoded, weight, right)

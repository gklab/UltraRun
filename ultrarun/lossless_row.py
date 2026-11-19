"""Reversible BF16 row packets for the single-position MTP input projection."""
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from .lossless_projection import DECODE


def encode_rows(weight):
    rows, width = weight.shape
    if weight.dtype != mx.bfloat16 or rows % 32 or width % 128 or rows * (width // 128) >= 1 << 24:
        raise ValueError("Expected BF16 rows aligned to 32 outputs and 128 inputs")
    # Each descriptor points to an original or compressed group of 128 weights.
    raw = weight.view(mx.uint16).reshape(-1, 32, 4)
    exponent = (raw >> 7) & 255
    base = mx.maximum(exponent.max(axis=(1, 2)).astype(mx.int32) - 15, 0).astype(mx.uint16)
    good = exponent.min(axis=(1, 2)) >= base
    count = int(mx.sum(good).item())
    desc = mx.zeros((raw.shape[0],), dtype=mx.uint32)
    if count:
        order = mx.sort(mx.argpartition((~good).astype(mx.uint8), count - 1)[:count])
        subset = raw[order]
        packed = ((subset & 127) | ((subset >> 8) & 128)).astype(mx.uint8)
        digits = mx.contiguous(packed).view(mx.uint32)[..., 0]
        delta = (((subset >> 7) & 255) - base[order, None, None])
        e = delta[..., 0] | (delta[..., 1] << 4) | (delta[..., 2] << 8) | (delta[..., 3] << 12)
        desc = desc.at[order].add((mx.arange(count, dtype=mx.uint32) << 8) | base[order].astype(mx.uint32))
    else:
        digits = mx.zeros((1, 32), dtype=mx.uint32)
        e = mx.zeros((1, 32), dtype=mx.uint16)
    missing = raw.shape[0] - count
    if missing:
        order = mx.sort(mx.argpartition(good.astype(mx.uint8), missing - 1)[:missing])
        escape = mx.contiguous(raw[order]).view(mx.uint32)
        desc = desc.at[order].add((mx.arange(missing, dtype=mx.uint32) << 8) | 255)
    else:
        escape = mx.zeros((1, 32, 2), dtype=mx.uint32)
    code = (mx.contiguous(digits), mx.contiguous(e), escape, desc.reshape(rows, width // 128))
    mx.eval(code)
    return LosslessRow(code)


DECODE_ROW = DECODE + r"""
inline uint2 row_packet(device const uint* digits, device const ushort* exps,
                        device const uint2* fallback, uint tag, uint lane) {
    uint base = tag & 255u;
    uint at = (tag >> 8) * 32 + lane;
    if (base == 255u) return fallback[at];
    uint m = digits[at];
    uint e = exps[at];
    return uint2(restore_pair(m & 65535u, e & 255u, base), restore_pair(m >> 16, e >> 8, base));
}
"""


@lru_cache
def unpack_kernel():
    return mx.fast.metal_kernel(
        name="local_mtp_unpacked_row_stream",
        input_names=["digits","exps","escaped","tags"], output_names=["weights"],
        header=DECODE_ROW, source=r"""
        uint index = thread_position_in_grid.x;
        uint packet = index / 32;
        if (packet >= tags_shape[0] * tags_shape[1]) return;
        auto fallback = reinterpret_cast<device const uint2*>(escaped);
        uint2 values = row_packet(digits, exps, fallback, tags[packet], index % 32);
        weights[2 * index] = values[0];
        weights[2 * index + 1] = values[1];
        """,
    )


def unpack_rows(code):
    rows, packets = code[-1].shape
    grid = ((rows * packets + 7) // 8) * 256
    return unpack_kernel()(
        inputs=code, grid=(grid, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, packets * 64)], output_dtypes=[mx.uint32],
    )[0].view(mx.bfloat16)


@lru_cache
def project_row_kernel():
    return mx.fast.metal_kernel(
        name="local_mtp_native_order_lossless_gemv",
        input_names=["x", "digits", "exps", "escaped", "tags"],
        output_names=["output"], header=DECODE_ROW,
        source=r"""
        uint lane = thread_position_in_grid.x % 32;
        uint first = (thread_position_in_grid.x / 32) * 4;
        uint packets = tags_shape[1];
        auto fallback = reinterpret_cast<device const uint2*>(escaped);
        float accum[4] = {};
        for (uint tile = 0; tile < packets; ++tile) {
            uint offset = (tile * 32 + lane) * 4;
            float4 input{float(x[offset]), float(x[offset + 1]), float(x[offset + 2]), float(x[offset + 3])};
            #pragma unroll
            for (uint row = 0; row < 4; ++row) {
                uint tag = tags[(first + row) * packets + tile];
                uint2 pairs = row_packet(digits, exps, fallback, tag, lane);
                float4 w = as_type<float4>(uint4(pairs[0] << 16, pairs[0] & 0xffff0000u,
                                               pairs[1] << 16, pairs[1] & 0xffff0000u));
                #pragma unroll
                for (uint item = 0; item < 4; ++item)
                    accum[row] = fma(w[item], input[item], accum[row]);
            }
        }
        // Keep the MLX 0.32.4 GEMV lane ordering before rounding to BF16.
        #pragma unroll
        for (uint row = 0; row < 4; ++row) {
            #pragma unroll
            for (uint shift = 16; shift != 0; shift >>= 1)
                accum[row] += simd_shuffle_down(accum[row], shift);
            if (lane == 0) output[first + row] = bfloat16_t(accum[row]);
        }
        """,
    )


@dataclass(frozen=True)
class LosslessRow:
    payload: tuple[mx.array, mx.array, mx.array, mx.array]

    def unpack(self):
        return unpack_rows(self.payload)

    def project_row(self, x):
        rows = self.payload[-1].shape[0]
        result, = project_row_kernel()(
            inputs=[x, *self.payload], grid=(rows * 8, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(1, rows)], output_dtypes=[mx.bfloat16],
        )
        return result

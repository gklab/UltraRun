"""Lossless BF16 packets for the packed 27B down projections."""
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from .packed_projection import FRAGMENT_COORDS

DECODE = r"""
inline uint restore_pair(uint m, uint e, uint base) {
    uint mantissa = (m & 0x007f) | ((m & 0x0080) << 8)
                    | ((m & 0x7f00) << 8) | ((m & 0x8000) << 16);
    return mantissa | ((base + (e & 15)) << 7) | ((base + (e >> 4)) << 23);
}

inline uint4 restore_packed_exponents(ushort4 digits, uint exponents, uint base) {
    uint4 m = uint4(digits);
    uint4 fields = uint4(exponents) >> uint4(0, 8, 16, 24);
    uint4 pairs = (fields & 15u) | ((fields & 240u) << 12u);
    // The exponent additions occupy independent uint16 lanes.
    uint4 exponent_bits = (pairs + (base | (base << 16u))) << 7u;
    uint4 mantissa_bits = (m & 0x007fu) | ((m & 0x7f80u) << 8u) | ((m & 0x8000u) << 16u);
    return mantissa_bits | exponent_bits;
}

inline uint4 decode_packet(device const ushort4* digits, device const uint* exponents,
                           device const uint4* escaped, uint descriptor, uint lane, bool packed_exponents = false) {
    uint base = descriptor & 255;
    uint index = (descriptor >> 8) * 32 + lane;
    if (base == 255) return escaped[index];
    ushort4 m = digits[index];
    uint e = exponents[index];
    if (packed_exponents) return restore_packed_exponents(m, e, base);
    return uint4(restore_pair(m[0], e & 255, base),
                 restore_pair(m[1], (e >> 8) & 255, base),
                 restore_pair(m[2], (e >> 16) & 255, base),
                 restore_pair(m[3], e >> 24, base));
}
"""


def encode_down(raw):
    blocks, tiles, fragment = raw.shape
    if raw.dtype != mx.uint16 or fragment != 64 or tiles % 4 or blocks * tiles // 4 >= 1 << 24:
        raise ValueError("Expected packed BF16 weights with four-tile packets and 24-bit packet indices")
    packets = raw.reshape(-1, 4, 32, 2)
    all_exponents = ((packets >> 7) & 255)
    maximum = all_exponents.max(axis=(1, 2, 3))
    base = mx.maximum(maximum.astype(mx.int32) - 15, 0).astype(mx.uint16)
    minimum = all_exponents.min(axis=(1, 2, 3))
    good = minimum >= base
    good_count = int(mx.sum(good).item())
    bad_count = good.size - good_count
    # A descriptor stores an exponent base and a dense index into its payload.
    # Base 255 denotes a verbatim packet, including every original exponent bit.
    descriptors = mx.zeros((packets.shape[0],), dtype=mx.uint32)
    if good_count:
        good_order = mx.sort(mx.argpartition((~good).astype(mx.uint8), good_count - 1)[:good_count])
        descriptors = descriptors.at[good_order].add(
            (mx.arange(good_count, dtype=mx.uint32) << 8) | base[good_order].astype(mx.uint32))
        kept = packets[good_order].transpose(0, 2, 1, 3)
        digits = ((kept & 127) | ((kept >> 8) & 128)).astype(mx.uint8)
        digits = mx.contiguous(digits).view(mx.uint16).reshape(good_count, 32, 4)
        exponents = ((kept >> 7) & 255) - base[good_order, None, None, None]
        exponents = mx.contiguous((exponents[..., 0] | (exponents[..., 1] << 4)).astype(mx.uint8))
    else:
        digits = mx.zeros((1, 32, 4), dtype=mx.uint16)
        exponents = mx.zeros((1, 32, 4), dtype=mx.uint8)
    if bad_count:
        bad_order = mx.sort(mx.argpartition(good.astype(mx.uint8), bad_count - 1)[:bad_count])
        descriptors = descriptors.at[bad_order].add((mx.arange(bad_count, dtype=mx.uint32) << 8) | 255)
        escaped = mx.contiguous(packets[bad_order].transpose(0, 2, 1, 3)).view(mx.uint32).reshape(bad_count, 32, 4)
    else:
        escaped = mx.zeros((1, 32, 4), dtype=mx.uint32)
    return LosslessDown((digits, exponents, escaped, descriptors.reshape(blocks, tiles // 4)))


@lru_cache
def unpack_kernel():
    return mx.fast.metal_kernel(
        name="local_unpack_bf16_packets4", input_names=["digits", "exponents", "escaped", "descriptor"],
        output_names=["raw"], header=DECODE,
        source=r"""
        uint i = thread_position_in_grid.x;
        uint packet = i / 32;
        if (packet >= descriptor_shape[0] * descriptor_shape[1]) return;
        uint lane = i % 32;
        uint4 pairs = decode_packet(reinterpret_cast<device const ushort4*>(digits),
                                    reinterpret_cast<device const uint*>(exponents),
                                    reinterpret_cast<device const uint4*>(escaped),
                                    descriptor[packet], lane);
        for (uint t = 0; t < 4; ++t) raw[(packet * 4 + t) * 32 + lane] = pairs[t];
        """,
    )


def decoded_bits(coded):
    blocks, packets = coded[-1].shape
    raw, = unpack_kernel()(
        inputs=coded, grid=(blocks * packets * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(blocks, packets * 4, 32)], output_dtypes=[mx.uint32],
    )
    return raw.view(mx.uint16)


@lru_cache
def row_major_kernel(transposed=False):
    writes = r"""
        uint width = descriptor_shape[0] * 8;
        uint feature = row_tile * 8 + row;
        uint column = (packet % packets_per_row) * 32 + col;
        for (uint tile = 0; tile < 4; ++tile) {
            uint dst = (column + tile * 8) * width + feature;
            raw[dst] = ushort(pairs[tile]);
            raw[dst + width] = ushort(pairs[tile] >> 16);
        }
    """ if transposed else r"""
        uint row_start = (row_tile * 8 + row) * packets_per_row * 16;
        uint first = row_start + (packet % packets_per_row) * 16 + col / 2;
        for (uint tile = 0; tile < 4; ++tile) raw[first + tile * 4] = pairs[tile];
    """
    return mx.fast.metal_kernel(
        name="local_unpack_lossless_down_transposed" if transposed else "local_unpack_lossless_down_row_major",
        input_names=["digits", "exponents", "escaped", "descriptor"], output_names=["raw"],
        header=DECODE,
        source=r"""
        uint i = thread_position_in_grid.x;
        uint packet = i / 32;
        uint packets_per_row = descriptor_shape[1];
        uint row_tile = packet / packets_per_row;
        if (row_tile >= descriptor_shape[0]) return;
        uint lane = i % 32;
        """ + FRAGMENT_COORDS + r"""
        uint4 pairs = decode_packet(reinterpret_cast<device const ushort4*>(digits),
                                    reinterpret_cast<device const uint*>(exponents),
                                    reinterpret_cast<device const uint4*>(escaped),
                                    descriptor[packet], lane);
        """ + writes,
    )


def unpack_row_major(payload, *, transposed=False):
    blocks, packets = payload[-1].shape
    shape = (packets * 32, blocks * 8) if transposed else (blocks * 8, packets * 16)
    raw, = row_major_kernel(transposed)(
        inputs=payload, grid=(blocks * packets * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[shape], output_dtypes=[mx.uint16 if transposed else mx.uint32],
    )
    return raw.view(mx.bfloat16)



@lru_cache
def project_kernel():
    return mx.fast.metal_kernel(
        name="local_lossless_packet4_mma",
        input_names=["x", "digits", "exponents", "escaped", "descriptor"], output_names=["partial"],
        header="#include <metal_simdgroup_matrix>\n" + DECODE,
        source=r"""
        uint lane = thread_position_in_grid.x % 32;
        uint output_tile = thread_position_in_grid.x / 32;
        uint split = thread_position_in_grid.y;
        uint packets = descriptor_shape[1];
        uint n = descriptor_shape[0] * 8;
        if (output_tile >= descriptor_shape[0]) return;
        """ + FRAGMENT_COORDS + r"""
        uint output_row = output_tile * 8 + row;
        auto m = reinterpret_cast<device const ushort4*>(digits);
        auto e = reinterpret_cast<device const uint*>(exponents);
        auto fallback = reinterpret_cast<device const uint4*>(escaped);
        auto inputs = reinterpret_cast<device const ushort2*>(x);
        metal::simdgroup_matrix<float, 8, 8> a, b, accum, tail;
        accum.thread_elements()[0] = accum.thread_elements()[1] = 0.0f;
        if (TOKENS > 8) tail.thread_elements()[0] = tail.thread_elements()[1] = 0.0f;

        uint end = min((split + 1) * 16, packets);
        for (uint packet = split * 16; packet < end; ++packet) {
            uint4 pairs = decode_packet(m, e, fallback, descriptor[output_tile * packets + packet],
                                        lane, TOKENS > 8 && TOKENS <= 16);
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                uint k = packet * 4 + j;
                uint pair = pairs[j];
                float2 w = as_type<float2>(uint2(pair << 16, pair & 0xffff0000u));
                float2 v = as_type<float2>(uint2(inputs[k * 32 + lane]) << 16);
                a.thread_elements()[0] = w[0];
                a.thread_elements()[1] = w[1];
                b.thread_elements()[0] = v[0];
                b.thread_elements()[1] = v[1];
                simdgroup_multiply_accumulate(accum, a, b, accum);
                if (TOKENS > 8) {
                    float2 next = as_type<float2>(uint2(inputs[(packets * 4 + k) * 32 + lane]) << 16);
                    b.thread_elements()[0] = next[0];
                    b.thread_elements()[1] = next[1];
                    simdgroup_multiply_accumulate(tail, a, b, tail);
                }
            }
        }
        if (col < TOKENS)
            partial[(split * TOKENS + col) * n + output_row] = accum.thread_elements()[0];
        if (col + 1 < TOKENS)
            partial[(split * TOKENS + col + 1) * n + output_row] = accum.thread_elements()[1];
        if (col + 8 < TOKENS)
            partial[(split * TOKENS + col + 8) * n + output_row] = tail.thread_elements()[0];
        if (col + 9 < TOKENS)
            partial[(split * TOKENS + col + 9) * n + output_row] = tail.thread_elements()[1];
        """,
    )


@dataclass(frozen=True)
class LosslessDown:
    payload: tuple[mx.array, mx.array, mx.array, mx.array]

    def partials(self, x, tokens):
        blocks, packets = self.payload[-1].shape
        splits = (packets + 15) // 16
        out, = project_kernel()(
            inputs=[x, *self.payload], template=[("TOKENS", tokens)],
            grid=(((blocks + 7) // 8) * 256, splits, 1), threadgroup=(256, 1, 1),
            output_shapes=[(splits, tokens, blocks * 8)], output_dtypes=[mx.float32],
        )
        return out

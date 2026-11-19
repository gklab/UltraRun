"""Layout-preserving BF16 operands for narrow Metal SIMD matrix products."""
from functools import lru_cache

import mlx.core as mx


@lru_cache
def fragment_order():
    lanes = []
    for lane in range(32):
        quartet = lane // 4
        row = (quartet & 4) + (lane // 2) % 4
        col = (quartet & 2) * 2 + (lane % 2) * 2
        lanes.extend((row * 8 + col, row * 8 + col + 1))
    inverse = [lanes.index(i) for i in range(64)]
    return mx.array(lanes, dtype=mx.int32), mx.array(inverse, dtype=mx.int32)


@lru_cache
def pack_weight_kernel():
    return mx.fast.metal_kernel(
        name="local_pack_bf16_direct", input_names=["weight"], output_names=["packed"],
        source=r"""
        uint i = thread_position_in_grid.x;
        uint k_tiles = weight_shape[1] / 8;
        uint tile = i / 32;
        uint row_tile = tile / k_tiles;
        if (row_tile >= weight_shape[0] / 8) return;
        uint lane = i % 32;
        """ + FRAGMENT_COORDS + r"""
        uint source_row = row_tile * 8 + row;
        uint source_pair = (tile % k_tiles) * 4 + col / 2;
        auto pairs = reinterpret_cast<device const uint*>(weight);
        packed[i] = pairs[source_row * k_tiles * 4 + source_pair];
        """,
    )


def pack_weight(weight):
    rows, width = weight.shape
    if weight.dtype != mx.bfloat16 or width % 8 or rows % 8:
        raise ValueError("Weight packing requires BF16 matrices aligned to 8 by 8 tiles")
    packed, = pack_weight_kernel()(
        inputs=[weight], grid=(rows * (width // 2), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows // 8, width // 8, 32)], output_dtypes=[mx.uint32],
    )
    return packed.view(mx.uint16)


@lru_cache
def unpack_kernel(gather_rows=False):
    return mx.fast.metal_kernel(
        name="local_gather_bf16_weight" if gather_rows else "local_unpack_bf16_weight",
        input_names=["packed", "inverse", "rows"] if gather_rows else ["packed", "inverse"],
        output_names=["row_major"],
        source=r"""
        uint i = thread_position_in_grid.x;
        uint k_tiles = packed_shape[1];
        uint pairs_per_row = k_tiles * 4;
        uint item = i / pairs_per_row;
        """ + (r"""
        if (item >= rows_shape[0]) return;
        uint row = uint(rows[item]);
        """ if gather_rows else r"""
        if (item >= packed_shape[0] * 8) return;
        uint row = item;
        """) + r"""
        uint pair = i % pairs_per_row;
        uint lane = inverse[(row % 8) * 8 + (pair % 4) * 2] / 2;
        uint tile = (row / 8) * k_tiles + pair / 4;
        auto source_pairs = reinterpret_cast<device const uint*>(packed);
        row_major[i] = source_pairs[tile * 32 + lane];
        """,
    )


def unpack_weight(packed, rows=None):
    row_tiles, k_tiles, _ = packed.shape
    count = row_tiles * 8 if rows is None else rows.shape[0]
    width = k_tiles * 8
    inputs = [packed, fragment_order()[1]]
    if rows is not None:
        inputs.append(rows)
    raw, = unpack_kernel(rows is not None)(
        inputs=inputs, grid=(count * (width // 2), 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(count, width // 2)], output_dtypes=[mx.uint32],
    )
    return raw.view(mx.bfloat16)


FRAGMENT_COORDS = """
    uint quartet = lane / 4;
    uint row = (quartet & 4) + (lane / 2) % 4;
    uint col = (quartet & 2) * 2 + (lane % 2) * 2;
"""


@lru_cache
def pack_input_kernel():
    return mx.fast.metal_kernel(
        name="local_pack_bf16_activations", input_names=["x"], output_names=["packed"],
        source=r"""
        uint i = thread_position_in_grid.x;
        uint width = x_shape[1];
        uint span = width * 8;
        if (i >= ((x_shape[0] + 7) / 8) * span) return;
        uint lane = i / 2 % 32;
        """ + FRAGMENT_COORDS + r"""
        uint token = (i / span) * 8 + col + i % 2;
        uint feature = ((i % span) / 64) * 8 + row;
        packed[i] = token < x_shape[0] ? x[token * width + feature] : bfloat16_t(0);
        """,
    )


PACKED_MMA_BODY = r"""
        uint k_tiles = weight_shape[1];
        uint n = weight_shape[0] * 8;
        if (output_tile >= weight_shape[0]) return;
        """ + FRAGMENT_COORDS + r"""
        uint output_row = output_tile * 8 + row;
        auto weights = reinterpret_cast<device const ushort2*>(weight);
        auto inputs = reinterpret_cast<device const ushort2*>(x);
        metal::simdgroup_matrix<float, 8, 8> a, b, accum, tail, up_accum, up_tail;
        accum.thread_elements()[0] = 0.0f;
        accum.thread_elements()[1] = 0.0f;
        if (paired) {
            up_accum.thread_elements()[0] = 0.0f;
            up_accum.thread_elements()[1] = 0.0f;
            if (TOKENS > 8) {
                up_tail.thread_elements()[0] = 0.0f;
                up_tail.thread_elements()[1] = 0.0f;
            }
        }
        if (TOKENS > 8) {
            tail.thread_elements()[0] = 0.0f;
            tail.thread_elements()[1] = 0.0f;
        }

        uint end = min((k_split + 1) * TILE, k_tiles);
        for (uint k = k_split * TILE; k < end; ++k) {
            // BF16 widening preserves all bits. Adjacent lanes read adjacent pairs.
            float2 w = as_type<float2>(uint2(weights[(output_tile * k_tiles + k) * 32 + lane]) << 16);
            float2 v = as_type<float2>(uint2(inputs[k * 32 + lane]) << 16);
            a.thread_elements()[0] = w[0];
            a.thread_elements()[1] = w[1];
            b.thread_elements()[0] = v[0];
            b.thread_elements()[1] = v[1];
            simdgroup_multiply_accumulate(accum, a, b, accum);
            if (paired) {
                uint up_tile = output_tile + weight_shape[0] / 2;
                float2 up = as_type<float2>(uint2(weights[(up_tile * k_tiles + k) * 32 + lane]) << 16);
                a.thread_elements()[0] = up[0];
                a.thread_elements()[1] = up[1];
                simdgroup_multiply_accumulate(up_accum, a, b, up_accum);
            }
            if (TOKENS > 8) {
                float2 next = as_type<float2>(uint2(inputs[(k_tiles + k) * 32 + lane]) << 16);
                b.thread_elements()[0] = next[0];
                b.thread_elements()[1] = next[1];
                if (paired) {
                    simdgroup_multiply_accumulate(up_tail, a, b, up_tail);
                    a.thread_elements()[0] = w[0];
                    a.thread_elements()[1] = w[1];
                }
                simdgroup_multiply_accumulate(tail, a, b, tail);
            }
        }
"""


@lru_cache
def packed_mma_kernel():
    return mx.fast.metal_kernel(
        name="local_packed_bf16_mma", input_names=["x", "weight"], output_names=["partial"],
        header="#include <metal_simdgroup_matrix>\n",
        source=r"""
        uint lane = thread_position_in_grid.x % 32;
        uint output_tile = thread_position_in_grid.x / 32;
        uint k_split = thread_position_in_grid.y;
        constexpr bool paired = false;
        """ + PACKED_MMA_BODY + r"""
        if (col < TOKENS)
            partial[(k_split * TOKENS + col) * n + output_row] = accum.thread_elements()[0];
        if (col + 1 < TOKENS)
            partial[(k_split * TOKENS + col + 1) * n + output_row] = accum.thread_elements()[1];
        if (col + 8 < TOKENS)
            partial[(k_split * TOKENS + col + 8) * n + output_row] = tail.thread_elements()[0];
        if (col + 9 < TOKENS)
            partial[(k_split * TOKENS + col + 9) * n + output_row] = tail.thread_elements()[1];
        """,
    )


@lru_cache
def packed_head_kernel():
    return mx.fast.metal_kernel(
        name="local_packed_head_fused_argmax", input_names=["x", "weight"],
        output_names=["score", "winner"], header="#include <metal_simdgroup_matrix>\n",
        source=r"""
        uint lane = thread_index_in_threadgroup % 32;
        uint output_tile = threadgroup_position_in_grid.x;
        uint k_split = thread_index_in_threadgroup / 32;
        constexpr bool paired = false;
        """ + PACKED_MMA_BODY + r"""

        threadgroup float partials[10 * (TOKENS > 8 ? 128 : 64)];
        uint slot = (k_split * (TOKENS > 8 ? 128 : 64)) + lane * 2;
        partials[slot] = accum.thread_elements()[0];
        partials[slot + 1] = accum.thread_elements()[1];
        if (TOKENS > 8) {
            partials[slot + 64] = tail.thread_elements()[0];
            partials[slot + 65] = tail.thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_split >= (TOKENS > 8 ? 2 : 1)) return;

        uint token = k_split * 8 + col;
        uint sum_slot = k_split * 64 + lane * 2;
        constexpr uint stride = TOKENS > 8 ? 128 : 64;
        float sum0 = 0.0f, sum1 = 0.0f;
        // Match the eight interleaved groups in MLX 0.32.4 column reduction.
        #pragma unroll
        for (uint group = 0; group < 8; ++group) {
            float subtotal0 = 0.0f, subtotal1 = 0.0f;
            #pragma unroll
            for (uint part = group; part < 10; part += 8) {
                subtotal0 += partials[part * stride + sum_slot];
                subtotal1 += partials[part * stride + sum_slot + 1];
            }
            sum0 += subtotal0;
            sum1 += subtotal1;
        }
        // Row bits are in lane bits 1, 2 and 4. Columns stay fixed.
        float value0 = float(bfloat16_t(sum0)), value1 = float(bfloat16_t(sum1));
        uint row0 = row, row1 = row;
        for (uint mask : {2u, 4u, 16u}) {
            float other0 = simd_shuffle_xor(value0, mask);
            float other1 = simd_shuffle_xor(value1, mask);
            uint index0 = simd_shuffle_xor(row0, mask);
            uint index1 = simd_shuffle_xor(row1, mask);
            if (other0 > value0 || (other0 == value0 && index0 < row0)) {
                value0 = other0;
                row0 = index0;
            }
            if (other1 > value1 || (other1 == value1 && index1 < row1)) {
                value1 = other1;
                row1 = index1;
            }
        }
        if (row == 0) {
            if (token < TOKENS) {
                score[token * weight_shape[0] + output_tile] = bfloat16_t(value0);
                winner[token * weight_shape[0] + output_tile] = output_tile * 8 + row0;
            }
            if (token + 1 < TOKENS) {
                score[(token + 1) * weight_shape[0] + output_tile] = bfloat16_t(value1);
                winner[(token + 1) * weight_shape[0] + output_tile] = output_tile * 8 + row1;
            }
        }
        """,
    )


def pack_activations(x):
    tokens, width = x.shape
    panels = (tokens + 7) // 8
    packed, = pack_input_kernel()(
        inputs=[x], grid=(panels * 8 * width, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(panels * width // 8, 64)], output_dtypes=[mx.bfloat16],
    )
    return packed


@lru_cache
def residual_norm_kernel():
    return mx.fast.metal_kernel(
        name="local_bf16_residual_norm_to_fragments",
        input_names=["x", "h", "weight", "inverse", "eps"],
        output_names=["residual", "packed"],
        source=r"""
        uint token = thread_position_in_grid.x / 256;
        uint tid = thread_index_in_threadgroup;
        uint lane = tid % 32;
        uint width = x_shape[1];
        uint panel = (token / 8) * width * 8;
        uint column = token % 8;
        if (token >= x_shape[0]) {
            for (uint feature = tid; feature < width; feature += 256)
                packed[panel + (feature / 8) * 64 + inverse[(feature % 8) * 8 + column]] = bfloat16_t(0);
            return;
        }

        threadgroup float sums[8];
        float sum = 0.0f;
        uint offset = token * width;
        for (uint feature = tid; feature < width; feature += 256) {
            uint i = offset + feature;
            bfloat16_t value = bfloat16_t(float(x[i]) + float(h[i]));
            residual[i] = value;
            sum = fma(float(value), float(value), sum);
        }
        sum = simd_sum(sum);
        if (lane == 0) sums[tid / 32] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        float scale = rsqrt(sum / float(width) + eps[0]);

        for (uint feature = tid; feature < width; feature += 256) {
            uint i = offset + feature;
            float value = float(bfloat16_t(float(x[i]) + float(h[i])));
            packed[panel + (feature / 8) * 64 + inverse[(feature % 8) * 8 + column]] =
                bfloat16_t(value * scale * weight[feature]);
        }
        """,
    )


def residual_norm_pack(x, h, weight, eps):
    tokens, width = x.shape
    panels = (tokens + 7) // 8
    return residual_norm_kernel()(
        inputs=[x, h, weight, fragment_order()[1], eps],
        grid=(panels * 8 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(tokens, width), (panels * width // 8, 64)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )


def packed_partials(packed_input, tokens, weight):
    if not 1 <= tokens <= 16:
        raise ValueError("Packed SIMD projections support 1 to 16 tokens")
    blocks, k_tiles, _ = weight.shape
    tile = 64
    splits = (k_tiles + tile - 1) // tile
    partial, = packed_mma_kernel()(
        inputs=[packed_input, weight], template=[("TOKENS", tokens), ("TILE", tile)],
        grid=(((blocks + 7) // 8) * 256, splits, 1), threadgroup=(256, 1, 1),
        output_shapes=[(splits, tokens, blocks * 8)], output_dtypes=[mx.float32],
    )
    return partial


def packed_project(x, weight):
    partial = packed_partials(pack_activations(x), x.shape[0], weight)
    return partial.sum(axis=0).astype(mx.bfloat16)


def packed_head_argmax(x, weight):
    tokens = x.shape[0]
    blocks = weight.shape[0]
    threads = 10 * 32
    scores, indices = packed_head_kernel()(
        inputs=[pack_activations(x), weight],
        template=[("TOKENS", tokens), ("TILE", 64)],
        grid=(threads * blocks, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(tokens, blocks), (tokens, blocks)],
        output_dtypes=[mx.bfloat16, mx.uint32],
    )
    tiles = mx.argmax(scores, axis=1)
    return mx.take_along_axis(indices, tiles[:, None], axis=1)[:, 0]


SWIGLU_BF16 = r"""
inline bfloat16_t swiglu_bf16(float gate_sum, float up_sum) {
    bfloat16_t gate = bfloat16_t(gate_sum);
    bfloat16_t up = bfloat16_t(up_sum);
    bfloat16_t sigmoid = bfloat16_t(1.0f / (1.0f + exp(-float(gate))));
    bfloat16_t swish = bfloat16_t(float(gate) * float(sigmoid));
    return bfloat16_t(float(swish) * float(up));
}
"""


@lru_cache
def paired_swiglu_kernel():
    return mx.fast.metal_kernel(
        name="local_packed_paired_swiglu", input_names=["x", "weight"], output_names=["packed"],
        header="#include <metal_simdgroup_matrix>\n" + SWIGLU_BF16,
        source=r"""
        uint lane = thread_index_in_threadgroup % 32;
        uint k_split = thread_index_in_threadgroup / 32;
        uint output_tile = threadgroup_position_in_grid.x;
        constexpr bool paired = true;
        """ + PACKED_MMA_BODY + r"""

        constexpr uint stride = 128;
        threadgroup float partials[2 * SPLITS * stride];
        uint gate_slot = k_split * stride + lane * 2;
        uint up_slot = gate_slot + SPLITS * stride;
        partials[gate_slot] = accum.thread_elements()[0];
        partials[gate_slot + 1] = accum.thread_elements()[1];
        partials[gate_slot + 64] = tail.thread_elements()[0];
        partials[gate_slot + 65] = tail.thread_elements()[1];
        partials[up_slot] = up_accum.thread_elements()[0];
        partials[up_slot + 1] = up_accum.thread_elements()[1];
        partials[up_slot + 64] = up_tail.thread_elements()[0];
        partials[up_slot + 65] = up_tail.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_split >= 2) return;

        uint token = k_split * 8 + col;
        uint sum_slot = k_split * 64 + lane * 2;
        float2 gate = float2(0.0f);
        float2 up = float2(0.0f);
        for (uint part = 0; part < SPLITS; ++part) {
            uint index = part * stride + sum_slot;
            gate += float2(partials[index], partials[index + 1]);
            index += SPLITS * stride;
            up += float2(partials[index], partials[index + 1]);
        }
        // These operand fragments already have the layout of the down projection's input.
        uint dst = (k_split * (weight_shape[0] / 2) + output_tile) * 64 + lane * 2;
        packed[dst] = token < TOKENS ? swiglu_bf16(gate.x, up.x) : bfloat16_t(0);
        packed[dst + 1] = token + 1 < TOKENS ? swiglu_bf16(gate.y, up.y) : bfloat16_t(0);
        """,
    )


@lru_cache
def swiglu_input_kernel():
    return mx.fast.metal_kernel(
        name="local_bf16_swiglu_to_fragments", input_names=["partial"], output_names=["packed"],
        header=SWIGLU_BF16,
        source=r"""
        uint i = thread_position_in_grid.x;
        uint width = partial_shape[2] / 2;
        uint span = width * 8;
        if (i >= ((partial_shape[1] + 7) / 8) * span) return;
        uint lane = i / 2 % 32;
        """ + FRAGMENT_COORDS + r"""
        uint token = (i / span) * 8 + col + i % 2;
        if (token >= partial_shape[1]) {
            packed[i] = bfloat16_t(0);
            return;
        }
        uint feature = ((i % span) / 64) * 8 + row;
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        uint stride = partial_shape[1] * partial_shape[2];
        uint offset = token * partial_shape[2] + feature;
        for (uint split = 0; split < partial_shape[0]; ++split) {
            gate_sum += partial[split * stride + offset];
            up_sum += partial[split * stride + offset + width];
        }
        packed[i] = swiglu_bf16(gate_sum, up_sum);
        """,
    )


def packed_feed_forward(x, input_weight, output_weight):
    return packed_feed_forward_from_fragments(pack_activations(x), x.shape[0], input_weight, output_weight)


@lru_cache
def supports_paired_swiglu():
    return mx.device_info()["device_name"] == "Apple M2 Ultra"


def paired_swiglu_activations(packed, tokens, input_weight):
    blocks = input_weight.shape[0] // 2
    tile = 64
    splits = (input_weight.shape[1] + tile - 1) // tile
    threads = splits * 32
    intermediate, = paired_swiglu_kernel()(
        inputs=[packed, input_weight],
        template=[("TOKENS", tokens), ("TILE", tile), ("SPLITS", splits)],
        grid=(blocks * threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(2 * blocks, 64)], output_dtypes=[mx.bfloat16],
    )
    return intermediate


def swiglu_activations(packed, tokens, input_weight):
    partial = packed_partials(packed, tokens, input_weight)
    width = input_weight.shape[0] * 4
    panels = (tokens + 7) // 8
    intermediate, = swiglu_input_kernel()(
        inputs=[partial], grid=(panels * 8 * width, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(panels * width // 8, 64)], output_dtypes=[mx.bfloat16],
    )
    return intermediate


def packed_swiglu(packed, tokens, input_weight):
    if 9 <= tokens <= 16 and input_weight.shape == (4352, 640, 64) and supports_paired_swiglu():
        return paired_swiglu_activations(packed, tokens, input_weight)
    return swiglu_activations(packed, tokens, input_weight)


def packed_feed_forward_from_fragments(packed, tokens, input_weight, output_weight):
    intermediate = packed_swiglu(packed, tokens, input_weight)
    result = packed_partials(intermediate, tokens, output_weight)
    return result.sum(axis=0).astype(mx.bfloat16)

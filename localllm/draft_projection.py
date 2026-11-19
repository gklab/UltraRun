"""BF16 MTP projection with cached FP32 embedding contributions."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from .lossless_row import DECODE_ROW, LosslessRow
from .memory import MemoryBudgetError


@lru_cache
def cached_mtp_kernel():
    return mx.fast.metal_kernel(
        name="local_mtp_cached_embedding_projection",
        input_names=["hidden", "weight", "base", "extra", "token"],
        output_names=["result"],
        source=r"""
        uint row = thread_position_in_grid.x / 32;
        uint lane = thread_position_in_grid.x % 32;
        uint width = weight_shape[1] / 4;
        if (row >= weight_shape[0]) return;
        auto weights = reinterpret_cast<device const ushort4*>(weight);
        auto input = reinterpret_cast<device const ushort4*>(hidden);
        float4 sums = {};
        for (uint col = lane; col < width; col += 32) {
            float4 w = as_type<float4>(uint4(weights[row * width + col]) << 16);
            float4 x = as_type<float4>(uint4(input[col]) << 16);
            sums = fma(w, x, sums);
        }
        float dot_product = simd_sum(sums.x + sums.y + sums.z + sums.w);
        if (lane == 0) {
            uint index = uint(token[0]);
            uint rows = weight_shape[0];
            float contribution = index < base_shape[0] ? base[index * rows + row]
                : extra[(index - base_shape[0]) * rows + row];
            result[row] = bfloat16_t(dot_product + contribution);
        }
        """,
    )


def cached_mtp(hidden, weight, table, token):
    rows = weight.shape[0]
    out, = cached_mtp_kernel()(
        inputs=[hidden, weight, *table, token],
        grid=(((rows + 7) // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1, rows)], output_dtypes=[mx.bfloat16],
    )
    return out


WIDTH = 5120
CAPACITY = 128
BATCH = 4
THREADS = 256

FLOAT4 = r"""
inline float4 load4(device const bfloat16_t* ptr, uint at) {
    return as_type<float4>(uint4(reinterpret_cast<device const ushort4*>(ptr)[at]) << 16);
}

inline float step4(float4 w, float4 x, float value) {
    #pragma unroll
    for (uint col = 0; col < 4; ++col) value = fma(w[col], x[col], value);
    return value;
}
"""


@lru_cache
def prefix_kernel():
    return mx.fast.metal_kernel(
        name="local_mtp_embedding_lane_prefix",
        input_names=["weights", "normalized"], output_names=["prefix"],
        header=FLOAT4, source=r"""
        uint i = thread_position_in_grid.x;
        uint lane = i % 32;
        uint first_row = (i / 32) * 4;
        uint token = thread_position_in_grid.y * BATCH;
        uint half_cols = normalized_shape[1] / 4;
        uint weight_stride = half_cols * 2;
        float accum[BATCH][4] = {};
        for (uint col = lane; col < half_cols; col += 32) {
            float4 values[BATCH];
            #pragma unroll
            for (uint sample = 0; sample < BATCH; ++sample)
                values[sample] = load4(normalized, (token + sample) * half_cols + col);
            #pragma unroll
            for (uint row = 0; row < 4; ++row) {
                float4 w = load4(weights, (first_row + row) * weight_stride + col);
                #pragma unroll
                for (uint sample = 0; sample < BATCH; ++sample)
                    accum[sample][row] = step4(w, values[sample], accum[sample][row]);
            }
        }
        #pragma unroll
        for (uint sample = 0; sample < BATCH; ++sample) {
            #pragma unroll
            for (uint row = 0; row < 4; ++row) {
                uint destination = (((token + sample) * weights_shape[0]) + first_row + row) * 32 + lane;
                prefix[destination] = accum[sample][row];
            }
        }
        """,
    )


@lru_cache
def continue_kernel():
    return mx.fast.metal_kernel(
        name="local_mtp_continue_lane_prefix",
        input_names=["weights", "input", "table", "token_slots", "token"], output_names=["result"],
        header=FLOAT4, source=r"""
        uint i = thread_position_in_grid.x;
        uint lane = i % 32;
        uint row = i / 32;
        const uint cols = 2560;
        int slot = token_slots[token[0]];
        float accum = 0.0f;
        uint begin = 0;
        if (slot >= 0) {
            accum = table[(uint(slot) * weights_shape[0] + row) * 32 + lane];
            begin = cols / 2;
        }
        auto row_weights = weights + row * cols * 4;
        #pragma unroll 2
        for (uint col = lane + begin; col < cols; col += 32)
            accum = step4(load4(row_weights, col), load4(input, col), accum);
        #pragma unroll
        for (uint shift = 16; shift != 0; shift >>= 1)
            accum += simd_shuffle_down(accum, shift);
        if (lane == 0) result[row] = bfloat16_t(accum);
        """,
    )


@dataclass(frozen=True)
class LanePrefixes:
    sums: mx.array
    slots: mx.array
    count: int


def prepare_lane_prefixes(weight, embedding, norm_weight, epsilon, ids, budget):
    if budget is None:
        return None
    selected = [token for token, _ in Counter(ids).most_common(CAPACITY)]
    padded = ((len(selected) + BATCH - 1) // BATCH) * BATCH
    # Each token retains one FP32 accumulator per output row and SIMD lane.
    reserve = padded * WIDTH * (32 * 4 + 4) + embedding.shape[0] * 4
    if mx.get_active_memory() + budget.cache_limit + reserve > budget.limit:
        return None
    budget.check()
    try:
        budget.check(force=True, required_bytes=reserve)
    except MemoryBudgetError:
        return None
    keys = mx.array(selected + selected[:1] * (padded - len(selected)), dtype=mx.int32)
    emb = embedding[keys]
    norm = mx.fast.rms_norm(emb, norm_weight, epsilon).astype(mx.bfloat16)
    sums, = prefix_kernel()(
        inputs=[weight, norm], template=[("BATCH", BATCH)],
        grid=(WIDTH * 8, padded // BATCH, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[(padded, WIDTH, 32)], output_dtypes=[mx.float32],
    )
    slots = mx.full((embedding.shape[0],), -1, dtype=mx.int32)
    slots = slots.at[keys[:len(selected)]].add(mx.arange(len(selected), dtype=mx.int32) + 1)
    mx.eval(sums, slots)
    return LanePrefixes(sums, slots, len(selected))


def continue_lane_prefix(weight, inputs, memo, token):
    result, = continue_kernel()(
        inputs=[weight, inputs, memo.sums, memo.slots, token],
        grid=(WIDTH * 32, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[(1, WIDTH)], output_dtypes=[mx.bfloat16],
    )
    return result


DENSE_READ = r"""
        uint cols = weight_shape[1] / 4;
        uint rows = weight_shape[0];
        auto raw = reinterpret_cast<device const ushort4*>(weight);
"""
CODED_READ = r"""
        uint rows = tags_shape[0];
        uint cols = tags_shape[1] * 32;
        auto fallback = reinterpret_cast<device const uint2*>(escaped);
"""
DENSE_LOAD = r"""
                float4 w = as_type<float4>(uint4(raw[(first + r) * cols + col]) << 16);
"""
CODED_LOAD = r"""
                uint note = tags[(first + r) * (cols/32) + col/32];
                uint2 packed = row_packet(digits, exps, fallback, note, lane);
                float4 w = as_type<float4>(uint4(packed.x<<16, packed.x & 0xffff0000u,
                                                 packed.y<<16, packed.y & 0xffff0000u));
"""


@lru_cache
def multi_gemv_kernel(coded):
    return mx.fast.metal_kernel(
        name=f"local_mtp_native_order_multi_gemv_{int(coded)}",
        input_names=["x","digits","exps","escaped","tags"] if coded else ["x","weight"],
        output_names=["output"],header=DECODE_ROW if coded else "",
        source=r"""
        uint i = thread_position_in_grid.x;
        uint lane = i % 32;
        uint first = (i / 32) * ROWS;
        """ + (CODED_READ if coded else DENSE_READ) + r"""
        if (first >= rows) return;
        auto in_vec = reinterpret_cast<device const ushort4*>(x);
        float acc[PATHS][ROWS] = {};
        for (uint col=lane; col<cols; col+=32) {
            float4 v[PATHS];
            #pragma unroll
            for (uint p=0; p<PATHS; ++p)
                v[p]=as_type<float4>(uint4(in_vec[p*cols+col])<<16);
            #pragma unroll
            for (uint r=0; r<ROWS; ++r) {
                """ + (CODED_LOAD if coded else DENSE_LOAD) + r"""
                #pragma unroll
                for (uint p=0; p<PATHS; ++p) {
                    #pragma unroll
                    for (uint j=0; j<4; ++j) acc[p][r] = fma(w[j], v[p][j], acc[p][r]);
                }
            }
        }
        #pragma unroll
        for (uint p=0; p<PATHS; ++p) {
            #pragma unroll
            for (uint r=0; r<ROWS; ++r) {
                #pragma unroll
                for (uint shift=16; shift!=0; shift>>=1)
                    acc[p][r]+=simd_shuffle_down(acc[p][r],shift);
                if (lane==0)
                    output[p*rows+first+r]=bfloat16_t(acc[p][r]);
            }
        }
        """,
    )


def multi_gemv(x, weight):
    """Read one weight fragment and update separate native-order dot products."""
    coded = isinstance(weight, LosslessRow)
    weights = weight.payload if coded else (weight,)
    rows = weights[-1].shape[0]
    rows_per_simd = 2
    groups = (rows // rows_per_simd + 7) // 8
    out, = multi_gemv_kernel(coded)(
        inputs=[x, *weights], template=[("PATHS", x.shape[0]), ("ROWS", rows_per_simd)],
        grid=(groups * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[(x.shape[0], rows)], output_dtypes=[mx.bfloat16],
    )
    return out

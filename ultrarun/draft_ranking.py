"""Joint argmax and normalization for a single draft vocabulary row."""
from functools import lru_cache

import mlx.core as mx


@lru_cache
def partial_ranking_kernel():
    return mx.fast.metal_kernel(
        name="local_partial_draft_ranking",
        input_names=["scores"],
        output_names=["indices", "maxima", "masses"],
        source=r"""
        uint group = threadgroup_position_in_grid.x;
        uint tid = thread_index_in_threadgroup;
        uint lane = tid % 32;
        uint first = group * 512 + tid;
        uint second = first + 256;
        uint size = scores_shape[1];
        float a = first < size ? float(scores[first]) : -INFINITY;
        float b = second < size ? float(scores[second]) : -INFINITY;
        threadgroup float group_maxima[8];
        threadgroup float group_masses[8];
        threadgroup uint group_indices[8];
        float local_maximum = simd_max(max(a, b));
        if (lane == 0) group_maxima[tid / 32] = local_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float top = simd_max(lane < 8 ? group_maxima[lane] : -INFINITY);
        uint index = a == top && first < size ? first : (b == top && second < size ? second : 0xffffffffu);
        index = simd_min(index);
        float mass = top == -INFINITY ? 0.0f : simd_sum(exp(a - top) + exp(b - top));
        if (lane == 0) {
            group_indices[tid / 32] = index;
            group_masses[tid / 32] = mass;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            uint best = simd_min(lane < 8 ? group_indices[lane] : 0xffffffffu);
            float total = simd_sum(lane < 8 ? group_masses[lane] : 0.0f);
            if (lane == 0) {
                indices[group] = best;
                maxima[group] = top;
                masses[group] = total;
            }
        }
        """,
    )


@lru_cache
def merge_ranking_kernel():
    return mx.fast.metal_kernel(
        name="local_merge_draft_ranking",
        input_names=["indices", "maxima", "masses"],
        output_names=["winner", "confidence"],
        source=r"""
        uint tid = thread_index_in_threadgroup;
        uint lane = tid % 32;
        uint size = maxima_shape[0];
        float local_maximum = -INFINITY;
        for (uint i = tid; i < size; i += 256) local_maximum = max(local_maximum, maxima[i]);
        local_maximum = simd_max(local_maximum);
        threadgroup float group_maxima[8];
        threadgroup float group_masses[8];
        threadgroup uint group_indices[8];
        if (lane == 0) group_maxima[tid / 32] = local_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float top = simd_max(lane < 8 ? group_maxima[lane] : -INFINITY);
        float mass = 0.0f;
        uint index = 0xffffffffu;
        for (uint i = tid; i < size; i += 256) {
            mass += masses[i] * exp(maxima[i] - top);
            if (maxima[i] == top) index = min(index, indices[i]);
        }
        mass = simd_sum(mass);
        index = simd_min(index);
        if (lane == 0) {
            group_indices[tid / 32] = index;
            group_masses[tid / 32] = mass;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            uint best = simd_min(lane < 8 ? group_indices[lane] : 0xffffffffu);
            float total = simd_sum(lane < 8 ? group_masses[lane] : 0.0f);
            if (lane == 0) {
                winner[0] = best;
                confidence[0] = -log(total);
            }
        }
        """,
    )


def draft_rank(scores):
    parts = (scores.shape[1] + 511) // 512
    partials = partial_ranking_kernel()(
        inputs=[scores], grid=(parts * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(parts,)] * 3, output_dtypes=[mx.uint32, mx.float32, mx.float32],
    )
    return merge_ranking_kernel()(
        inputs=partials, grid=(256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1,), (1,)], output_dtypes=[mx.uint32, mx.float32],
    )

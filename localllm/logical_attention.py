"""Cyclic GQA reduction by logical ancestor position for short BF16 verification."""
from functools import lru_cache

import mlx.core as mx


PARTITIONS = 128


LOAD_KV = r"""
    uint ki = kv_head * keys_strides[1] + physical * keys_strides[2] + feature;
    uint vi = kv_head * values_strides[1] + physical * values_strides[2] + feature;
    float key[8], value[8];
    if (VECTOR_QKV) {
        load_bf16x8(reinterpret_cast<device const uint4*>(keys + ki), key);
        load_bf16x8(reinterpret_cast<device const uint4*>(values + vi), value);
    } else {
        #pragma unroll
        for (uint f = 0; f < 8; ++f) {
            key[f] = float(keys[ki + f]);
            value[f] = float(values[vi + f]);
        }
    }
"""

LOAD_SUFFIX = (LOAD_KV.replace("physical *", "(physical - fixed) *")
               .replace("keys", "suffix_keys").replace("values", "suffix_values"))

ACCUMULATE = r"""
    float dot = 0;
    #pragma unroll
    for (uint f = 0; f < 8; ++f)
        dot += q[query][f] * key[f];
    float score = simd_sum(dot);
    float new_max = max(max_score[query], score);
    float prev_weight = fast::exp(max_score[query] - new_max);
    float new_weight = fast::exp(score - new_max);
    max_score[query] = new_max;
    mass[query] = mass[query] * prev_weight + new_weight;
    #pragma unroll
    for (uint f = 0; f < 8; ++f)
        accum[query][f] = accum[query][f] * prev_weight + new_weight * value[f];
"""


def accumulate_range(end, load):
    return r"""
        for (; physical < """ + end + r"""; physical += PARTITIONS) {
        """ + load + r"""
            #pragma unroll
            for (uint query = 0; query < 2; ++query) {
        """ + ACCUMULATE + r"""
            }
        }
    """


@lru_cache
def partial_kernel(segmented=False, reserved=False):
    names = ["queries", "keys", "values", "scale", "path"]
    kernel_name = "local_logical_ancestor_gqa"
    prefix_size = "uint prefix = keys_shape[2] - tokens;"
    prefix_loop = accumulate_range("prefix", LOAD_KV)
    load_tail = LOAD_KV
    if segmented:
        names.extend(("suffix_keys", "suffix_values"))
        kernel_name = "local_prompt_suffix_ancestor_gqa"
        suffix = "suffix_keys_shape[2] - (SINGLE_QUERY ? 0 : tokens)"
        if reserved:
            names.append("write_meta")
            kernel_name = "local_reserved_suffix_ancestor_gqa"
            suffix = "uint(write_meta[0]) + uint(write_meta[1])"
        prefix_size = f"uint fixed = keys_shape[2]; uint prefix = fixed + {suffix};"
        prefix_loop = accumulate_range("fixed", LOAD_KV) + accumulate_range("prefix", LOAD_SUFFIX)
        load_tail = LOAD_SUFFIX
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=names,
        output_names=["partials", "masses", "maxima"],
        ensure_row_contiguous=False,
        header=r"""
        inline void load_bf16x8(device const uint4* source, thread float* output) {
            uint4 bits = source[0];
            float4 low = as_type<float4>(bits << 16);
            float4 high = as_type<float4>(bits & 0xffff0000u);
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                output[2 * i] = low[i];
                output[2 * i + 1] = high[i];
            }
        }
        """,
        source=r"""
        uint lane = thread_position_in_grid.x % 32;
        uint group = thread_position_in_grid.x / 32;
        uint split = thread_position_in_grid.y;
        uint tokens = SINGLE_QUERY ? 1 : queries_shape[2];
        uint heads = queries_shape[1];
        uint per_head = PAIR_HEADS ? tokens : (tokens + 1) / 2;
        uint head = (group / per_head) * (PAIR_HEADS ? 2 : 1);
        if (head >= heads) return;
        uint first_token = (group % per_head) * (PAIR_HEADS ? 1 : 2);
        uint kv_head = head / (heads / keys_shape[1]);
        """ + prefix_size + r"""
        uint feature = lane * 8;
        float q[2][8], accum[2][8] = {};
        float max_score[2] = {-3.402823466e38f, -3.402823466e38f};
        float mass[2] = {};

        #pragma unroll
        for (uint query = 0; query < 2; ++query) {
            uint query_head = head + (PAIR_HEADS ? query : 0);
            uint token = min(first_token + (PAIR_HEADS ? 0 : query), tokens - 1);
            uint source = query_head * queries_strides[1] + token * queries_strides[2] + feature;
            if (VECTOR_QKV) {
                load_bf16x8(reinterpret_cast<device const uint4*>(queries + source), q[query]);
                #pragma unroll
                for (uint f = 0; f < 8; ++f) q[query][f] *= scale;
            } else {
                #pragma unroll
                for (uint f = 0; f < 8; ++f)
                    q[query][f] = scale * float(queries[source + f]);
            }
        }

        uint physical = split;
        """ + prefix_loop + r"""

        // The short tail is smaller than one partition cycle. Map the
        // partition to the ancestor's physical KV slot after the dense prefix.
        uint tail = (split + PARTITIONS - prefix % PARTITIONS) % PARTITIONS;
        #pragma unroll
        for (uint query = 0; query < 2; ++query) {
            uint token = first_token + (PAIR_HEADS ? 0 : query);
            if (token >= tokens) continue;
            uint query_head = head + (PAIR_HEADS ? query : 0);
            uint route = TREE ? token * path_shape[1] : 0;
            uint length = TREE ? path[route] : token + 1;
            if (!SINGLE_QUERY && tail < length) {
                uint physical = prefix + (TREE ? path[route + 1 + tail] : tail);
        """ + load_tail + ACCUMULATE + r"""
            }

            uint output_row = query_head * tokens + token;
            uint stats_index = output_row * PARTITIONS + split;
            if (lane == 0) {
                masses[stats_index] = mass[query];
                maxima[stats_index] = max_score[query];
            }
            uint tile = (output_row * (32 / CLUSTER) + lane / CLUSTER) * PARTITIONS + split;
            #pragma unroll
            for (uint f = 0; f < 8; ++f)
                partials[tile * CLUSTER * 8 + (lane % CLUSTER) * 8 + f] = bfloat16_t(accum[query][f]);
        }
        """,
    )


@lru_cache
def reduce_kernel():
    return mx.fast.metal_kernel(
        name="local_logical_ancestor_gqa_reduce",
        input_names=["partials", "masses", "maxima"],
        output_names=["output"],
        source=r"""
        uint group = thread_position_in_grid.x / 32;
        uint lane = thread_position_in_grid.x % 32;
        uint row = group / 32;
        uint feature_group = group % 32;
        uint stats_base = row * PARTITIONS;
        float largest = -3.402823466e38f;
        for (uint part = lane; part < PARTITIONS; part += 32)
            largest = max(largest, maxima[stats_base + part]);
        largest = simd_max(largest);

        float denominator = 0;
        float numerators[8] = {};
        uint tile_base = (row * (32 / CLUSTER) + feature_group / CLUSTER) * PARTITIONS;
        for (uint part = lane; part < PARTITIONS; part += 32) {
            uint stats_index = stats_base + part;
            float weight = fast::exp(maxima[stats_index] - largest);
            denominator += weight * masses[stats_index];
            uint at = ((tile_base + part) * CLUSTER + feature_group % CLUSTER) * 8;
            #pragma unroll
            for (uint f = 0; f < 8; ++f)
                numerators[f] += weight * float(partials[at + f]);
        }
        denominator = simd_sum(denominator);
        #pragma unroll
        for (uint f = 0; f < 8; ++f) {
            float sum = simd_sum(numerators[f]);
            if (lane == 0)
                output[row * 256 + feature_group * 8 + f] = bfloat16_t(denominator == 0 ? sum : sum / denominator);
        }
        """,
    )


@lru_cache
def linear_route():
    return mx.zeros((1, 1), dtype=mx.uint32)


def logical_attention(q, k, v, scale, tree, *, aligned_bf16=False, prompt_kv=None):
    route = linear_route() if tree is None else tree.logical_route
    return attention_from_route(q, k, v, scale, route, tree=tree is not None,
                                aligned_bf16=aligned_bf16, prompt_kv=prompt_kv)


def attention_from_route(q, k, v, scale, route, *, tree, aligned_bf16=False, prompt_kv=None, write_meta=None):
    heads, tokens = q.shape[1:3]
    single_query = tokens == 1 and prompt_kv is not None and not tree
    pair_heads = single_query or tokens >= 9
    groups = heads // 2 * tokens if pair_heads else heads * ((tokens + 1) // 2)
    cluster = 4 if pair_heads else 32
    shape = (heads, tokens, PARTITIONS)
    inputs = [q, k, v, scale, route] if prompt_kv is None else [q, *prompt_kv, scale, route, k, v]
    if write_meta is not None:
        inputs.append(write_meta)
    partial = partial_kernel(prompt_kv is not None, write_meta is not None)(
        inputs=inputs,
        template=[("PARTITIONS", PARTITIONS), ("CLUSTER", cluster),
                  ("PAIR_HEADS", pair_heads), ("TREE", tree), ("SINGLE_QUERY", single_query),
                  ("VECTOR_QKV", aligned_bf16 and q.dtype == k.dtype == v.dtype == mx.bfloat16)],
        grid=(((groups + 3) // 4) * 128, PARTITIONS, 1), threadgroup=(128, 1, 1),
        output_shapes=[(*shape, 256), shape, shape],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )
    result, = reduce_kernel()(
        inputs=partial, template=[("PARTITIONS", PARTITIONS), ("CLUSTER", cluster)],
        grid=(heads * tokens * 1024, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[q.shape], output_dtypes=[mx.bfloat16],
    )
    return result

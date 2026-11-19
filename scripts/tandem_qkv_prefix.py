"""Specialize the paired BF16 projection for an eight-query Tandem prefix."""

from functools import lru_cache
import inspect

import mlx.core as mx

from ultrarun import lossless_gate_up as paired
from ultrarun.packed_projection import pack_activations
from scripts.bench_verbatim_ffn import replace_once


PREFIX_OUTPUT = r"""
        if (output_tile < QUERY_TILES) {
            uint half_query = QUERY_TILES * 8;
            uint pitch = half_query * 2;
            uint index = token * pitch + output_tile * 8 + row;
            query[index] = bfloat16_t(first.x);
            query[index + half_query] = bfloat16_t(second.x);
            query[index + pitch] = bfloat16_t(first.y);
            query[index + pitch + half_query] = bfloat16_t(second.y);
        } else {
            uint pitch = (half_blocks - QUERY_TILES) * 8;
            uint index = token * pitch + (output_tile - QUERY_TILES) * 8 + row;
            if (token < TOKENS) {
                key[index] = bfloat16_t(first.x);
                value[index] = bfloat16_t(second.x);
            }
            if (token + 1 < TOKENS) {
                key[index + pitch] = bfloat16_t(first.y);
                value[index + pitch] = bfloat16_t(second.y);
            }
        }
"""


@lru_cache
def kernel_factory():
    # Keep the deployed dot products, partials layout and FP32 reduction order.
    source = inspect.getsource(paired.paired_kernel)
    source = replace_once(source, '"project": LOCAL_REDUCTION + PROJECT_SUM + PROJECT_OUTPUT,',
                          '"project": PREFIX_REDUCTION + PROJECT_SUM + PREFIX_OUTPUT,')
    source = replace_once(source, 'else ["output"]),', 'else ["query", "key", "value"]),')
    assert source.count("if (TOKENS > 8) {") == 2
    source = source.replace("if (TOKENS > 8) {", "if (tail_active) {")
    source = replace_once(source, "        metal::simdgroup_matrix<float, 8, 8> a, b, accum, tail, up_accum, up_tail;",
                          "        bool tail_active = output_tile >= QUERY_TILES;\n"
                          "        metal::simdgroup_matrix<float, 8, 8> a, b, accum, tail, up_accum, up_tail;")
    reduce = replace_once(paired.LOCAL_REDUCTION, "if (TOKENS > 8) {", "if (tail_active) {")
    reduce = replace_once(reduce, "(TOKENS > 8 ? 2 : 1)", "(tail_active ? 2 : 1)")
    scope = {**vars(paired), "PREFIX_REDUCTION": reduce, "PREFIX_OUTPUT": PREFIX_OUTPUT}
    exec(compile(source.replace("local_lossless_paired_", "local_tandem_qkv_prefix_"), __file__, "exec"), scope)
    return scope["paired_kernel"]


def project_prefix(attention, x, queries):
    """Read each weight tile once, evaluating Q for eight rows and K/V for all rows."""
    width = x.shape[1]
    assert queries == 8 and queries < width <= 16
    payload = attention.qkv.payload
    blocks, packets, words = payload[-1].shape
    query_tiles, remainder = divmod(attention.half_q, 8)
    assert remainder == 0 and (blocks - query_tiles) * 8 == attention.kv_rows
    splits = (packets + 15) // 16
    threads = splits * 32
    q, k, v = kernel_factory()("project", False, words == 1)(
        inputs=[pack_activations(x[0]), *payload],
        template=[("TOKENS", width), ("SPLITS", splits), ("QUERY_TILES", query_tiles)],
        grid=(threads * blocks, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(queries, 2 * attention.half_q), (width, attention.kv_rows), (width, attention.kv_rows)],
        output_dtypes=[mx.bfloat16] * 3,
    )
    return q[None], k[None], v[None]

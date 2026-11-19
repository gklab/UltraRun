"""Qwen3.8-27B text forward directly over the original safetensors."""
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
import json

import mlx.core as mx
from mlx import nn

from .attention_io import (
    K_HEADS, Q_HEADS, ROPE, WIDTH, normalize_rotate_qk, pack_gated_attention, update_attention_suffix,
)
from .delta import DeltaRollback, causal_conv4, recurrent, reference
from .draft_projection import LanePrefixes, cached_mtp, continue_lane_prefix, multi_gemv, prepare_lane_prefixes
from .logical_attention import attention_from_route, linear_route, logical_attention
from .lossless_gate_up import LosslessPaired, encode_paired, unpack_joint
from .lossless_projection import LosslessDown, decoded_bits, encode_down, unpack_row_major
from .lossless_row import LosslessRow, encode_rows
from .lossless_vocabulary import VocabularyOrder, encode_vocabulary
from .lossless_vocabulary import vocabulary_rows as lossless_vocabulary_rows
from .memory import GIB, MemoryBudgetError
from .packed_delta import gated_projection, projected_convolution
from .packed_projection import (
    pack_activations, pack_weight, packed_feed_forward_from_fragments, packed_head_argmax, packed_partials,
    packed_project, packed_swiglu, residual_norm_pack, unpack_weight,
)
from .projection import split_gemv, tiled_argmax

COMPILED_DELTA_WIDTHS = (6, 11)
COMPILED_TREE_WIDTHS = (8, 11, 16)
# The server warms these widths before serving requests.
DIRECT_DELTA_WIDTHS = frozenset(COMPILED_DELTA_WIDTHS + COMPILED_TREE_WIDTHS)
PREFILL_WINDOW_TOKENS = 4096
VERIFY_SUBMISSION_LAYERS = 8


def project(x, weight, skinny=False):
    if skinny and 1 < x.shape[0] <= 16 and weight.shape[0] >= weight.shape[1]:
        # On M2 Ultra this orientation selects a faster short-M MLX kernel.
        if weight.shape[0] < 100_000 or x.shape[0] <= 4:
            return mx.contiguous((weight @ x.T).T)
    return x @ weight.T


@mx.compile
def swiglu(gate, up):
    return nn.silu(gate) * up


@mx.compile
def delta_coefficients(a, b, log_a, bias):
    decay = mx.exp(-mx.exp(log_a) * nn.softplus(a.astype(mx.float32) + bias))
    return decay, mx.sigmoid(b).astype(mx.float32)


def unit_norm(x):
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.sum(mx.square(x), axis=-1, keepdims=True) + 1e-6)


@mx.compile
def delta_output(y, gate):
    return (y.astype(mx.float32) * nn.silu(gate.astype(mx.float32))).astype(y.dtype)


def split_causal_attention(q, k, v, scale):
    """Use the short-query Metal path with the same logical causal positions."""
    tokens = q.shape[2]
    prefix = k.shape[2] - tokens
    chunk = min((tokens + 1) // 2, 4)
    pieces = []
    for start in range(0, tokens, chunk):
        end = min(start + chunk, tokens)
        pieces.append(mx.fast.scaled_dot_product_attention(
            q[:, :, start:end], k[:, :, :prefix + end], v[:, :, :prefix + end], scale=scale, mask="causal",
        ))
    return mx.concatenate(pieces, axis=2)


@dataclass
class LayerState:
    keys: mx.array | None = None
    values: mx.array | None = None
    conv: mx.array | None = None
    delta: mx.array | None = None
    replay: tuple | None = None
    draft_lanes: LanePrefixes | None = None
    logical_attention: bool = False
    compile_delta_tree: bool = False
    pending_kv: tuple[int, mx.array] | None = None
    prompt_kv: tuple[mx.array, mx.array] | None = None
    draft_prompt_kv: tuple[mx.array, mx.array] | None = None


@dataclass(frozen=True)
class DraftVocabulary:
    prefix: LosslessPaired
    extra: mx.array | None


@dataclass(frozen=True)
class _PreparedDeltaTree:
    windows: mx.array
    delta_route: mx.array
    multi_fork: bool
    scheduled: bool = False


class Layer:
    def __init__(self, weights, config, kind, merge, group_rows, use_reference, skinny,
                 reserve_kv=False, vector_bf16=False, fused_delta=False, wide_verify=False):
        self.gdn = kind == "linear_attention"
        self.c = config
        self.eps = config["rms_norm_eps"]
        self.w = weights
        self.group_rows = group_rows
        self.use_reference = use_reference
        self.skinny = skinny
        self.reserve_kv = reserve_kv
        self.reserve_single_kv = False
        self.vector_bf16 = vector_bf16
        self.fused_delta = fused_delta
        self.wide_verify = wide_verify
        self.packed_ffn = None
        self.packed_weights = set()
        self.fused_ffn = False
        self.residual_norm_epsilon = None
        self.fused_delta_io = False
        self.direct_delta_sequence = False
        self.compact_delta = False
        self.compiled_delta = None
        self.compact_tree = False
        self.short_tokens = 8
        self.split_attention = False
        self.fused_attention_io = False
        for name in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            key = f"{name}.weight"
            if key in weights:
                weights[key] = 1 + weights[key].astype(mx.float32)
        if self.gdn:
            w = weights.pop("linear_attn.conv1d.weight")
            weights["conv"] = mx.contiguous(w.transpose(0, 2, 1))
            for name in ("linear_attn.A_log", "linear_attn.dt_bias"):
                weights[name] = weights[name].astype(mx.float32)
            self.merge("delta_ab", ["linear_attn.in_proj_a", "linear_attn.in_proj_b"])
        if merge:
            self.merge("ffn_in", ["mlp.gate_proj", "mlp.up_proj"])
            if self.gdn:
                self.merge("delta_qkvz", ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"])
            else:
                self.merge("attn_qkv", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])

    def merge(self, target, names):
        weights = [self.w.pop(f"{name}.weight") for name in names]
        self.w[target] = mx.concatenate(weights, axis=0)

    def prefill_copy(self):
        layer = copy(self)
        layer.w = dict(self.w)
        for name in self.packed_weights:
            weight = self.w[name]
            if isinstance(weight, LosslessPaired):
                layer.w[name] = unpack_joint(weight.payload, row_major=True)
            elif isinstance(weight, LosslessDown):
                layer.w[name] = unpack_row_major(weight.payload)
            else:
                layer.w[name] = unpack_weight(weight)
        layer.packed_weights = set()
        return layer

    def project(self, x, name):
        weight = self.w[name]
        if isinstance(weight, LosslessRow):
            if x.shape[0] == 1 and x.dtype == mx.bfloat16:
                return weight.project_row(x)
            return project(x, weight.unpack(), self.skinny)
        if isinstance(weight, LosslessPaired):
            if x.shape[0] <= self.short_tokens and x.dtype == mx.bfloat16:
                packed = pack_activations(x)
                if name == "mlp.down_proj.weight":
                    return weight.partials(packed, x.shape[0]).sum(axis=0).astype(mx.bfloat16)
                return weight.project(packed, x.shape[0])
            return project(x, unpack_joint(weight.payload, row_major=True), self.skinny)
        if isinstance(weight, LosslessDown):
            if x.shape[0] <= self.short_tokens and x.dtype == mx.bfloat16:
                return weight.partials(pack_activations(x), x.shape[0]).sum(axis=0).astype(mx.bfloat16)
            return project(x, unpack_row_major(weight.payload), self.skinny)
        if name in self.packed_weights:
            if x.shape[0] <= self.short_tokens and x.dtype == mx.bfloat16:
                return packed_project(x, weight)
            weight = unpack_weight(weight)
        if name == "ffn_in" and self.packed_ffn is not None and x.shape[0] in (5, 7):
            return packed_project(x, self.packed_ffn)
        if (self.wide_verify and x.shape[0] == 5 and x.dtype == weight.dtype == mx.bfloat16
                and weight.shape in ((16384, 5120), (5120, 17408))):
            return split_gemv(x, weight)
        if self.vector_bf16 and x.dtype == weight.dtype == mx.bfloat16 and 2 <= x.shape[0] <= 4:
            rows, width = weight.shape
            if (width == 5120 and 8192 <= rows <= 34816) or (rows, width) == (5120, 17408):
                return split_gemv(x, weight)
        return project(x, weight, self.skinny)

    def forward(self, x, cache, offset, keep_replay=False, tree=None, last_only=False):
        if (self.compiled_delta is not None and keep_replay and not last_only
                and x.shape[0] <= self.short_tokens and x.dtype == mx.bfloat16
                and cache.conv is not None and cache.delta is not None):
            if tree is None and x.shape[0] in COMPILED_DELTA_WIDTHS:
                return self.compiled_delta(self, x, cache)
            if (tree is not None and self.compact_tree and cache.compile_delta_tree
                    and x.shape[0] in COMPILED_TREE_WIDTHS and tree.spine is not None):
                return self.compiled_delta(self, x, cache, tree)
        w = self.w
        if (not self.gdn and cache.logical_attention and cache.prompt_kv is not None
                and self.fused_attention_io and not last_only
                and 2 <= x.shape[0] <= self.short_tokens and x.dtype == mx.bfloat16
                and self.residual_norm_epsilon is not None
                and isinstance(w.get("attn_qkv"), LosslessPaired)
                and isinstance(w.get("ffn_in"), LosslessPaired)
                and isinstance(w.get("mlp.down_proj.weight"), LosslessDown)
                and "self_attn.o_proj.weight" in self.packed_weights):
            if self.reserve_kv and x.shape[0] in (8, 16):
                return self.reserved_attention(x, cache, offset, tree)
            project_qkv, finish = compile_attention_projections(self.eps)
            raw = project_qkv(x, w["input_layernorm.weight"], w["attn_qkv"].payload)
            output, gate = self.attention(x, cache, offset, tree, projected_qkv=raw, return_gate=True)
            out_weight = w["self_attn.o_proj.weight"]
            out_weight = out_weight.payload if isinstance(out_weight, LosslessPaired) else out_weight
            return finish(x, output, gate, out_weight, w["post_attention_layernorm.weight"],
                          self.residual_norm_epsilon, w["ffn_in"].payload, w["mlp.down_proj.weight"].payload)
        h = self.attention_result(x, cache, offset, keep_replay, tree, last_only)
        if last_only:
            x = x[-1:]
        if (self.residual_norm_epsilon is not None and x.shape[0] <= self.short_tokens
                and x.dtype == h.dtype == mx.bfloat16):
            x, packed = residual_norm_pack(x, h, self.w["post_attention_layernorm.weight"],
                                           self.residual_norm_epsilon)
            return x + self.feed_forward_from_fragments(packed, x.shape[0])
        return self.ffn_residual(x + h)

    def reserved_attention(self, x, cache, offset, tree):
        prompt_length = cache.prompt_kv[0].shape[2]
        write = offset - prompt_length
        replay_count = 0
        slots = empty_attention_replay()
        if cache.pending_kv is not None:
            start, absolute_slots = cache.pending_kv
            write = start - prompt_length
            replay_count = absolute_slots.shape[0]
            slots = mx.pad(absolute_slots - prompt_length, [(0, 16 - replay_count)])
        # Replay rows immediately precede the new batch in the logical history.
        write_meta = mx.array([write, replay_count], dtype=mx.int32)
        positions = mx.array([offset], dtype=mx.int32) if tree is None else tree.positions
        route = linear_route() if tree is None else tree.logical_route

        end = offset + x.shape[0] - prompt_length
        capacity = ((end + 511) // 512) * 512
        keys, values = cache.keys, cache.values
        if keys is None:
            keys = values = mx.zeros((1, K_HEADS, capacity, WIDTH), dtype=x.dtype)
        elif capacity > keys.shape[2]:
            padding = [(0, 0), (0, 0), (0, capacity - keys.shape[2]), (0, 0)]
            keys, values = mx.pad(keys, padding), mx.pad(values, padding)

        w = self.w
        out_weight = w["self_attn.o_proj.weight"]
        out_weight = out_weight.payload if isinstance(out_weight, LosslessPaired) else out_weight
        weights = (w["input_layernorm.weight"], w["attn_qkv"].payload, w["self_attn.q_norm.weight"],
                   w["self_attn.k_norm.weight"], out_weight, w["post_attention_layernorm.weight"],
                   self.residual_norm_epsilon, w["ffn_in"].payload, w["mlp.down_proj.weight"].payload)
        forward = compile_reserved_attention(self.eps, self.c["rope_parameters"]["rope_theta"], tree is not None)
        output, cache.keys, cache.values = forward(
            x, keys, values, cache.prompt_kv, positions, write_meta, route, slots, weights,
        )
        cache.pending_kv = None
        return output

    def attention_result(self, x, cache, offset, keep_replay=False, tree=None, last_only=False):
        h = mx.fast.rms_norm(x, self.w["input_layernorm.weight"], self.eps).astype(x.dtype)
        if self.gdn:
            return self.delta(h, cache, keep_replay, tree)
        return self.attention(h, cache, offset, tree, last_only=last_only)

    def ffn_residual(self, x):
        h = mx.fast.rms_norm(x, self.w["post_attention_layernorm.weight"], self.eps).astype(x.dtype)
        if self.fused_ffn and h.shape[0] <= self.short_tokens and h.dtype == mx.bfloat16:
            return x + self.feed_forward_from_fragments(pack_activations(h), h.shape[0])
        if "ffn_in" in self.w:
            gate, up = mx.split(self.project(h, "ffn_in"), 2, axis=-1)
        else:
            gate = self.project(h, "mlp.gate_proj.weight")
            up = self.project(h, "mlp.up_proj.weight")
        return x + self.project(swiglu(gate, up), "mlp.down_proj.weight")

    def feed_forward_from_fragments(self, packed, tokens):
        gate_up, down = self.w["ffn_in"], self.w["mlp.down_proj.weight"]
        coded_down = isinstance(down, (LosslessDown, LosslessPaired))
        if isinstance(gate_up, LosslessPaired):
            intermediate = gate_up.project(packed, tokens, swiglu=True)
        elif coded_down:
            intermediate = packed_swiglu(packed, tokens, gate_up)
        else:
            return packed_feed_forward_from_fragments(packed, tokens, gate_up, down)
        partial = down.partials(intermediate, tokens) if coded_down else packed_partials(intermediate, tokens, down)
        return partial.sum(axis=0).astype(mx.bfloat16)

    def delta(self, x, cache, keep_replay, tree, *, projected_ab=None):
        c, w = self.c, self.w
        size = x.shape[0]
        hk, hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        key_size, conv_size = hk * dk, 2 * hk * dk + hv * dv
        taps = c["linear_conv_kernel_dim"] - 1
        history = cache.conv
        if history is None:
            history = mx.zeros((taps, conv_size), dtype=x.dtype)
        fused = self.fused_delta and not self.use_reference and taps == 3 and size <= self.short_tokens
        fused_io = (self.fused_delta_io and fused and tree is None and x.dtype == mx.bfloat16
                    and not isinstance(w.get("delta_qkvz"), LosslessPaired))
        if fused_io:
            qkv, z, seq = projected_convolution(x, w["delta_qkvz"], history, w["conv"])
        else:
            weight = w.get("delta_qkvz")
            if (self.direct_delta_sequence and fused and size in DIRECT_DELTA_WIDTHS
                    and history.dtype == x.dtype == mx.bfloat16
                    and isinstance(weight, LosslessPaired)):
                seq, z = weight.project_sequence(pack_activations(x), size, history)
            else:
                if weight is not None:
                    qkv, z = mx.split(self.project(x, "delta_qkvz"), [conv_size], axis=-1)
                else:
                    qkv = self.project(x, "linear_attn.in_proj_qkv.weight")
                    z = self.project(x, "linear_attn.in_proj_z.weight")
                seq = mx.concatenate((history, qkv), axis=0)
            if fused:
                qkv = causal_conv4(seq, w["conv"], None if tree is None else tree.windows)
            elif tree is not None:
                qkv = nn.silu(mx.conv1d(seq[tree.windows], w["conv"], groups=conv_size)[:, 0])
            else:
                qkv = nn.silu(mx.conv1d(seq[None], w["conv"], groups=conv_size)[0])
        ab = self.project(x, "delta_ab") if projected_ab is None else projected_ab
        a, b = mx.split(ab, 2, axis=-1)
        if tree is None:
            cache.conv = mx.contiguous(seq[-taps:])
        q, k, v = mx.split(qkv, [key_size, 2 * key_size], axis=-1)
        q, k = q.reshape(size, hk, dk), k.reshape(size, hk, dk)
        if not fused:
            q, k = unit_norm(q) * (dk ** -0.5), unit_norm(k)
        v = v.reshape(size, hv, dv)
        decay, beta = delta_coefficients(a, b, w["linear_attn.A_log"], w["linear_attn.dt_bias"])
        state = cache.delta
        if state is None:
            state = mx.zeros((hv, dv, dk), dtype=mx.float32)
        trace = (tree is not None or keep_replay) and size <= self.short_tokens and not self.use_reference
        scheduled = getattr(tree, "scheduled", False)
        prepared_route = tree.delta_route if scheduled or isinstance(tree, _PreparedDeltaTree) else None
        tree_spine = tree.spine if prepared_route is None and tree is not None and self.compact_tree else None
        compact_tree = tree_spine is not None or prepared_route is not None
        trajectory = None
        replay = None
        if self.use_reference:
            y, cache.delta = reference(q, k, v, decay, beta, state)
        elif trace and self.compact_delta and (tree is None or compact_tree):
            y, cache.delta, updates, keys = recurrent(
                q, k, v, decay, beta, state, self.group_rows, normalize=fused, record_updates=True,
                parents=None if tree is None or prepared_route is not None else tree.parents, spine=tree_spine,
                multi_fork=compact_tree and tree.multi_fork, prepared_route=prepared_route,
                scheduled_tree=scheduled,
            )
            replay = DeltaRollback(state, updates, keys, decay)
        elif trace:
            y, trajectory = recurrent(q, k, v, decay, beta, state, self.group_rows,
                                       trace=True, normalize=fused,
                                       parents=None if tree is None else tree.parents)
            cache.delta = trajectory[-1]
        else:
            y, cache.delta = recurrent(q, k, v, decay, beta, state, self.group_rows, normalize=fused)
        if keep_replay or tree is not None:
            if not trace:
                replay = (q, k, v, decay, beta, state)
            cache.replay = (seq, trajectory, replay)
        z = z.reshape(size, hv, dv)
        if fused_io:
            return gated_projection(y, z, w["linear_attn.norm.weight"], self.eps, w["linear_attn.out_proj.weight"])
        y = mx.fast.rms_norm(y, w["linear_attn.norm.weight"], self.eps)
        if fused:
            y = delta_output(y, z)
        else:
            y = (y.astype(mx.float32) * nn.silu(z.astype(mx.float32))).astype(x.dtype)
        return self.project(y.reshape(size, hv * dv), "linear_attn.out_proj.weight")

    def accept(self, cache, offset, total, keep):
        if not self.gdn:
            if keep != total and not self.reserve_kv:
                cache.keys = mx.contiguous(cache.keys[:, :, :offset + keep])
                cache.values = mx.contiguous(cache.values[:, :, :offset + keep])
            return
        seq, trajectory, replay = cache.replay
        cache.replay = None
        if keep != total:
            taps = self.c["linear_conv_kernel_dim"] - 1
            cache.conv = mx.contiguous(seq[keep:keep + taps])
            if trajectory is not None:
                cache.delta = trajectory[keep - 1]
            elif isinstance(replay, DeltaRollback):
                cache.delta = replay.restore(keep)
            else:
                q, k, v, decay, beta, state = replay
                _, cache.delta = recurrent(
                    q[:keep], k[:keep], v[:keep], decay[:keep], beta[:keep], state, self.group_rows
                )

    def attention(self, x, cache, offset, tree=None, last_only=False, *, projected_qkv=None, return_gate=False):
        c, w = self.c, self.w
        size, head_dim = x.shape[0], c["head_dim"]
        hq, hk = c["num_attention_heads"], c["num_key_value_heads"]
        fused = (self.fused_attention_io and not last_only and 2 <= size <= self.short_tokens
                 and x.dtype == mx.bfloat16)
        if "attn_qkv" in w:
            raw = self.project(x, "attn_qkv") if projected_qkv is None else projected_qkv
            q, k, v = mx.split(
                raw, [2 * hq * head_dim, (2 * hq + hk) * head_dim], axis=-1
            )
        else:
            q = self.project(x, "self_attn.q_proj.weight")
            k = self.project(x, "self_attn.k_proj.weight")
            v = self.project(x, "self_attn.v_proj.weight")
        q, gate = mx.split(q.reshape(size, hq, 2 * head_dim), 2, axis=-1)
        rope = c["rope_parameters"]
        # The three mRoPE position streams are identical for text-only inputs.
        if fused:
            positions = offset if tree is None else tree.positions
            q, k = normalize_rotate_qk(raw, w["self_attn.q_norm.weight"], w["self_attn.k_norm.weight"],
                                       positions, rope["rope_theta"], self.eps, tree=tree is not None)
        else:
            if last_only:
                q, gate = q[-1:], gate[-1:]
            q = mx.fast.rms_norm(q, w["self_attn.q_norm.weight"], self.eps).astype(x.dtype)
            k = mx.fast.rms_norm(k.reshape(size, hk, head_dim), w["self_attn.k_norm.weight"], self.eps).astype(x.dtype)
            args = dict(dims=int(head_dim * rope["partial_rotary_factor"]), traditional=False,
                        base=rope["rope_theta"], scale=1.0, offset=offset)
            if tree is None:
                q_args = {**args, "offset": offset + size - 1} if last_only else args
                q = mx.fast.rope(q.transpose(1, 0, 2)[None], **q_args)
                k = mx.fast.rope(k.transpose(1, 0, 2)[None], **args)
            else:
                args["offset"] = tree.positions
                q = mx.fast.rope(q[:, :, None, :], **args)[:, :, 0, :].transpose(1, 0, 2)[None]
                k = mx.fast.rope(k[:, :, None, :], **args)[:, :, 0, :].transpose(1, 0, 2)[None]
        v = v.reshape(size, hk, head_dim).transpose(1, 0, 2)[None]
        prompt_kv = cache.prompt_kv
        segmented_draft = (prompt_kv is not None and self.reserve_single_kv and not self.fused_attention_io
                           and q.shape[2] == 1 and tree is None and x.dtype == mx.bfloat16
                           and head_dim == 256 and hq % (2 * hk) == 0)
        logical = (cache.logical_attention and fused) or segmented_draft
        if prompt_kv is not None and not logical:
            # Standard attention reads a contiguous history.
            keys, values = prompt_kv
            cache.keys = keys if cache.keys is None else mx.concatenate((keys, cache.keys), axis=2)
            cache.values = values if cache.values is None else mx.concatenate((values, cache.values), axis=2)
            cache.prompt_kv = prompt_kv = None
        prompt_length = 0 if prompt_kv is None else prompt_kv[0].shape[2]
        write_at = offset - prompt_length
        if cache.pending_kv is not None:
            # Repack accepted side-branch rows in the same update as the new batch.
            write_at, slots = cache.pending_kv
            write_at -= prompt_length
            if prompt_length:
                slots = slots - prompt_length
            k = mx.concatenate((mx.take(cache.keys, slots, axis=2), k), axis=2)
            v = mx.concatenate((mx.take(cache.values, slots, axis=2), v), axis=2)
            cache.pending_kv = None
        if self.reserve_kv and (size > 1 or self.reserve_single_kv):
            end = offset + size - prompt_length
            capacity = ((end + 511) // 512) * 512
            cache.keys = self.append_reserved(cache.keys, k, write_at, capacity)
            cache.values = self.append_reserved(cache.values, v, write_at, capacity)
            k, v = cache.keys[:, :, :end], cache.values[:, :, :end]
        else:
            if offset:
                old_keys, old_values = cache.keys, cache.values
                if self.reserve_kv:
                    old_keys, old_values = old_keys[:, :, :write_at], old_values[:, :, :write_at]
                k = mx.concatenate((old_keys, k), axis=2)
                v = mx.concatenate((old_values, v), axis=2)
            cache.keys, cache.values = k, v
        scale = head_dim ** -0.5
        if logical:
            # Short Q/K and reserved K/V have 256 contiguous BF16 values per head.
            if segmented_draft and cache.draft_prompt_kv is not None:
                # Only change what the draft queries. Writes and RoPE use the full prompt offset.
                prompt_kv = cache.draft_prompt_kv
            out = logical_attention(q, k, v, scale, tree, aligned_bf16=True, prompt_kv=prompt_kv)
        elif tree is not None and tree.spine is not None:
            spine = tree.spine
            end = offset + spine
            lead_q, lead_k, lead_v = q[:, :, :spine], k[:, :, :end], v[:, :, :end]
            if self.split_attention and offset and spine >= 6:
                lead = split_causal_attention(lead_q, lead_k, lead_v, scale)
            else:
                lead = mx.fast.scaled_dot_product_attention(lead_q, lead_k, lead_v, scale=scale, mask="causal")
            tail = mx.fast.scaled_dot_product_attention(
                q[:, :, spine:], k, v, scale=scale, mask=tree.attention[spine:],
            )
            out = mx.concatenate((lead, tail), axis=2)
        elif self.split_attention and tree is None and not last_only and offset and 6 <= size <= self.short_tokens:
            out = split_causal_attention(q, k, v, scale)
        else:
            mask = tree.attention if tree is not None else ("causal" if size > 1 and not last_only else None)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        if fused:
            if return_gate:
                return out, gate
            packed = pack_gated_attention(out, gate)
            weight = w["self_attn.o_proj.weight"]
            weight = weight.payload if isinstance(weight, LosslessPaired) else weight
            return project_attention_output(packed, size, weight)
        out = out[0].transpose(1, 0, 2) * mx.sigmoid(gate)
        out = out.reshape(q.shape[2], -1)
        return self.project(out, "self_attn.o_proj.weight") if projected_qkv is None else out

    def accept_tree(self, cache, tree, offset, path, indices):
        if self.gdn:
            sequence, trajectory, replay = cache.replay
            cache.replay = None
            if trajectory is not None:
                cache.delta = trajectory[path[-1]]
            elif path[-1] != getattr(tree, "checkpoint", tree.spine - 1):
                cache.delta = replay.restore(len(path), indices)
            cache.conv = mx.contiguous(sequence[tree.windows[path[-1], 1:]])
            return
        if self.reserve_kv and path == list(range(len(path))):
            return
        if cache.prompt_kv is not None:
            offset -= cache.prompt_kv[0].shape[2]
        end = offset + len(path)
        for name in ("keys", "values"):
            buffer = getattr(cache, name)
            if self.reserve_kv:
                retained = mx.take(buffer, offset + indices, axis=2)
                updated = mx.slice_update(buffer, retained, mx.array([offset], dtype=mx.int32), axes=[2])
            elif path == list(range(len(path))):
                updated = mx.contiguous(buffer[:, :, :end])
            else:
                retained = mx.take(buffer, offset + indices, axis=2)
                updated = mx.concatenate((buffer[:, :, :offset], retained), axis=2)
            setattr(cache, name, updated)

    @staticmethod
    def append_reserved(buffer, update, offset, capacity):
        if buffer is None:
            padding = capacity - update.shape[2]
            return mx.pad(update, [(0, 0), (0, 0), (0, padding), (0, 0)]) if padding else update
        if capacity > buffer.shape[2]:
            padding = capacity - buffer.shape[2]
            buffer = mx.pad(buffer, [(0, 0), (0, 0), (0, padding), (0, 0)])
        return mx.slice_update(buffer, update, mx.array([offset], dtype=mx.int32), axes=[2])


@lru_cache(maxsize=2)
def compile_attention_projections(eps):
    """Reuse short projection graphs across layer weights and K/V history lengths."""
    @mx.compile
    def project_qkv(x, norm, weights):
        return project_attention_qkv(x, norm, weights, eps)

    return project_qkv, mx.compile(finish_attention_projections)


def project_attention_qkv(x, norm, weights, eps):
    hidden = mx.fast.rms_norm(x, norm, eps).astype(x.dtype)
    return LosslessPaired(weights).project(pack_activations(hidden), x.shape[0])


def project_attention_output(packed, tokens, weight):
    if isinstance(weight, tuple):
        return LosslessPaired(weight).project(packed, tokens)
    return packed_partials(packed, tokens, weight).sum(axis=0).astype(packed.dtype)


def finish_attention_projections(x, attention, gate, out_weight, norm, epsilon, gate_up, down):
    tokens = x.shape[0]
    packed = pack_gated_attention(attention, gate)
    output = project_attention_output(packed, tokens, out_weight)
    residual, normalized = residual_norm_pack(x, output, norm, epsilon)
    intermediate = LosslessPaired(gate_up).project(normalized, tokens, swiglu=True)
    projected = LosslessDown(down).partials(intermediate, tokens).sum(axis=0).astype(x.dtype)
    return residual + projected


@lru_cache
def empty_attention_replay():
    return mx.zeros((16,), dtype=mx.int32)


@lru_cache(maxsize=4)
def compile_reserved_attention(eps, rope_base, tree):
    @mx.compile
    def forward(x, keys, values, prompt_kv, positions, write_meta, route, slots, weights):
        norm, qkv, q_norm, k_norm, out_weight, ff_norm, ff_eps, gate_up, down = weights
        size = x.shape[0]
        raw = project_attention_qkv(x, norm, qkv, eps)
        q_gate, _, v = mx.split(raw, [2 * Q_HEADS * WIDTH, (2 * Q_HEADS + K_HEADS) * WIDTH], axis=-1)
        _, gate = mx.split(q_gate.reshape(size, Q_HEADS, 2 * WIDTH), 2, axis=-1)
        q, k = normalize_rotate_qk(raw, q_norm, k_norm, positions, rope_base, eps, tree=tree)
        v = v.reshape(size, K_HEADS, WIDTH).transpose(1, 0, 2)[None]
        keys, values = update_attention_suffix(keys, values, k, v, write_meta, slots)
        attention = attention_from_route(q, keys, values, WIDTH ** -0.5, route, tree=tree,
                                         aligned_bf16=True, prompt_kv=prompt_kv, write_meta=write_meta)
        output = finish_attention_projections(x, attention, gate, out_weight, ff_norm, ff_eps, gate_up, down)
        return output, keys, values

    return forward


def compile_delta_layer(example):
    template = copy(example)
    template.w = {}
    template.compiled_delta = None
    template.residual_norm_epsilon = None
    signature = tuple((name, type(weight)) for name, weight in example.w.items())

    def run(x, conv, previous, epsilon, weights, tree=None):
        layer = copy(template)
        layer.w = {name: kind(weight) if kind in (LosslessDown, LosslessPaired) else weight
                   for (name, kind), weight in zip(signature, weights)}
        layer.residual_norm_epsilon = epsilon
        state = LayerState(conv=conv, delta=previous)
        # DeltaNet uses the convolution history and recurrent state for position.
        output = layer.forward(x, state, 0, keep_replay=True, tree=tree)
        sequence, _, replay = state.replay
        return output, state.conv, state.delta, sequence, replay.updates, replay.keys, replay.decay

    @mx.compile
    def forward(x, conv, previous, epsilon, weights):
        return run(x, conv, previous, epsilon, weights)

    @lru_cache(maxsize=4)
    def compile_tree(multi_fork, scheduled=False):
        @mx.compile
        def forward_tree(x, conv, previous, epsilon, weights, windows, route):
            return run(x, conv, previous, epsilon, weights, _PreparedDeltaTree(windows, route, multi_fork, scheduled))
        return forward_tree

    tree_steps = (compile_tree(False), compile_tree(True)) if example.compact_tree else ()

    def step(layer, x, cache, tree=None):
        previous = cache.delta
        weights = tuple(layer.w[name].payload if kind in (LosslessDown, LosslessPaired) else layer.w[name]
                        for name, kind in signature)
        inputs = x, cache.conv, previous, layer.residual_norm_epsilon, weights
        if tree is None:
            result = forward(*inputs)
        else:
            step_tree = compile_tree(tree.multi_fork, True) if getattr(tree, "scheduled", False) else (
                tree_steps[tree.multi_fork])
            result = step_tree(*inputs, tree.windows, tree.delta_route)
        output, cache.conv, cache.delta, sequence, updates, keys, decay = result
        cache.replay = (sequence, None, DeltaRollback(previous, updates, keys, decay))
        return output

    return step


class _CheckpointWeights:
    """Open each shard once, retaining only the tensors still needed by the loader."""

    def __init__(self, directory, mapping, names, budget):
        self.directory = directory
        self.budget = budget
        self.remaining = {name: shard for name, shard in mapping.items() if name in names}
        self.unopened = {}
        for name, shard in self.remaining.items():
            self.unopened.setdefault(shard, []).append(name)
        missing = sorted(shard for shard in self.unopened if not (directory / shard).is_file())
        if missing:
            raise FileNotFoundError(f"{len(missing)} weight shards missing in {directory}: {', '.join(missing)}")
        self.pending = {}

    def _open(self, shard):
        tensors = mx.load(str(self.directory / shard))
        for name in self.unopened.pop(shard):
            if name not in tensors:
                raise ValueError(f"Missing checkpoint tensor in {shard}: {name}")
            weight = tensors[name]
            if weight.dtype != mx.bfloat16 and not (weight.dtype == mx.float32 and weight.ndim == 1):
                raise ValueError(f"Unexpected tensor dtype: {name} = {weight.dtype}")
            self.pending[name] = weight
        del tensors
        if self.budget:
            self.budget.check()

    def pop(self, name):
        shard = self.remaining[name]
        if shard in self.unopened:
            self._open(shard)
        del self.remaining[name]
        return self.pending.pop(name)

    def take_prefix(self, prefix):
        return {
            name.removeprefix(prefix): self.pop(name)
            for name in tuple(self.remaining) if name.startswith(prefix)
        }


class Model:
    def __init__(self, directory, *, merge=False, group_rows=4, use_reference=False, mtp=False,
                 skinny=False, reserve_kv=False, vector_bf16=False, fused_delta=False,
                 wide_verify=False, packed_ffn=False, packed_main=False, packed_head=False, packed_head_only=False,
                 packed_attention=False, fused_ffn=False, fast_draft=False, fused_delta_io=False,
                 split_attention=False, fused_attention_io=False, fused_residual=False, compact_delta=False,
                 wide_blocks=False, compact_tree=False, draft_embedding_rows=0, lossless_down=False,
                 lossless_gate_up=False, lossless_inputs=False, lossless_attention_outputs=False,
                 lossless_vocabulary=False, paired_down=False, lossless_draft_rows=0,
                 shared_vocabulary=False, compiled_delta=False, layer_prefill=False, lossless_mtp_ffn=False,
                 async_verify=False, direct_delta_sequence=False, budget=None):
        if (packed_ffn or packed_main) and (not merge or budget is None):
            raise ValueError("Packed projections need merged weights and an active memory budget")
        if packed_ffn and packed_main:
            raise ValueError("Choose either duplicate FFN packing or packed main projections")
        if layer_prefill and not packed_main:
            raise ValueError("Layer-wise prefill needs packed main weights")
        if packed_head_only and not packed_head:
            raise ValueError("Single-layout vocabulary needs a packed vocabulary projection")
        if lossless_vocabulary and not packed_head:
            raise ValueError("Lossless vocabulary needs a packed vocabulary projection")
        if shared_vocabulary and not (lossless_vocabulary and lossless_draft_rows):
            raise ValueError("Shared vocabulary needs lossless target and draft vocabularies")
        if lossless_down and not packed_main:
            raise ValueError("Lossless down projection needs packed main weights")
        if paired_down and not lossless_down:
            raise ValueError("Paired down projection needs lossless down weights")
        if lossless_gate_up and not fused_ffn:
            raise ValueError("Lossless gate/up projections need fused FFN")
        if lossless_inputs and not packed_main:
            raise ValueError("Lossless attention inputs need packed main weights")
        if lossless_attention_outputs and not packed_attention:
            raise ValueError("Lossless attention outputs need packed attention weights")
        if packed_attention and not packed_main:
            raise ValueError("Packed attention needs packed main projections")
        if fused_ffn and not packed_main:
            raise ValueError("Fused FFN needs packed main projections")
        if fused_residual and not fused_ffn:
            raise ValueError("Fused residual normalization needs fused FFN")
        if wide_blocks and not (packed_main and compact_delta):
            raise ValueError("Wide blocks need packed main projections and compact DeltaNet rollback")
        if compact_tree and not compact_delta:
            raise ValueError("Compact trees need compact DeltaNet rollback")
        if compiled_delta and not (compact_delta and fused_delta and fused_residual and not use_reference):
            raise ValueError("Compiled DeltaNet needs GPU recurrence, compact rollback and fused residuals")
        if draft_embedding_rows and (not mtp or budget is None):
            raise ValueError("Cached draft embeddings need MTP and an active memory budget")
        if lossless_draft_rows and (not mtp or budget is None):
            raise ValueError("Lossless draft vocabulary needs MTP and an active memory budget")
        if fused_delta_io and not (packed_attention and fused_delta):
            raise ValueError("Fused DeltaNet projections need packed attention and fused DeltaNet")
        config = json.loads((directory / "config.json").read_text())
        self.config = config["text_config"]
        self.skinny = skinny
        self.wide_verify = wide_verify
        self.fast_draft = fast_draft
        self.short_tokens = 16 if wide_blocks else 8
        self.compact_tree = compact_tree
        c = self.config
        if config["model_type"] != "qwen3_5" or config.get("quantization_config") or c["dtype"] != "bfloat16":
            raise ValueError("Expected a Qwen3.5-family original BF16 checkpoint")
        if c.get("attention_bias") or c.get("output_gate_type", "swish") != "swish":
            raise ValueError("Unsupported attention configuration")
        if c["linear_key_head_dim"] % 32 or c["linear_num_value_heads"] % c["linear_num_key_heads"]:
            raise ValueError("Unsupported Gated DeltaNet head geometry")
        self.fused_attention_io = (
            fused_attention_io and packed_attention and not use_reference
            and (c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]) == (Q_HEADS, K_HEADS, WIDTH)
            and c["rope_parameters"]["partial_rotary_factor"] * WIDTH == ROPE
            and mx.__version__ == "0.32.4" and mx.device_info()["device_name"] == "Apple M2 Ultra"
        )
        prefix = "model.language_model."
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        if budget:
            budget.checkpoint(index.get("metadata", {}).get("total_size"))
        mapping = index["weight_map"]
        names = {k for k in mapping if k.startswith(prefix) or k == "lm_head.weight"}
        if mtp:
            if c.get("mtp_num_hidden_layers") != 1 or c.get("mtp_use_dedicated_embeddings"):
                raise ValueError("Expected one native MTP layer with shared embeddings")
            names.update(k for k in mapping if k.startswith("mtp."))
        weights = _CheckpointWeights(directory, mapping, names, budget)
        self.packed_main_count = 0
        self.lossless_down_count = 0
        self.paired_down_count = 0
        self.lossless_gate_up_count = 0
        self.lossless_input_count = 0
        self.lossless_attention_output_count = 0
        if packed_main:
            residual_norm_epsilon = mx.array([c["rms_norm_eps"]], dtype=mx.float32) if fused_residual else None
            layouts = {"ffn_in": (34816, 5120), "mlp.down_proj.weight": (5120, 17408),
                       "delta_qkvz": (16384, 5120)}
            if packed_attention:
                layouts.update({
                    "attn_qkv": (14336, 5120),
                    "self_attn.o_proj.weight": (5120, 6144),
                    "linear_attn.out_proj.weight": (5120, 6144),
                })

            def pack_main_layer(layer):
                for name, shape in layouts.items():
                    weight = layer.w.get(name)
                    if weight is None or weight.dtype != mx.bfloat16 or weight.shape != shape:
                        continue
                    packed = pack_weight(weight)
                    mx.eval(packed)
                    encode_gate = lossless_gate_up and name == "ffn_in"
                    encode_input = lossless_inputs and name in ("delta_qkvz", "attn_qkv")
                    encode_attention_output = lossless_attention_outputs and name in (
                        "self_attn.o_proj.weight", "linear_attn.out_proj.weight",
                    )
                    encode_paired_down = paired_down and name == "mlp.down_proj.weight"
                    if encode_gate or encode_input or encode_attention_output or encode_paired_down:
                        paired = encode_paired(packed, compact_descriptors=encode_gate or encode_input or encode_attention_output)
                        if not mx.array_equal(unpack_joint(paired.payload), packed).item():
                            raise ValueError(f"Lossless packet validation failed for {name}")
                        layer.w[name] = paired
                        self.lossless_gate_up_count += encode_gate
                        self.lossless_input_count += encode_input
                        self.lossless_attention_output_count += encode_attention_output
                        self.lossless_down_count += encode_paired_down
                        self.paired_down_count += encode_paired_down
                    elif lossless_down and name == "mlp.down_proj.weight":
                        coded = encode_down(packed)
                        if not mx.array_equal(decoded_bits(coded.payload), packed).item():
                            raise ValueError("Lossless down packet validation failed")
                        layer.w[name] = coded
                        self.lossless_down_count += 1
                    else:
                        layer.w[name] = packed
                    layer.packed_weights.add(name)
                    del weight, packed
                    budget.check()
                    self.packed_main_count += 1
                layer.fused_ffn = fused_ffn and {"ffn_in", "mlp.down_proj.weight"} <= layer.packed_weights
                if layer.fused_ffn:
                    layer.residual_norm_epsilon = residual_norm_epsilon
                layer.fused_delta_io = (
                    fused_delta_io and {"delta_qkvz", "linear_attn.out_proj.weight"} <= layer.packed_weights
                    and layer.w["linear_attn.norm.weight"].dtype == mx.bfloat16
                    and not isinstance(layer.w["linear_attn.out_proj.weight"], LosslessPaired)
                )

        self.layers = []
        for i, kind in enumerate(c["layer_types"]):
            layer_weights = weights.take_prefix(f"{prefix}layers.{i}.")
            layer = Layer(layer_weights, c, kind, merge, group_rows, use_reference, skinny,
                          reserve_kv, vector_bf16, fused_delta, wide_verify)
            layer.split_attention = split_attention
            layer.fused_attention_io = self.fused_attention_io and not layer.gdn
            layer.compact_delta = compact_delta
            layer.compact_tree = compact_tree
            layer.direct_delta_sequence = direct_delta_sequence
            layer.short_tokens = self.short_tokens
            mx.eval(*layer.w.values())
            # Release this layer's dense weights before materializing the next layer.
            if packed_main:
                pack_main_layer(layer)
            if budget:
                budget.check()
            self.layers.append(layer)
        embedding_name = f"{prefix}embed_tokens.weight"
        # A separate embedding matrix is not needed while encoding the output head.
        defer_embedding = packed_head_only and not c["tie_word_embeddings"] and not packed_ffn
        if not defer_embedding:
            self.embedding = weights.pop(embedding_name)
        self.head = self.embedding if c["tie_word_embeddings"] else weights.pop("lm_head.weight")
        self.vocab_size = self.head.shape[0]
        self.norm = 1 + weights.pop(f"{prefix}norm.weight").astype(mx.float32)
        if mtp:
            self.mtp_layer = Layer(
                weights.take_prefix("mtp.layers.0."),
                c, "full_attention", merge, group_rows, False, skinny, reserve_kv,
            )
            self.mtp_layer.reserve_single_kv = reserve_kv
            self.mtp_fc = weights.pop("mtp.fc.weight")
            self.mtp_norms = [
                1 + weights.pop(f"mtp.{name}.weight").astype(mx.float32)
                for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm")
            ]
            mx.eval(self.mtp_fc, *self.mtp_norms, *self.mtp_layer.w.values())
        mx.eval(self.head, self.norm)
        if not defer_embedding:
            mx.eval(self.embedding)
        self.packed_ffn_layers = 0
        self.lossless_mtp_ffn = False
        self.lossless_vocabulary = lossless_vocabulary
        self.compiled_delta = compiled_delta
        self.direct_delta_sequence = direct_delta_sequence
        self.async_verify = async_verify
        self.layer_prefill = layer_prefill
        self.packed_head = None
        self.draft_vocabulary = None
        self.draft_vocabulary_rows = 0
        draft_rows = min(lossless_draft_rows, self.vocab_size) // 16 * 16
        self.vocabulary_order = VocabularyOrder(self.vocab_size, draft_rows) if shared_vocabulary and draft_rows else None
        self.fused_head_argmax = False
        self.draft_embedding = None
        self.budget = budget
        if packed_ffn:
            # Leave room for prefill, the KV cache, and the temporary packing buffer.
            spare = budget.limit - mx.get_active_memory() - budget.cache_limit - 6 * GIB
            for layer in self.layers:
                weight = layer.w["ffn_in"]
                if weight.shape != (34816, 5120):
                    continue
                if spare < 2 * weight.nbytes:
                    break
                layer.packed_ffn = pack_weight(weight)
                mx.eval(layer.packed_ffn)
                spare -= weight.nbytes
                budget.check()
                self.packed_ffn_layers += 1
        if packed_head:
            if budget is None:
                raise ValueError("Packed vocabulary projection needs an active memory budget")
            if self.head.dtype != mx.bfloat16 or any(dimension % 8 for dimension in self.head.shape):
                raise ValueError("Expected a BF16 vocabulary projection aligned to 8 by 8 tiles")
            spare = budget.limit - mx.get_active_memory() - budget.cache_limit - 6 * GIB
            if spare < (2 if lossless_vocabulary else 1) * self.head.nbytes:
                raise MemoryBudgetError("Insufficient working-set headroom for the packed vocabulary projection.")
            if lossless_vocabulary:
                self.packed_head = encode_vocabulary(self.head, budget, self.vocabulary_order)
            else:
                self.packed_head = pack_weight(self.head)
                mx.eval(self.packed_head)
            self.fused_head_argmax = (
                self.head.shape == (248320, 5120) and mx.__version__ == "0.32.4"
                and mx.device_info()["device_name"] == "Apple M2 Ultra"
            )
            if packed_head_only:
                self.head = None
                mx.clear_cache()
            budget.check()
        if defer_embedding:
            self.embedding = weights.pop(embedding_name)
            mx.eval(self.embedding)
            budget.check()
        if weights.remaining:
            raise ValueError(f"Unrecognized text weights: {sorted(weights.remaining)}")
        if draft_rows:
            if self.vocabulary_order is not None:
                self.draft_vocabulary = self.vocabulary_order.draft_view(self.packed_head)
            else:
                weight_bytes = 2 * draft_rows * c["hidden_size"]
                reserve = 6 * GIB + 6 * weight_bytes
                if mx.get_active_memory() + budget.cache_limit + reserve > budget.limit:
                    raise MemoryBudgetError("Insufficient headroom for the lossless draft vocabulary.")
                weight = self.vocabulary_rows(mx.arange(draft_rows, dtype=mx.int32))
                self.draft_vocabulary = encode_vocabulary(weight, budget)
                del weight
                mx.clear_cache()
                budget.check(force=True)
            self.draft_vocabulary_rows = draft_rows
        if (lossless_mtp_ffn and mtp and fast_draft and merge and budget is not None
                and mx.__version__ == "0.32.4" and mx.device_info()["device_name"] == "Apple M2 Ultra"):
            weight = self.mtp_layer.w["ffn_in"]
            reserve = 8 * weight.nbytes
            if weight.shape == (34816, 5120) and mx.get_active_memory() + budget.cache_limit + reserve <= budget.limit:
                budget.check(force=True, required_bytes=reserve)
                coded = encode_rows(weight)
                if not mx.array_equal(coded.unpack().view(mx.uint16), weight.view(mx.uint16)).item():
                    raise ValueError("Lossless MTP FFN row validation failed")
                self.mtp_layer.w["ffn_in"] = coded
                self.lossless_mtp_ffn = True
                del weight
                mx.clear_cache()
                budget.check(force=True)
        if draft_embedding_rows:
            rows = min(draft_embedding_rows, self.embedding.shape[0])
            width = self.embedding.shape[1]
            reserve = 8 * rows * width + 6 * width * width + GIB
            if mx.get_active_memory() + budget.cache_limit + reserve > budget.limit:
                raise MemoryBudgetError("Insufficient headroom for the MTP embedding cache.")
            self.draft_embedding_fc = mx.contiguous(self.mtp_fc[:, :width]).astype(mx.float32)
            self.draft_hidden_fc = mx.contiguous(self.mtp_fc[:, width:])
            mx.eval(self.draft_embedding_fc, self.draft_hidden_fc)
            self.draft_embedding = self.project_draft_embeddings(mx.arange(rows, dtype=mx.int32))
        if compiled_delta:
            compiled_steps = {}
            for layer in self.layers:
                if not layer.gdn or layer.residual_norm_epsilon is None:
                    continue
                signature = tuple((name, type(weight)) for name, weight in layer.w.items())
                if signature not in compiled_steps:
                    compiled_steps[signature] = compile_delta_layer(layer)
                layer.compiled_delta = compiled_steps[signature]
        self.resident_gib = mx.get_active_memory() / GIB

    def project_draft_embeddings(self, tokens):
        chunks = []
        for start in range(0, tokens.shape[0], 2048):
            inputs = self.embedding[tokens[start:start + 2048]]
            norm = mx.fast.rms_norm(inputs, self.mtp_norms[0], self.config["rms_norm_eps"]).astype(inputs.dtype)
            projected = norm.astype(mx.float32) @ self.draft_embedding_fc.T
            mx.eval(projected)
            self.budget.check()
            chunks.append(projected)
        table = mx.concatenate(chunks)
        mx.eval(table)
        self.budget.check()
        return table

    def draft_embedding_table(self, shortlist, draft_vocab):
        prefix = min(self.draft_embedding.shape[0], draft_vocab, shortlist.shape[0])
        base = self.draft_embedding[:prefix]
        extras = shortlist.shape[0] - prefix
        if not extras:
            return base, base[:1]
        reserve = 8 * extras * base.shape[1] + GIB
        if mx.get_active_memory() + self.budget.cache_limit + reserve > self.budget.limit:
            return None
        return base, self.project_draft_embeddings(shortlist[prefix:])

    def cache(self):
        return [LayerState() for _ in self.layers]

    def prefill(self, tokens, cache, chunk, *, offset=0):
        window_chunks = PREFILL_WINDOW_TOKENS // chunk
        if not self.layer_prefill or chunk <= self.short_tokens or tokens.size <= chunk or window_chunks < 2:
            for start in range(0, tokens.size, chunk):
                position = offset + start
                yield position, self.hidden(tokens[start:start + chunk], cache, position)
            return
        window_size = chunk * window_chunks
        for start in range(0, tokens.size, window_size):
            window = tokens[start:start + window_size]
            position = offset + start
            if window.size <= chunk:
                yield position, self.hidden(window, cache, position)
                break
            blocks = self.prefill_window(window, cache, position, chunk)
            for i, hidden in enumerate(blocks):
                yield position + i * chunk, hidden
            del blocks

    def prefill_window(self, tokens, cache, offset, chunk):
        # Keep the original token blocks and reuse only the current layer's row-major weights.
        blocks = [self.embedding[tokens[start:start + chunk]] for start in range(0, tokens.size, chunk)]
        # Batch only the verified 512-token blocks. Keep the partial tail's existing path.
        complete_blocks = tokens.size // chunk
        complete_tokens = complete_blocks * chunk
        features = self.config["hidden_size"] + self.config["intermediate_size"]
        join_ffn = (complete_blocks > 1 and chunk == 512
                    and 32 * complete_tokens * features <= 4 * GIB)
        # The 4K panel was checked on the optimized M2 Ultra profile.
        join_delta = (join_ffn and self.fused_attention_io and offset == 0
                      and complete_blocks == 8 and tokens.size == complete_tokens)
        for i, (layer, state) in enumerate(zip(self.layers, cache), start=1):
            prepared = layer.prefill_copy()
            delta_panel = join_delta and isinstance(layer.w.get("delta_qkvz"), LosslessPaired)
            if delta_panel:
                residuals = []
                for start in range(0, complete_blocks, 4):
                    panel = mx.concatenate(blocks[start:start + 4], axis=0)
                    norm = mx.fast.rms_norm(panel, prepared.w["input_layernorm.weight"], prepared.eps)
                    norm = norm.astype(panel.dtype)
                    # Retain the 512-row reduction for the small a/b projection.
                    ab = mx.concatenate([prepared.project(block, "delta_ab")
                                         for block in mx.split(norm, 4, axis=0)], axis=0)
                    output = panel + prepared.delta(norm, state, False, None, projected_ab=ab)
                    self.evaluate(output, [state])
                    residuals.append(output)
                    del panel, norm, ab
                    self.budget.check()
                ffn_input = mx.concatenate(residuals, axis=0)
                del residuals
            else:
                for j, hidden in enumerate(blocks):
                    current = prepared if hidden.shape[0] > layer.short_tokens else layer
                    position = offset + j * chunk
                    if join_ffn and j < complete_blocks:
                        output = hidden + current.attention_result(hidden, state, position)
                    else:
                        output = current.forward(hidden, state, position)
                    self.evaluate(output, [state])
                    blocks[j] = output
                    self.budget.check()
                del current
            if join_ffn:
                # Attention has advanced the cache for every block; the FFN is row-independent.
                if not delta_panel:
                    ffn_input = mx.concatenate(blocks[:complete_blocks], axis=0)
                output = prepared.ffn_residual(ffn_input)
                del ffn_input
                mx.eval(output)
                blocks[:complete_blocks] = [output[start:start + chunk] for start in range(0, complete_tokens, chunk)]
                self.budget.check()
            del prepared
            if i % 8 == 0:
                mx.clear_cache()
                self.budget.check(force=True)
        return blocks

    def hidden(self, tokens, cache, offset, keep_replay=False, tree=None):
        if tree is not None:
            tree_limit = self.short_tokens if self.compact_tree and tree.spine is not None else 8
            if tokens.shape[0] > tree_limit or any(layer.use_reference for layer in self.layers):
                raise ValueError(f"Tree verification requires at most {tree_limit} GPU-backed nodes")
        x = self.embedding[tokens]
        width = tokens.shape[0]
        stage_prefill = self.packed_main_count and width > self.short_tokens
        stage_verify = self.async_verify and keep_replay and 2 <= width <= self.short_tokens
        for end, (layer, state) in enumerate(zip(self.layers, cache), 1):
            x = layer.forward(x, state, offset, keep_replay, tree)
            if stage_prefill and end % 4 == 0:
                self.evaluate(x, cache[end - 4:end])
                self.budget.check()
            elif stage_verify and end % VERIFY_SUBMISSION_LAYERS == 0 and end < len(self.layers):
                # Start executing this group while building the remaining verification graph.
                self.evaluate(x, cache[end - VERIFY_SUBMISSION_LAYERS:end], asynchronous=True)
        return x

    def logits(self, x):
        x = mx.fast.rms_norm(x, self.norm, self.config["rms_norm_eps"]).astype(x.dtype)
        if self.head is not None:
            return project(x, self.head, self.skinny)
        return self.packed_vocabulary_logits(x, skinny=self.skinny)

    def vocabulary_rows(self, rows):
        if self.head is not None:
            return self.head[rows]
        return self.packed_vocabulary_rows(rows)

    def packed_vocabulary_rows(self, rows):
        if self.lossless_vocabulary:
            if self.vocabulary_order is not None:
                rows = self.vocabulary_order.to_storage(rows)
            return lossless_vocabulary_rows(self.packed_head, rows)
        return unpack_weight(self.packed_head, rows)

    def prepare_draft_vocabulary(self, shortlist, base_rows):
        prefix = self.draft_vocabulary_rows
        if prefix and base_rows >= prefix:
            extra = self.vocabulary_rows(shortlist[prefix:]) if shortlist.shape[0] > prefix else None
            if extra is not None:
                mx.eval(extra)
            return DraftVocabulary(self.draft_vocabulary, extra)
        rows = self.vocabulary_rows(shortlist)
        mx.eval(rows)
        return rows

    def draft_scores(self, x, weight):
        if isinstance(weight, DraftVocabulary):
            if x.shape[0] == 1 and x.dtype == mx.bfloat16:
                prefix = weight.prefix.project(pack_activations(x), 1)
            else:
                prefix = project(x, unpack_joint(weight.prefix.payload, row_major=True))
            if weight.extra is None:
                return prefix
            return mx.concatenate((prefix, project(x, weight.extra)), axis=-1)
        if weight is not None:
            return project(x, weight)
        return self.packed_vocabulary_logits(x)

    def packed_vocabulary_blocks(self):
        # Up to 16,256 rows (159 MiB for the 27B head) in each row-major temporary.
        rows_per_block = 16256
        for start in range(0, self.vocab_size, rows_per_block):
            end = min(start + rows_per_block, self.vocab_size)
            if self.lossless_vocabulary:
                rows = mx.arange(start, end, dtype=mx.int32)
                yield start, self.packed_vocabulary_rows(rows)
            else:
                yield start, unpack_weight(self.packed_head[start // 8:end // 8])

    def packed_vocabulary_logits(self, x, *, skinny=False):
        # Limit the lifetime of the row-major temporary to one vocabulary block.
        scores = []
        skinny = skinny and (self.vocab_size < 100_000 or x.shape[0] <= 4)
        for _, rows in self.packed_vocabulary_blocks():
            block_scores = project(x, rows, skinny)
            mx.eval(block_scores)
            del rows
            scores.append(block_scores)
            self.budget.check()
        return mx.concatenate(scores, axis=-1)

    def predictions(self, x, *, observe_vocabulary_tiles=None):
        if self.packed_head is not None and 2 <= x.shape[0] <= self.short_tokens and x.dtype == mx.bfloat16:
            x = mx.fast.rms_norm(x, self.norm, self.config["rms_norm_eps"]).astype(x.dtype)
            if self.lossless_vocabulary:
                packed = pack_activations(x)
                if self.fused_head_argmax:
                    if self.vocabulary_order is not None:
                        if observe_vocabulary_tiles is not None:
                            scores, indices = self.packed_head.argmax_tiles(packed, x.shape[0])
                            observe_vocabulary_tiles(scores, indices)
                            return self.vocabulary_order.argmax_from_tiles(scores, indices)
                        return self.vocabulary_order.argmax(self.packed_head, packed, x.shape[0])
                    return self.packed_head.argmax(packed, x.shape[0])
                scores = self.packed_head.project(packed, x.shape[0])
                if self.vocabulary_order is not None:
                    scores = self.vocabulary_order.token_order(scores)
                return mx.argmax(scores, axis=-1)
            if self.fused_head_argmax:
                return packed_head_argmax(x, self.packed_head)
            return mx.argmax(packed_project(x, self.packed_head), axis=-1)
        if self.head is not None and self.wide_verify and x.shape[0] == 5 and x.dtype == self.head.dtype == mx.bfloat16:
            x = mx.fast.rms_norm(x, self.norm, self.config["rms_norm_eps"]).astype(x.dtype)
            return tiled_argmax(x, self.head, 62080)
        return mx.argmax(self.logits(x), axis=-1)

    def forward(self, tokens, cache, offset, logits=True):
        x = self.hidden(tokens, cache, offset)
        if not logits:
            return x[-1:]
        return self.logits(x[-1:])[0]

    def prepare_draft_lanes(self, ids, budget):
        if (self.mtp_fc.shape != (5120, 10240) or self.mtp_fc.dtype != mx.bfloat16
                or mx.__version__ != "0.32.4" or mx.device_info()["device_name"] != "Apple M2 Ultra"):
            return None
        norm = self.mtp_norms[0]
        return prepare_lane_prefixes(self.mtp_fc, self.embedding, norm, self.config["rms_norm_eps"], ids, budget)

    def draft(self, hidden, next_tokens, cache, offset, *, cached_embedding=None):
        embed_norm, hidden_norm, head_norm = self.mtp_norms
        eps = self.config["rms_norm_eps"]
        hidden = mx.fast.rms_norm(hidden, hidden_norm, eps).astype(hidden.dtype)
        if cached_embedding is None:
            inputs = self.embedding[next_tokens]
            inputs = mx.fast.rms_norm(inputs, embed_norm, eps).astype(inputs.dtype)
            joined = mx.concatenate((inputs, hidden), axis=-1)
            if cache.draft_lanes is not None and joined.shape[0] == 1 and joined.dtype == mx.bfloat16:
                x = continue_lane_prefix(self.mtp_fc, joined, cache.draft_lanes, next_tokens)
            else:
                x = project(joined, self.mtp_fc)
        else:
            x = cached_mtp(hidden, self.draft_hidden_fc, *cached_embedding)
        x = self.mtp_layer.forward(x, cache, offset, last_only=self.fast_draft and x.shape[0] > 1)
        return mx.fast.rms_norm(x, head_norm, eps).astype(x.dtype)

    def draft_root_pair(self, hidden, tokens, cache, offset, head):
        hidden, scores = self.draft_batch(hidden, tokens, (cache, copy(cache)), (offset, offset), head)
        return hidden[:1], scores

    def draft_batch(self, hidden, tokens, caches, offsets, head):
        """Score two or three MTP paths with shared weights and independent KV state."""
        layer = self.mtp_layer
        w, eps = layer.w, layer.eps
        embed_norm, hidden_norm, head_norm = self.mtp_norms
        inputs = mx.fast.rms_norm(self.embedding[tokens], embed_norm, eps).astype(mx.bfloat16)
        hidden = mx.fast.rms_norm(hidden, hidden_norm, eps).astype(mx.bfloat16)
        if hidden.shape[0] == 1:
            hidden = mx.concatenate((hidden,) * len(caches))
        x = multi_gemv(mx.concatenate((inputs, hidden), axis=-1), self.mtp_fc)
        h = mx.fast.rms_norm(x, w["input_layernorm.weight"], eps).astype(mx.bfloat16)
        qkv = multi_gemv(h, w["attn_qkv"])
        a = mx.concatenate([layer.attention(h[i:i + 1], cache, offset, projected_qkv=qkv[i:i + 1])
                            for i, (cache, offset) in enumerate(zip(caches, offsets))])
        x = x + multi_gemv(a, w["self_attn.o_proj.weight"])
        h = mx.fast.rms_norm(x, w["post_attention_layernorm.weight"], eps).astype(mx.bfloat16)
        gate, up = mx.split(multi_gemv(h, w["ffn_in"]), 2, axis=-1)
        x = x + multi_gemv(swiglu(gate, up), w["mlp.down_proj.weight"])
        h = mx.fast.rms_norm(x, head_norm, eps).astype(mx.bfloat16)
        scores = head.prefix.project(pack_activations(h), len(caches))
        if head.extra is not None:
            extras = mx.concatenate([project(h[i:i + 1], head.extra) for i in range(len(caches))])
            scores = mx.concatenate((scores, extras), axis=-1)
        return h, scores

    def accept(self, cache, offset, total, keep):
        for layer, state in zip(self.layers, cache):
            layer.accept(state, offset, total, keep)

    def accept_tree(self, cache, tree, offset, path, *, defer_kv=False):
        indices = mx.array(path, dtype=mx.int32)
        pending = None
        if defer_kv:
            first_moved = next((i for i, node in enumerate(path) if node != i), len(path))
            if first_moved < len(path):
                pending = offset + first_moved, offset + indices[first_moved:]
        deferred = False
        for layer, state in zip(self.layers, cache):
            if pending is not None and not layer.gdn and layer.reserve_kv:
                state.pending_kv = pending
                deferred = True
            else:
                layer.accept_tree(state, tree, offset, path, indices)
        if defer_kv:
            return deferred

    @staticmethod
    def evaluate(output, cache, *, asynchronous=False):
        state = [v for layer in cache for v in (layer.keys, layer.values, layer.conv, layer.delta)
                 if v is not None]
        evaluate = mx.async_eval if asynchronous else mx.eval
        evaluate(output, *state)

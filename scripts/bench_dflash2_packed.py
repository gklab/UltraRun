"""Test losslessly packed FFN and projection weights in the BF16 Tandem draft."""

import argparse
from collections import Counter
from copy import copy
from hashlib import sha256
import inspect
import json
from math import exp
from pathlib import Path

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.lossless_gate_up import LosslessPaired, unpack_joint
from ultrarun.memory import GIB, metal_job
from ultrarun.packed_projection import pack_activations, pack_weight
from ultrarun.server import warm_verification_graphs
from scripts.bench_dflash2 import ROOT, bind_draft, load_draft, load_target, pages, prefill, tiled_candidates
from scripts.bench_dflash2_tandem import draft_runner, exercise, reference, streams_at_prefix, verified_directories
from scripts.bench_dspine import checked_paired, checked_split_k, paired_rows
from scripts.bench_verbatim_ffn import replace_once
from scripts.dflash2_ranked_head import CompactShortlist


def payload_bytes(weights):
    return sum(array.nbytes for weight in weights for array in weight.payload)


class PackedFFN:
    def __init__(self, original, name):
        self.original = original
        weights = (original.gate_proj.weight, original.up_proj.weight, original.down_proj.weight)
        self.gate_up = checked_paired(mx.concatenate(weights[:2], axis=0), name + " gate/up")
        self.down = checked_split_k(weights[2], name + " down")
        self.bytes = payload_bytes((self.gate_up, self.down))

    def __call__(self, x):
        rows = x.shape[1]
        if x.shape[0] != 1 or rows > 16:
            return self.original(x)
        mid = self.gate_up.project(pack_activations(x[0]), rows, swiglu=True)
        return self.down.partials(mid, rows).sum(axis=0).astype(mx.bfloat16)[None]


class PackedLinear:
    def __init__(self, original, name, *, split_k=False):
        self.original = original
        self.split_k = split_k
        check = checked_split_k if split_k else checked_paired
        self.weight = check(original.weight, name)
        self.bytes = payload_bytes((self.weight,))

    def __call__(self, x):
        rows = x.shape[1]
        if x.shape[0] != 1 or rows > 16:
            return self.original(x)
        if self.split_k:
            packed = pack_activations(x[0])
            return self.weight.partials(packed, rows).sum(axis=0).astype(mx.bfloat16)[None]
        return paired_rows(self.weight, x)


class PackedAttention:
    """Keep upstream sliding-window attention and combine its short projections."""

    def __init__(self, attention, name):
        assert attention.is_sliding and not attention.is_causal
        self.original = attention
        q, k, v = (layer.weight for layer in (attention.q_proj, attention.k_proj, attention.v_proj))
        self.half_q, self.kv_rows = q.shape[0] // 2, k.shape[0]
        self.qkv = checked_paired(mx.concatenate((q[:self.half_q], k, q[self.half_q:], v), axis=0), name + " QKV")
        self.kv = LosslessPaired((*self.qkv.payload[:-1], mx.contiguous(self.qkv.payload[-1][self.half_q // 8:])))
        if not mx.array_equal(unpack_joint(self.kv.payload), pack_weight(mx.concatenate((k, v), axis=0))).item():
            raise ValueError(f"{name} K/V descriptor alias did not round-trip")
        self.output = PackedLinear(attention.o_proj, name + " output")
        self.bytes = payload_bytes((self.qkv,)) + self.kv.payload[-1].nbytes + self.output.bytes

    def split_prefix_projection(self, x, queries):
        descriptor = self.qkv.payload[-1][:self.half_q // 8]
        query_weight = LosslessPaired((*self.qkv.payload[:-1], descriptor))
        query = paired_rows(query_weight, x[:, :queries])
        key, value = mx.split(paired_rows(self.kv, x), 2, axis=-1)
        return query, key, value

    def __call__(self, x, ctx, rope, cache, *, query_rows=None, prefix_projection=None):
        attention = self.original
        batch, width, _ = x.shape
        if batch != 1 or width > 16:
            if query_rows is not None:
                raise ValueError("Prefix queries require one draft of at most sixteen positions")
            return attention(x, ctx, rope, cache)
        keep = attention.sliding_window - 1
        if ctx.shape[1] > keep:
            cache.offset += ctx.shape[1] - keep
            ctx = ctx[:, -keep:]
        count = ctx.shape[1]
        offset = cache.offset
        half, kv = self.half_q, self.kv_rows
        queries = width if query_rows is None else query_rows
        if queries == width:
            left, prop_k, right, prop_v = mx.split(
                paired_rows(self.qkv, x), [half, half + kv, 2 * half + kv], axis=-1,
            )
            query = mx.concatenate((left, right), axis=-1)
        else:
            # K/V still cover the whole proposal block. Q and output need only
            # the prefix, and the Q view shares the actual BF16 payloads.
            if not 0 < queries < width:
                raise ValueError("Invalid draft query prefix length")
            if prefix_projection is None:
                query, prop_k, prop_v = self.split_prefix_projection(x, queries)
            else:
                query, prop_k, prop_v = prefix_projection(self, x, queries)
        if count <= 16:
            ctx_k, ctx_v = mx.split(paired_rows(self.kv, ctx), 2, axis=-1)
        else:
            ctx_k, ctx_v = attention.k_proj(ctx), attention.v_proj(ctx)

        def reshape(value, rows, heads):
            return value.reshape(batch, rows, heads, -1)

        q = attention.q_norm(reshape(query, queries, attention.n_heads)).transpose(0, 2, 1, 3)
        ck = attention.k_norm(reshape(ctx_k, count, attention.n_kv_heads)).transpose(0, 2, 1, 3)
        cv = reshape(ctx_v, count, attention.n_kv_heads).transpose(0, 2, 1, 3)
        pk = attention.k_norm(reshape(prop_k, width, attention.n_kv_heads)).transpose(0, 2, 1, 3)
        pv = reshape(prop_v, width, attention.n_kv_heads).transpose(0, 2, 1, 3)
        q, ck, pk = rope(q, offset=offset + count), rope(ck, offset=offset), rope(pk, offset=offset + count)
        key, value = cache.update_and_fetch(ck, cv)
        context_len = key.shape[2]
        keys = mx.concatenate((key, pk), axis=2)
        values = mx.concatenate((value, pv), axis=2)
        query_position = context_len + mx.arange(queries)[:, None]
        key_position = mx.arange(context_len + width)[None]
        mask = (query_position - key_position < attention.sliding_window) | (key_position >= context_len)
        output = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attention.scale, mask=mask)
        return self.output(output.transpose(0, 2, 1, 3).reshape(batch, queries, -1))


def pack_variants(draft, budget):
    ffn, all_projections = copy(draft), copy(draft)
    ffn_layers, all_layers = [], []
    ffn_bytes = other_bytes = 0
    for index, base in enumerate(draft.layers):
        budget.check(force=True, required_bytes=2 * GIB)
        name = f"Tandem layer {index}"
        layer = copy(base)
        layer.mlp = PackedFFN(base.mlp, name)
        ffn_bytes += layer.mlp.bytes
        ffn_layers.append(layer)
        full = copy(layer)
        full.self_attn = PackedAttention(base.self_attn, name)
        other_bytes += full.self_attn.bytes
        for label in ("attention_conv", "mlp_conv"):
            convolution = copy(getattr(base, label))
            convolution.kernel_projection = PackedLinear(convolution.kernel_projection, f"{name} {label}")
            other_bytes += convolution.kernel_projection.bytes
            setattr(full, label, convolution)
        all_layers.append(full)
        mx.clear_cache()
    ffn.layers, all_projections.layers = ffn_layers, all_layers
    all_projections.fc = PackedLinear(draft.fc, "Tandem context projection", split_k=True)
    other_bytes += all_projections.fc.bytes
    assert all(old.mlp is not new.mlp for old, new in zip(draft.layers, ffn.layers, strict=True))
    assert all(a.mlp is b.mlp for a, b in zip(ffn.layers, all_projections.layers, strict=True))
    print("packed", json.dumps({
        "ffn_extra_bytes": ffn_bytes, "other_extra_bytes": other_bytes, "weight_bits_equal": True,
        "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics(),
    }), flush=True)
    return ffn, all_projections


def compare_logits(drafts, prefilled, context):
    _, first, features = prefilled
    previous = None
    for name, draft in drafts.items():
        cache = draft.make_cache()
        for state in cache:
            state.offset = context - features.shape[1]
        ids = mx.array([[first] + [draft.config.mask_token_id] * (draft.config.block_size - 1)], dtype=mx.int32)
        logits = draft.compute_logits(draft.hidden_states(ids, features, cache, logits_start=1))
        if previous is not None:
            # GPU kernels keep original BF16 weights. Independently check whether
            # their dot product order changes candidate logits in the first block.
            reference = previous[:, :logits.shape[1]]
            active = mx.isfinite(logits)
            delta = mx.where(active, logits.astype(mx.float32) - reference.astype(mx.float32), 0.0)
            equal = (logits.view(mx.uint16) == reference.view(mx.uint16)) & active
            identical = (mx.sum(equal) / mx.sum(active)).item()
            same_argmax = mx.sum(mx.argmax(logits, axis=-1) == mx.argmax(reference, axis=-1)).item()
            print("proposal_logits", json.dumps({
                "arm": name, "equal_fraction": identical, "argmax_equal_rows": same_argmax,
                "rows": logits.shape[1], "max_abs_diff": mx.max(mx.abs(delta)).item(),
            }), flush=True)
        else:
            mx.eval(logits)
            previous = logits


def short_vocabulary(draft, model, prompt):
    """Use 32K shared BF16 rows and add rare IDs already present in the prompt."""
    assert draft.config.output_multiplier > 0 and not draft.config.final_logit_softcapping
    variant = copy(draft)
    shortlist = CompactShortlist(model, prompt)
    assert shortlist.rows == 32768
    variant.lm_head = lambda hidden: shortlist.masked_logits(hidden)[None]
    print("short_vocabulary", json.dumps({
        "base_rows": shortlist.rows, "prompt_rows": shortlist.extras.size,
        "extra_bytes": 0 if shortlist.weights is None else shortlist.weights.nbytes,
    }), flush=True)
    return variant


class FinalPrefixLayer:
    """Retain all attention sources, then evaluate just the causal output prefix."""

    def __init__(self, layer, width):
        self.layer = layer
        self.width = width
        assert layer.attention_conv.kernel_size == layer.mlp_conv.kernel_size == 2

    def __call__(self, x, ctx, rope, cache):
        layer, width = self.layer, self.width
        normalized = layer.input_layernorm(x)
        values, kernel = layer.attention_conv.prepare(normalized)
        mixed = layer.self_attn(values, ctx, rope, cache)
        h = x[:, :width] + layer.attention_conv.finish(mixed[:, :width], kernel[:, :width])
        values, kernel = layer.mlp_conv.prepare(layer.post_attention_layernorm(h))
        return h + layer.mlp_conv.finish(layer.mlp(values), kernel)


def trimmed_output(draft, width):
    variant = copy(draft)
    variant.layers = [*draft.layers[:-1], FinalPrefixLayer(draft.layers[-1], width)]
    return variant


def verifier_cost_ratio():
    """Price both widths from recorded BF16 packed Tandem measurements."""
    def arm(file, label):
        rows = [json.loads(line[7:]) for line in (ROOT / "runs" / file).read_text().splitlines()
                if line.startswith("result ")]
        return next(row for row in rows if row["arm"] == label)

    narrow = arm("tandem-dflash2-packed-code-ab.log", "packed_projections")
    wide = arm("tandem-dflash2-fork-sites-code-ab.log", "fork_sites")
    assert (narrow["verifier_budget"], wide["verifier_budget"]) == (8, 16)
    overhead = (wide["phase_s"]["draft"] + wide["phase_s"]["commit"]) / wide["rounds"]
    round_times = [1000 * (row["phase_s"]["verify"] / row["rounds"] + overhead) for row in (narrow, wide)]
    ratio = round_times[1] / round_times[0]
    print("verify_price", json.dumps({"ms_8_16": round_times, "ratio": ratio}), flush=True)
    return ratio


def fork_site_allocator(*, wide_ratio=None, with_mass=False, adjust=None, skip_unavailable=False):
    # Preserve the same priority queue and width. DeltaNet can reuse a saved
    # parent state for any sibling of that parent in the resulting DFS order.
    source = inspect.getsource(reference.candidate_tree)
    for old, new in (
        ("forks=0", "fork_sites=set()"),
        ("if side and forks>=FORK_LIMIT:", "if side and parent not in fork_sites and len(fork_sites)>=FORK_LIMIT:"),
        ("forks+=side", "if side:\n            fork_sites.add(parent)"),
    ):
        source = replace_once(source, old, new)
    if adjust is not None:
        source = replace_once(source, "enumerate(logp[depth][previous])",
                              "enumerate(adjust(depth, logp[depth][previous]))")
    if skip_unavailable:
        source = replace_once(source, "                heapq.heappush(queue,(cost-logprob,parent,depth,rank))",
                              """                if logprob != -float("inf"):
                    heapq.heappush(queue,(cost-logprob,parent,depth,rank))""")
    if wide_ratio is not None or with_mass:
        # Allocation order is prefix-closed. The first eight popped nodes are
        # exactly the narrow tree before the DFS layout changes their order.
        source = replace_once(source, "    tokens=[pending]",
                              "    tokens=[pending]\n    expected=narrow=1.0")
        source = replace_once(source, "        tokens.append(candidates[depth][rank])",
                              """        tokens.append(candidates[depth][rank])
        expected += exp(-cost)
        if len(tokens) == 8:
            narrow = expected""")
    if wide_ratio is not None:
        source = replace_once(source, "    order=[]",
                              """    if len(tokens) > 8 and expected < wide_ratio * narrow:
        tokens = tokens[:8]
        children = [[child for child in row if child < 8] for row in children[:8]]
    order=[]""")
    scope = {**vars(reference), "exp": exp, "wide_ratio": wide_ratio, "adjust": adjust}
    if with_mass:
        if wide_ratio is not None:
            raise ValueError("Mass is reported before adaptive width selection")
        source = replace_once(source, "spine if spine<len(tokens) else None",
                              "spine if spine<len(tokens) else None, expected")
    exec(compile(source, __file__, "exec"), scope)
    return scope["candidate_tree"]


def fork_site_runner(block, stats, *, wide_ratio=None, route_digest=None, lattice=None, calibration=None):
    allocate = fork_site_allocator(wide_ratio=wide_ratio, adjust=None if calibration is None else calibration.adjust)

    def checked(*args):
        tokens, parents, spine = allocate(*args)
        children = Counter(parents[1:])
        assert sum(n > 1 for n in children.values()) <= reference.FORK_LIMIT
        extra = sum(n - 1 for n in children.values())
        stats["calls"] += 1
        stats["more_than_two_side_children"] += extra > reference.FORK_LIMIT
        stats["max_side_children"] = max(stats["max_side_children"], extra)
        stats["selected_wide"] += len(tokens) > 8
        return tokens, parents, spine

    return draft_runner(block, allocate=checked, route_digest=route_digest, lattice=lattice,
                        observe=None if calibration is None else calibration.observe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, choices=(8, 16), default=8)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--compare-vocab", action="store_true", help="compare full and 32K-plus-prompt draft vocabularies")
    variants.add_argument("--trim-output", action="store_true", help="compare full and eight-row final draft stages")
    variants.add_argument("--compare-fork-sites", action="store_true", help="allow extra siblings at two saved parents")
    variants.add_argument("--compare-adaptive-width", action="store_true",
                          help="compare fixed 8/16 against probability-priced width selection")
    variants.add_argument("--compare-ranked-head", action="store_true",
                          help="compare full 8-node scoring with sliced and storage-order scoring")
    parser.add_argument("--upstream", type=Path, default=Path("/private/tmp/localllm-ref/dflash2/model_mlx.py"))
    args = parser.parse_args()
    if (args.trim_output or args.compare_ranked_head) and args.cap != 8:
        parser.error("Prefix scoring requires --cap 8")
    path = verified_directories(("tandem_b16",))["tandem_b16"]
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    before = pages()
    ratio = verifier_cost_ratio() if args.compare_adaptive_width else None
    with metal_job(80, 2) as budget:
        model = load_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=9 * GIB)
        stock, upstream = load_draft(args.upstream, path)
        assert stock.config.block_size == 16
        upstream.rank_candidates = tiled_candidates
        bind_draft(model, stock, {})
        ffn, packed = pack_variants(stock, budget)
        if args.trim_output:
            draft = short_vocabulary(packed, model, trace["input"])
            drafts = {"full_output": draft, "trimmed_output": trimmed_output(draft, args.cap)}
        elif args.compare_vocab:
            short = short_vocabulary(packed, model, trace["input"])
            drafts = {"packed_projections": packed, "packed_short_vocab": short}
        elif args.compare_fork_sites:
            drafts = {"side_children": packed, "fork_sites": packed}
        elif args.compare_adaptive_width:
            drafts = dict.fromkeys(("fixed8", "fixed16", "adaptive"), packed)
        elif args.compare_ranked_head:
            assert stock.config.output_multiplier == 1.0 and not stock.config.final_logit_softcapping
            trimmed = trimmed_output(packed, args.cap)
            drafts = {"full_output": packed, "prefix_output": trimmed, "mapped_rank": trimmed}
        else:
            drafts = {"stock": stock, "packed_ffn": ffn, "packed_projections": packed}
        warm_verification_graphs(model, budget)
        layout = model.fused_attention_io
        model.fused_attention_io = False
        try:
            prefilled = prefill(model, stock, trace["input"])
        finally:
            model.fused_attention_io = layout
        assert prefilled[1] == trace["output"][0]
        print("prefill", json.dumps({"peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
        compare_logits(drafts, prefilled, len(trace["input"]))
        runner = draft_runner(stock.config.block_size)
        arms = {name: (draft, upstream, [False], runner) for name, draft in drafts.items()}
        fork_stats = {}
        digests = {}
        if args.compare_fork_sites:
            fork_stats["fork_sites"] = Counter()
            arms["fork_sites"] = packed, upstream, [False], fork_site_runner(stock.config.block_size, fork_stats["fork_sites"])
        caps = dict.fromkeys(arms, args.cap)
        if args.compare_adaptive_width:
            for name in arms:
                stats = fork_stats[name] = Counter()
                allocator = fork_site_runner(stock.config.block_size, stats, wide_ratio=ratio if name == "adaptive" else None)
                arms[name] = packed, upstream, [False], allocator
            caps = {"fixed8": 8, "fixed16": 16, "adaptive": 16}
        if args.compare_ranked_head:
            from scripts.dflash2_ranked_head import lattice_for, profile

            profile(model, packed, prefilled, len(trace["input"]))
            mapped = lattice_for(model, stock.config.block_size)
            for name, draft in drafts.items():
                fork_stats[name] = Counter()
                digests[name] = sha256()
                allocator = fork_site_runner(stock.config.block_size, fork_stats[name],
                                             route_digest=digests[name], lattice=mapped if name == "mapped_rank" else None)
                arms[name] = draft, upstream, [False], allocator
        warm = streams_at_prefix(model, arms, trace, prefilled, caps, budget, 65)
        ready, _ = exercise(warm, 65, report_progress=False)
        assert all(result["matches_reference"] for result in ready.values())
        for stats in fork_stats.values():
            stats.clear()
        if args.compare_ranked_head:
            # Recreate the digest callbacks after warmup to count only measured rounds.
            for name, (draft, draft_module, checked, _) in arms.items():
                digests[name] = sha256()
                allocator = fork_site_runner(stock.config.block_size, fork_stats[name], route_digest=digests[name],
                                             lattice=mapped if name == "mapped_rank" else None)
                arms[name] = draft, draft_module, checked, allocator
        mx.clear_cache()
        budget.check(force=True)
        mx.reset_peak_memory()
        thermal, start_faults = thermal_state(), pages()
        if args.trim_output:
            digests = {name: sha256() for name in drafts}
            arms = {
                name: (draft, upstream, checked, draft_runner(draft.config.block_size, route_digest=digests[name]))
                for name, (draft, upstream, checked, _) in arms.items()
            }
        streams = streams_at_prefix(model, arms, trace, prefilled, caps, budget, 1024)
        results, elapsed = exercise(streams, 1024, report_progress=True)
        for name, stats in results.items():
            seconds = elapsed[name]
            print("result", json.dumps({
                "arm": name, **stats, "decode_s": seconds, "decode_tok_s": stats["tokens"] / seconds,
                "advance_required_at_100": 100 * seconds / stats["rounds"],
            }), flush=True)
        assert all(result["matches_reference"] for result in results.values())
        if args.compare_fork_sites:
            print("fork_sites", json.dumps(fork_stats["fork_sites"]), flush=True)
        for name, stats in fork_stats.items():
            assert stats["calls"] == results[name]["rounds"]
        if args.compare_adaptive_width:
            print("width_selection", json.dumps(fork_stats), flush=True)
        if args.trim_output or args.compare_ranked_head:
            routes = {name: digest.hexdigest() for name, digest in digests.items()}
            print("routes", json.dumps({"digest": routes, "equal": len(set(routes.values())) == 1}), flush=True)
            if args.compare_ranked_head:
                assert len(set(routes.values())) == 1
        after = pages()
        print("host", json.dumps({
            "peak_gib": mx.get_peak_memory() / GIB, "thermal": [thermal, thermal_state()],
            "decode_faults": {key: val - start_faults[key] for key, val in after.items()},
            "whole_faults": {key: val - before[key] for key, val in after.items()}, **budget.metrics(),
        }), flush=True)


if __name__ == "__main__":
    main()

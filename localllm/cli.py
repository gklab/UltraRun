"""Text deployment, kernel verification, and reproducible performance runs."""
import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter

from jinja2 import Environment, StrictUndefined, nodes
from jinja2.ext import Extension
import mlx.core as mx
from tokenizers import Tokenizer

from .delta import DeltaRollback, causal_conv4, recurrent, reference
from .draft_ranking import draft_rank
from .draft_short_head import REQUEST_RESERVE_BYTES, RequestDraftHead, retained_prompt_rows
from .draft_vocab_prefetch import PREFETCH_RESERVE_BYTES, DraftVocabPrefetch
from .frontier import NODE_LIMIT as FRONTIER_NODE_LIMIT, FrontierCalibration, build_frontier
from .host import thermal_state
from .lossless_gate_up import encode_paired, unpack_joint
from .lossless_projection import decoded_bits, encode_down, unpack_row_major
from .lossless_vocabulary import VocabularyOrder
from .model import (
    COMPILED_DELTA_WIDTHS, DraftVocabulary, LayerState, Model, delta_output, project, split_causal_attention, swiglu,
    unit_norm,
)
from .projection import split_gemv, tiled_argmax
from .packed_projection import (
    pack_activations, pack_weight, packed_feed_forward, packed_partials, packed_project, paired_swiglu_activations,
    residual_norm_pack, supports_paired_swiglu, swiglu_activations, unpack_weight,
)
from .packed_delta import gated_projection, projected_convolution
from .memory import DEFAULT_CACHE_GIB, DEFAULT_MEMORY_GIB, MemoryBudgetError, metal_job
from .phrase_history import PhraseHistory
from .spec_tree import VerificationTree, accepted_path

PROMPT_TOKENS = 4096
OUTPUT_TOKENS = 1024
MAX_CONFIRMED_DRAFT_ROWS = 128
FRONTIER_SCHEDULES = ("frontier", "frontier_prefetch")
DRAFT_SCHEDULES = ("standard", "logical", *FRONTIER_SCHEDULES)


class GenerationBlock(Extension):
    """Render the official template's assistant span as ordinary output."""
    tags = {"generation"}

    def parse(self, parser):
        line = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body).set_lineno(line)


def prompt_tokens(tokenizer, directory, prompt, thinking):
    return chat_tokens(tokenizer, directory, [{"role": "user", "content": prompt}], thinking)


def chat_tokens(tokenizer, directory, messages, thinking):
    def raise_exception(message):
        raise ValueError(message)

    template = Environment(undefined=StrictUndefined, extensions=[GenerationBlock]).from_string(
        (directory / "chat_template.jinja").read_text()
    )
    text = template.render(
        messages=messages, add_generation_prompt=True,
        enable_thinking=thinking, tools=None, raise_exception=raise_exception,
    )
    return tokenizer.encode(text, add_special_tokens=False).ids


def bench_prompt(tokenizer, prompt_file):
    content = (prompt_file or Path(inspect.__file__)).read_text()
    ids = tokenizer.encode(content, add_special_tokens=False).ids
    if len(ids) < PROMPT_TOKENS:
        raise ValueError(f"The benchmark needs {PROMPT_TOKENS} input tokens, received {len(ids)}")
    return ids[:PROMPT_TOKENS]


def draft_shortlist(model, ids, draft_vocab):
    return mx.array(sorted(set(range(min(draft_vocab, model.vocab_size))) | set(ids)), dtype=mx.int32)


def extend_draft_shortlist(model, head, extra_ids, additions):
    ids = extra_ids + additions
    order = sorted(range(len(ids)), key=ids.__getitem__)
    extra_ids = [ids[i] for i in order]
    rows = model.vocabulary_rows(mx.array(additions, dtype=mx.int32))
    if head.extra is not None:
        rows = mx.concatenate((head.extra, rows), axis=0)
    rows = mx.contiguous(rows[mx.array(order, dtype=mx.int32)])
    shortlist = mx.concatenate((mx.arange(model.draft_vocabulary_rows, dtype=mx.int32),
                                 mx.array(extra_ids, dtype=mx.int32)))
    mx.eval(rows, shortlist)
    return shortlist, DraftVocabulary(head.prefix, rows), extra_ids


def generate(model, ids, limit, chunk, stop=(), draft_depth=0, lookup_depth=0, draft_vocab=0,
             budget=None, tree_depth=0, tree_tail_gap=None, draft_confidence=None, device_draft=False,
             draft_extension_confidence=None, draft_root_branch_gap=None, draft_branch_positions=1,
             fused_draft_ranking=False, draft_panel_checkpoint=False, draft_branch_depth=2,
             draft_second_branch=False, draft_multi_fork=False, *, draft_confirmed_vocab=False,
             draft_lane_cache=False, draft_paired_root=False, draft_schedule="standard",
             session=None, on_token=None):
    if draft_schedule not in DRAFT_SCHEDULES:
        raise ValueError("Unknown draft schedule")
    applied_schedule = "standard"
    if (draft_schedule != "standard" and budget is not None
            and (session is not None or (len(ids) == PROMPT_TOKENS and limit == OUTPUT_TOKENS))
            and model.fused_attention_io and model.compact_tree and model.short_tokens >= FRONTIER_NODE_LIMIT
            and model.lossless_mtp_ffn and model.draft_embedding is None
            and 0 < model.draft_vocabulary_rows <= draft_vocab and draft_depth >= 3 and device_draft
            and not (lookup_depth or tree_depth or tree_tail_gap is not None or draft_second_branch
                     or draft_multi_fork or fused_draft_ranking or draft_panel_checkpoint)
            and "MLX_SDPA_BLOCKS" not in os.environ):
        applied_schedule = draft_schedule
    if (applied_schedule == "frontier_prefetch"
            and not (draft_confirmed_vocab and model.fused_head_argmax and model.vocabulary_order is not None)):
        applied_schedule = "standard"
    hit = session.take(ids) if session is not None else None
    cached = len(hit.tokens) if hit is not None else 0
    short_head_rows = retained_prompt_rows(model, ids) if applied_schedule == "frontier_prefetch" else None
    if budget:
        extra_bytes = 0
        if applied_schedule in FRONTIER_SCHEDULES:
            cfg = model.config
            # Up to one independent BF16 MTP K/V pair per non-root node.
            extra_bytes = (FRONTIER_NODE_LIMIT - 1) * 4 * cfg["num_key_value_heads"] * cfg["head_dim"] * (
                len(ids) + limit + 511
            )
        if applied_schedule == "frontier_prefetch":
            extra_bytes += PREFETCH_RESERVE_BYTES
        mx.reset_peak_memory()
        context = len(ids) + limit
        block = min(chunk, max(1, len(ids) - cached))
        if short_head_rows is not None:
            try:
                budget.admit(model, context, block, draft_depth, extra_bytes=extra_bytes + REQUEST_RESERVE_BYTES)
            except MemoryBudgetError:
                short_head_rows = None
        if short_head_rows is None:
            budget.admit(model, context, block, draft_depth, extra_bytes=extra_bytes)
    cache = hit.layers if hit is not None else model.cache()
    mtp_cache = hit.draft if hit is not None else LayerState()
    hidden = hit.hidden if hit is not None else None
    del hit
    tokens = mx.array(ids, dtype=mx.int32)
    start = perf_counter()
    prepared_draft = None
    if draft_depth and draft_vocab and model.draft_embedding is not None:
        shortlist = draft_shortlist(model, ids, draft_vocab)
        prepared_draft = shortlist, model.draft_embedding_table(shortlist, draft_vocab)
    if cached and cached < len(ids) and draft_depth:
        # The saved MTP state ends one pair before the main model's prefix.
        primed = model.draft(hidden, tokens[cached:cached + 1], mtp_cache, cached - 1)
        model.evaluate(primed, [mtp_cache])
    if cached == len(ids) and hidden is not None:
        output = mx.argmax(model.logits(hidden[-1:])[0])
        model.evaluate(output, cache)
    blocks = (model.prefill(tokens[cached:], cache, chunk, offset=cached) if cached
              else model.prefill(tokens, cache, chunk))
    for offset, hidden in blocks:
        end = offset + hidden.shape[0]
        last = end == len(ids)
        pairs = min(end, len(ids) - 1) - offset
        if draft_depth and pairs:
            primed = model.draft(hidden[:pairs], tokens[offset + 1:offset + 1 + pairs], mtp_cache, offset)
            model.evaluate(primed[-1:], [mtp_cache])
        output = mx.argmax(model.logits(hidden[-1:])[0]) if last else hidden[-1:]
        model.evaluate(output, cache)
        if budget:
            budget.check()
    next_id = output.item()
    prefill_peak = None
    if budget:
        prefill_peak = max(mx.get_peak_memory(), mx.get_active_memory())
        mx.reset_peak_memory()
    first = perf_counter()
    if on_token is not None:
        on_token(next_id)
    if session is not None:
        session.remember_prompt(ids, cache, mtp_cache, hidden[-1:])
    if draft_depth or lookup_depth:
        prompt_kv_layers = 0
        if applied_schedule != "standard":
            for layer, state in zip(model.layers, cache):
                state.logical_attention = layer.fused_attention_io
                state.compile_delta_tree = layer.gdn and applied_schedule == "frontier_prefetch"
                if (applied_schedule == "frontier_prefetch" and state.logical_attention
                        and layer.reserve_kv and state.keys is not None and next_id not in stop):
                    state.prompt_kv = state.keys[:, :, :len(ids)], state.values[:, :, :len(ids)]
                    state.keys = state.values = None
                    prompt_kv_layers += 1
        if (applied_schedule == "frontier_prefetch" and mtp_cache.keys is not None
                and model.mtp_layer.reserve_single_kv and next_id not in stop):
            # MTP consumes the next token alongside its preceding target hidden state.
            pairs = len(ids) - 1
            mtp_cache.prompt_kv = mtp_cache.keys[:, :, :pairs], mtp_cache.values[:, :, :pairs]
            mtp_cache.keys = mtp_cache.values = None
        if (draft_lane_cache and draft_depth and device_draft and prepared_draft is None
                and len(ids) >= 4096 and limit >= 1024 and next_id not in stop):
            mtp_cache.draft_lanes = model.prepare_draft_lanes(ids, budget)
        short_head = (RequestDraftHead(model, short_head_rows)
                      if short_head_rows is not None and next_id not in stop else None)
        hooks = {"session": session, "on_token": on_token} if session is not None or on_token is not None else {}
        generated, extra = speculative(
            model, cache, mtp_cache, ids, hidden[-1:], next_id, limit, stop,
            draft_depth, lookup_depth, draft_vocab, budget, tree_depth, tree_tail_gap, draft_confidence,
            device_draft, draft_extension_confidence, draft_root_branch_gap, draft_branch_positions,
            fused_draft_ranking, draft_panel_checkpoint, draft_branch_depth, draft_second_branch,
            prepared_draft, draft_multi_fork, draft_confirmed_vocab=draft_confirmed_vocab,
            draft_paired_root=draft_paired_root, draft_schedule=applied_schedule, short_head=short_head, **hooks,
        )
        if mtp_cache.draft_lanes is not None:
            extra["draft_lane_cache_rows"] = mtp_cache.draft_lanes.count
        if prompt_kv_layers:
            extra["target_prompt_kv_layers"] = prompt_kv_layers
        if mtp_cache.prompt_kv is not None:
            extra["draft_prompt_kv_tokens"] = mtp_cache.prompt_kv[0].shape[2]
    else:
        extra = {"target_decode_passes": 0}
        generated = [next_id]
        for index in range(limit - 1):
            if next_id in stop:
                break
            if session is None:
                output = model.forward(output.reshape((1,)), cache, len(ids) + index)
            else:
                hidden = model.hidden(output.reshape((1,)), cache, len(ids) + index)
                output = model.logits(hidden[-1:])[0]
            output = mx.argmax(output)
            model.evaluate(output, cache)
            if budget:
                budget.check()
            next_id = output.item()
            generated.append(next_id)
            if on_token is not None:
                on_token(next_id)
            extra["target_decode_passes"] += 1
            count = index + 2
            if not stop and (count % 128 == 0 or count == limit):
                print(f"decode {count}/{limit}", file=sys.stderr, flush=True)
        if session is not None and len(generated) > 1:
            session.remember_output(ids, generated, cache, mtp_cache, hidden[-1:])
    finished = perf_counter()
    decode_peak = max(mx.get_peak_memory(), mx.get_active_memory())
    phase_peaks = {
        "prefill_peak_gib": prefill_peak / 2**30,
        "decode_peak_gib": decode_peak / 2**30,
    } if prefill_peak is not None else {}
    return generated, {
        "input_tokens": len(ids), "output_tokens": len(generated),
        **({"cached_prompt_tokens": cached, "prefill_tokens": len(ids) - cached} if session is not None else {}),
        "draft_schedule": applied_schedule,
        "ttft_s": first - start,
        "prefill_tok_s": len(ids) / (first - start),
        "decode_tok_s": (len(generated) - 1) / (finished - first),
        "end_to_end_s": finished - start,
        "end_to_end_tok_s": len(generated) / (finished - start),
        "peak_gib": max(prefill_peak or 0, decode_peak) / 2**30,
        "fused_head_argmax": model.fused_head_argmax,
        "packed_head_only": model.head is None,
        "lossless_down_matrices": model.lossless_down_count,
        "paired_down_matrices": model.paired_down_count,
        "lossless_gate_up_matrices": model.lossless_gate_up_count,
        "lossless_input_matrices": model.lossless_input_count,
        "lossless_attention_output_matrices": model.lossless_attention_output_count,
        "fused_attention_io": model.fused_attention_io,
        "lossless_vocabulary": model.lossless_vocabulary,
        "lossless_draft_vocab_rows": model.draft_vocabulary_rows,
        "lossless_mtp_ffn": model.lossless_mtp_ffn,
        "shared_vocabulary": model.vocabulary_order is not None,
        "compiled_delta": model.compiled_delta,
        "direct_delta_sequence": model.direct_delta_sequence,
        "layer_prefill": model.layer_prefill,
        **phase_peaks,
        **(budget.metrics() if budget else {}),
        **extra,
    }


def lookup(history, maximum):
    """Reuse a continuation from a prior occurrence of the current suffix."""
    for length in (12, 8, 6, 4):
        suffix_start = len(history) - length
        suffix = history[suffix_start:]
        for start in range(suffix_start - length - 1, -1, -1):
            if history[start:start + length] == suffix:
                end = start + length
                return history[end:min(end + maximum, suffix_start)]
    return []


@mx.compile
def greedy_log_probability(scores):
    logits = scores.astype(mx.float32)
    centered = logits - logits.max(axis=-1, keepdims=True)
    return -mx.logsumexp(centered, axis=-1)


def ranked_alternates(choices, max_gap, limit):
    prefix = 0.0
    ranked = []
    for position, choice in enumerate(choices):
        token, gap, log_probability = choice[:3]
        prefix += log_probability
        alternate_log_probability = prefix - gap
        if gap <= max_gap and alternate_log_probability > -math.inf:
            ranked.append((alternate_log_probability, position, int(token)))
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    return [(position, token) for _, position, token in ranked[:limit]]


def draft_on_device(model, hidden, cache, offset, head, shortlist, count, confidence_floor,
                    extension_floor, root_branch_gap, branch_positions, fused_ranking, panel_checkpoint,
                    second_branch=False, embedding_table=None, multi_fork=False, *, root_preview=None):
    fork = replace(cache)
    candidates, probabilities = [], []
    choices, snapshots = [], []
    proposed = []
    prefix = 0.0
    branches = []
    paired_scores = None
    root_side = None
    for i in range(count):
        scores = paired_scores if i == 1 and paired_scores is not None else model.draft_scores(hidden, head)
        scored = i + 1
        examine = root_branch_gap is not None and i < branch_positions and scores.shape[-1] > 1
        needs_probability = scored < count and confidence_floor is not None
        use_probability = needs_probability or (examine and branch_positions > 1)
        if fused_ranking and use_probability:
            proposal, probability = draft_rank(scores)
        else:
            proposal = mx.argmax(scores, axis=-1)
            probability = greedy_log_probability(scores) if use_probability else None
        if examine:
            vocabulary = mx.arange(scores.shape[-1])
            remaining_scores = mx.where(vocabulary == proposal[0], -mx.inf, scores)
            second = mx.argmax(remaining_scores, axis=-1)
            gap = scores[0, proposal[0]].astype(mx.float32) - scores[0, second[0]].astype(mx.float32)
            token = shortlist[second] if shortlist is not None else second
            if i == 0 and root_preview is not None:
                root_side = token
            if probability is None:
                probability = mx.zeros((1,), dtype=mx.float32)
            choice = [token.astype(mx.float32), gap.reshape(1), probability]
            if second_branch and scores.shape[-1] > 2:
                third = mx.argmax(mx.where(vocabulary == second[0], -mx.inf, remaining_scores), axis=-1)
                third_gap = scores[0, proposal[0]].astype(mx.float32) - scores[0, third[0]].astype(mx.float32)
                third_token = shortlist[third] if shortlist is not None else third
                choice.append(mx.where(third_gap <= root_branch_gap, third_token.astype(mx.int32), -1).astype(mx.float32))
            choices.append(mx.concatenate(choice))
            snapshots.append((hidden, replace(fork)))
        cache_slot = proposal
        if shortlist is not None:
            proposal = shortlist[proposal]
        candidates.append(proposal)
        if needs_probability:
            probabilities.append(probability)
        check_prefix = scored % 5 == 0 or (panel_checkpoint and scored == 7)
        checkpoint = count > 7 and scored < count and check_prefix and bool(probabilities)
        if scored == count or checkpoint:
            tokens = mx.concatenate(candidates)
            choice = (mx.stack(choices),) if choices else ()
            confidence = (mx.concatenate(probabilities),) if probabilities else ()
            preview = root_preview if root_preview is not None else ()
            mx.eval(tokens, *confidence, *choice, *preview, cache.keys, cache.values)
            accepted_length = len(candidates)
            stopped = False
            if confidence:
                for j, log_probability in enumerate(confidence[0].tolist(), start=1):
                    prefix += log_probability
                    if len(proposed) + j >= 2 and prefix < confidence_floor:
                        accepted_length = j
                        stopped = True
                        break
            proposed.extend(tokens.tolist()[:accepted_length])
            if choice:
                choices_host = choice[0].tolist()[:len(proposed)]
                for position, alternate in ranked_alternates(choices_host, root_branch_gap, 2 if multi_fork else 1):
                    row = choices_host[position]
                    third = int(row[3]) if len(row) == 4 else -1
                    branches.append((position, alternate, *snapshots[position], third))
                choices.clear()
                snapshots.clear()
            if stopped or (checkpoint and extension_floor is not None and prefix < extension_floor):
                return proposed, True, scored, branches
            candidates.clear()
            probabilities.clear()
        if scored < count:
            cached = (embedding_table, cache_slot) if embedding_table is not None else None
            if i == 0 and root_side is not None:
                hidden, scores = model.draft_root_pair(hidden, mx.concatenate((proposal, root_side)), fork, offset, head)
                paired_scores = scores[:1]
                root_preview.append(shortlist[mx.argmax(scores[1:], axis=-1)])
            else:
                hidden = model.draft(hidden, proposal, fork, offset + i, cached_embedding=cached)
    return proposed, False, count, branches


def speculative(model, cache, mtp_cache, ids, hidden, pending, limit, stop, depth, lookup_depth,
                draft_vocab, budget, tree_depth, tree_tail_gap, draft_confidence, device_draft,
                draft_extension_confidence, draft_root_branch_gap, draft_branch_positions, fused_draft_ranking,
                draft_panel_checkpoint, draft_branch_depth, draft_second_branch, prepared_draft, draft_multi_fork,
                *, draft_confirmed_vocab=False, draft_paired_root=False, draft_schedule="standard", short_head=None,
                session=None, on_token=None):
    generated = [pending]
    offset = len(ids)
    pair_ids = [pending]
    mtp_offset = offset - 1
    phrases = PhraseHistory(ids) if draft_schedule == "frontier_prefetch" else None
    calibration = FrontierCalibration() if draft_schedule == "frontier_prefetch" else None
    stats = {
        "target_decode_passes": 0, "drafted": 0, "accepted": 0,
        "lookup_drafted": 0, "lookup_accepted": 0,
        "draft_s": 0.0, "target_s": 0.0, "accept_s": 0.0,
        "tree_rounds": 0, "max_tree_nodes": 0, "branch_selected": 0, "branch_accepted": 0,
        "second_branch_rounds": 0, "second_branch_selected": 0,
        "multi_fork_rounds": 0,
        "confidence_stops": 0,
        "draft_scores_evaluated": 0,
    }
    branch_depth = min(depth, draft_branch_positions) - 1 + draft_branch_depth if draft_root_branch_gap is not None else 0
    acceptance = [0] * (max(depth, tree_depth, lookup_depth, branch_depth) + 1)
    branch_attempts = [0] * draft_branch_positions
    branch_hits = [0] * draft_branch_positions
    branch_accepted_lengths = [0] * (draft_branch_depth + 1)
    device_branches = 1 + draft_multi_fork
    extra_candidates = (device_branches * draft_branch_depth + draft_second_branch
                        if draft_root_branch_gap is not None else tree_depth)
    draft_lengths = [0] * (max(depth + extra_candidates, lookup_depth) + 1)
    use_frontier = draft_schedule in FRONTIER_SCHEDULES
    if use_frontier:
        acceptance = [0] * (model.short_tokens + 1)
        draft_lengths = [0] * (model.short_tokens + 1)
        branch_attempts = [0] * model.short_tokens
        branch_hits = [0] * model.short_tokens
        branch_accepted_lengths = [0] * (model.short_tokens + 1)
        stats["draft_frontier_rounds"] = 0
    confidence_floor = math.log(draft_confidence) if draft_confidence is not None else None
    extension_floor = math.log(draft_extension_confidence) if draft_extension_confidence is not None else None
    branch_budget = model.short_tokens - 1 if model.compact_tree else 7
    draft_head = model.head
    shortlist = None
    embedding_table = None
    if depth and draft_vocab:
        if prepared_draft is None:
            shortlist = draft_shortlist(model, ids, draft_vocab)
        else:
            shortlist, embedding_table = prepared_draft
        draft_head = model.prepare_draft_vocabulary(shortlist, draft_vocab)
    remaining_draft_rows = 0
    vocab_prefetch = None
    if draft_confirmed_vocab and isinstance(draft_head, DraftVocabulary) and embedding_table is None:
        base_rows = model.draft_vocabulary_rows
        extra_ids = shortlist[base_rows:].tolist()
        known_extras = set(extra_ids)
        remaining_draft_rows = MAX_CONFIRMED_DRAFT_ROWS
        stats.update(draft_vocab_updates=0, draft_vocab_added=0)
        if draft_schedule == "frontier_prefetch":
            vocab_prefetch = DraftVocabPrefetch(model, ids, extra_ids, remaining_draft_rows, stats)
            stats["target_kv_deferred_rounds"] = 0
    pair_root = (draft_paired_root and mtp_cache.draft_lanes is not None and model.lossless_mtp_ffn
                 and draft_root_branch_gap is not None and draft_branch_positions == 1
                 and draft_branch_depth == 2 and not (draft_multi_fork or draft_second_branch)
                 and isinstance(draft_head, DraftVocabulary) and shortlist is not None and embedding_table is None)
    if pair_root:
        stats.update(draft_paired_root_batches=0, draft_paired_root_reused=0)
    next_report = 128
    while len(generated) < limit and generated[-1] not in stop:
        step_start = perf_counter()
        if short_head is not None:
            short_head.retain(pair_ids)
        if phrases is not None:
            phrases.extend(pair_ids)
        if vocab_prefetch is not None:
            shortlist, draft_head = vocab_prefetch.update(shortlist, draft_head, pair_ids, budget)
        elif remaining_draft_rows:
            additions = list(dict.fromkeys(token for token in pair_ids
                                           if token >= base_rows and token not in known_extras))[:remaining_draft_rows]
            if additions:
                # Row extraction, join and the stable token-ID reorder can overlap in memory.
                new_rows = len(extra_ids) + len(additions)
                row_bytes = 2 * model.config["hidden_size"]
                scratch = (2 * new_rows + len(additions)) * row_bytes + 4 * (base_rows + 3 * new_rows)
                if budget is not None and mx.get_active_memory() + budget.cache_limit + scratch > budget.limit:
                    remaining_draft_rows = 0
                else:
                    shortlist, draft_head, extra_ids = extend_draft_shortlist(model, draft_head, extra_ids, additions)
                    known_extras.update(additions)
                    remaining_draft_rows -= len(additions)
                    stats["draft_vocab_updates"] += 1
                    stats["draft_vocab_added"] += len(additions)
        remaining = limit - len(generated) - 1
        proposals = lookup(ids + generated, min(lookup_depth, remaining)) if lookup_depth and remaining else []
        from_lookup = bool(proposals)
        count = min(depth, remaining)
        alternatives = []
        alternate_parents = []
        third_node = None
        branch_count = min(tree_depth, remaining) if not from_lookup else 0
        branch_parent = 0
        log_confidence = 0.0
        if session is not None and depth and not count:
            # The final verification may skip drafting. Advance the real MTP
            # cache so the next turn can continue from the committed prefix.
            primed = model.draft(hidden, mx.array(pair_ids, dtype=mx.int32), mtp_cache, mtp_offset)
            model.evaluate(primed, [mtp_cache])
            mtp_offset += len(pair_ids)
        if count and device_draft:
            root = model.draft(hidden, mx.array(pair_ids, dtype=mx.int32), mtp_cache, mtp_offset)[-1:]
            mtp_offset += len(pair_ids)
            if use_frontier and count >= 3:
                proposals, alternatives, alternate_parents, scored = build_frontier(
                    model, root, mtp_cache, mtp_offset, pending, remaining, draft_head, shortlist,
                    phrases=phrases, calibration=calibration, short_head=short_head,
                )
                stats["draft_scores_evaluated"] += scored
                stats["draft_frontier_rounds"] += 1
                for parent in alternate_parents:
                    if parent <= len(proposals):
                        branch_attempts[parent] += 1
            else:
                preview = [] if pair_root and count >= 2 else None
                proposals, stopped, scored, branches = draft_on_device(
                    model, root, mtp_cache, mtp_offset, draft_head, shortlist, count, confidence_floor, extension_floor,
                    draft_root_branch_gap, draft_branch_positions, fused_draft_ranking, draft_panel_checkpoint,
                    draft_second_branch, embedding_table, draft_multi_fork, root_preview=preview,
                )
                if preview:
                    stats["draft_paired_root_batches"] += 1
                stats["confidence_stops"] += stopped
                stats["draft_scores_evaluated"] += scored
                for index, (parent, alternate, branch_hidden, fork, third) in enumerate(branches):
                    available = branch_budget - len(proposals) - len(alternatives)
                    if available <= 0:
                        break
                    # Reserve one node for a qualifying second fork.
                    available -= available > 1 and index + 1 < len(branches)
                    alternatives.append(alternate)
                    alternate_parents.append(parent)
                    use_third = third >= 0 and available > 1
                    slots = min(draft_branch_depth, available - use_third, remaining - parent)
                    if slots == 2 and parent == 0 and preview:
                        alternate_parents.append(len(proposals) + len(alternatives))
                        alternatives.append(preview[0].item())
                        stats["draft_scores_evaluated"] += 1
                        stats["draft_paired_root_reused"] += 1
                    elif slots > 1:
                        next_hidden = model.draft(branch_hidden, mx.array([alternate], dtype=mx.int32),
                                                 fork, mtp_offset + parent)
                        suffix, _, scored, _ = draft_on_device(
                            model, next_hidden, fork, mtp_offset + parent + 1, draft_head, shortlist,
                            slots - 1, confidence_floor, None, None, 1, fused_draft_ranking, False,
                            embedding_table=embedding_table,
                        )
                        first = len(proposals) + len(alternatives)
                        alternate_parents.extend(range(first, first + len(suffix)))
                        alternatives.extend(suffix)
                        stats["draft_scores_evaluated"] += scored
                    if use_third:
                        alternatives.append(third)
                        alternate_parents.append(parent)
                        third_node = len(proposals) + len(alternatives)
                        stats["second_branch_rounds"] += 1
                    branch_attempts[parent] += 1
        elif count:
            draft_hidden = model.draft(hidden, mx.array(pair_ids, dtype=mx.int32), mtp_cache, mtp_offset)[-1:]
            mtp_offset += len(pair_ids)
            fork = replace(mtp_cache)
            root_hidden = draft_hidden
            for i in range(0 if from_lookup else count):
                scores = model.draft_scores(draft_hidden, draft_head)
                stats["draft_scores_evaluated"] += 1
                proposal = mx.argmax(scores, axis=-1)
                tail_branch = tree_tail_gap is not None and i == count - 2
                if (i == 0 and branch_count) or tail_branch:
                    alternative = mx.argmax(mx.where(mx.arange(scores.shape[-1]) == proposal[0],
                                                     -mx.inf, scores), axis=-1)
                if tail_branch:
                    gap = (scores[0, proposal[0]].astype(mx.float32) -
                           scores[0, alternative[0]].astype(mx.float32))
                    if gap.item() <= tree_tail_gap:
                        alternative_token = shortlist[alternative] if shortlist is not None else alternative
                        alternatives.append(alternative_token.item())
                        branch_parent = i
                if shortlist is not None:
                    proposal = shortlist[proposal]
                if confidence_floor is not None and i + 1 < count:
                    probability = greedy_log_probability(scores)
                    mx.eval(proposal, probability)
                    log_confidence += probability.item()
                proposals.append(proposal.item())
                if alternatives:
                    break
                if (confidence_floor is not None and 2 <= i + 1 < count
                        and log_confidence < confidence_floor):
                    stats["confidence_stops"] += 1
                    break
                if i + 1 < count:
                    draft_hidden = model.draft(draft_hidden, proposal, fork, mtp_offset + i)
            model.evaluate(draft_hidden, [mtp_cache])
            if branch_count:
                branch = replace(mtp_cache)
                branch_hidden = root_hidden
                for i in range(branch_count):
                    token = shortlist[alternative] if shortlist is not None else alternative
                    alternatives.append(token.item())
                    if i + 1 < branch_count:
                        branch_hidden = model.draft(branch_hidden, token, branch, mtp_offset + i)
                        alternative = mx.argmax(model.draft_scores(branch_hidden, draft_head), axis=-1)
                        stats["draft_scores_evaluated"] += 1
        target_start = perf_counter()
        tokens = [pending, *proposals, *alternatives]
        batch = mx.array(tokens, dtype=mx.int32)
        tree = None
        if alternatives:
            if device_draft:
                parents = [-1, *range(len(proposals)), *alternate_parents]
            else:
                parents = [-1, *range(len(proposals)), branch_parent,
                           *range(len(proposals) + 1, len(tokens) - 1)]
            spine = len(proposals) + 1 if device_draft else None
            tree = VerificationTree(parents, offset, model.config["linear_conv_kernel_dim"] - 1, spine=spine)
        verified = model.hidden(batch, cache, offset, keep_replay=bool(proposals), tree=tree)
        if vocab_prefetch is None:
            predictions = model.predictions(verified)
        else:
            predictions = model.predictions(verified, observe_vocabulary_tiles=vocab_prefetch.capture)
        hints = vocab_prefetch.hints if vocab_prefetch is not None else None
        model.evaluate([predictions, *hints] if hints is not None else predictions, cache)
        accept_start = perf_counter()
        if budget:
            budget.check()
        expected = predictions.tolist()
        if tree is not None:
            path = accepted_path(parents, tokens, expected)
            if session is not None:
                # Do not commit the stop token or later nodes. The template
                # will supply the separator when the next user message arrives.
                end = next((i for i, node in enumerate(path[1:], 1) if tokens[node] in stop), len(path))
                path = path[:end]
            keep = len(path)
            pending = expected[path[-1]]
            pair_ids = [tokens[i] for i in path[1:]] + [pending]
            if (vocab_prefetch is not None and len(generated) + keep < limit
                    and not any(token in stop for token in pair_ids)):
                stats["target_kv_deferred_rounds"] += model.accept_tree(cache, tree, offset, path, defer_kv=True)
            else:
                model.accept_tree(cache, tree, offset, path)
            hidden = verified[mx.array(path, dtype=mx.int32)]
            stats["tree_rounds"] += 1
            stats["multi_fork_rounds"] += tree.multi_fork
            stats["max_tree_nodes"] = max(stats["max_tree_nodes"], len(tokens))
            if path[-1] > len(proposals):
                stats["branch_selected"] += 1
                if device_draft:
                    branch_parent = parents[next(node for node in path if node > len(proposals))]
                stats["branch_accepted"] += keep - 1 - (branch_parent if device_draft else 0)
                if device_draft:
                    branch_hits[branch_parent] += 1
                    branch_accepted_lengths[keep - 1 - branch_parent] += 1
                if path[-1] == third_node:
                    stats["second_branch_selected"] += 1
        else:
            keep = 1
            for proposed, actual in zip(proposals, expected):
                if proposed != actual or (session is not None and proposed in stop):
                    break
                keep += 1
            if proposals:
                model.accept(cache, offset, len(proposals) + 1, keep)
            pending = expected[keep - 1]
            pair_ids = proposals[:keep - 1] + [pending]
            hidden = verified[:keep]
        if vocab_prefetch is not None:
            vocab_prefetch.observe(path if tree is not None else range(keep))
        if calibration is not None:
            calibration.observe(path if tree is not None else range(keep), expected)
        offset += keep
        matched = keep - 1
        stats["target_decode_passes"] += 1
        stats["drafted"] += len(proposals) + len(alternatives)
        draft_lengths[len(proposals) + len(alternatives)] += 1
        stats["accepted"] += matched
        acceptance[matched] += 1
        stats["draft_s"] += target_start - step_start
        stats["target_s"] += accept_start - target_start
        stats["accept_s"] += perf_counter() - accept_start
        if from_lookup:
            stats["lookup_drafted"] += len(proposals)
            stats["lookup_accepted"] += matched
        for token in pair_ids:
            generated.append(token)
            if on_token is not None:
                on_token(token)
            if token in stop:
                break
        if not stop and (len(generated) >= next_report or len(generated) == limit):
            print(f"decode {len(generated)}/{limit}", file=sys.stderr, flush=True)
            next_report += 128
    if (session is not None and len(generated) > 1 and offset == len(ids) + len(generated) - 1
            and not any(state.pending_kv is not None for state in cache)):
        if depth:
            pairs = offset - 1 - mtp_offset
            assert 0 <= pairs < len(pair_ids)
            if pairs:
                primed = model.draft(hidden[:pairs], mx.array(pair_ids[:pairs], dtype=mx.int32), mtp_cache, mtp_offset)
                model.evaluate(primed, [mtp_cache])
            if budget:
                budget.check()
        session.remember_output(ids, generated, cache, mtp_cache, hidden[-1:])
    stats["accepted_per_pass_histogram"] = acceptance
    stats["drafts_per_pass_histogram"] = draft_lengths
    if phrases is not None:
        stats["draft_phrase_nodes"] = phrases.added_nodes
        stats["draft_phrase_rounds"] = phrases.rounds
        stats["draft_phrase_anchor_rounds"] = phrases.anchor_rounds
        stats["draft_phrase_early_anchor_rounds"] = phrases.early_anchor_rounds
    if calibration is not None:
        stats["draft_path_calibration_samples"] = calibration.samples
        stats["draft_path_counterfactual_samples"] = calibration.counterfactual_samples
        stats["draft_path_reorder_rounds"] = calibration.reorder_rounds
        stats["draft_path_reserve_nodes"] = calibration.reserve_nodes
        stats["draft_path_fourth_rank_nodes"] = calibration.fourth_rank_nodes
        stats["draft_width_kept_narrow"] = calibration.kept_narrow
        stats["draft_width_early_narrow_rounds"] = calibration.early_narrow
        stats["draft_width_skipped_score_rows"] = calibration.skipped_rows
    if short_head is not None:
        stats.update(short_head.metrics())
    if use_frontier or draft_root_branch_gap is not None:
        stats["branch_attempts_by_position"] = branch_attempts
        stats["branch_hits_by_position"] = branch_hits
        stats["branch_accepted_lengths"] = branch_accepted_lengths
    return generated, stats


def verify_kernel():
    mx.random.seed(378)
    q = mx.random.normal((9, 2, 128)) / 128
    k = mx.random.normal((9, 2, 128)) / (128 ** 0.5)
    v = mx.random.normal((9, 6, 128)).astype(mx.bfloat16)
    decay = mx.random.uniform(0.5, 1.0, (9, 6))
    beta = mx.random.uniform(0.0, 1.0, (9, 6))
    initial = mx.random.normal((6, 128, 128)) * 0.1
    expected, expected_state = reference(q, k, v, decay, beta, initial)
    for rows in (1, 2, 4, 8):
        actual, state = recurrent(q, k, v, decay, beta, initial, rows)
        assert mx.allclose(actual, expected, atol=0.001, rtol=0.008).item()
        assert mx.allclose(state, expected_state, atol=1e-6, rtol=1e-5).item()
        prefill, halfway = recurrent(q[:8], k[:8], v[:8], decay[:8], beta[:8], initial, rows)
        step, final = recurrent(q[8:], k[8:], v[8:], decay[8:], beta[8:], halfway, rows)
        assert mx.array_equal(mx.concatenate((prefill, step)), actual).item()
        assert mx.array_equal(final, state).item()
        traced, trajectory = recurrent(q, k, v, decay, beta, initial, rows, trace=True)
        assert mx.array_equal(traced, actual).item()
        assert mx.array_equal(trajectory[-2], halfway).item()
        assert mx.array_equal(trajectory[-1], final).item()
    print("DeltaNet FP32 state and BF16 output match the reference recurrence.")
    raw_q, raw_k = q.astype(mx.bfloat16), k.astype(mx.bfloat16)
    for keys, queries, normalize in ((k, q, False), (raw_k, raw_q, True)):
        inputs = (queries[:8], keys[:8], v[:8], decay[:8], beta[:8], initial, 4)
        traced, trajectory = recurrent(*inputs, trace=True, normalize=normalize)
        recorded, last, updates, normalized_keys = recurrent(*inputs, normalize=normalize, record_updates=True)
        rollback = DeltaRollback(initial, updates, normalized_keys, decay[:8])
        assert mx.array_equal(recorded.view(mx.uint16), traced.view(mx.uint16)).item()
        assert mx.array_equal(last.view(mx.uint32), trajectory[-1].view(mx.uint32)).item()
        for keep in range(1, 9):
            restored = rollback.restore(keep)
            assert mx.array_equal(restored.view(mx.uint32), trajectory[keep - 1].view(mx.uint32)).item()
    print("DeltaNet rank-one rollback restores every accepted prefix bit-for-bit.")
    standard, state = recurrent(unit_norm(raw_q) * (128 ** -0.5), unit_norm(raw_k),
                                v, decay, beta, initial, 4, trace=True)
    fused, trace = recurrent(raw_q, raw_k, v, decay, beta, initial, 4, trace=True, normalize=True)
    assert mx.allclose(fused, standard, atol=0.001, rtol=0.008).item()
    assert mx.allclose(trace, state, atol=2e-6, rtol=2e-5).item()
    parents = [-1, 0, 1, 0, 3]
    tree = VerificationTree(parents, 2, 3)
    output, states = recurrent(raw_q[:5], raw_k[:5], v[:5], decay[:5], beta[:5],
                               initial, 4, normalize=True, parents=tree.parents)
    for node, path in enumerate(tree.paths):
        rows = mx.array(path, dtype=mx.int32)
        linear, final = recurrent(raw_q[rows], raw_k[rows], v[rows], decay[rows], beta[rows],
                                  initial, 4, normalize=True)
        assert mx.array_equal(output[node], linear[-1]).item()
        assert mx.array_equal(states[node], final).item()
    assert tree.attention[4].tolist() == [True, True, True, False, False, True, True]
    assert accepted_path(parents, [10, 11, 12, 13, 14], [13, 0, 0, 14, 0]) == [0, 3, 4]
    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
    gathers = mx.array(nodes, dtype=mx.int32)
    inputs = (raw_q[gathers], raw_k[gathers], v[gathers], decay[gathers], beta[gathers], initial, 4)
    for branches in ([0, 12, 13, 14], [5, 12, 13, 14], [11, 12, 13, 14],
                     [0, 12, 0, 14], [5, 12, 5, 14], [0, 12, 5, 14],
                     [11, 12, 0, 14], [9, 12, 2, 14],
                     [0, 12, 12, 14], [5, 12, 13, 12], [0, 12, 12, 0]):
        compact_tree = VerificationTree([-1, *range(11), *branches], 2, 3, spine=12)
        traced, trajectory = recurrent(*inputs, trace=True, normalize=True, parents=compact_tree.parents)
        output, last, updates, keys = recurrent(*inputs, normalize=True, parents=compact_tree.parents,
                                                record_updates=True, spine=compact_tree.spine,
                                                multi_fork=compact_tree.multi_fork)
        assert mx.array_equal(traced.view(mx.uint16), output.view(mx.uint16)).item()
        assert mx.array_equal(trajectory[11].view(mx.uint32), last.view(mx.uint32)).item()
        rollback = DeltaRollback(initial, updates, keys, decay[gathers])
        for index, path in enumerate(compact_tree.paths):
            restored = rollback.restore(len(path), mx.array(path, dtype=mx.int32))
            assert mx.array_equal(trajectory[index].view(mx.uint32), restored.view(mx.uint32)).item()
    print("Compact recurrence and rollback match every state with one or two fork points, including nested forks.")
    from mlx import nn
    sequence = mx.random.normal((7, 160)).astype(mx.bfloat16)
    taps = mx.random.normal((160, 4, 1)).astype(mx.bfloat16)
    standard_conv = nn.silu(mx.conv1d(sequence[None], taps, groups=160)[0]).astype(mx.float32)
    fused_conv = causal_conv4(sequence, taps).astype(mx.float32)
    assert mx.allclose(fused_conv, standard_conv, atol=0.002, rtol=0.01).item()
    tree_sequence = mx.concatenate((sequence, sequence[:1]), axis=0)
    indexed = causal_conv4(tree_sequence, taps, tree.windows)
    for node in range(len(tree.paths)):
        linear = causal_conv4(tree_sequence[tree.windows[node]], taps)
        assert mx.array_equal(indexed[node], linear[0]).item()
    print("Short-block DeltaNet convolution and normalized recurrence match their references.")
    weight = (mx.random.normal((19, 1536)) / 32).astype(mx.bfloat16)
    for size in (2, 3, 4, 5):
        x = mx.random.normal((size, 1536)).astype(mx.bfloat16)
        vector = split_gemv(x, weight).astype(mx.float32)
        expected = (x @ weight.T).astype(mx.float32)
        assert mx.allclose(vector, expected, atol=0.002, rtol=0.008).item()
    print("Vectorized BF16 projection matches MLX matmul.")
    scores = mx.array([[0, 1], [3, 0], [3, 5], [2, 5], [1, 2]], dtype=mx.bfloat16)
    assert tiled_argmax(mx.eye(2, dtype=mx.bfloat16), scores, 2).tolist() == [1, 2]
    print("Tiled full-vocabulary argmax preserves first-index ties.")
    original = weight[:16]
    packed = pack_weight(original)
    assert mx.array_equal(unpack_weight(packed).view(mx.uint16), original.view(mx.uint16)).item()
    for size in (1, 3, 5, 7, 8):
        x = mx.random.normal((size, 1536)).astype(mx.bfloat16)
        actual = packed_project(x, packed).astype(mx.float32)
        expected = (x @ original.T).astype(mx.float32)
        assert mx.allclose(actual, expected, atol=0.002, rtol=0.008).item()
    print("Packed SIMD matrix projection preserves BF16 weight bits and matches MLX within tolerance.")
    coded = encode_down(packed)
    assert mx.array_equal(decoded_bits(coded.payload), packed).item()
    assert mx.array_equal(unpack_row_major(coded.payload).view(mx.uint16), original.view(mx.uint16)).item()
    for width in (6, 11, 16):
        x_packed = pack_activations(mx.random.normal((width, original.shape[1])).astype(mx.bfloat16))
        expected = packed_partials(x_packed, width, packed)
        actual = coded.partials(x_packed, width)
        assert mx.array_equal(expected.view(mx.uint32), actual.view(mx.uint32)).item(), width
    for raw in (mx.full((1, 4, 64), 0x3f80, dtype=mx.uint16),
                (mx.arange(256, dtype=mx.uint16) * 127).reshape(1, 4, 64)):
        assert mx.array_equal(decoded_bits(encode_down(raw).payload), raw).item()
    print("Lossless down projections restore BF16 bits and preserve FP32 partials.")
    norm_weight = mx.random.uniform(0.5, 1.5, (5120,))
    eps = 1e-6
    for size in (1, 5, 8, 9, 16):
        residual_input, attention = [mx.random.normal((size, 5120)).astype(mx.bfloat16) for _ in range(2)]
        residual, normalized = residual_norm_pack(residual_input, attention, norm_weight, mx.array([eps]))
        expected_residual = residual_input + attention
        expected_normalized = mx.fast.rms_norm(expected_residual, norm_weight, eps).astype(mx.bfloat16)
        assert mx.array_equal(residual, expected_residual).item()
        assert mx.allclose(normalized.astype(mx.float32), pack_activations(expected_normalized).astype(mx.float32),
                           atol=0.001, rtol=0.008).item()
    print("Fused residual normalization preserves the BF16 residual and packs short blocks within tolerance.")
    gate_up = pack_weight((mx.random.normal((80, 1536)) / 64).astype(mx.bfloat16))
    down = pack_weight((mx.random.normal((24, 40)) / 8).astype(mx.bfloat16))
    x = mx.random.normal((5, 1536)).astype(mx.bfloat16)
    gate, up = mx.split(packed_project(x, gate_up), 2, axis=-1)
    actual = packed_feed_forward(x, gate_up, down).astype(mx.float32)
    expected = packed_project(swiglu(gate, up), down).astype(mx.float32)
    assert mx.allclose(actual, expected, atol=0.002, rtol=0.012).item()
    print("Fused FFN accumulates and packs SwiGLU for the second projection within BF16 tolerance.")
    if supports_paired_swiglu():
        paired_gate_up = pack_weight(mx.random.normal((64, 5120)).astype(mx.bfloat16))
        for tokens in range(9, 17):
            x_packed = pack_activations(mx.random.normal((tokens, 5120)).astype(mx.bfloat16))
            regular = swiglu_activations(x_packed, tokens, paired_gate_up)
            paired = paired_swiglu_activations(x_packed, tokens, paired_gate_up)
            assert mx.array_equal(regular.view(mx.uint16), paired.view(mx.uint16)).item(), tokens
        print("Paired gate/up MMA preserves SwiGLU operand bits at widths 9–16.")
        joint = encode_paired(paired_gate_up)
        assert mx.array_equal(unpack_joint(joint.payload), paired_gate_up).item()
        assert mx.array_equal(unpack_joint(joint.payload, row_major=True).view(mx.uint16),
                              unpack_weight(paired_gate_up).view(mx.uint16)).item()
        for tokens in (6, 11, 16):
            x_packed = pack_activations(mx.random.normal((tokens, 5120)).astype(mx.bfloat16))
            standard = swiglu_activations(x_packed, tokens, paired_gate_up)
            output = joint.project(x_packed, tokens, swiglu=True)
            assert mx.array_equal(output.view(mx.uint16), standard.view(mx.uint16)).item()
            expected = packed_partials(x_packed, tokens, paired_gate_up).sum(axis=0).astype(mx.bfloat16)
            assert mx.array_equal(joint.project(x_packed, tokens).view(mx.uint16), expected.view(mx.uint16)).item()
            assert mx.array_equal(joint.argmax(x_packed, tokens), mx.argmax(expected, axis=-1)).item()
        paired_down = pack_weight(mx.random.normal((80, 17408)).astype(mx.bfloat16))
        joint_down = encode_paired(paired_down)
        for tokens in (1, 8, 9, 16):
            fragments = pack_activations(mx.random.normal((tokens, 17408)).astype(mx.bfloat16))
            expected = packed_partials(fragments, tokens, paired_down)
            assert mx.array_equal(joint_down.partials(fragments, tokens).view(mx.uint32),
                                  expected.view(mx.uint32)).item(), tokens
        zero = pack_activations(mx.zeros((11, 5120), dtype=mx.bfloat16))
        assert mx.all(joint.argmax(zero, 11) == 0).item()
        for raw in (mx.full((2, 4, 64), 0x3f80, dtype=mx.uint16),
                    (mx.arange(512, dtype=mx.uint16) * 127).reshape(2, 4, 64)):
            assert mx.array_equal(unpack_joint(encode_paired(raw).payload), raw).item()
        print("Joint gate/up packets preserve the original BF16 weights and SwiGLU bits.")
    for size in (9, 13, 16):
        wide = mx.random.normal((size, 1536)).astype(mx.bfloat16)
        for apply in (lambda block: packed_project(block, packed),
                      lambda block: packed_feed_forward(block, gate_up, down)):
            joined = mx.concatenate((apply(wide[:8]), apply(wide[8:])))
            assert mx.array_equal(apply(wide).view(mx.uint16), joined.view(mx.uint16)).item()
    print("Two token fragments share one weight read and match separate BF16 projections bit-for-bit.")
    projection = pack_weight((mx.random.normal((224, 1536)) / 64).astype(mx.bfloat16))
    history = sequence[:3]
    raw_qkv, raw_gate = mx.split(packed_project(x, projection), [160], axis=-1)
    raw_sequence = mx.concatenate((history, raw_qkv), axis=0)
    expected_conv = causal_conv4(raw_sequence, taps).astype(mx.float32)
    fused_conv, fused_gate, fused_sequence = projected_convolution(x, projection, history, taps)
    assert mx.allclose(fused_sequence.astype(mx.float32), raw_sequence.astype(mx.float32),
                       atol=0.001, rtol=0.008).item()
    assert mx.allclose(fused_gate.astype(mx.float32), raw_gate.astype(mx.float32),
                       atol=0.001, rtol=0.008).item()
    assert mx.allclose(fused_conv.astype(mx.float32), expected_conv, atol=0.003, rtol=0.015).item()
    value, z = [mx.random.normal((5, 2, 128)).astype(mx.bfloat16) for _ in range(2)]
    norm = (1 + mx.random.normal((128,)) / 16).astype(mx.bfloat16)
    weight = pack_weight((mx.random.normal((32, 256)) / 32).astype(mx.bfloat16))
    expected = packed_project(delta_output(mx.fast.rms_norm(value, norm, 1e-6), z).reshape(5, 256), weight)
    actual = gated_projection(value, z, norm, 1e-6, weight)
    assert mx.allclose(actual.astype(mx.float32), expected.astype(mx.float32),
                       atol=0.006, rtol=0.012).item()
    print("DeltaNet fused projection boundaries preserve convolution history and BF16 outputs within tolerance.")
    key, value = [mx.random.normal((1, 2, 37, 32)).astype(mx.bfloat16) for _ in range(2)]
    for width in (6, 7, 8, 9, 12, 16):
        query = mx.random.normal((1, 4, width, 32)).astype(mx.bfloat16)
        expected = mx.fast.scaled_dot_product_attention(query, key, value, scale=32 ** -0.5, mask="causal")
        actual = split_causal_attention(query, key, value, 32 ** -0.5)
        assert mx.allclose(actual.astype(mx.float32), expected.astype(mx.float32),
                           atol=0.003, rtol=0.012).item()
    print("Split attention agrees with the full causal attention reference.")
    scores = mx.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=mx.bfloat16)
    assert mx.allclose(greedy_log_probability(scores), mx.array([-math.log(4)] * 2)).item()
    for width in (1, 513, 32771, 248320):
        scores = (mx.random.normal((1, width)) * 2).astype(mx.bfloat16)
        scores = mx.where((mx.arange(width) == 0) | (mx.arange(width) == width - 1),
                          mx.array(16, dtype=mx.bfloat16), scores)
        token, confidence = draft_rank(scores)
        assert token.tolist() == [0]
        assert mx.allclose(confidence, greedy_log_probability(scores), atol=2e-6, rtol=2e-6).item()
    print("Fused draft ranking preserves first-index ties and agrees with the reference log probability.")
    choices = [[10, 0.95, -0.40], [11, 0.25, -0.50], [12, 0.15, -0.65]]
    assert ranked_alternates(choices, 1.0, 2) == [(1, 11), (0, 10)]
    assert ranked_alternates(choices[:1], 1.0, 1) == [(0, 10)]
    assert ranked_alternates(choices, 0.2, 1) == [(2, 12)]
    assert ranked_alternates(choices, 0.1, 1) == []
    order = VocabularyOrder(64, 32)
    ids = mx.arange(64, dtype=mx.int32)
    weights = mx.broadcast_to(((ids == 20) | (ids == 38))[:, None], (64, 5120)).astype(mx.bfloat16)
    shared = encode_paired(pack_weight(weights[order.to_token_ids(ids)]))
    x = pack_activations(mx.ones((2, 5120), dtype=mx.bfloat16))
    assert order.argmax(shared, x, 2).tolist() == [20, 20]
    original_draft = encode_paired(pack_weight(weights[:32]))
    assert mx.array_equal(order.draft_view(shared).project(x, 2), original_draft.project(x, 2)).item()
    print("Shared vocabulary preserves token order on ties and matches an independent draft projection.")


def verify_model(model, directory, tokenizer):
    ids = prompt_tokens(tokenizer, directory, "What is the capital of France?", False)
    tokens = mx.array(ids, dtype=mx.int32)
    cache = model.cache()
    whole = model.forward(tokens, cache, 0)
    model.evaluate(whole, cache)
    cache = model.cache()
    output = model.forward(tokens[:-1], cache, 0, logits=False)
    model.evaluate(output, cache)
    cached = model.forward(tokens[-1:], cache, len(ids) - 1)
    model.evaluate(cached, cache)
    first, second = mx.argmax(whole).item(), mx.argmax(cached).item()
    print(json.dumps({
        "prefill_argmax": first, "cached_argmax": second,
        "next_token": tokenizer.decode([first]),
        "mean_abs_logit_delta": mx.mean(mx.abs(whole.astype(mx.float32) - cached)).item(),
    }, ensure_ascii=False))
    if first != second:
        raise SystemExit("Cached and uncached inference disagree on the verification prompt")
    if model.packed_head is not None:
        original = model.head
        if original is None:
            index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
            key = "model.language_model.embed_tokens.weight" if model.config["tie_word_embeddings"] else "lm_head.weight"
            original = model.embedding if model.config["tie_word_embeddings"] else mx.load(str(directory / index[key]))[key]
        for start, restored in model.packed_vocabulary_blocks():
            expected_rows = original[start:start + restored.shape[0]]
            assert mx.array_equal(restored.view(mx.uint16), expected_rows.view(mx.uint16)).item()
        width = original.shape[1]
        if model.head is None:
            last = model.vocab_size - 1
            rows = mx.array([last, min(34, last), 0, min(32768, last), last], dtype=mx.int32)
            gathered = model.vocabulary_rows(rows)
            assert mx.array_equal(gathered.view(mx.uint16), original[rows].view(mx.uint16)).item()
            x = mx.random.normal((1, width)).astype(mx.bfloat16)
            normalized = mx.fast.rms_norm(x, model.norm, model.config["rms_norm_eps"]).astype(x.dtype)
            expected_logits = project(normalized, original, model.skinny)
            assert mx.array_equal(model.logits(x).view(mx.uint16), expected_logits.view(mx.uint16)).item()
            print(json.dumps({"packed_head_only_rows_exact": True, "single_token_logits_exact": True}))
        del original
        x = mx.random.normal((5, width)).astype(mx.bfloat16)
        expected = model.logits(x).astype(mx.float32)
        actual = model.predictions(x)
        score = mx.take_along_axis(expected, actual[:, None], axis=-1)[:, 0]
        max_gap = mx.max(mx.max(expected, axis=-1) - score).item()
        if max_gap > 0.05:
            raise SystemExit(f"Packed vocabulary argmax exceeds BF16 tolerance: {max_gap}")
        print(json.dumps({"full_vocabulary_weight_bits_match": True,
                          "packed_vocabulary_argmax_max_logit_gap": max_gap}))
        if model.fused_head_argmax:
            for tokens_in_batch in range(2, model.short_tokens + 1):
                x = mx.random.normal((tokens_in_batch, width)).astype(mx.bfloat16)
                normalized = mx.fast.rms_norm(x, model.norm, model.config["rms_norm_eps"]).astype(x.dtype)
                if model.lossless_vocabulary:
                    logits = model.packed_head.project(pack_activations(normalized), tokens_in_batch)
                    if model.vocabulary_order is not None:
                        logits = model.vocabulary_order.token_order(logits)
                else:
                    logits = packed_project(normalized, model.packed_head)
                expected = mx.argmax(logits, axis=-1)
                assert mx.array_equal(expected, model.predictions(x)).item(), tokens_in_batch
            print(json.dumps({"fused_head_argmax_exact_widths": [2, model.short_tokens]}))
    if model.draft_vocabulary_rows:
        rows = model.draft_vocabulary_rows
        extras = [rows, model.vocab_size - 1] if rows < model.vocab_size else []
        for additional in ([], extras):
            shortlist = draft_shortlist(model, additional, rows)
            original = model.vocabulary_rows(shortlist)
            coded = model.prepare_draft_vocabulary(shortlist, rows)
            x = mx.random.normal((1, model.config["hidden_size"])).astype(mx.bfloat16)
            expected = model.draft_scores(x, original)
            actual = model.draft_scores(x, coded)
            assert mx.argmax(expected).item() == mx.argmax(actual).item()
            assert mx.allclose(expected, actual, atol=0.001, rtol=0.008).item()
        print(json.dumps({"lossless_draft_vocab_rows": rows, "draft_vocab_argmax_matches": True}))
    if model.fast_draft:
        source = model.hidden(tokens, model.cache(), 0)
        optimized = LayerState()
        uncropped = LayerState()
        full = model.fast_draft
        try:
            model.fast_draft = False
            expected = model.draft(source[:-1], tokens[1:], uncropped, 0)[-1:]
        finally:
            model.fast_draft = full
        actual = model.draft(source[:-1], tokens[1:], optimized, 0)
        assert mx.array_equal(optimized.keys, uncropped.keys).item()
        assert mx.array_equal(optimized.values, uncropped.values).item()
        diff = mx.mean(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
        assert diff < 0.05, diff
        print(json.dumps({"draft_kv_exact": True, "draft_hidden_mae": diff}))
        if model.mtp_layer.reserve_single_kv:
            position = 512
            shape = (1, model.config["num_key_value_heads"], position, model.config["head_dim"])
            before = LayerState(
                keys=mx.random.normal(shape).astype(mx.bfloat16),
                values=mx.random.normal(shape).astype(mx.bfloat16),
            )
            reserved, concatenated = replace(before), replace(before)
            model.mtp_layer.reserve_single_kv = False
            try:
                expected = model.draft(source[-1:], tokens[-1:], concatenated, position)
            finally:
                model.mtp_layer.reserve_single_kv = True
            actual = model.draft(source[-1:], tokens[-1:], reserved, position)
            assert reserved.keys.shape[2] == reserved.values.shape[2] == 1024
            assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()
            for name in ("keys", "values"):
                with_reserve = getattr(reserved, name)[:, :, :position + 1]
                assert mx.array_equal(with_reserve, getattr(concatenated, name)).item()
                assert mx.array_equal(with_reserve[:, :, :position], getattr(before, name)).item()
            print(json.dumps({"mtp_single_kv_exact": True, "capacity_before": position, "capacity_after": 1024}))
    if model.draft_embedding is not None:
        source = source if model.fast_draft else model.hidden(tokens, model.cache(), 0)
        prefix = source[-1:]
        rows = model.draft_embedding.shape[0]
        ids = draft_shortlist(model, [model.embedding.shape[0] - 1], rows)
        table = model.draft_embedding_table(ids, rows)
        assert table is not None
        errors = []
        for index in (min(17, rows - 1), ids.shape[0] - 1):
            token = ids[index:index + 1]
            slot = mx.array([index], dtype=mx.int32)
            expected = model.draft(prefix, token, LayerState(), 0).astype(mx.float32)
            actual = model.draft(prefix, token, LayerState(), 0, cached_embedding=(table, slot)).astype(mx.float32)
            errors.append(mx.mean(mx.abs(expected - actual)).item())
        assert max(errors) < 0.01, errors
        print(json.dumps({"cached_draft_embedding_mae": errors, "cached_rows": rows}))


def verify_tree_model(model, directory, tokenizer, split_spine=False, multi_fork=False, nested=False):
    ids = prompt_tokens(tokenizer, directory, "The capital of France is", False)
    offset = len(ids)
    prefix = model.cache()
    hidden = model.hidden(mx.array(ids, dtype=mx.int32), prefix, 0)
    model.evaluate(hidden, prefix)
    parents = [-1, *range(8), 2, 9, 10, 9, 12, 13, 14] if nested else [-1, 0, 1, 0, 1 if multi_fork else 3]
    spine = 9 if nested else (3 if split_spine else None)
    tree = VerificationTree(parents, offset, model.config["linear_conv_kernel_dim"] - 1,
                            spine=spine)
    tokens = tokenizer.encode(" Paris Lyon London France England", add_special_tokens=False).ids
    assert len(tokens) == 5
    if nested:
        tokens = (tokens * 4)[:len(parents)]
    cache = [replace(layer) for layer in prefix]
    packed = model.hidden(mx.array(tokens, dtype=mx.int32), cache, offset, tree=tree)
    scores = model.logits(packed).astype(mx.float32)
    model.evaluate(scores, cache)

    other_cache = [replace(layer) for layer in prefix]
    path = list(tree.paths[-1])
    siblings = [i for i in range(len(tokens)) if i not in path]
    changed = tokens.copy()
    changed[siblings[0]], changed[siblings[1]] = changed[siblings[1]], changed[siblings[0]]
    varied = model.hidden(mx.array(changed, dtype=mx.int32), other_cache, offset, tree=tree)
    independent = mx.array(path, dtype=mx.int32)
    assert mx.array_equal(packed[independent], varied[independent]).item()

    serial = [replace(layer) for layer in prefix]
    linear = model.hidden(mx.array([tokens[i] for i in path], dtype=mx.int32), serial, offset)
    linear_scores = model.logits(linear).astype(mx.float32)
    delta = mx.mean(mx.abs(linear_scores - scores[independent])).item()
    model.accept_tree(cache, tree, offset, path)
    next_token = mx.array([tokens[0]], dtype=mx.int32)
    got = model.forward(next_token, cache, offset + len(path)).astype(mx.float32)
    want = model.forward(next_token, serial, offset + len(path)).astype(mx.float32)
    next_delta = mx.mean(mx.abs(got - want)).item()
    assert delta < 0.15 and next_delta < 0.15, (delta, next_delta)
    assert mx.argmax(got).item() == mx.argmax(want).item()
    print(json.dumps({"tree_logits_mae": delta, "next_token_logits_mae": next_delta,
                      "branch_isolation": True, "committed_next_token_matches": True,
                      "multi_fork": tree.multi_fork, "nested": nested}))


def verify_layer_prefill(model, directory, tokenizer):
    chunk = 512
    ids = prompt_tokens(tokenizer, directory, "def step(value):\n    return value + 1\n\n" * 128, False)
    for length in (chunk + 8, 2 * chunk + 17):
        tokens = mx.array(ids[:length], dtype=mx.int32)
        assert tokens.size == length
        expected_cache, actual_cache = model.cache(), model.cache()
        actual_blocks = list(model.prefill(tokens, actual_cache, chunk))
        for start, output in actual_blocks:
            expected = model.hidden(tokens[start:start + chunk], expected_cache, start)
            model.evaluate(expected, expected_cache)
            assert mx.array_equal(expected.view(mx.uint16), output.view(mx.uint16)).item(), length
        for expected, actual in zip(expected_cache, actual_cache):
            for name in ("keys", "values", "conv", "delta"):
                a, b = getattr(expected, name), getattr(actual, name)
                assert (a is None) == (b is None)
                if a is not None:
                    word = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
                    assert mx.array_equal(a.view(word), b.view(word)).item(), (length, name)
        model.budget.check(force=True)
    print("Layer-wise prefill matches block outputs and all caches with short and batched FFN tails.")


def verify_compiled_delta(model):
    layer = next(layer for layer in model.layers if layer.compiled_delta is not None)
    cfg = model.config
    conv = mx.zeros((cfg["linear_conv_kernel_dim"] - 1, layer.w["conv"].shape[0]), dtype=mx.bfloat16)
    delta = mx.zeros((cfg["linear_num_value_heads"], cfg["linear_value_head_dim"],
                      cfg["linear_key_head_dim"]), dtype=mx.float32)
    step = layer.compiled_delta
    for width in COMPILED_DELTA_WIDTHS:
        if width > layer.short_tokens:
            continue
        x = mx.random.normal((width, cfg["hidden_size"])).astype(mx.bfloat16)
        eager = LayerState(conv=conv, delta=delta)
        compiled = replace(eager)
        layer.compiled_delta = None
        try:
            expected = layer.forward(x, eager, 4096, keep_replay=True)
        finally:
            layer.compiled_delta = step
        output = layer.forward(x, compiled, 4096, keep_replay=True)
        pairs = ((expected, output), (eager.conv, compiled.conv), (eager.delta, compiled.delta),
                 (eager.replay[2].restore(3), compiled.replay[2].restore(3)))
        for a, b in pairs:
            word = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
            assert mx.array_equal(a.view(word), b.view(word)).item()
        model.budget.check()
    print("Compiled DeltaNet matches eager outputs, recurrent state and rollback.")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("weights/Qwen3.8-27B"))
    parser.add_argument("--chunk", type=positive, default=512)
    parser.add_argument("--group-rows", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--skinny", action="store_true")
    parser.add_argument("--reserve-kv", action="store_true")
    parser.add_argument("--vector-bf16", action="store_true")
    parser.add_argument("--fused-delta", action="store_true")
    parser.add_argument("--fused-delta-io", action="store_true")
    parser.add_argument("--compact-delta", action="store_true",
                        help="record rank-one updates for short linear DeltaNet rollback")
    parser.add_argument("--compiled-delta", action="store_true",
                        help="cache DeltaNet layer graphs for six- and eleven-position linear verification")
    parser.add_argument("--compact-tree", action="store_true",
                        help="use rank-one rollback with a single linear branch")
    parser.add_argument("--wide-blocks", action="store_true",
                        help="share packed BF16 projections across up to 16 verification tokens")
    parser.add_argument("--split-attention", action="store_true")
    parser.add_argument("--fused-attention-io", action="store_true",
                        help="fuse short-block attention normalization, rotation and output packing")
    parser.add_argument("--wide-verify", action="store_true")
    packing = parser.add_mutually_exclusive_group()
    packing.add_argument("--packed-ffn", action="store_true")
    packing.add_argument("--packed-main", action="store_true")
    parser.add_argument("--layer-prefill", action="store_true",
                        help="reuse unpacked weights across prompt blocks one layer at a time")
    parser.add_argument("--lossless-down", action="store_true",
                        help="store packed down projection weights as reversible BF16 packets")
    parser.add_argument("--paired-down", action="store_true",
                        help="share activation fragments across two lossless down projection output tiles")
    parser.add_argument("--lossless-gate-up", action="store_true",
                        help="store packed gate/up projections as joint reversible BF16 packets")
    parser.add_argument("--lossless-inputs", action="store_true",
                        help="store packed attention and DeltaNet input projections as reversible BF16 packets")
    parser.add_argument("--lossless-attention-outputs", action="store_true",
                        help="store packed attention and DeltaNet output projections as reversible BF16 packets")
    parser.add_argument("--packed-head", action="store_true",
                        help="use packed BF16 for full-vocabulary short-block verification")
    parser.add_argument("--packed-head-only", action="store_true",
                        help="keep only the packed layout of the full vocabulary projection")
    parser.add_argument("--lossless-vocabulary", action="store_true",
                        help="store the packed vocabulary as reversible BF16 packets")
    parser.add_argument("--packed-attention", action="store_true",
                        help="pack the remaining attention projections for short blocks")
    parser.add_argument("--fused-ffn", action="store_true")
    parser.add_argument("--fused-residual", action="store_true",
                        help="combine residual addition, RMSNorm and BF16 fragment packing for short blocks")
    parser.add_argument("--fast-draft", action="store_true",
                        help="compute only the final draft hidden state while updating the complete draft cache")
    parser.add_argument("--lossless-mtp-ffn", action="store_true",
                        help="use reversible BF16 MTP feed-forward weights for single-token drafts")
    parser.add_argument("--memory-gib", type=positive, default=DEFAULT_MEMORY_GIB)
    parser.add_argument("--cache-gib", type=nonnegative, default=DEFAULT_CACHE_GIB)
    parser.add_argument("--draft-depth", type=int, choices=range(16), default=0)
    parser.add_argument("--tree-depth", type=int, choices=range(4), default=0,
                        help="maximum length of a second MTP branch")
    parser.add_argument("--tree-tail-gap", type=float,
                        help="maximum logit margin for branching within the existing candidate budget")
    parser.add_argument("--lookup-depth", type=int, choices=range(33), default=0)
    parser.add_argument("--draft-vocab", type=positive, default=0,
                        help="draft-only shortlist of the first N vocabulary IDs plus the input IDs")
    parser.add_argument("--lossless-draft-vocab", action="store_true",
                        help="cache the base draft shortlist in reversible BF16 packets")
    parser.add_argument("--draft-confirmed-vocab", action="store_true",
                        help="add up to 128 confirmed output IDs to the BF16 draft shortlist")
    parser.add_argument("--draft-lane-cache", action="store_true",
                        help="reuse exact BF16 MTP embedding accumulators on supported long requests")
    parser.add_argument("--draft-paired-root", action="store_true",
                        help="read each MTP weight fragment once for the root and its backup")
    parser.add_argument("--draft-schedule", choices=DRAFT_SCHEDULES, default="standard",
                        help="select logical attention, bounded frontier, or vocabulary prefetch for 4096/1024")
    parser.add_argument("--shared-vocabulary", action="store_true",
                        help="use the same BF16 packets for target verification and the draft shortlist")
    parser.add_argument("--draft-confidence", type=float,
                        help="stop extending a linear draft when its cumulative probability falls below this value")
    parser.add_argument("--draft-extension-confidence", type=float,
                        help="minimum cumulative probability at each five-token checkpoint to extend a device draft")
    parser.add_argument("--draft-root-branch-gap", type=float,
                        help="max logit gap for a second candidate on the draft branch")
    parser.add_argument("--draft-branch-positions", type=int, choices=range(1, 6), default=1,
                        help="choose branch points from the first N draft positions")
    parser.add_argument("--draft-branch-depth", type=int, choices=range(1, 6), default=2,
                        help="maximum length of a device-drafted alternative branch")
    parser.add_argument("--draft-second-branch", action="store_true",
                        help="try a third-ranked candidate at the same fork when its gap qualifies")
    parser.add_argument("--draft-multi-fork", action="store_true",
                        help="verify alternate continuations at up to two different draft positions")
    parser.add_argument("--device-draft", action="store_true",
                        help="read back an entire linear draft at once before target verification")
    parser.add_argument("--fused-draft-ranking", action="store_true",
                        help="compute the draft argmax and log probability together")
    parser.add_argument("--cache-draft-embedding", action="store_true",
                        help="precompute FP32 embedding contributions for short-listed MTP proposals")
    parser.add_argument("--draft-panel-checkpoint", action="store_true",
                        help="reapply the extension threshold before an eighth candidate")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-kernel")
    commands.add_parser("verify-model")
    bench = commands.add_parser("bench")
    bench.add_argument("--runs", type=positive, default=1)
    bench.add_argument("--jsonl", type=Path)
    bench.add_argument("--prompt-file", type=Path)
    chat = commands.add_parser("generate")
    chat.add_argument("prompt")
    chat.add_argument("--max-tokens", type=positive, default=OUTPUT_TOKENS)
    chat.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if (args.packed_ffn or args.packed_main) and not args.merge:
        parser.error("packed projections require --merge")
    if args.packed_head_only and not args.packed_head:
        parser.error("--packed-head-only requires --packed-head")
    if args.lossless_vocabulary and not args.packed_head:
        parser.error("--lossless-vocabulary requires --packed-head")
    if args.lossless_down and not args.packed_main:
        parser.error("--lossless-down requires --packed-main")
    if args.paired_down and not args.lossless_down:
        parser.error("--paired-down requires --lossless-down")
    if args.lossless_gate_up and not args.fused_ffn:
        parser.error("--lossless-gate-up requires --fused-ffn")
    if args.lossless_inputs and not args.packed_main:
        parser.error("--lossless-inputs requires --packed-main")
    if args.lossless_attention_outputs and not args.packed_attention:
        parser.error("--lossless-attention-outputs requires --packed-attention")
    if args.packed_attention and not args.packed_main:
        parser.error("--packed-attention requires --packed-main")
    if args.fused_delta_io and not (args.packed_attention and args.fused_delta):
        parser.error("--fused-delta-io requires --packed-attention and --fused-delta")
    if args.fused_ffn and not args.packed_main:
        parser.error("--fused-ffn requires --packed-main")
    if args.fused_residual and not args.fused_ffn:
        parser.error("--fused-residual requires --fused-ffn")
    if args.wide_blocks and not (args.packed_main and args.compact_delta):
        parser.error("--wide-blocks requires --packed-main and --compact-delta")
    if args.compact_tree and not args.compact_delta:
        parser.error("--compact-tree requires --compact-delta")
    if args.compiled_delta and not (args.compact_delta and args.fused_delta and args.fused_residual):
        parser.error("--compiled-delta requires --compact-delta, --fused-delta and --fused-residual")
    if args.layer_prefill and not args.packed_main:
        parser.error("--layer-prefill requires --packed-main")
    if args.draft_depth > 8 and not args.wide_blocks:
        parser.error("draft depths above eight require --wide-blocks")
    if args.fast_draft and not args.draft_depth:
        parser.error("--fast-draft requires MTP")
    if args.lossless_draft_vocab and not (args.draft_depth and args.draft_vocab):
        parser.error("--lossless-draft-vocab requires MTP and --draft-vocab")
    if args.draft_confirmed_vocab and (not args.lossless_draft_vocab or args.cache_draft_embedding):
        parser.error("--draft-confirmed-vocab requires --lossless-draft-vocab without --cache-draft-embedding")
    if args.shared_vocabulary and not (args.lossless_vocabulary and args.lossless_draft_vocab):
        parser.error("--shared-vocabulary requires lossless target and draft vocabularies")
    if args.tree_depth and (not args.draft_depth or args.draft_depth + args.tree_depth > 7):
        parser.error("--tree-depth needs MTP and at most seven total candidate nodes")
    if args.tree_tail_gap is not None:
        if args.tree_depth or not 2 <= args.draft_depth <= 7:
            parser.error("--tree-tail-gap needs a draft depth of two to seven and no extra branch")
        if not math.isfinite(args.tree_tail_gap) or args.tree_tail_gap < 0:
            parser.error("--tree-tail-gap must be finite and nonnegative")
    if args.draft_confidence is not None:
        if not 0 < args.draft_confidence < 1:
            parser.error("--draft-confidence must be between zero and one")
        if args.draft_depth < 3 or args.tree_depth or args.tree_tail_gap is not None:
            parser.error("--draft-confidence requires at least three linear MTP candidates")
    if args.device_draft and (not args.draft_depth or args.lookup_depth or args.tree_depth
                              or args.tree_tail_gap is not None):
        parser.error("--device-draft requires linear MTP without lookup")
    if args.fused_draft_ranking and not args.device_draft:
        parser.error("--fused-draft-ranking requires --device-draft")
    if args.cache_draft_embedding and (not args.device_draft or not args.draft_vocab):
        parser.error("--cache-draft-embedding requires --device-draft and --draft-vocab")
    if args.draft_extension_confidence is not None:
        if not args.device_draft or args.draft_depth <= 7 or args.draft_confidence is None:
            parser.error("--draft-extension-confidence needs device drafting above seven with --draft-confidence")
        if not args.draft_confidence <= args.draft_extension_confidence < 1:
            parser.error("--draft-extension-confidence must be at least --draft-confidence and below one")
    if args.draft_panel_checkpoint and args.draft_extension_confidence is None:
        parser.error("--draft-panel-checkpoint requires --draft-extension-confidence")
    if args.draft_root_branch_gap is not None:
        if not args.device_draft:
            parser.error("--draft-root-branch-gap needs device drafting")
        if not math.isfinite(args.draft_root_branch_gap) or args.draft_root_branch_gap < 0:
            parser.error("--draft-root-branch-gap must be finite and nonnegative")
    if args.draft_branch_positions > 1 and args.draft_root_branch_gap is None:
        parser.error("--draft-branch-positions above one needs --draft-root-branch-gap")
    if args.draft_branch_depth != 2 and args.draft_root_branch_gap is None:
        parser.error("--draft-branch-depth requires --draft-root-branch-gap")
    if args.draft_second_branch and args.draft_root_branch_gap is None:
        parser.error("--draft-second-branch requires --draft-root-branch-gap")
    if args.draft_multi_fork and (not args.compact_tree or args.draft_branch_positions < 2 or args.draft_second_branch):
        parser.error("--draft-multi-fork needs --compact-tree, at least two branch positions, and no --draft-second-branch")
    try:
        with metal_job(args.memory_gib, args.cache_gib) as budget:
            if args.command == "verify-kernel":
                verify_kernel()
            else:
                run(args, budget)
    except MemoryBudgetError as error:
        parser.exit(1, f"{error}\n")


def run(args, budget):
    start = perf_counter()
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    model = Model(args.model, merge=args.merge, group_rows=args.group_rows,
                  mtp=bool(args.draft_depth), skinny=args.skinny, reserve_kv=args.reserve_kv,
                  vector_bf16=args.vector_bf16, fused_delta=args.fused_delta,
                  wide_verify=args.wide_verify, packed_ffn=args.packed_ffn,
                  packed_main=args.packed_main, packed_head=args.packed_head,
                  packed_head_only=args.packed_head_only,
                  lossless_down=args.lossless_down,
                  paired_down=args.paired_down,
                  lossless_gate_up=args.lossless_gate_up,
                  lossless_inputs=args.lossless_inputs,
                  lossless_attention_outputs=args.lossless_attention_outputs,
                  lossless_vocabulary=args.lossless_vocabulary,
                  packed_attention=args.packed_attention, fused_ffn=args.fused_ffn,
                  fast_draft=args.fast_draft, fused_delta_io=args.fused_delta_io,
                  split_attention=args.split_attention, fused_attention_io=args.fused_attention_io,
                  fused_residual=args.fused_residual,
                  compact_delta=args.compact_delta, wide_blocks=args.wide_blocks,
                  compact_tree=args.compact_tree,
                  draft_embedding_rows=args.draft_vocab if args.cache_draft_embedding else 0,
                  lossless_draft_rows=args.draft_vocab if args.lossless_draft_vocab else 0,
                  shared_vocabulary=args.shared_vocabulary,
                  compiled_delta=args.compiled_delta,
                  layer_prefill=args.layer_prefill,
                  lossless_mtp_ffn=args.lossless_mtp_ffn,
                  budget=budget)
    print(f"BF16 text model loaded in {perf_counter() - start:.2f}s", file=sys.stderr)
    if args.command == "verify-model":
        verify_model(model, args.model, tokenizer)
        if model.compiled_delta:
            verify_compiled_delta(model)
        if model.layer_prefill:
            verify_layer_prefill(model, args.model, tokenizer)
        if args.tree_depth or args.tree_tail_gap is not None or args.draft_root_branch_gap is not None:
            verify_tree_model(model, args.model, tokenizer, split_spine=args.draft_root_branch_gap is not None,
                              multi_fork=args.draft_multi_fork)
        if args.draft_multi_fork and args.wide_blocks:
            verify_tree_model(model, args.model, tokenizer, nested=True)
        return
    if args.command == "generate":
        ids = prompt_tokens(tokenizer, args.model, args.prompt, args.thinking)
        limits = json.loads((args.model / "generation_config.json").read_text())
        eos = limits["eos_token_id"]
        stop = (eos,) if isinstance(eos, int) else tuple(eos)
        if len(ids) + args.max_tokens > model.config["max_position_embeddings"]:
            raise SystemExit("prompt and generation budget exceed the model context limit")
        generated, stats = generate(
            model, ids, args.max_tokens, args.chunk, stop, args.draft_depth, args.lookup_depth, args.draft_vocab,
            budget, args.tree_depth, args.tree_tail_gap, args.draft_confidence, args.device_draft,
            args.draft_extension_confidence, args.draft_root_branch_gap, args.draft_branch_positions,
            args.fused_draft_ranking, args.draft_panel_checkpoint, args.draft_branch_depth, args.draft_second_branch,
            args.draft_multi_fork, draft_confirmed_vocab=args.draft_confirmed_vocab,
            draft_lane_cache=args.draft_lane_cache, draft_paired_root=args.draft_paired_root,
            draft_schedule=args.draft_schedule,
        )
        print(tokenizer.decode(generated, skip_special_tokens=True))
        print(json.dumps(stats), file=sys.stderr)
        return

    ids = bench_prompt(tokenizer, args.prompt_file)
    source = Path(__file__).resolve().parent
    runtime_hash = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(source.glob("*.py")))).hexdigest()
    # Warm both dispatch paths outside timed runs and discard their caches.
    generate(
        model, ids[: min(args.chunk, 64)], 4, args.chunk, stop=(-1,),
        draft_depth=args.draft_depth, lookup_depth=args.lookup_depth, draft_vocab=args.draft_vocab,
        budget=budget, tree_depth=args.tree_depth, tree_tail_gap=args.tree_tail_gap,
        draft_confidence=args.draft_confidence, device_draft=args.device_draft,
        draft_extension_confidence=args.draft_extension_confidence,
        draft_root_branch_gap=args.draft_root_branch_gap,
        draft_branch_positions=args.draft_branch_positions,
        fused_draft_ranking=args.fused_draft_ranking,
        draft_panel_checkpoint=args.draft_panel_checkpoint,
        draft_branch_depth=args.draft_branch_depth,
        draft_second_branch=args.draft_second_branch,
        draft_multi_fork=args.draft_multi_fork,
        draft_confirmed_vocab=args.draft_confirmed_vocab, draft_lane_cache=args.draft_lane_cache,
        draft_paired_root=args.draft_paired_root, draft_schedule=args.draft_schedule,
    )
    results = []
    for run in range(1, args.runs + 1):
        mx.clear_cache()
        thermal_before = thermal_state()
        generated, stats = generate(
            model, ids, OUTPUT_TOKENS, args.chunk, draft_depth=args.draft_depth,
            lookup_depth=args.lookup_depth, draft_vocab=args.draft_vocab, budget=budget,
            tree_depth=args.tree_depth, tree_tail_gap=args.tree_tail_gap,
            draft_confidence=args.draft_confidence, device_draft=args.device_draft,
            draft_extension_confidence=args.draft_extension_confidence,
            draft_root_branch_gap=args.draft_root_branch_gap,
            draft_branch_positions=args.draft_branch_positions,
            fused_draft_ranking=args.fused_draft_ranking,
            draft_panel_checkpoint=args.draft_panel_checkpoint,
            draft_branch_depth=args.draft_branch_depth,
            draft_second_branch=args.draft_second_branch,
            draft_multi_fork=args.draft_multi_fork,
            draft_confirmed_vocab=args.draft_confirmed_vocab, draft_lane_cache=args.draft_lane_cache,
            draft_paired_root=args.draft_paired_root, draft_schedule=args.draft_schedule,
        )
        stats.update({
            "run": run, "model": str(args.model), "weight_dtype": "BF16",
            "thermal_before": thermal_before, "thermal_after": thermal_state(),
            "runtime_hash": runtime_hash,
            "mlx": mx.__version__, "device": mx.device_info()["device_name"],
            "metal_command_buffer": {
                name: os.environ.get(name) for name in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")
            },
            "chunk": args.chunk, "group_rows": args.group_rows, "merge": args.merge,
            "skinny": args.skinny,
            "reserve_kv": args.reserve_kv,
            "vector_bf16": args.vector_bf16,
            "fused_delta": args.fused_delta,
            "fused_delta_io": args.fused_delta_io,
            "compact_delta": args.compact_delta,
            "compact_tree": args.compact_tree,
            "wide_blocks": args.wide_blocks,
            "split_attention": args.split_attention,
            "wide_verify": args.wide_verify,
            "packed_ffn_layers": model.packed_ffn_layers,
            "packed_main_matrices": model.packed_main_count,
            "packed_head": model.packed_head is not None,
            "packed_attention": args.packed_attention,
            "fused_ffn": args.fused_ffn,
            "fused_residual": args.fused_residual,
            "fast_draft": args.fast_draft,
            "resident_gib": model.resident_gib,
            "draft_depth": args.draft_depth,
            "tree_depth": args.tree_depth,
            "tree_tail_gap": args.tree_tail_gap,
            "lookup_depth": args.lookup_depth,
            "draft_vocab": args.draft_vocab,
            "draft_confidence": args.draft_confidence,
            "draft_extension_confidence": args.draft_extension_confidence,
            "draft_root_branch_gap": args.draft_root_branch_gap,
            "draft_branch_positions": args.draft_branch_positions,
            "fused_draft_ranking": args.fused_draft_ranking,
            "cache_draft_embedding": args.cache_draft_embedding,
            "draft_panel_checkpoint": args.draft_panel_checkpoint,
            "draft_branch_depth": args.draft_branch_depth,
            "draft_second_branch": args.draft_second_branch,
            "draft_multi_fork": args.draft_multi_fork,
            "device_draft": args.device_draft,
            "prompt_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "token_hash": hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
        })
        line = json.dumps(stats)
        print(line, flush=True)
        if args.jsonl:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.jsonl.open("a") as file:
                file.write(line + "\n")
        results.append(stats["decode_tok_s"])
    print(f"Median decode {statistics.median(results):.2f} tok/s", file=sys.stderr)


if __name__ == "__main__":
    main()

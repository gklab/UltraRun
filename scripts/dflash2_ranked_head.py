"""Select DFlash2 candidates directly from the lossless BF16 head layout."""

import inspect
import json
from statistics import median
from time import perf_counter

import mlx.core as mx

from ultrarun.lossless_vocabulary import VocabularyOrder
from ultrarun.packed_projection import pack_activations
from scripts.bench_dflash2_tandem import reference
from scripts.bench_verbatim_ffn import replace_once
from scripts.dflash2_top16 import bounded_top16, mapped_top16, reference_top16


def rank_ordered(logits):
    ids = bounded_top16(logits)
    return ids, mx.take_along_axis(logits, ids, axis=-1)


def storage_scores(target, hidden):
    assert hidden.dtype == mx.bfloat16 and hidden.shape[0] == 1
    return target.packed_head.project(pack_activations(hidden[0]), hidden.shape[1])


def check_scores(left, right):
    assert mx.array_equal(left[0], right[0]).item(), "Different token IDs or candidate order"
    assert mx.array_equal(left[1].view(mx.uint16), right[1].view(mx.uint16)).item(), "Different BF16 scores"


class CompactShortlist:
    """Score the shared draft rows and prompt extras without expanding masked IDs."""

    def __init__(self, target, prompt):
        self.rows = target.vocabulary_order.draft_rows
        self.size = target.vocab_size
        self.head = target.vocabulary_order.draft_view(target.packed_head)
        extra_ids = sorted({token for token in prompt if token >= self.rows})
        self.extras = mx.array(extra_ids, dtype=mx.int32)
        self.weights = target.vocabulary_rows(self.extras) if self.extras.size else None
        # Keep the largest masked IDs for stable ties if a score is -inf.
        selected = {token: i for i, token in enumerate(extra_ids)}
        masked = 0
        for token in range(self.size - 1, self.rows - 1, -1):
            if token not in selected:
                selected[token] = self.extras.size
                masked += 1
                if masked == 16:
                    break
        suffix = sorted(selected)
        self.suffix_ids = mx.array(suffix, dtype=mx.int32)
        self.suffix_positions = mx.array([selected[token] for token in suffix], dtype=mx.int32)
        mx.eval(self.suffix_ids, self.suffix_positions)
        if self.weights is not None:
            mx.eval(self.weights)

    def scores(self, hidden):
        rows = hidden.shape[1]
        hidden = hidden[0]
        scores = self.head.project(pack_activations(hidden), rows)
        extras = hidden @ self.weights.T if self.weights is not None else None
        return scores, extras

    def masked_logits(self, hidden):
        """Materialize the original representation for the existing diagnostic."""
        scores, extras = self.scores(hidden)
        omitted = mx.full((hidden.shape[1], self.size - self.rows), -mx.inf, dtype=mx.bfloat16)
        full = mx.concatenate((scores, omitted), axis=1)
        if extras is not None:
            full = full.at[:, self.extras].maximum(extras)
        return full

    def rank(self, hidden):
        scores, extras = self.scores(hidden)
        prefix_ids, prefix = rank_ordered(scores)
        if not self.suffix_ids.size:
            return prefix_ids, prefix
        masked = mx.full((hidden.shape[1], 1), -mx.inf, dtype=mx.bfloat16)
        if extras is not None:
            masked = mx.concatenate((extras, masked), axis=-1)
        suffix = masked[:, self.suffix_positions]
        combined = mx.concatenate((prefix, suffix), axis=-1)
        suffix_ids = mx.broadcast_to(self.suffix_ids, suffix.shape)
        ids = mx.concatenate((prefix_ids, suffix_ids), axis=-1)
        winners = reference_top16(combined)
        return mx.take_along_axis(ids, winners, axis=-1), mx.take_along_axis(combined, winners, axis=-1)


def profile_shortlist(target, compact):
    """Pair the existing mask, compact merge, and full head on fixed BF16 inputs."""
    hidden = mx.random.normal((1, 7, target.config["hidden_size"])).astype(mx.bfloat16)
    mx.eval(hidden)
    methods = {
        "masked": lambda: rank_ordered(compact.masked_logits(hidden)),
        "compact": lambda: compact.rank(hidden),
        "full": lambda: mapped_top16(storage_scores(target, hidden), target.vocabulary_order),
    }
    check_scores(methods["masked"](), methods["compact"]())
    samples = {name: [] for name in methods}
    names = tuple(methods)
    for step in range(12):
        shift = step % len(names)
        for name in names[shift:] + names[:shift]:
            start = perf_counter()
            mx.eval(*methods[name]())
            samples[name].append((perf_counter() - start) * 1000)
    print("compact_head_probe", json.dumps({"rows": hidden.shape[1], "mask_bitwise": True,
          "median_ms": {name: median(durations) for name, durations in samples.items()}}), flush=True)


def profile(target, draft, prepared, context):
    """Check original logits and signed-zero/NaN ties before timing both heads."""
    _, first, prefix = prepared
    special = mx.array([0.0, -0.0, 0.5, 1.0, float("inf"), -float("inf"), float("nan")],
                       dtype=mx.bfloat16)
    order = VocabularyOrder(2048, 512)
    logical = mx.take(special, mx.arange(4096).reshape(2, 2048) % special.size)
    physical = logical[:, order.to_token_ids(mx.arange(order.size))]
    check_scores(rank_ordered(logical), mapped_top16(physical, order))

    cache = draft.make_cache()
    for state in cache:
        state.offset = context - prefix.shape[1]
    block = draft.config.block_size
    ids = mx.array([[first] + [draft.config.mask_token_id] * (block - 1)], dtype=mx.int32)
    hidden = draft.hidden_states(ids, prefix, cache, logits_start=1)
    mx.eval(hidden)
    order = target.vocabulary_order
    for width in (7, 15):
        h = hidden[:, :width]
        storage = storage_scores(target, h)
        mx.eval(storage)
        original = rank_ordered(draft.compute_logits(h)[0])
        check_scores(original, mapped_top16(storage, order))
        operations = {
            "baseline_sort": lambda: rank_ordered(order.token_order(storage)),
            "mapped_sort": lambda: mapped_top16(storage, order),
            "baseline_head": lambda: rank_ordered(draft.compute_logits(h)[0]),
            "mapped_head": lambda: mapped_top16(storage_scores(target, h), order),
        }
        readings = {key: [] for key in operations}
        functions = list(operations)
        for trial in range(20):
            for name in functions[trial % 4:] + functions[:trial % 4]:
                started = perf_counter()
                mx.eval(*operations[name]())
                readings[name].append((perf_counter() - started) * 1000)
        print("ranked_head", json.dumps({
            "width": width, "bitwise": True, "median_ms": {key: median(v) for key, v in readings.items()},
        }), flush=True)


def lattice_for(target, block, *, retain=False, compact=None):
    """Reuse the original conditional lattice with pre-ranked BF16 unary values."""
    def rank_hidden(hidden):
        if compact is not None:
            return compact.rank(hidden)
        return mapped_top16(storage_scores(target, hidden), target.vocabulary_order)

    def original_logits(draft, hidden):
        return draft.compute_logits(hidden)[0] if compact is None else compact.masked_logits(hidden)

    source = inspect.getsource(reference.lattice)
    source = replace_once(
        source,
        """    logits = draft.compute_logits(hidden)
    candidates = upstream.rank_candidates(logits, draft.config.selector_top_k)[0]""",
        "    candidates, unary = rank_hidden(hidden)",
    )
    source = replace_once(source, "    unary = mx.take_along_axis(logits[0], candidates, axis=-1)\n", "")
    source = replace_once(
        source,
        "        verify_top16(logits[0])",
        """        logits = original_logits(draft, hidden)[None]
        verify_top16(logits[0])
        check_scores(rank_ordered(logits[0]), (candidates, unary))""",
    )
    if retain:
        source = replace_once(source, "    return choices,weights",
                              "    return choices, weights, candidates, unary, contextual")
    scope = {**vars(reference), "DRAFT_WIDTH": block, "rank_hidden": rank_hidden,
             "original_logits": original_logits, "rank_ordered": rank_ordered, "check_scores": check_scores}
    exec(compile(source, __file__, "exec"), scope)
    return scope["lattice"]

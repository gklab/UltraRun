"""Measure DFlash2 conditional ranks at every anchor of a saved BF16 continuation."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from ultrarun.memory import GIB, metal_job
from ultrarun.server import warm_verification_graphs
from ultrarun.spec_tree import VerificationTree
from scripts.bench_dflash2 import ROOT, bind_draft, load_draft, load_target, observe_layers, pages, prefill


WIDTH = 16
CASE = "chinese_technical_report"


def capture_teacher(target, draft, trace, cache):
    taps = draft.config.target_layer_ids
    captured = dict.fromkeys(taps)

    def observe(layer, _offset, hidden):
        captured[layer] = hidden

    observe_layers(target, taps, observe)
    expected = trace["output"]
    context = len(trace["input"])
    conv = target.config["linear_conv_kernel_dim"] - 1
    features, disagreements = [], []
    for start in range(0, len(expected) - 1, WIDTH):
        stop = min(start + WIDTH, len(expected) - 1)
        width = stop - start
        offset = context + start
        tree = VerificationTree([-1, *range(width - 1)], offset, conv, spine=width - 1)
        inputs = mx.array(expected[start:stop], dtype=mx.int32)
        hidden = target.hidden(inputs, cache, offset, keep_replay=True, tree=tree)
        predictions = target.predictions(hidden)
        tapped = mx.concatenate([captured[i] for i in taps], axis=-1)[None]
        target.evaluate([predictions, tapped], cache)
        disagreements.extend(i for i, (a, b) in enumerate(zip(predictions.tolist(), expected[start + 1:stop + 1]),
                                                                start + 1) if a != b)
        target.accept_tree(cache, tree, offset, list(range(width)))
        target.evaluate(tapped, cache)
        features.append(tapped)
        target.budget.check()
    hidden = mx.concatenate(features, axis=1)
    mx.eval(hidden)
    return hidden, disagreements


def conditional_ranks(selector, hidden, logits, candidates, teacher):
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    query = selector.hidden_projection(hidden)
    pred = selector.predecessor_codebook(teacher[:-1])[None, :, None, :]
    successors = selector.successor_codebook(candidates)
    scores = unary + mx.sum(pred * query[:, :, None, :] * successors, axis=-1)
    present = candidates == teacher[None, 1:, None]
    chosen_slot = mx.argmax(present, axis=-1, keepdims=True)
    chosen_score = mx.take_along_axis(scores, chosen_slot, axis=-1)
    earlier = mx.arange(candidates.shape[-1]) < chosen_slot
    beats_truth = (scores > chosen_score) | ((scores == chosen_score) & earlier)
    return mx.where(mx.any(present, axis=-1), 1 + mx.sum(beats_truth, axis=-1), candidates.shape[-1] + 1)


def capture_ranks(draft, upstream, prompt_hidden, teacher_hidden, trace, budget):
    cache = draft.make_cache()
    for state in cache:
        state.offset = len(trace["input"]) - prompt_hidden.shape[1]
    truth = trace["output"]
    ranks = []
    for anchor in range(len(truth) - 1):
        width = min(WIDTH, len(truth) - anchor)
        context = prompt_hidden if anchor == 0 else teacher_hidden[:, anchor - 1:anchor]
        masks = [draft.config.mask_token_id] * (width - 1)
        inputs = mx.array([[truth[anchor], *masks]], dtype=mx.int32)
        hidden = draft.hidden_states(inputs, context, cache, logits_start=1)
        logits = draft.compute_logits(hidden)
        candidates = upstream.rank_candidates(logits, draft.config.selector_top_k)
        teacher = mx.array(truth[anchor:anchor + width], dtype=mx.int32)
        step = conditional_ranks(draft.candidate_selector, hidden, logits, candidates, teacher)
        ranks.append(step[0].tolist())
        budget.check()
        if (anchor + 1) % 256 == 0:
            print("ranks_progress", anchor + 1, flush=True)
    return ranks


def modeled_advance(ranks, mode):
    nodes, branches, length = 1, 0, 0
    for rank in ranks:
        if rank > 16 or (mode == "top1" and rank != 1):
            break
        nodes += rank if mode == "two_ranked_forks" else 1
        branches += rank > 1
        if nodes > WIDTH or (mode == "two_ranked_forks" and branches > 2):
            break
        length += 1
    return length + 1


def fewest_rounds(reach):
    last = len(reach)
    distance = [0] + [last] * last
    for anchor, limit in enumerate(reach):
        for end in range(anchor + 1, min(last, anchor + limit) + 1):
            distance[end] = min(distance[end], distance[anchor] + 1)
    return distance[-1]


def summarize(ranks, roofline):
    results = {}
    for mode in ("top1", "two_ranked_forks", "any_top16_oracle"):
        reach = [min(len(ranks) - i, modeled_advance(row, mode)) for i, row in enumerate(ranks)]
        rounds = fewest_rounds(reach)
        results[mode] = {
            "oracle_minimum_rounds": rounds, "mean_confirmed_at_all_anchors": sum(reach) / len(reach),
            "anchors_capable_of_10": sum(value >= 10 for value in reach),
            "tok_s_at_modeled_floor": len(ranks) * 1000 / (rounds * roofline),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_bytes())[CASE]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    costs = (ROOT / "runs/dflash2-acceptance-budget.log").read_text().splitlines()
    floor = next(row["modeled_round_ms"] for tag, _, data in (line.partition(" ") for line in costs)
                 if tag == "threshold" for row in (json.loads(data),) if row["width"] == WIDTH)
    before = pages()
    with metal_job(80, 2) as budget:
        target = load_target(budget)
        budget.admit(target, 5120, 512, False, extra_bytes=6 * GIB)
        draft, upstream = load_draft(args.upstream, ROOT / "weights/Qwen3.8-27B-DFlash2")
        bind_draft(target, draft, {})
        budget.admit(target, 5120, 512, False, extra_bytes=2 * GIB)
        warm_verification_graphs(target, budget)
        started = perf_counter()
        cache, first, prompt_hidden = prefill(target, draft, trace["input"])
        assert first == trace["output"][0]
        teacher_hidden, disagreements = capture_teacher(target, draft, trace, cache)
        del cache
        mx.clear_cache()
        print("teacher", json.dumps({"seconds": perf_counter() - started, "len": teacher_hidden.shape[1],
                                    "verification_differences": disagreements}), flush=True)
        ranks = capture_ranks(draft, upstream, prompt_hidden, teacher_hidden, trace, budget)
        assert len(ranks) == 1023 and all(len(row) == min(15, 1023 - i) for i, row in enumerate(ranks))
        budget.check(force=True)
        after = pages()
        result = {"case": CASE, "width": WIDTH, "modeled_round_floor_ms": floor,
                  "modes": summarize(ranks, floor), "root_ranks": Counter(row[0] for row in ranks),
                  "teacher_forced_differences": disagreements, "peak_gib": mx.get_peak_memory() / GIB,
                  "host_delta": {key: after[key] - before[key] for key in before}, **budget.metrics()}
        print("result", json.dumps(result), flush=True)
        args.results.write_text(json.dumps({**result, "ranks": ranks}) + "\n")


if __name__ == "__main__":
    main()

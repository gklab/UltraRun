"""Compare terminal versus layerwise conditioning on actual native MTP proposals."""

import argparse
from collections import Counter
import json
from math import exp
from statistics import median
from time import perf_counter

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from scripts.audit_dspine_complement import capture_mtp
from scripts.bench_dflash2 import pages
from scripts.bench_dspine import (
    DSpineDraft, ROOT, verify_checkpoint, with_packed_ffn, with_packed_projections, with_prompt_vocabulary,
)
from scripts.bench_dspine_adaptive import advance
from scripts.bench_midlayer_tree import batch_layout
from scripts.bench_ready import benchmark_target, prepare_prefix


SEED_LENGTHS = (2, 4)
METHODS = ("terminal_only", "layerwise_codes", "codes_and_embeddings")
WIDTHS = (8, 16)
GATE_MASSES = (0.0, 0.35, 0.55, 0.75)
PRICES = {8: 74.77, 16: 94.16}


def complete(draft, model, pending, prefix, seeds, previous, mode):
    started = perf_counter()
    options, scores, rms = draft.lattice(
        model, pending, prefix, draft.config.block_size,
        extra_senders=previous, seeds=seeds if mode != "terminal_only" else (),
        seed_embeddings=mode == "codes_and_embeddings",
    )
    conditional = scores / mx.maximum(rms[:, None, :], draft.eps)
    logp = conditional - mx.logsumexp(conditional, axis=-1, keepdims=True)
    indices = mx.argmax(conditional, axis=-1)
    best = mx.take_along_axis(logp, indices[..., None], axis=-1)[..., 0]
    mx.eval(options, indices, best)
    choices, picks, confidence = options.tolist(), indices.tolist(), best.tolist()
    result, probabilities = [pending, *seeds], []
    last = -1  # Explicit MTP predecessor in the extra sender column.
    for i in range(len(seeds), len(choices)):
        selected = picks[i][last]
        result.append(choices[i][selected])
        probabilities.append(confidence[i][last])
        last = selected
    return result, probabilities, (perf_counter() - started) * 1000


def graft(nodes, parents, chain, cap):
    """Append a chain where possible, preserving all existing MTP candidates."""
    children = [dict() for _ in nodes]
    for i, (parent, token) in enumerate(zip(parents[1:], nodes[1:], strict=True), 1):
        children[parent][token] = i
    extra = sum(max(0, len(row) - 1) for row in children)
    at = 0
    result = list(nodes)
    edges = list(parents)
    for token in chain[1:]:
        child = children[at].get(token)
        if child is None:
            side = bool(children[at])
            if len(result) >= cap or (side and extra >= 2):
                break
            child = len(result)
            children[at][token] = child
            extra += side
            children.append({})
            edges.append(at)
            result.append(token)
        at = child
    return result, edges


def audit(model, draft, trace, native, prefix, budget):
    """Use the recorded main (unverified) MTP spine as the seed."""
    ids, context = trace["output"], len(trace["input"])
    measured = []
    for offset_str, saved in sorted(trace["rounds"].items(), key=lambda item: int(item[0])):
        offset = int(offset_str)
        start = offset - context
        truth = ids[start + 1:start + 17]
        if not truth:
            continue
        nodes, parents, path = batch_layout(saved)
        fork = next((i for i in range(1, len(parents)) if parents[i] != i - 1), len(parents))
        assert nodes[0] == ids[start] and len(path) == native[start]["advance"]
        local = [tuple(x[:, :, :offset] for x in pair) for pair in prefix]
        existing_width = 16 if len(nodes) > 8 else 8
        for count in SEED_LENGTHS:
            if fork < count + 1 or len(truth) <= count:
                continue
            seeds = tuple(nodes[1:count + 1])
            senders = [nodes[0], *seeds, *([nodes[0]] * (draft.config.block_size - count - 2))]
            previous = mx.array(senders, dtype=mx.int32)[:, None]
            seed_mass = saved["nodes"][saved["order"][count]][3]
            sample = {"offset": start, "seed_length": count, "seed_log_mass": seed_mass,
                      "correct_seed": list(seeds) == truth[:count], "mtp": len(path), "width": existing_width,
                      "nodes": len(nodes)}
            shift = (len(measured) // len(SEED_LENGTHS)) % len(METHODS)
            for mode in METHODS[shift:] + METHODS[:shift]:
                chain, probabilities, draft_ms = complete(draft, model, nodes[0], local, seeds, previous, mode)
                solo = {w: advance(chain[:w], [-1, *range(len(chain[:w]) - 1)], truth, w) for w in WIDTHS}
                merged = {}
                for width in WIDTHS:
                    if width < len(nodes):
                        continue
                    leaves, edges = graft(nodes, parents, chain, width)
                    kept = advance(leaves, edges, truth, len(leaves))
                    assert kept >= len(path)
                    merged[width] = {"advance": kept, "width": 16 if len(leaves) > 8 else 8,
                                     "nodes_added": len(leaves) - len(nodes)}
                sample[mode] = {"ms": draft_ms, "solo": solo, "graft": merged,
                                "first_suffix_hit": len(seeds) < len(truth) and chain[count + 1] == truth[count],
                                "suffix_prior_logp": sum(probabilities[:2])}
            measured.append(sample)
            budget.check()
        if len(measured) and len(measured) % 128 == 0:
            print("anchor_pairs", len(measured), flush=True)
    return measured


def summarize(measured, native):
    token_price = sum(PRICES[16 if v["width"] > 8 else 8] + 13.5 for v in native.values()) / 1023
    results = {}
    for count in SEED_LENGTHS:
        points = [sample for sample in measured if sample["seed_length"] == count]
        correct = [sample for sample in points if sample["correct_seed"]]
        reports = {}
        for mode in METHODS:
            cost = median(row[mode]["ms"] for row in points)
            totals = {
                "draft_median_ms": cost, "first_suffix_hits_on_correct_seeds": sum(
                    row[mode]["first_suffix_hit"] for row in correct),
                "solo_advances_width16_on_correct_seeds": sum(row[mode]["solo"][16] for row in correct),
                "mean_suffix_log_mass": sum(row[mode]["suffix_prior_logp"] for row in points) / len(points),
                "gates": {},
            }
            for probability in GATE_MASSES:
                active = [row for row in points if exp(row["seed_log_mass"]) >= probability]
                choices = Counter()
                for row in active:
                    same = row[mode]["graft"][row["width"]]["advance"] - row["mtp"]
                    wide = row[mode]["graft"][16]
                    more = wide["advance"] - row["mtp"]
                    optimistic_ms = more * token_price - (PRICES[wide["width"]] - PRICES[row["width"]])
                    choices.update(same_extra=same, wide_extra=more, same_gain_rounds=int(same > 0))
                    choices["ideal_grafting_gross_ms"] += max(same * token_price, optimistic_ms, 0)
                skip = sum(row["nodes"] == 16 for row in active)
                totals["gates"][probability] = {
                    "calls": len(active), **choices,
                    "optimistic_net_ms": choices["ideal_grafting_gross_ms"] - cost * len(active),
                    "full_batches_skipped": skip,
                    "optimistic_net_ms_skip_full": choices["ideal_grafting_gross_ms"] - cost * (len(active) - skip),
                }
            reports[mode] = totals
        results[count] = {"anchors": len(points), "correct_mtp_seeds": len(correct), "modes": reports}
    return {"token_price_ms": token_price, "seed_length": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("code", "chinese_technical_report", "english_technical_tutorial"),
                        default="code")
    args = parser.parse_args()
    verify_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[args.workload]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    before, thermal = pages(), thermal_state()
    with metal_job(80, 2) as budget:
        draft = DSpineDraft()
        mx.eval(draft.weights, draft.code_projection, draft.sender)
        draft = with_packed_projections(with_packed_ffn(draft, budget), budget)
        model = benchmark_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=2 * GIB)
        draft = with_prompt_vocabulary(draft, model, trace["input"])
        draft.context_window = (128, 2048)
        native, context = capture_mtp(model, draft, trace, prepare_prefix(model, draft, trace, budget), budget)
        print("mtp_replayed", len(native), "exact_output_tokens", len(trace["output"]), flush=True)
        mx.clear_cache()
        budget.check(force=True)
        mx.reset_peak_memory()
        measured = audit(model, draft, trace, native, context, budget)
        outcome = summarize(measured, native)
        after = pages()
        print("result", json.dumps({
            "workload": args.workload, **outcome, "thermal": [thermal, thermal_state()],
            "peak_gib": mx.get_peak_memory() / GIB,
            "faults": {key: after[key] - before[key] for key in before}, **budget.metrics(),
        }), flush=True)


if __name__ == "__main__":
    main()

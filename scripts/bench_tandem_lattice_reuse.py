"""Reuse a scored Tandem tail, including after a bonus token outside the old shortlist."""

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.server import warm_verification_graphs
from ultrarun.spec_tree import VerificationTree, accepted_path
from scripts.bench_dflash2 import (
    SOURCE_SHA256, bind_draft, load_draft, load_target, observe_layers, pages, prefill, tiled_candidates,
)
from scripts.bench_dflash2_packed import fork_site_allocator, pack_variants, trimmed_output
from scripts.bench_dflash2_tandem import copy_verifier, exercise, verified_directories
from scripts.bench_ready import ROOT, digest
from scripts.dflash2_ranked_head import lattice_for


CAP = 8
MODES = ("fresh7", "stored_predecessor", "rescored_predecessor")


@dataclass
class Lattice:
    options: list
    scores: list
    candidate_ids: mx.array
    unary: mx.array
    context: mx.array

    def root_scores(self, selector, depth, token):
        """Apply the original BF16 Markov operations to one new predecessor."""
        predecessor = selector.predecessor_codebook(mx.array([token], dtype=mx.int32))[0]
        successors = selector.successor_codebook(self.candidate_ids[depth])
        edge = mx.sum(predecessor * self.context[depth] * successors, axis=-1)
        scores = (self.unary[depth] + edge).astype(mx.float32)
        return (scores - mx.logsumexp(scores)).tolist()


def cutoff_from_measurements():
    log = ROOT / "runs/tandem-dflash2-ranked-head-code-ab.log"
    row = next(report for line in log.read_text().splitlines() if line.startswith("result ")
               for report in (json.loads(line[7:]),) if report["arm"] == "mapped_rank")
    phases = row["phase_s"]
    # No future output is available here. Price an extra verifier call by
    # the average cost and advancement measured with the fixed-width backend.
    return row["mean_confirmed"] * (phases["verify"] + phases["commit"]) / sum(phases.values())


def reuse_plan(block, confirmed, selector, width, cutoff, allow_unseen, allocate, stats):
    depth = len(confirmed)
    options = block.options[depth:]
    if min(width, len(options) + 1) < cutoff:
        return None
    bonus = confirmed[-1]
    previous = block.options[depth - 1]
    if bonus in previous:
        first = block.scores[depth][previous.index(bonus)]
        kind = "stored"
    elif allow_unseen:
        started = perf_counter()
        first = block.root_scores(selector, depth, bonus)
        stats["rescored_ms"] += (perf_counter() - started) * 1000
        kind = "rescored"
    else:
        return None
    stats[f"{kind}_considered"] += 1
    plan = allocate(bonus, options, [[first], *block.scores[depth + 1:]], width)
    if plan[-1] < cutoff:
        return None
    return plan[:3], kind


def run(target, draft, upstream, trace, prepared, budget, mode, limit, cutoff, *, verify_row=False):
    initial, first, features = prepared
    cache = [replace(state) for state in initial]
    context = len(trace["input"])
    offset = context
    expected = trace["output"][:limit]
    output = [first]
    draft_cache = draft.make_cache()
    for state in draft_cache:
        state.offset = offset - features.shape[1]
    fresh = lattice_for(target, draft.config.block_size, retain=mode != "fresh7")
    allocate = fork_site_allocator(with_mass=True)
    taps = draft.config.target_layer_ids
    conv = target.config["linear_conv_kernel_dim"] - 1
    remembered, confirmed = None, ()
    checked = [False]
    phases, stats, widths = Counter(), Counter(), Counter()
    while len(output) < limit:
        assert all(state.offset == offset - features.shape[1] for state in draft_cache)
        cap = min(CAP, limit - len(output) + 1)
        started = perf_counter()
        reuse = None
        if remembered is not None:
            assert confirmed[-1] == output[-1]
            reuse = reuse_plan(remembered, confirmed, draft.candidate_selector, cap, cutoff,
                               mode == "rescored_predecessor", allocate, stats)
        if reuse is not None:
            (tokens, parents, spine), kind = reuse
            queued = features
            remembered = None  # At most one skipped forward between cache updates.
        else:
            prepared_block = fresh(draft, upstream, output[-1], features, draft_cache, checked)
            if mode != "fresh7":
                remembered = Lattice(*prepared_block)
                if verify_row:
                    probe = remembered.root_scores(draft.candidate_selector, 1, remembered.options[0][3])
                    difference = max(abs(a - b) for a, b in zip(probe, remembered.scores[1][3], strict=True))
                    assert difference == 0, ("cached candidate row changed", difference)
                    print("shifted_row", json.dumps({"bitwise": True, "selector_rank": draft.config.selector_rank}),
                          flush=True)
                    verify_row = False
            choices, weights = prepared_block[:2]
            tokens, parents, spine, _ = allocate(output[-1], choices, weights, cap)
            assert all(state.offset == offset for state in draft_cache)
            stats["fresh"] += 1
        drafted_at = perf_counter()
        tapped = {}
        observe_layers(target, taps, lambda layer, _offset, hidden: tapped.__setitem__(layer, hidden))
        tree = VerificationTree(parents, offset, conv, spine=spine) if spine is not None else None
        hidden = target.hidden(mx.array(tokens, dtype=mx.int32), cache, offset, keep_replay=True, tree=tree)
        prediction = target.predictions(hidden)
        target.evaluate([prediction, *tapped.values()], cache)
        predicted = prediction.tolist()
        verified_at = perf_counter()
        path = accepted_path(parents, tokens, predicted)[:limit - len(output)]
        keep = len(path)
        confirmed = [tokens[node] for node in path[1:]] + [predicted[path[-1]]]
        assert confirmed == expected[len(output):len(output) + keep], (mode, len(output), confirmed)
        if tree is None:
            target.accept(cache, offset, len(tokens), keep)
        else:
            target.accept_tree(cache, tree, offset, path)
        indices = mx.array(path, dtype=mx.int32)
        features = mx.concatenate([tapped[i][indices] for i in taps], axis=-1)[None]
        if reuse is not None:
            stats[f"{kind}_reused"] += 1
            stats[f"{kind}_confirmed"] += keep
            features = mx.concatenate((queued, features), axis=1)
            stats["max_queued_rows"] = max(stats["max_queued_rows"], features.shape[1])
        target.evaluate(features, cache)
        budget.check()
        offset += keep
        output.extend(confirmed)
        phases.update(draft=drafted_at - started, verify=verified_at - drafted_at, commit=perf_counter() - verified_at)
        widths[len(tokens)] += 1
        yield len(output)
    assert output == expected
    rounds = sum(widths.values())
    assert rounds == stats["fresh"] + stats["stored_reused"] + stats["rescored_reused"]
    return {"mode": mode, "tokens": len(output), "sha256": digest(output), "matches_reference": True,
            "rounds": rounds, "widths": widths, "phases_s": phases, "reuse": stats}


def compare(model, draft, upstream, trace, initial, budget, cutoff):
    variants = {name: trimmed_output(draft, CAP) if name == "fresh7" else draft for name in MODES}

    def streams(limit, *, test_row=False):
        return {name: run(copy_verifier(model), variant, upstream, trace, initial, budget, name, limit, cutoff,
                          verify_row=test_row and name == "rescored_predecessor")
                for name, variant in variants.items()}

    _, elapsed = exercise(streams(65, test_row=True), 65, report_progress=False)
    print("warm", json.dumps({"elapsed_s": elapsed}), flush=True)
    mx.clear_cache()
    budget.check(force=True)
    mx.reset_peak_memory()
    thermal, before = thermal_state(), pages()
    results, seconds = exercise(streams(1024), 1024, report_progress=True)
    for name, result in results.items():
        print("result", json.dumps({**result, "decode_s": seconds[name], "decode_tok_s": 1023 / seconds[name]}),
              flush=True)
    assert len({result["sha256"] for result in results.values()}) == 1
    print("decode_memory", json.dumps({"peak_gib": mx.get_peak_memory() / GIB,
                                       "thermal": [thermal, thermal_state()],
                                       "faults": {k: after - before[k] for k, after in pages().items()},
                                       **budget.metrics()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/private/tmp/localllm-ref/dflash2/model_mlx.py"))
    module = parser.parse_args().upstream
    if sha256(module.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected DFlash2 model source")
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert (len(trace["input"]), len(trace["output"])) == (4096, 1024)
    path = verified_directories(("tandem_b16",))["tandem_b16"]
    before = pages()
    cutoff = cutoff_from_measurements()
    print("reuse_cutoff", json.dumps({"expected_tokens_per_round": cutoff}), flush=True)
    with metal_job(64, 2) as budget:
        model = load_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=8 * GIB)
        stock, upstream = load_draft(module, path)
        assert stock.config.block_size == 16 and stock.config.selector_top_k == 16
        assert stock.config.output_multiplier == 1 and not stock.config.final_logit_softcapping
        upstream.rank_candidates = tiled_candidates
        bind_draft(model, stock, {})
        draft = pack_variants(stock, budget)[1]
        warm_verification_graphs(model, budget)
        original = model.fused_attention_io
        model.fused_attention_io = False
        try:
            initial = prefill(model, stock, trace["input"])
        finally:
            model.fused_attention_io = original
        assert initial[1] == trace["output"][0]
        peak = mx.get_peak_memory() / GIB
        compare(model, draft, upstream, trace, initial, budget, cutoff)
        print("host", json.dumps({"preparation_peak_gib": peak,
                                  "faults": {k: after - before[k] for k, after in pages().items()},
                                  **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

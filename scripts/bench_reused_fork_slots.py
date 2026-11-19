"""Evaluate reusable DeltaNet fork registers with a BF16 Tandem verifier."""

from collections import Counter
from copy import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import FunctionType

import mlx.core as mx

from ultrarun.delta import DeltaRollback, recurrent
from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.server import warm_verification_graphs
from ultrarun.spec_tree import VerificationTree
from ultrarun.tree_schedule import order_for_slots, slot_plan
from scripts.bench_delta_checkpoint import activation_batch
from scripts.bench_dflash2 import SOURCE_SHA256, bind_draft, load_draft, load_target, pages, prefill, tiled_candidates
from scripts.bench_dflash2_packed import fork_site_allocator, pack_variants
from scripts.bench_dflash2_tandem import draft_runner, exercise, reference, streams_at_prefix, verified_directories
from scripts.bench_ready import ROOT
from scripts.dflash2_ranked_head import lattice_for

from verify_bf16_wide_panels import same, snapshot_after, state_parity


WIDTH = 16
FIXED_PARENTS = [-1, *range(9), 7, 5, 3, 2, 1, 0]


def relayout(parents):
    order, required = order_for_slots(parents)
    new_index = {old: new for new, old in enumerate(order)}
    arranged = [-1, *(new_index[parents[old]] for old in order[1:])]
    spine = next(i for i, parent in enumerate(arranged[1:], 1) if parent != i - 1)
    assert slot_plan(arranged)[2] == required
    return order, arranged, spine, required


def count_forks(parents):
    return len({parent for node, parent in enumerate(parents[1:], 1) if parent != node - 1})


def allocator(stats, *, scheduled):
    base = fork_site_allocator()
    full = FunctionType(reference.candidate_tree.__code__, {**vars(reference), "FORK_LIMIT": WIDTH})

    def run(*args):
        stats["rounds"] += 1
        if not scheduled:
            return base(*args)
        tokens, parents, spine = full(*args)
        forks = count_forks(parents)
        if forks <= 2:
            return tokens, parents, spine
        _, _, slots = slot_plan(parents)
        if slots > 2:
            order, parents, spine, slots = relayout(parents)
            tokens = [tokens[i] for i in order]
            if slots > 2:
                stats["fallback"] += 1
                return base(*args)
            stats["reordered"] += 1
        assert forks == count_forks(parents)
        stats["scheduled"] += 1
        stats["maximum_fork_parents"] = max(stats["maximum_fork_parents"], forks)
        stats["maximum_slots"] = max(stats["maximum_slots"], slots)
        return tokens, parents, spine
    return run


def tree_factory(parents, offset, taps, *, spine):
    return VerificationTree(parents, offset, taps, spine=spine, reuse_forks=count_forks(parents) > 2)


def recurrence_check():
    # Six distinct forks, using one slot after reordering, plus a separate
    # tree with two overlapping register lifetimes.
    for source in (FIXED_PARENTS, [-1, 0, 1, 1, 0, 4, 4, 0]):
        _, parents, spine, slots = relayout(source)
        tree = VerificationTree(parents, 4096, 3, spine=spine, reuse_forks=True)
        data = activation_batch(len(parents))
        original = mx.random.normal((48, 128, 128), dtype=mx.float32) / 64
        plain, trajectory = recurrent(*data, original, 4, trace=True, normalize=True, parents=tree.parents)
        output, stored, updates, keys = recurrent(
            *data, original, 4, normalize=True, record_updates=True,
            prepared_route=tree.delta_route, scheduled_tree=True, multi_fork=tree.multi_fork,
        )
        assert same(plain, output) and same(stored, trajectory[tree.checkpoint])
        rollback = DeltaRollback(original, updates, keys, data[3])
        states = [rollback.restore(len(path), mx.array(path, dtype=mx.int32)) for path in tree.paths]
        assert same(mx.stack(states), trajectory)
        print("recurrence", json.dumps({"width": len(parents), "fork_parents": count_forks(parents),
              "live_slots": slots, "checkpoint_and_all_rollback_states_bitwise": True}), flush=True)


def fixed(model, prefix, trace, budget):
    offset = len(trace["input"])
    order, parents, spine, slots = relayout(FIXED_PARENTS)
    raw = trace["output"][:WIDTH]
    gold_ids = mx.array(raw, dtype=mx.int32)
    new_ids = mx.array([raw[i] for i in order], dtype=mx.int32)
    order_on_gpu = mx.array(order, dtype=mx.int32)
    baseline = VerificationTree(FIXED_PARENTS, offset, 3)
    baseline.spine = 10  # Preserve layout for attention while materializing recurrent states.
    scheduled = VerificationTree(parents, offset, 3, spine=spine, reuse_forks=True)
    dense = copy(model)
    dense.layers = [copy(layer) for layer in model.layers]
    for layer in dense.layers:
        layer.compact_tree = False
        layer.compiled_delta = None
    follow = mx.array(trace["output"][WIDTH:WIDTH + 8], dtype=mx.int32)
    accepted_paths = ([*range(6), 11], list(range(baseline.spine)))
    for accepted, deferred in ((path, mode) for path in accepted_paths for mode in (False, True)):
        moved = [order.index(node) for node in accepted]
        assert tuple(moved) == scheduled.paths[moved[-1]]
        through = offset + len(accepted)
        gold = snapshot_after(dense, prefix, gold_ids, offset, single=True, tree=baseline,
                              keep=accepted, defer=deferred)
        candidate = snapshot_after(model, prefix, new_ids, offset, single=True, tree=scheduled,
                                   keep=moved, defer=deferred)
        assert all(same(a[order_on_gpu], b) for a, b in zip(gold[:2], candidate[:2], strict=True))
        if not deferred:
            assert not any(state_parity(gold[2], candidate[2], through=through).values())
        left, right = [snapshot_after(model, branch[2], follow, through) for branch in (gold, candidate)]
        assert all(same(a, b) for a, b in zip(left[:2], right[:2], strict=True))
        assert not any(state_parity(left[2], right[2], through=through + len(follow)).values())
        del gold, candidate, left, right
        budget.check()
    times = [[], []]
    arms = ((dense, gold_ids, baseline), (model, new_ids, scheduled))
    for trial in range(12):
        for arm in (trial % 2, 1 - trial % 2):
            engine, tokens, tree = arms[arm]
            started = perf_counter()
            snapshot_after(engine, prefix, tokens, offset, single=True, tree=tree)
            times[arm].append(1000 * (perf_counter() - started))
        budget.check()
    print("fixed", json.dumps({
        "width": WIDTH, "fork_parents": count_forks(parents), "live_slots": slots,
        "dense_ms": median(times[0]), "reused_ms": median(times[1]),
        "paired_saved_ms": median(a - b for a, b in zip(*times, strict=True)),
        "hidden_prediction_and_immediate_deferred_commit_bitwise": True,
        "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics(),
    }), flush=True)


def live(model, draft, upstream, trace, prepared, budget):
    lattice = lattice_for(model, draft.config.block_size)

    def streams(limit):
        names = ("two_fork_parents", "reused_slots")
        stats = {name: Counter() for name in names}
        arms = {}
        for name in names:
            scheduled = name == "reused_slots"
            runner = draft_runner(draft.config.block_size, lattice=lattice,
                                  allocate=allocator(stats[name], scheduled=scheduled),
                                  tree_factory=tree_factory if scheduled else None)
            arms[name] = draft, upstream, [False], runner
        return streams_at_prefix(model, arms, trace, prepared, dict.fromkeys(names, WIDTH), budget, limit), stats

    warm, _ = exercise(streams(65)[0], 65, report_progress=False)
    assert all(result["matches_reference"] for result in warm.values())
    mx.clear_cache()
    budget.check(force=True)
    mx.reset_peak_memory()
    thermal, before = thermal_state(), pages()
    runners, stats = streams(1024)
    results, elapsed = exercise(runners, 1024, report_progress=True)
    assert stats["reused_slots"]["scheduled"] > 0
    assert len({row["sha256"] for row in results.values()}) == 1
    for name, row in results.items():
        assert row["matches_reference"] and row["rounds"] == stats[name]["rounds"]
        print("result", json.dumps({"arm": name, "decode_tok_s": row["tokens"] / elapsed[name],
                                    "scheduler": stats[name], **row}), flush=True)
    print("decode_host", json.dumps({"peak_gib": mx.get_peak_memory() / GIB,
                                     "thermal": [thermal, thermal_state()],
                                     "faults": {k: val - before[k] for k, val in pages().items()},
                                     **budget.metrics()}), flush=True)


def main():
    source = Path("/private/tmp/localllm-ref/dflash2/model_mlx.py")
    assert sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    draft_path = verified_directories(("tandem_b16",))["tandem_b16"]
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert (len(trace["input"]), len(trace["output"])) == (4096, 1024)
    before = pages()
    with metal_job(64, 2) as budget:
        mx.random.seed(754)
        recurrence_check()
        mx.clear_cache()
        budget.check(force=True)
        model = load_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=10 * GIB)
        stock, upstream = load_draft(source, draft_path)
        assert stock.config.block_size == WIDTH
        upstream.rank_candidates = tiled_candidates
        bind_draft(model, stock, {})
        draft = pack_variants(stock, budget)[1]
        warm_verification_graphs(model, budget)
        layout, layers = model.fused_attention_io, model.layers
        model.layers = list(layers)
        model.fused_attention_io = False
        try:
            prepared = prefill(model, stock, trace["input"])
        finally:
            model.fused_attention_io = layout
            model.layers = layers
        assert prepared[1] == trace["output"][0]
        del stock
        fixed(model, prepared[0], trace, budget)
        print("prepare", json.dumps({"peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
        live(model, draft, upstream, trace, prepared, budget)
        print("host", json.dumps({k: val - before[k] for k, val in pages().items()}), flush=True)


if __name__ == "__main__":
    main()

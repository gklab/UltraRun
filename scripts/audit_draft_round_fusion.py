"""Bound same-round BF16 draft fusion on the saved code continuation."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
WIDTHS = (8, 16)
WORKLOAD = "code"
AFTER_ROOT = 1023
TOP = 16


def log_rows(name, tag):
    prefix = f"{tag} "
    return [json.loads(line[len(prefix):]) for line in (RUNS / name).read_text().splitlines()
            if line.startswith(prefix)]


def costs():
    verify = log_rows("delta-direct-sequence-code.log", "verifier")
    target = {row["width"]: row["base_ms"] for row in verify}
    assert set(target) == set(WIDTHS)
    tandem = next(row for row in log_rows("tandem-dflash2-packed-code-ab.log", "result")
                  if row["arm"] == "packed_projections")
    spine = next(row for row in log_rows("dspine-bf16-code.log", "result")
                 if row["width"] == 16 and row["tree_proposals"])
    floor = json.loads((RUNS / "bf16-100-tps-budget-after-output-pack.json").read_text())

    def timed(result, field):
        return {phase: result[field][phase] * 1000 / result["rounds"] for phase in ("draft", "commit")}

    return target, {"tandem": timed(tandem, "phase_s"), "dspine": timed(spine, "phases_s")}, (
        floor["workloads"][WORKLOAD]["draft_and_commit_ms"]), floor["weights"]["stream_ms"]


def extend(rows, slots, forks, *, optimistic=False, count_parents=False):
    """With known truth, choose the cheapest candidate at each depth.

    In the ranked mode, a rank k requires the winning token and k-1 siblings.
    A missing row or a rank outside the selector always ends this path.
    """
    kept = 0
    for column in range(max(map(len, rows), default=0)):
        rank = min((row[column] for row in rows if column < len(row)), default=TOP + 1)
        if rank > TOP:
            break
        price = 1 if optimistic else rank
        forks_used = int(price > 1) if count_parents else price - 1
        if price > slots or forks_used > forks:
            break
        slots -= price
        forks -= forks_used
        kept += 1
    return kept


def least_time(actions):
    """A verifier may intentionally take any nonempty prefix of its verified path."""
    count = len(actions)
    elapsed = [0.0] + [float("inf")] * count
    path = [None] * (count + 1)
    for start, choices in enumerate(actions):
        for label, advance, milliseconds in choices:
            price = elapsed[start] + milliseconds
            for stop in range(start + 1, min(start + advance, count) + 1):
                if price < elapsed[stop]:
                    elapsed[stop] = price
                    path[stop] = start, label
    turns = Counter()
    stop = count
    while stop:
        start, label = path[stop]
        turns[label] += 1
        stop = start
    return {"rounds": sum(turns.values()), "ms": elapsed[-1], "tok_s": count * 1000 / elapsed[-1],
            "routes": turns}


def recorded_rounds(trace):
    """Join only archives with the identical emitted tokens and verifier route."""
    count, context = len(trace["output"]) - 1, len(trace["input"])
    archive = json.loads((RUNS / "dspine-mtp-complement.json").read_text())
    assert archive["result"]["workload"] == WORKLOAD and len(archive["anchors"]) == count == AFTER_ROOT
    native = {int(offset): row for offset, row in archive["native"].items()}
    anchor = 0
    for offset, record in sorted(trace["rounds"].items(), key=lambda pair: int(pair[0])):
        assert int(offset) == anchor + context
        nodes, order, path = (record[key] for key in ("nodes", "order", "path"))
        assert [nodes[i][1] for i in path] == trace["output"][anchor:anchor + len(path)]
        assert native[anchor]["advance"] == len(path) and native[anchor]["width"] == len(order)
        children = Counter(nodes[i][0] for i in order[1:])
        native[anchor]["free_forks"] = max(0, 2 - sum(n - 1 for n in children.values()))
        native[anchor]["free_parents"] = max(0, 2 - sum(n > 1 for n in children.values()))
        native[anchor]["graft_allowed"] = children[path[-1]] == 0
        anchor += len(path)
    assert anchor == count and len(native) == len(trace["rounds"])
    return native, [row["ranks"] for row in archive["anchors"]]


def replay(native, candidates):
    verify, draft, mtp, read = costs()
    singles = (("tandem",), ("dspine",))
    joined = (*singles, ("tandem", "dspine"))
    scenarios = (
        ("mtp_only", (), False, False, "total", "edges"),
        ("one_draft_per_round", singles, False, False, "total", "edges"),
        ("joined_ranked", joined, True, False, "total", "edges"),
        ("joined_ranked_two_parents", joined, True, False, "total", "parents"),
        ("joined_ranked_unlimited_forks", joined, True, False, "total", "unlimited"),
        ("joined_ranked_unlimited_forks_free_draft", joined, True, False, "verify", "unlimited"),
        ("joined_top16_free_siblings", joined, True, True, "total", "edges"),
        ("joined_top16_free_draft", joined, True, True, "verify", "edges"),
        ("joined_top16_read_only", joined, True, True, "read", "edges"),
    )
    results = {}
    for name, groups, graft, optimistic, charge, fork_type in scenarios:
        count_parents = fork_type == "parents"
        fork_limit = max(WIDTHS) if fork_type == "unlimited" else 2
        actions = []
        for start in range(AFTER_ROOT):
            choices = []

            def add(label, width, advance, overhead):
                ms = read if charge == "read" else verify[width] + (overhead if charge == "total" else 0.0)
                choices.append((f"{label}:{width}", advance, ms))

            base = native.get(start)
            if base:
                band = next(width for width in WIDTHS if width >= base["width"])
                add("mtp", band, base["advance"], mtp)
            for group in groups:
                label = "+".join(group)
                rows = [candidates[source][start] for source in group]
                drafting = sum(draft[source]["draft"] for source in group)
                commit = min(draft[source]["commit"] for source in group)
                for width in WIDTHS:
                    kept = extend(rows, width - 1, fork_limit, optimistic=optimistic, count_parents=count_parents)
                    add(label, width, kept + 1, drafting + commit)
                    if graft and base and base["graft_allowed"] and base["width"] < width:
                        depth = base["advance"] - 1
                        slots = width - base["width"]
                        free_forks = {"edges": base["free_forks"], "parents": base["free_parents"],
                                      "unlimited": fork_limit}[fork_type]
                        extra = extend([row[depth:] for row in rows], slots, free_forks, optimistic=optimistic,
                                       count_parents=count_parents)
                        if extra:
                            add(f"mtp+{label}", width, base["advance"] + extra, mtp + drafting)
            actions.append(choices)
        results[name] = least_time(actions)
    return {"verifier_ms": verify, "draft_and_commit_ms": draft, "mtp_ms": mtp, "read_ms": read,
            "estimated": results}


def profile_tandem(trace):
    import mlx.core as mx

    from ultrarun.host import thermal_state
    from ultrarun.memory import GIB, metal_job
    from ultrarun.server import warm_verification_graphs
    from scripts.audit_dflash2_branches import conditional_ranks
    from scripts.audit_dspine_complement import capture_target_suffix
    from scripts.bench_dflash2 import bind_draft, load_draft, load_target, pages, prefill, tiled_candidates
    from scripts.bench_dflash2_packed import pack_variants
    from scripts.bench_dflash2_tandem import verified_directories

    path = verified_directories(("tandem_b16",))["tandem_b16"]
    upstream = Path("/private/tmp/localllm-ref/dflash2/model_mlx.py")
    before, thermal = pages(), thermal_state()
    with metal_job(80, 2) as budget:
        target = load_target(budget)
        budget.admit(target, 5120, 512, False, extra_bytes=8 * GIB)
        draft, module = load_draft(upstream, path)
        assert draft.config.block_size == 16 and draft.config.selector_top_k == TOP
        module.rank_candidates = tiled_candidates
        bind_draft(target, draft, {})
        draft = pack_variants(draft, budget)[1]
        warm_verification_graphs(target, budget)
        cache, first, prompt = prefill(target, draft, trace["input"])
        teacher_start = perf_counter()
        observed, hidden = capture_target_suffix(target, draft.config.target_layer_ids, trace, cache, first, budget)
        print("teacher", json.dumps({"rounds": len(observed), "replayed_tokens": len(trace["output"]),
                                     "captured_rows": hidden.shape[1],
                                     "seconds": perf_counter() - teacher_start}), flush=True)
        del cache
        mx.clear_cache()
        states = draft.make_cache()
        for state in states:
            state.offset = len(trace["input"]) - prompt.shape[1]
        ranks = []
        width = draft.config.block_size
        expected = trace["output"]
        for anchor in range(len(expected) - 1):
            context = prompt if anchor == 0 else hidden[:, anchor - 1:anchor]
            offset = len(trace["input"]) + anchor
            assert all(state.offset == offset - context.shape[1] for state in states)
            inputs = mx.array([[expected[anchor]] + [draft.config.mask_token_id] * (width - 1)], dtype=mx.int32)
            prediction = draft.hidden_states(inputs, context, states, logits_start=1)
            logits = draft.compute_logits(prediction)
            options = module.rank_candidates(logits, TOP)
            truth = expected[anchor:anchor + width]
            padded = mx.array(truth + [0] * (width - len(truth)), dtype=mx.int32)
            step = conditional_ranks(draft.candidate_selector, prediction, logits, options, padded)
            ranks.append(step[0, :len(truth) - 1].tolist())
            budget.check()
            if len(ranks) % 256 == 0:
                print("draft_anchors", len(ranks), flush=True)
        assert len(ranks) == AFTER_ROOT
        after = pages()
        return ranks, {"peak_gib": mx.get_peak_memory() / GIB, "thermal": [thermal, thermal_state()],
                       "faults": {key: after[key] - before[key] for key in before}, **budget.metrics()}


def native_fork_room(trace):
    """Count stored first misses extra siblings could cover without changing the width band."""
    measures = Counter()
    for offset, row in trace["rounds"].items():
        nodes, order, path = (row[key] for key in ("nodes", "order", "path"))
        measures["rounds"] += 1
        terminal = path[-1]
        nxt = int(offset) - len(trace["input"]) + len(path)
        if nxt >= len(trace["output"]):
            continue
        target = trace["output"][nxt]
        assert target == row["prediction"][order.index(terminal)]
        children = Counter(nodes[i][0] for i in order[1:])
        extra = sum(n - 1 for n in children.values())
        forks = sum(n > 1 for n in children.values())
        if extra < 2 or not children[terminal] or (children[terminal] == 1 and forks >= 2):
            continue
        ranked = [token for token, _ in row["seen"].get(str(terminal), ())]
        cap = WIDTHS[0] if len(order) <= WIDTHS[0] else WIDTHS[-1]
        measures["newly_eligible"] += 1
        for top in (4, 8):
            if target in ranked[:top]:
                measures[f"top{top}"] += 1
                if len(order) < cap:
                    measures[f"top{top}_room"] += 1
    return measures


def tandem_reuse_bound(trace, archive):
    """Price a single stale tail after a fresh block on known conditional ranks."""
    assert archive["input"] == trace["input"] and archive["output"] == trace["output"]
    rows = archive["tandem"]
    count = len(rows)
    full = next(row for row in log_rows("tandem-dflash2-ranked-head-code-ab.log", "result")
                if row["arm"] == "full_output")
    phase = full["phase_s"]
    fresh_ms = 1000 * sum(phase.values()) / full["rounds"]
    stale_ms = 1000 * (phase["verify"] + phase["commit"]) / full["rounds"]
    outcomes = {}
    for free_siblings in (False, True):
        for policy in ("fresh", "stored", "rescored"):
            elapsed = [0.0] + [float("inf")] * count
            previous = [None] * (count + 1)
            for start, ranks in enumerate(rows):
                latest = 1 + extend([ranks], 7, 2, optimistic=free_siblings, count_parents=True)
                billed = elapsed[start] + fresh_ms
                for fresh in range(1, min(count - start, latest) + 1):
                    mid = start + fresh
                    if billed < elapsed[mid]:
                        elapsed[mid] = billed
                        previous[mid] = (start, "fresh")
                    if policy == "fresh" or fresh >= len(ranks):
                        continue
                    outside = ranks[fresh - 1] > TOP
                    if outside and policy == "stored":
                        continue
                    extra = extend([ranks[fresh:]], 7, 2, optimistic=free_siblings, count_parents=True)
                    for end in range(mid + 1, min(count, mid + extra + 1) + 1):
                        if billed + stale_ms < elapsed[end]:
                            elapsed[end] = billed + stale_ms
                            previous[end] = (start, "reuse_outside" if outside else "reuse_seen")
            end, route = count, Counter()
            while end:
                end, label = previous[end]
                route[label] += 1
            band = "free_siblings" if free_siblings else "rank_paid"
            outcomes[f"{band}_{policy}"] = {"tok_s": count * 1000 / elapsed[-1], "routes": route}
    return {"anchors": count, "fresh_ms": fresh_ms, "reuse_ms": stale_ms, "oracle": outcomes}


def overlap_rounds(ranks, width, history, *, free_siblings=False, rescore=False):
    """Choose round boundaries using only blocks drafted at those boundaries."""
    count = len(ranks)
    assert all(len(row) == min(15, count - at) for at, row in enumerate(ranks))
    # Two different boundaries at the same position can have different tail
    # candidates. Keep their costs separately until the old blocks expire.
    pending = [{} for _ in range(count + 1)]
    pending[0][()] = 0
    for start, states in enumerate(pending[:-1]):
        for saved, rounds in states.items():
            aligned = [ranks[start][:width - 1]]
            for past in saved:
                past_ranks = ranks[past]
                advance = start - past
                aligned.append([
                    rank if rescore or past_ranks[depth - 1] <= TOP else TOP + 1
                    for depth, rank in enumerate(past_ranks[advance:advance + width - 1], advance)
                ])
            reach = 1 + extend(aligned, width - 1, 2, optimistic=free_siblings, count_parents=True)
            retained = (start, *saved)[:history]
            for stop in range(start + 1, min(count, start + reach) + 1):
                remaining = tuple(at for at in retained if stop - at < len(ranks[at]))
                current = pending[stop].get(remaining, count + 1)
                if rounds + 1 < current:
                    pending[stop][remaining] = rounds + 1
        states.clear()
    return min(pending[-1].values())


def overlap_audit(traces, archive):
    code = traces[WORKLOAD]
    assert archive["input"] == code["input"] and archive["output"] == code["output"]
    original = json.loads((RUNS / "dflash2-bf16-branch-ranks.json").read_text())
    assert original["case"] == "chinese_technical_report" and not original["teacher_forced_differences"]
    for workload, ranks in ((WORKLOAD, archive["tandem"]), (original["case"], original["ranks"])):
        assert (len(traces[workload]["input"]), len(traces[workload]["output"]), len(ranks)) == (4096, 1024, 1023)
        for width in WIDTHS:
            for free_siblings in (False, True):
                counts = {"fresh": overlap_rounds(ranks, width, 0, free_siblings=free_siblings)}
                for history in (1, 2):
                    for rescore in (False, True):
                        method = "rescored" if rescore else "stored"
                        counts[f"{history}_previous_{method}"] = overlap_rounds(
                            ranks, width, history, free_siblings=free_siblings, rescore=rescore)
                assert all(rounds <= counts["fresh"] for rounds in counts.values())
                print("overlap_bound", json.dumps({
                    "workload": workload, "width": width, "free_siblings": free_siblings,
                    "minimum_rounds": counts,
                    "round_ms_needed_at_100": {key: len(ranks) * 10 / value for key, value in counts.items()},
                }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--reuse", action="store_true", help="reanalyze captured draft ranks on CPU")
    modes.add_argument("--native-fork-room", action="store_true", help="count old MTP misses with room for one sibling")
    modes.add_argument("--tandem-reuse-bound", action="store_true", help="price Tandem lattice reuse on stored ranks")
    modes.add_argument("--anchor-overlap", action="store_true", help="bound fusing previous and fresh score rows")
    args = parser.parse_args()
    path = RUNS / "bf16-draft-round-fusion.json"
    traces = json.loads((RUNS / "frontier-target-margin-audit.json").read_text())
    if args.native_fork_room:
        print(json.dumps({name: native_fork_room(trace) for name, trace in traces.items()}), flush=True)
        return
    trace = traces[WORKLOAD]
    if args.tandem_reuse_bound:
        print(json.dumps(tandem_reuse_bound(trace, json.loads(path.read_text()))), flush=True)
        return
    if args.anchor_overlap:
        overlap_audit(traces, json.loads(path.read_text()))
        return
    assert (len(trace["input"]), len(trace["output"])) == (4096, 1024)
    native, dspine = recorded_rounds(trace)
    if args.reuse:
        stored = json.loads(path.read_text())
        assert stored["input"] == trace["input"] and stored["output"] == trace["output"]
        tandem, memory = stored["tandem"], stored["memory"]
    else:
        tandem, memory = profile_tandem(trace)
    analysis = replay(native, {"tandem": tandem, "dspine": dspine})
    print("result", json.dumps(analysis), flush=True)
    print("memory", json.dumps(memory), flush=True)
    if not args.reuse:
        path.write_text(json.dumps({"input": trace["input"], "output": trace["output"], "tandem": tandem,
                                   "memory": memory, "analysis": analysis}) + "\n")


if __name__ == "__main__":
    main()

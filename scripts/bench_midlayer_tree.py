"""Measure oracle removal of speculative branches partway through BF16 verification."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import median
from time import perf_counter


ROOT = Path(__file__).resolve().parent.parent
WIDTH = 8
CUTS = (16, 32, 48)


def trace_audit(traces):
    for workload, trace in traces.items():
        rounds = list(trace["rounds"].values())
        wide = [r for r in rounds if len(r["order"]) > WIDTH]
        eligible = [r for r in wide if len(r["path"]) <= WIDTH]
        # Give every eligible round the full measured 16-to-8 cost difference.
        # Assume proportional savings after layer 32 and free branch selection.
        half_gap_ms = (94.16 - 74.77) / 2
        saved = half_gap_ms * len(eligible) / 1000
        target = sum(r["target_s"] for r in rounds)
        print("trace", json.dumps({
            "workload": workload, "rounds": len(rounds), "wide_rounds": len(wide),
            "preserves_all_confirmed_with_oracle_at_width8": len(eligible),
            "lost_if_all_wide_rounds_are_capped": sum(max(0, len(r["path"]) - WIDTH) for r in wide),
            "modeled_midpoint_saving_s": saved, "fraction_of_recorded_target_s": saved / target,
        }), flush=True)


def batch_layout(record):
    nodes, order = record["nodes"], record["order"]
    location = {old: new for new, old in enumerate(order)}
    parents = [-1, *(location[nodes[i][0]] for i in order[1:])]
    path = [location[i] for i in record["path"]]
    return [nodes[i][1] for i in order], parents, path


def shortlist(parents, path):
    assert len(path) <= WIDTH
    keep = set(path)
    # Start from the already confirmed chain and fill using original verification order.
    for i, parent in enumerate(parents[1:], 1):
        if len(keep) == WIDTH:
            break
        if parent in keep:
            keep.add(i)
    assert len(keep) == WIDTH
    rows = sorted(keep)
    mapped = {old: new for new, old in enumerate(rows)}
    compact = [-1, *(mapped[parents[i]] for i in rows[1:])]
    return rows, compact, [mapped[i] for i in path]


def gpu_probe(traces):
    import mlx.core as mx

    from localllm.host import thermal_state
    from localllm.memory import GIB, metal_job
    from localllm.model import VERIFY_SUBMISSION_LAYERS
    from localllm.spec_tree import VerificationTree, accepted_path
    from scripts.bench_dflash2 import pages
    from scripts.bench_ready import benchmark_target
    from scripts.bench_verify_buffer_limits import partition_prompt

    def layout(parents, offset, conv):
        fork = next((i for i in range(1, len(parents)) if parents[i] != i - 1), None)
        return VerificationTree(parents, offset, conv, spine=fork) if fork is not None else None

    def forward(model, states, hidden, offset, tree, first, last):
        for i in range(first, last):
            hidden = model.layers[i].forward(hidden, states[i], offset, True, tree)
            stop = i + 1
            if model.async_verify and stop % VERIFY_SUBMISSION_LAYERS == 0 and stop < len(states):
                model.evaluate(hidden, states[stop - VERIFY_SUBMISSION_LAYERS:stop], asynchronous=True)
        return hidden

    def commit(model, states, tree, offset, width, path, start, stop):
        if tree is None:
            assert path == list(range(len(path)))
            for i in range(start, stop):
                model.layers[i].accept(states[i], offset, width, len(path))
        else:
            indices = mx.array(path, dtype=mx.int32)
            for i in range(start, stop):
                model.layers[i].accept_tree(states[i], tree, offset, path, indices)

    def bitwise(a, b):
        if a is None or b is None:
            return a is b
        dtype = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32 if a.dtype == mx.float32 else a.dtype
        return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a.view(dtype), b.view(dtype)).item()

    def used_kv(state, field, through):
        array = getattr(state, field)
        if array is not None and state.prompt_kv is not None:
            return array[:, :, :through - state.prompt_kv[0].shape[2]]
        return array

    def mismatches(a, b, through):
        return {field: [i for i, (left, right) in enumerate(zip(a, b, strict=True))
                        if not bitwise(used_kv(left, field, through) if field in ("keys", "values") else getattr(left, field),
                                       used_kv(right, field, through) if field in ("keys", "values") else getattr(right, field))]
                for field in ("keys", "values", "delta", "conv")}

    def measure(model, snapshot, trace, offset, record, conv, budget):
        batch, parents, path = batch_layout(record)
        chosen, narrow, mapped_path = shortlist(parents, path)
        full_tree = layout(parents, offset, conv)
        thin_tree = layout(narrow, offset, conv)
        rows = mx.array(chosen, dtype=mx.int32)
        tokens = mx.array(batch, dtype=mx.int32)
        through = offset + len(path)
        output_offset = through - len(trace["input"])
        continuation = mx.array(trace["output"][output_offset:output_offset + 8], dtype=mx.int32)
        layers = len(model.layers)

        def run(cut, *, synchronize=False, keep=False):
            states = [replace(s) for s in snapshot]
            if cut == layers:
                x = model.hidden(tokens, states, offset, keep_replay=True, tree=full_tree)
            else:
                x = forward(model, states, model.embedding[tokens], offset, full_tree, 0, cut)
                if synchronize:
                    model.evaluate(x, states[:cut])
                x = forward(model, states, x[rows], offset, thin_tree, cut, layers)
            y = model.predictions(x)
            model.evaluate([x, y], states)
            if keep:
                commit(model, states, full_tree, offset, len(batch), path, 0, cut)
                if cut < layers:
                    commit(model, states, thin_tree, offset, len(chosen), mapped_path, cut, layers)
                model.evaluate([x, y], states)
            return x, y, states

        x, y, states = run(layers, keep=True)
        assert y.tolist() == record["prediction"]
        following = model.hidden(continuation, states, through, keep_replay=True)
        model.evaluate(following, states)
        previous = x[rows], y[rows]
        for depth in CUTS:
            a, b, narrowed = run(depth, keep=True)
            assert bitwise(previous[0], a) and bitwise(previous[1], b), (offset, depth, "retained rows")
            c = model.hidden(continuation, narrowed, through, keep_replay=True)
            model.evaluate(c, narrowed)
            mismatch = mismatches(states, narrowed, through + continuation.shape[0])
            assert bitwise(following, c) and not any(mismatch.values()), (offset, depth, mismatch)
        del states, narrowed
        arms = [(layers, False), *((cut, False) for cut in CUTS), (32, True)]
        times = {key: [] for key in arms}
        for i in range(10):
            shift = i % len(arms)
            for depth, synchronize in arms[shift:] + arms[:shift]:
                begin = perf_counter()
                run(depth, synchronize=synchronize)
                times[depth, synchronize].append(1000 * (perf_counter() - begin))
            budget.check()
        print("fixed", json.dumps({
            "offset": offset, "wide": len(batch), "narrow": len(chosen), "accepted": len(path),
            "side_commit": path != list(range(len(path))),
            "hidden_prediction_and_next_step_bitwise": True,
            "median_ms": {f"{key[0]}{'_sync' if key[1] else ''}": median(values) for key, values in times.items()},
            "paired_delta_ms": {
                f"{key[0]}{'_sync' if key[1] else ''}": median(a - b for a, b in zip(values, times[layers, False], strict=True))
                for key, values in times.items() if key != (layers, False)
            },
            "thermal": thermal_state(), "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics(),
        }), flush=True)

    selected = {}
    trace = traces["code"]
    records = [(int(offset), r) for offset, r in sorted(trace["rounds"].items(), key=lambda kv: int(kv[0]))]
    for offset, r in records:
        if len(r["order"]) != 16 or len(r["path"]) > WIDTH:
            continue
        _, _, path = batch_layout(r)
        category = path != list(range(len(path)))
        selected.setdefault(category, offset)
        if len(selected) == 2:
            break
    assert len(selected) == 2
    before = pages()
    with metal_job(80, 2) as budget:
        target = benchmark_target(budget)
        budget.admit(target, 5120, 512, False, extra_bytes=2 * GIB)
        cache = target.cache()
        for _, h in target.prefill(mx.array(trace["input"], dtype=mx.int32), cache, 512):
            target.evaluate(h, cache)
            budget.check()
        assert mx.argmax(target.logits(h[-1:]), axis=-1).item() == trace["output"][0]
        partition_prompt(target, cache, len(trace["input"]))
        conv = target.config["linear_conv_kernel_dim"] - 1
        print("prefill", json.dumps({"peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
        for offset, record in records:
            if offset in selected.values():
                measure(target, cache, trace, offset, record, conv, budget)
            if offset >= max(selected.values()):
                break
            batch, parents, path = batch_layout(record)
            tree = layout(parents, offset, conv)
            hidden = target.hidden(mx.array(batch, dtype=mx.int32), cache, offset, keep_replay=True, tree=tree)
            predictions = target.predictions(hidden)
            target.evaluate(predictions, cache)
            assert accepted_path(parents, batch, predictions.tolist()) == path
            assert predictions.tolist() == record["prediction"]
            commit(target, cache, tree, offset, len(batch), path, 0, len(target.layers))
            target.evaluate(predictions, cache)
            budget.check()
        after = pages()
        print("host", json.dumps({name: after[name] - total for name, total in before.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    traces = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())
    trace_audit(traces)
    if not args.trace_only:
        gpu_probe(traces)


if __name__ == "__main__":
    main()

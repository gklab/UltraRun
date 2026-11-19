"""Measure reusing the dense DeltaNet state slot for an earlier verified node."""

import argparse
from collections import Counter
import json
from math import exp
from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx

from localllm.delta import DeltaRollback, recurrent
from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from localllm.spec_tree import VerificationTree
from scripts.bench_dflash2 import pages


LAYERS = 48
CASES = (
    (8, None, (0,)),
    (8, None, (0, 1)),
    (8, None, (0, 1, 2, 3)),
    (8, None, tuple(range(8))),
    (8, ([-1, 0, 1, 2, 3, 1, 5, 4], 5), (0, 1, 5, 6)),
    (16, ([-1, *range(10), 1, 11, 4, 13, 14], 11), (0, 1, 11, 12)),
    (16, ([-1, *range(10), 1, 11, 4, 13, 14], 11), (0, 1, 2, 3, 4, 13, 14, 15)),
)


def activation_batch(width):
    q = mx.random.normal((width, 16, 128), dtype=mx.bfloat16)
    k = mx.random.normal(q.shape, dtype=mx.bfloat16)
    v = mx.random.normal((width, 48, 128), dtype=mx.bfloat16)
    decay = mx.random.uniform(low=0.70, high=1.0, shape=(width, 48))
    beta = mx.random.uniform(low=0.05, high=0.80, shape=decay.shape)
    return q, k, v, decay, beta


def equal(a, b):
    bits = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return mx.array_equal(a.view(bits), b.view(bits)).item()


def round_inputs(width, layout):
    if layout is None:
        return {}, width - 1
    parents, spine = layout
    tree = VerificationTree(parents, 4096, 3, spine=spine)
    return {"prepared_route": tree.delta_route, "multi_fork": tree.multi_fork}, spine - 1


def verify_and_commit(data, state, route, node, path, replay_path, capture):
    output, chosen, updates, keys = recurrent(
        *data, state, 4, normalize=True, record_updates=True, checkpoint=capture, **route,
    )
    if node == path[-1]:
        return output, chosen, chosen, updates, keys
    previous = chosen if node in path else state
    count = len(path) - (path.index(node) + 1 if node in path else 0)
    committed = DeltaRollback(previous, updates, keys, data[3]).restore(count, replay_path)
    return output, committed, chosen, updates, keys


def remaining(path, node, linear):
    start = path.index(node) + 1 if node in path else 0
    return None if linear and start == 0 else mx.array(path[start:], dtype=mx.int32)


def prepare_round(data, route, path, node, *, checkpoint=False):
    suffix = remaining(path, node, not route)
    capture = mx.array([node], dtype=mx.int32) if checkpoint else None
    mx.eval(*(a for a in (route.get("prepared_route"), suffix, capture) if a is not None))

    def run(state):
        return verify_and_commit(data, state, route, node, path, suffix, capture)

    return run


def run_layers(states, run):
    readings = []
    for state in states:
        output, committed, chosen, _, _ = run(state)
        readings.extend((output, committed, chosen))
    mx.eval(readings)


def compare(width, layout, path, states):
    data = activation_batch(width)
    mx.eval(data)
    route, terminal = round_inputs(width, layout)
    saved = min(1, terminal)
    baseline = prepare_round(data, route, path, terminal)
    checkpoint = prepare_round(data, route, path, saved, checkpoint=True)

    base = baseline(states[0])
    test = checkpoint(states[0])
    assert all(equal(base[i], test[i]) for i in (0, 1, 3, 4)), (width, path)
    parents = None if layout is None else mx.array(layout[0], dtype=mx.int32)
    _, trajectory = recurrent(*data, states[0], 4, trace=True, parents=parents, normalize=True)
    assert equal(test[2], trajectory[saved]) and equal(test[1], trajectory[path[-1]]), (width, path)
    next_input = activation_batch(8)
    base_next = recurrent(*next_input, base[1], 4, normalize=True)
    test_next = recurrent(*next_input, test[1], 4, normalize=True)
    assert all(equal(a, b) for a, b in zip(base_next, test_next, strict=True)), (width, path)
    del trajectory, base_next, test_next

    run_layers(states, baseline)
    run_layers(states, checkpoint)
    times = []
    for repeat in range(12):
        pair = [0.0, 0.0]
        for arm in ((1, 0) if repeat % 2 else (0, 1)):
            start = perf_counter()
            run_layers(states, (baseline, checkpoint)[arm])
            pair[arm] = 1000 * (perf_counter() - start)
        times.append(pair)
    print("case", json.dumps({"width": width, "path": path, "spine_checkpoint": terminal,
          "early_checkpoint": saved, "bit_exact": True, "layers": len(states),
          "baseline_ms": median(t[0] for t in times), "candidate_ms": median(t[1] for t in times),
          "paired_gain_ms": median(a - b for a, b in times)}), flush=True)


def replay_trace(states, budget):
    file = Path(__file__).resolve().parents[1] / "runs/frontier-target-margin-audit.json"
    cases = json.loads(file.read_text())
    data_by_width = {}
    for workload, trace in cases.items():
        elapsed_ms, restore_calls, replay_steps = Counter(), Counter(), Counter()
        rounds = 0
        for record in trace["rounds"].values():
            nodes, order = record["nodes"], record["order"]
            width = len(order)
            if width < 2:
                continue
            where = {node: i for i, node in enumerate(order)}
            parents = [-1, *(where[nodes[i][0]] for i in order[1:])]
            path = [where[node] for node in record["path"]]
            assert path[0] == 0 and all(parents[b] == a for a, b in zip(path, path[1:]))
            spine = next((i for i in range(1, width) if parents[i] != i - 1), width)
            route, terminal = round_inputs(width, None if spine == width else (parents, spine))

            reach = [exp(nodes[i][3]) for i in order]
            stop = reach.copy()
            for i, parent in enumerate(parents[1:], 1):
                stop[parent] -= reach[i]
            likely = max(range(width), key=lambda i: 4 * max(0, stop[i]) + reach[i] * (1 + nodes[order[i]][2]))
            choices = {"baseline": terminal, "second_node": min(1, terminal), "probability": likely}
            if width not in data_by_width:
                data_by_width[width] = activation_batch(width)
                mx.eval(data_by_width[width])
            arms = {name: prepare_round(data_by_width[width], route, path, node, checkpoint=node != terminal)
                    for name, node in choices.items()}
            original = arms["baseline"](states[0])
            for name, node in choices.items():
                steps = len(path) - (path.index(node) + 1 if node in path else 0)
                restore_calls[name] += steps > 0
                replay_steps[name] += steps
                alternative = arms[name](states[0])
                assert all(equal(original[i], alternative[i]) for i in (0, 1)), (workload, width, path, node)
            # Alternate the warm and measured arm order to limit time drift.
            names = tuple(arms)
            names = names[rounds % 3:] + names[:rounds % 3]
            for name in reversed(names):
                run_layers(states, arms[name])
            for name in names:
                start = perf_counter()
                run_layers(states, arms[name])
                elapsed_ms[name] += 1000 * (perf_counter() - start)
            rounds += 1
            budget.check()
        print("replay", json.dumps({"workload": workload, "rounds": rounds, "layers": len(states),
              "elapsed_ms": elapsed_ms, "restore_rounds": restore_calls,
              "steps_to_restore": replay_steps, "state_and_output_exact": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="use recorded MTP paths and synthetic recurrence inputs")
    args = parser.parse_args()
    mx.random.seed(4927)
    before = pages()
    thermal = thermal_state()
    with metal_job(4, 1) as budget:
        states = [mx.random.normal((48, 128, 128)) * 0.01 for _ in range(LAYERS)]
        mx.eval(states)
        if args.replay:
            replay_trace(states, budget)
        else:
            for case in CASES:
                compare(*case, states)
                budget.check(force=True)
        after = pages()
        print("host", json.dumps({"peak_gib": mx.get_peak_memory() / GIB,
              "thermal": [thermal, thermal_state()],
              "faults": {k: after[k] - before[k] for k in before}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()

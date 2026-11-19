"""Price wide MTP trees that add continuations copied from the current request."""

import argparse
from collections import Counter, defaultdict, deque
import heapq
import json
import math
from pathlib import Path
from statistics import median

from localllm.phrase_history import PhraseHistory


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "runs/frontier-target-margin-audit.json"
NARROW = 8
HOT_PRICE = 1.45


class RecentWidthPrice:
    """Estimate an eligible phrase's width cost from completed rounds of this request."""

    def __init__(self):
        self.recent = deque(maxlen=36)

    @staticmethod
    def band(width):
        return 8 if width == 8 else 12 if 8 < width <= 12 else 16 if 12 < width <= 16 else None

    def record(self, width, target_seconds, round_seconds):
        band = self.band(width)
        if band is None or not 0 < target_seconds <= round_seconds:
            self.recent.append(None)
            return
        self.recent.append((band, target_seconds, round_seconds - target_seconds))

    def __call__(self, width):
        band = self.band(width)
        assert band in (12, 16)
        narrow, wide, overhead = [], [], []
        for sample in self.recent:
            if sample is None:
                continue
            kind, target, other = sample
            overhead.append(other)
            if kind == NARROW:
                narrow.append(target)
            elif kind == band:
                wide.append(target)
        if len(narrow) < 6 or len(wide) < 3:
            return None
        rest = median(overhead)
        return min(1.65, max(1.0, (median(wide) + rest) / (median(narrow) + rest)))


def sparse_tree(parents, reach, width):
    """Select the most likely prefix-closed subset, with at most width nodes."""
    children = [[] for _ in parents]
    for index, parent in enumerate(parents[1:], 1):
        children[parent].append(index)
    heap = [(0.0, 0)]
    selected = []
    while heap and len(selected) < width:
        _, node = heapq.heappop(heap)
        selected.append(node)
        for child in children[node]:
            heapq.heappush(heap, (-reach[child], child))
    return sorted(selected)


def phrase_features(nodes, native, history, ranked):
    parent, token, _, _ = nodes[native]
    path = []
    while parent:
        path.append(nodes[parent][1])
        parent = nodes[parent][0]
    source = history.continuation(tuple(reversed(path)))
    assert source is not None and source[0][0] == token
    _, suffix, support = source
    return int(suffix >= 6), int(support > 1), int(token in ranked.get(nodes[native][0], ()))


def observe_phrase(nodes, native, path, cell, table):
    verified = set(path)
    for i in range(native, len(nodes)):
        if nodes[i][0] in verified:
            key = cell if i == native else "tail"
            table[key][int(i in verified)] += 1


def phrase_probability(counts, prior):
    return (counts[1] + 12 * prior) / (counts[0] + counts[1] + 12)


def propose_prune(nodes, selection_mass, cell, table, price, *, repack):
    native = len(selection_mass)
    probabilities = [math.exp(min(0.0, mass)) for mass in selection_mass]
    for i in range(native, len(nodes)):
        parent = nodes[i][0]
        key = cell if i == native else "tail"
        prior = 0.80 if key == "tail" else 0.35 + 0.10 * cell[0] + 0.05 * cell[1] + 0.15 * cell[2]
        probabilities.append(probabilities[parent] * phrase_probability(table.get(key, Counter()), prior))
    if len(nodes) <= NARROW:
        return None
    kept = (sparse_tree([n[0] for n in nodes], probabilities, NARROW) if repack else list(range(NARROW)))
    narrow_value = sum(probabilities[i] for i in kept)
    return kept if sum(probabilities) < price * narrow_value else None


def candidates(trace):
    history = PhraseHistory(trace["input"])
    start, end = len(trace["input"]), 0
    for offset, record in sorted(trace["rounds"].items(), key=lambda pair: int(pair[0])):
        head = int(offset) - start
        history.extend(trace["output"][end:head + 1])
        end = head + 1
        nodes, native = record["nodes"], record["native"]
        if len(nodes) <= native or record["anchor"]:
            continue
        ranks = {int(p): [token for token, _ in rows[:4]] for p, rows in record["seen"].items()}
        cell = phrase_features(nodes, native, history, ranks)
        yield record, cell


def fit_phrase_table(samples, held_out):
    table = defaultdict(Counter)
    for name, records in samples.items():
        if name != held_out:
            for r, cell in records:
                observe_phrase(r["nodes"], r["native"], r["path"], cell, table)
    return table


def replay(traces):
    samples = {name: list(candidates(trace)) for name, trace in traces.items()}
    for name, trace in traces.items():
        table = fit_phrase_table(samples, name)
        for price, repack in ((1.25, False), (HOT_PRICE, False), (HOT_PRICE, True)):
            outcome = Counter()
            for r, cell in samples[name]:
                nodes, path = r["nodes"], r["path"]
                if len(nodes) <= NARROW:
                    continue
                outcome["wide_phrases"] += 1
                narrow = propose_prune(nodes, r["chosen"], cell, table, price, repack=repack)
                if narrow is None:
                    continue
                retained = set(narrow)
                after = next((i for i, node in enumerate(path) if node not in retained), len(path))
                outcome["pruned"] += 1
                outcome["zero_loss"] += after == len(path)
                outcome["lost_tokens"] += len(path) - after
            cost = (price - 1) * outcome["pruned"] - len(trace["rounds"]) * outcome["lost_tokens"] / 1023
            print("replay", name, json.dumps({"price": price, "repack": repack, **outcome,
                                              "net_narrow_round_equivalents": cost}),
                  flush=True)


def replay_recent(traces):
    samples = {name: list(candidates(trace)) for name, trace in traces.items()}
    for name, trace in traces.items():
        table = fit_phrase_table(samples, name)
        cells = {id(record): cell for record, cell in samples[name]}
        pricing = RecentWidthPrice()
        outcomes = Counter()
        prices = []
        for _, record in sorted(trace["rounds"].items(), key=lambda row: int(row[0])):
            nodes, path = record["nodes"], record["path"]
            width = len(nodes)
            if width > NARROW and id(record) in cells:
                price = pricing(width)
                if price is not None:
                    prices.append(price)
                    keep = propose_prune(nodes, record["chosen"], cells[id(record)], table, price, repack=False)
                    if keep is not None:
                        after = next((i for i, node in enumerate(path) if node >= NARROW), len(path))
                        outcomes["pruned"] += 1
                        outcomes["zero_loss"] += after == len(path)
                        outcomes["lost_tokens"] += len(path) - after
                else:
                    outcomes["no_estimate"] += 1
            # The archive contains target-only times. Use a fixed 14 ms for other work in replay.
            target = record["target_s"]
            pricing.record(width, target, target + 0.014)
        print("recent_replay", name, json.dumps({**outcomes, "estimates": len(prices),
              "median_price": median(prices) if prices else None}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", choices=("code", "english_technical_tutorial", "chinese_technical_report"),
                        help="run a 4096/1024 paired request using the hot price")
    parser.add_argument("--warm-before", nargs="?", type=int, const=1, default=0, choices=(1, 2, 3),
                        help="run complete control decodes before timing")
    parser.add_argument("--recent-price", action="store_true", help="use the measured request-local width price")
    parser.add_argument("--include-fixed", action="store_true", help="also time the fixed-price schedule")
    args = parser.parse_args()
    if args.include_fixed and not (args.live and args.recent_price):
        parser.error("--include-fixed requires --live and --recent-price")
    traces = json.loads(TRACE.read_text())
    if args.live:
        from scripts.bench_frontier_phrase_live import live
        live(traces, args.live, args.warm_before, args.recent_price, args.include_fixed)
    elif args.recent_price:
        replay_recent(traces)
    else:
        replay(traces)


if __name__ == "__main__":
    main()

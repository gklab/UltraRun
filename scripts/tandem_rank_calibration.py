"""Estimate the success probabilities of Tandem candidates from verified prefixes."""

from bisect import bisect_right
from collections import Counter, defaultdict
from math import exp, inf, log


CONFIDENCE_BINS = (0.45, 0.65, 0.8, 0.9, 0.97)
PRIOR_STRENGTH = 32
MIN_OBSERVATIONS = 32


class RankCalibration:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = defaultdict(Counter)
        self.coverage = defaultdict(Counter)
        self.observations = 0
        self.adjusted_rows = 0

    @staticmethod
    def row_order(depth, row):
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))
        band = min(2, (depth + 2) // 3)
        bucket = bisect_right(CONFIDENCE_BINS, exp(row[order[0]]))
        return (band, bucket), order

    def adjust(self, depth, row):
        if self.observations < MIN_OBSERVATIONS:
            return row
        cell, order = self.row_order(depth, row)
        support = self.coverage[cell[0]]
        counts = self.counts[cell]
        # The likelihood of a valid continuation is below one when the true
        # token can be outside the selector. Leave that probability unassigned.
        included = (PRIOR_STRENGTH + support[True]) / (PRIOR_STRENGTH + support.total())
        prior = PRIOR_STRENGTH * included
        denominator = PRIOR_STRENGTH + counts.total()
        adjusted = [0.0] * len(row)
        for rank, index in enumerate(order):
            probability = (prior * exp(row[index]) + counts[rank]) / denominator
            adjusted[index] = log(probability) if probability else -inf
        self.adjusted_rows += 1
        return adjusted

    def observe(self, options, logp, tokens, path, predictions):
        # Only learn from the actual committed prefix, including its first miss.
        previous = 0
        for depth, node in enumerate(path):
            if depth >= len(options):
                break
            row = options[depth]
            actual = predictions[node]
            cell, order = self.row_order(depth, logp[depth][previous])
            if actual in row:
                previous = row.index(actual)
                self.counts[cell][order.index(previous)] += 1
                self.coverage[cell[0]][True] += 1
            else:
                self.counts[cell][len(row)] += 1
                self.coverage[cell[0]][False] += 1
            self.observations += 1
            if depth + 1 < len(path):
                assert tokens[path[depth + 1]] == actual and actual in row

    def metrics(self):
        return {"observations": self.observations, "outside_top16": sum(x[False] for x in self.coverage.values()),
                "adjusted_rows": self.adjusted_rows}

"""Merge fresh and just-verified Tandem score rows at the same token position."""

from collections import Counter
from math import exp, log


OLD_WEIGHT = 0.25


class ScoreRows:
    """Map each predecessor to its scores only when the tree expands it."""

    def __init__(self, candidates, predecessors, fresh, old, stats):
        self.candidates = candidates
        self.predecessors = predecessors
        self.fresh = fresh
        self.old = old
        self.stats = stats

    @staticmethod
    def log_probabilities(source, predecessor):
        candidates, rows, index = source
        position = index.get(predecessor)
        if position is None:
            return {}
        return dict(zip(candidates, rows[position], strict=True))

    def __getitem__(self, slot):
        predecessor = self.predecessors[slot]
        fresh = self.log_probabilities(self.fresh, predecessor)
        old = self.log_probabilities(self.old, predecessor) if self.old is not None else {}
        missing = -float("inf")
        if not fresh or not old:
            self.stats["old_only_rows"] += bool(old)
            source = fresh or old
            return [source.get(token, missing) for token in self.candidates]
        self.stats["mixed_rows"] += 1
        values = [(1 - OLD_WEIGHT) * exp(fresh.get(token, missing)) + OLD_WEIGHT * exp(old.get(token, missing))
                  for token in self.candidates]
        return [log(p) if p > 0 else missing for p in values]


class Overlap:
    """Keep one prior block, with the exact number of verified steps to align it."""

    def __init__(self):
        self.previous = None
        self.current = None
        self.advance = 0
        self.stats = Counter()

    @staticmethod
    def rows(root, options, scores):
        predecessors = [[root], *options[:-1]]
        return [(choices, weights, {token: i for i, token in enumerate(prior)})
                for choices, weights, prior in zip(options, scores, predecessors, strict=True)]

    def merge(self, root, options, scores):
        self.current = options
        fresh = self.rows(root, options, scores)
        old = [] if self.previous is None else self.previous[self.advance:]
        self.previous = fresh
        self.stats["rounds"] += 1
        if not old:
            return options, scores

        self.stats["overlap_rounds"] += 1
        choices, weights = [], []
        predecessor = [root]
        for depth, row in enumerate(fresh):
            prior = old[depth] if depth < len(old) else None
            ordered = list(dict.fromkeys((*row[0], *prior[0]))) if prior is not None else row[0]
            self.stats["extra_candidates"] += len(ordered) - len(row[0])
            weights.append(ScoreRows(ordered, predecessor, row, prior, self.stats))
            choices.append(ordered)
            predecessor = ordered
        return choices, weights

    def observe(self, _options, _scores, tokens, path, _predictions):
        self.advance = len(path)
        self.stats["accepted_old_only"] += sum(tokens[node] not in self.current[depth]
                                                for depth, node in enumerate(path[1:]))

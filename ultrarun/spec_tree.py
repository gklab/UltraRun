"""Topologically ordered speculative candidates and their causal paths."""
from functools import cached_property

import mlx.core as mx

from .tree_schedule import slot_plan


class VerificationTree:
    def __init__(self, parents, offset, conv_taps, *, spine=None, reuse_forks=False):
        if not parents or parents[0] != -1 or any(not 0 <= p < i for i, p in enumerate(parents[1:], 1)):
            raise ValueError("Expected one root followed by children of earlier nodes")
        if spine is not None and (not 1 <= spine < len(parents) or parents[1:spine] != list(range(spine - 1))):
            raise ValueError("Expected a contiguous linear spine followed by branch nodes")
        self.scheduled = reuse_forks
        self.checkpoint = None if spine is None else spine - 1
        if reuse_forks:
            if spine is None:
                raise ValueError("A scheduled verification tree requires a spine")
            saves, restores, slots = slot_plan(parents)
            if slots > 2:
                raise ValueError("The scheduled tree needs more than two live fork states")
            self.commands = [(saved + 1) * 4 + restored + 1 for saved, restored in zip(saves, restores, strict=True)]
            self.multi_fork = slots == 2
        elif spine is not None:
            # A jump can restore an earlier side-branch node as well as a spine node.
            forks = {parents[spine]}
            for index, parent in enumerate(parents[spine + 1:], spine + 1):
                if parent != index - 1:
                    forks.add(parent)
            if len(forks) > 2:
                raise ValueError("Expected at most two saved branch states for compact verification")
            self.multi_fork = len(forks) == 2
        else:
            self.multi_fork = False
        self.parents = mx.array(parents, dtype=mx.int32)
        self.spine = spine
        paths = [(0,)]
        for i, parent in enumerate(parents[1:], 1):
            paths.append((*paths[parent], i))
        self.paths = paths
        if reuse_forks:
            # Keep the deepest state in the one existing dense checkpoint.
            self.checkpoint = max(range(len(paths)), key=lambda i: len(paths[i]))
        self.positions = mx.array([offset + len(path) - 1 for path in paths], dtype=mx.int32)
        descendants = mx.array([[i in path for i in range(len(paths))] for path in paths])
        self.attention = mx.concatenate((mx.ones((len(paths), offset), dtype=mx.bool_), descendants), axis=1)
        prefix = list(range(conv_taps))
        windows = [(prefix + [conv_taps + i for i in path])[-(conv_taps + 1):] for path in paths]
        self.windows = mx.array(windows, dtype=mx.int32)

    @cached_property
    def logical_route(self):
        width = len(self.paths)
        return mx.array([[len(path), *path, *([0] * (width - len(path)))] for path in self.paths],
                        dtype=mx.uint32)

    @cached_property
    def delta_route(self):
        """Store the spine and two fork positions before the per-node restore slots."""
        if self.scheduled:
            return mx.array([self.checkpoint, *self.commands], dtype=mx.int32)
        forks = {}
        slots = [-1] * self.spine
        for index, path in enumerate(self.paths[self.spine:], self.spine):
            parent = path[-2]
            slot = forks.setdefault(parent, len(forks)) if index == self.spine or parent != index - 1 else -1
            slots.append(slot)
        positions = tuple(forks)
        return mx.array([self.spine, positions[0], positions[-1], *slots], dtype=mx.int32)


def accepted_path(parents, tokens, predictions):
    """Walk only children equal to the complete target vocabulary argmax."""
    children = {}
    for node, (parent, token) in enumerate(zip(parents[1:], tokens[1:]), 1):
        children[(parent, token)] = node
    path = [0]
    while (child := children.get((path[-1], predictions[path[-1]]))) is not None:
        path.append(child)
    return path

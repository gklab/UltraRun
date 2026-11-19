"""Find draft continuations in the request's prompt and confirmed tokens."""
from collections import defaultdict


SUFFIX_LENGTHS = (8, 6, 4, 3)
MAX_OCCURRENCES = 12
MAX_PHRASE_TOKENS = 5
MAX_ANCHORED_TOKENS = 15


class PhraseHistory:
    def __init__(self, tokens):
        self.tokens = []
        self.indices = {length: defaultdict(list) for length in SUFFIX_LENGTHS}
        self.added_nodes = 0
        self.rounds = 0
        self.anchor_rounds = 0
        self.early_anchor_rounds = 0
        self.extend(tokens)

    def extend(self, tokens):
        for token in tokens:
            end = len(self.tokens)
            for length, index in self.indices.items():
                if end >= length:
                    occurrences = index[tuple(self.tokens[-length:])]
                    occurrences.append(end)
                    if len(occurrences) > MAX_OCCURRENCES:
                        del occurrences[0]
            self.tokens.append(token)

    def continuation(self, hypothetical, *, max_tokens=MAX_PHRASE_TOKENS):
        suffix = (*self.tokens[-SUFFIX_LENGTHS[0]:], *hypothetical)
        for length, index in self.indices.items():
            occurrences = index.get(suffix[-length:], ())
            if not occurrences or (length == 3 and len(occurrences) < 2):
                continue
            first = self.tokens[occurrences[0]]
            if any(self.tokens[start] != first for start in occurrences):
                continue

            sources = [self.tokens[start:start + max_tokens] for start in occurrences]
            result = [first]
            for offset in range(1, max_tokens):
                next_tokens = {source[offset] for source in sources if offset < len(source)}
                if len(next_tokens) != 1:
                    break
                result.append(next_tokens.pop())
            return tuple(result), length, len(occurrences)
        return None

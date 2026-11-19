"""Bound the live recurrent states of a depth-first verification tree."""


def order_for_slots(parents):
    """Place the most register-demanding subtree last at each fork."""
    children = [[] for _ in parents]
    for node, parent in enumerate(parents[1:], 1):
        children[parent].append(node)
    slots = [0] * len(parents)
    for node in reversed(range(len(parents))):
        siblings = children[node]
        if len(siblings) > 1:
            # The final child can reuse the parent's register immediately.
            # A stable tie prefers to hold the first child for last.
            last = max(siblings, key=slots.__getitem__)
            siblings.remove(last)
            siblings.append(last)
            slots[node] = max(slots[last], 1 + max(slots[child] for child in siblings[:-1]))
        elif siblings:
            slots[node] = slots[siblings[0]]
    stack = [0]
    order = []
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(children[node]))
    return order, slots[0]


def slot_plan(parents):
    """Return per-position restore and save slots, freeing after the last use."""
    last_jump = {parent: node for node, parent in enumerate(parents[1:], 1) if parent != node - 1}
    live = {}
    saves = [-1] * len(parents)
    restores = [-1] * len(parents)
    peak = 0
    for node, parent in enumerate(parents):
        if node and parent != node - 1:
            restores[node] = live[parent]
            if last_jump[parent] == node:
                del live[parent]
        if node in last_jump:
            used = set(live.values())
            slot = next(i for i in range(len(used) + 1) if i not in used)
            saves[node] = live[node] = slot
            peak = max(peak, slot + 1)
    return saves, restores, peak

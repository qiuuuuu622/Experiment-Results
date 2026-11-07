# SPDX-License-Identifier: Apache-2.0
"""Host-side planning, expressed in index-key coordinates (never physical IDs)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexerShape:
    rows: int
    max_keys: int
    ratio: int = 1
    topk: int = 512
    score_budget: int = 1 << 30

    def __post_init__(self):
        if self.rows < 1 or self.max_keys < 1:
            raise ValueError("rows and max_keys must be positive")
        if self.ratio not in (1, 2) or self.topk != 512:
            raise ValueError("Only ratio 1/2 and topk 512 are supported")
        if self.score_budget < self.max_keys * 4:
            raise ValueError("One score row exceeds workspace budget")

    def chunks(self, group_sizes=None):
        """Keep complete request groups unless a group alone exceeds the budget."""
        groups = tuple(group_sizes or (1,) * self.rows)
        if sum(groups) != self.rows or any(g < 1 for g in groups):
            raise ValueError("Positive group sizes must partition all query rows")
        width = (self.max_keys + 63) // 64 * 64
        capacity = self.score_budget // (4 * width)
        if capacity < 1:
            raise ValueError("One aligned score row exceeds workspace budget")
        chunks, start, end = [], 0, 0
        for size in groups:
            if end + size - start > capacity and end > start:
                chunks.append((start, end))
                start = end
            while size > capacity:
                chunks.append((end, end + capacity))
                end += capacity
                start = end
                size -= capacity
            end += size
        if end > start:
            chunks.append((start, end))
        return tuple(chunks)

    def original_token_span(self, index_position):
        """A compressed key represents a span, not a unique original token ID."""
        if index_position < 0:
            return (-1, -1)
        return (index_position * self.ratio, (index_position + 1) * self.ratio)


def candidate_positions(blocks, visible_keys, block_size=8):
    """Small host oracle: canonical candidates expanded in index-key space."""
    if visible_keys < 0:
        raise ValueError("visible_keys cannot be negative")
    canonical = sorted(
        {b for b in blocks if 0 <= b < (visible_keys + block_size - 1) // block_size}
    )
    return [
        p
        for b in canonical
        for p in range(b * block_size, min((b + 1) * block_size, visible_keys))
    ]


def shared_candidate_groups(groups, counts, blocks):
    """Return longest-prefix-first groups, or None for distinct candidate sets."""
    reordered = []
    for group in groups:
        live = [row for row in group if row >= 0]
        if not live:
            raise ValueError("A shared group must contain a valid query row")
        live.sort(key=lambda row: counts[row], reverse=True)
        first = live[0]
        for row in live:
            count = counts[row]
            if count < 0 or count > len(blocks[row]) * 8:
                raise ValueError("Candidate count exceeds canonical block capacity")
            n = (count + 7) // 8
            if blocks[row][:n] != blocks[first][:n]:
                return None
        reordered.append(live + [-1] * (len(group) - len(live)))
    return reordered

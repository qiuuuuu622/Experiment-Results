# SPDX-License-Identifier: Apache-2.0
"""Candidate-sharing eligibility is checked cheaply on the host before launch."""

import pytest

from vllm_dsv41_indexer.contract import shared_candidate_groups


def test_partial_tail_shares_longest_canonical_prefix_without_candidate_union():
    blocks = [[0, 100, 100], [0, 5, 9], [2, 100, 100], [2, 8, 100]]
    groups = [[0, 1, -1, -1], [2, 3, -1, -1]]
    assert shared_candidate_groups(groups, [8, 17, 0, 9], blocks) == [
        [1, 0, -1, -1],
        [3, 2, -1, -1],
    ]


def test_distinct_candidate_sets_require_rowwise_fallback():
    assert shared_candidate_groups([[0, 1]], [8, 16], [[1, 100], [0, 5]]) is None


def test_invalid_count_or_empty_group_cannot_be_planned():
    with pytest.raises(ValueError, match="capacity"):
        shared_candidate_groups([[0]], [9], [[0]])
    with pytest.raises(ValueError, match="valid query"):
        shared_candidate_groups([[-1]], [0], [[0]])

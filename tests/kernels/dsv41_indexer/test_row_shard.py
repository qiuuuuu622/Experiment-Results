# SPDX-License-Identifier: Apache-2.0
"""Host planning must cover every query once without splitting verify requests."""

import pytest

from vllm_dsv41_indexer.row_shard import RowPlan


def test_ragged_request_groups_remain_whole_after_input_reorder():
    requests = [9, 5, 9, 1, 3, 5, 5, 3, 8, 5, 9]
    plan = RowPlan.make(requests)
    assert sorted(x for rank in plan.rank_rows for x in rank) == list(range(11))
    owners = {}
    for rank, rows in enumerate(plan.rank_rows):
        for row in rows:
            owners.setdefault(requests[row], rank)
            assert owners[requests[row]] == rank
    packed = [
        x for rows in plan.rank_rows for x in rows + (-1,) * (plan.capacity - len(rows))
    ]
    assert [packed[x] for x in plan.inverse] == list(range(11))


def test_single_request_keeps_empty_ranks_and_split_is_explicit():
    whole = RowPlan.make([7] * 6)
    assert [len(x) for x in whole.rank_rows] == [6, 0, 0, 0]
    split = RowPlan.make([7] * 6, split_rows=True)
    assert [len(x) for x in split.rank_rows] == [2, 2, 1, 1]
    assert split.mode == "split_rows"


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 17, 129])
def test_nondivisible_split_does_not_lose_or_duplicate_rows(rows):
    plan = RowPlan.make(range(rows), split_rows=True)
    assert sorted(plan.inverse) == sorted(
        r * plan.capacity + i
        for r, rs in enumerate(plan.rank_rows)
        for i in range(len(rs))
    )
    assert sorted(x for rank in plan.rank_rows for x in rank) == list(range(rows))


@pytest.mark.parametrize("requests,world", [([], 4), ([0], 0), ([-1], 4)])
def test_invalid_ownership_metadata_fails_before_any_collective(requests, world):
    with pytest.raises(ValueError):
        RowPlan.make(requests, world)

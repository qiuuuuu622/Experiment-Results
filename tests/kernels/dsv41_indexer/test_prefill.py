# SPDX-License-Identifier: Apache-2.0
"""Prefill request aliases, causal windows and prefix-only score safety."""

import pytest


def test_prefill_alias_candidates_and_nan_tails_keep_logical_coordinates():
    """Distinct query candidate sets may alias the same physical request cache."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SM90 GPU required")
    from benchmarks.kernels.dsv41_indexer.common import check_vllm

    check_vllm()
    from benchmarks.kernels.dsv41_indexer.verify_prefill import make_case
    from vllm_dsv41_indexer.prefill import PrefillIndexer

    inputs, candidates = make_case(4, 4, 1031, ratio=2)
    inputs.q.zero_()
    inputs.table[1:].copy_(inputs.table[0].expand(3, -1))
    inputs.lengths.copy_(
        torch.tensor([1031, 19, 5, 0], device=inputs.q.device, dtype=torch.int32)
    )
    candidates.fill_(-1)
    candidates[:, :4] = torch.tensor(
        [[1, 1, 128, 99999], [0, 2, 2, -1], [0, 10, -1, -1], [0, 2, -1, -1]],
        device=inputs.q.device,
        dtype=torch.int32,
    )
    op = PrefillIndexer(inputs, backend="indirect", candidate_blocks=candidates)
    op.scores.fill_(float("nan"))
    op.forward()
    expected = [
        list(range(8, 16)) + list(range(1024, 1031)),
        list(range(8)) + [16, 17, 18],
        list(range(5)),
        [],
    ]
    for row, positions in enumerate(expected):
        actual = [p for p in op.output[row].tolist() if p >= 0]
        assert sorted(actual) == positions
        assert len(actual) == len(set(actual))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.forward()
    # Candidate contents and causal bounds change inside a stable request plan.
    candidates.fill_(-1)
    candidates[:, 0] = 0
    inputs.lengths.fill_(3)
    graph.replay()
    assert op.output.sort(1).values[:, -3:].tolist() == [[0, 1, 2]] * 4

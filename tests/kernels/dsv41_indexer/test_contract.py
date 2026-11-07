# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard TP8's replicated-index-head contract and preserved upstream kernels.

The cheapest check for accidental head partitioning is a CPU contract test;
actual CUDA/cache mapping checks live in the standalone GPU verifier.
"""

import hashlib
import json
from pathlib import Path

import pytest

from vllm_dsv41_indexer import IndexerParallel
from vllm_dsv41_indexer.contract import IndexerShape, candidate_positions


def test_score_budget_preserves_groups_and_splits_only_oversized_group():
    shape = IndexerShape(13, 1024, score_budget=4 * 1024 * 4)
    assert shape.chunks([2, 2, 6, 3]) == ((0, 4), (4, 8), (8, 10), (10, 13))


def test_candidates_are_index_coordinates_and_partial_block_is_clipped():
    assert candidate_positions([0, 2, 2, -1, 100], 19) == list(range(8)) + [16, 17, 18]
    assert candidate_positions([0], 0) == []
    assert IndexerShape(1, 10, ratio=2).original_token_span(3) == (6, 8)


def test_invalid_plans_fail_before_execution():
    with pytest.raises(ValueError):
        IndexerShape(1, 1024, score_budget=100)
    with pytest.raises(ValueError):
        IndexerShape(3, 1024).chunks([1, 1])


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_every_tp_rank_retains_all_32_index_heads(world):
    for rank in range(world):
        contract = IndexerParallel(world, rank)
        assert contract.local_index_heads == 32
        contract.check_query_shape((6, 32, 128))
        # Main-attention eight-head sharding must never reach this indexer.
        for heads in (8, 32 // 8, 64):
            with pytest.raises(ValueError):
                contract.check_query_shape((6, heads, 128))


@pytest.mark.parametrize("pcp,dcp", [(8, 1), (1, 8), (2, 2)])
def test_reject_cp_sharded_cache_contract(pcp, dcp):
    with pytest.raises(ValueError, match="PCP1 and DCP1"):
        IndexerParallel(8, 0, pcp, dcp)


@pytest.mark.parametrize("world,rank", [(8, 8), (8, -1), (3, 0)])
def test_reject_rank_outside_supported_tp_group(world, rank):
    with pytest.raises(ValueError):
        IndexerParallel(world, rank)


def test_vendored_topk_and_dependencies_match_pinned_ant_sources():
    root = Path(__file__).resolve().parents[3] / "vllm_dsv41_indexer/third_party/ant"
    manifest = json.loads((root / "sources.json").read_text())
    for file in manifest["files"]:
        assert (
            hashlib.sha256((root / file["path"]).read_bytes()).hexdigest()
            == file["sha256"]
        )


def test_shared_score_arena_and_ragged_groups_preserve_source_results():
    """A small imposed budget catches accidental per-chunk full allocations."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SM90 GPU required")
    from benchmarks.kernels.dsv41_indexer.common import check_vllm

    check_vllm()
    from benchmarks.kernels.dsv41_indexer.inputs import make_inputs
    from vllm_dsv41_indexer.module import ChunkedIndexer, FullIndexer, IndexerInputs

    length = 65536
    q, _, w, cache, table, req, lens, _ = make_inputs(
        dict(
            rows=13,
            compressed_length=length,
            rows_per_request=1,
            page_size=64,
            candidate_blocks=2048,
        )
    )
    req.copy_(
        torch.tensor(
            [0] + [1] * 2 + [2] * 4 + [3] * 6, device=q.device, dtype=torch.int32
        )
    )
    lens.sub_(torch.arange(13, device=q.device, dtype=torch.int32))
    inputs = IndexerInputs(q, w, cache, table, req, lens, IndexerShape(13, length))
    full = FullIndexer(inputs, group_rows=4, source=True)
    separate = FullIndexer(inputs, group_rows=4, source=True, fuse_source_max=False)
    limited = IndexerInputs(
        q,
        w,
        cache,
        table,
        req,
        lens,
        IndexerShape(13, length, score_budget=length * 4 * 4),
    )
    chunked = ChunkedIndexer(limited, group_rows=4, source=True)
    assert chunked.arena.numel() * 4 <= limited.shape.score_budget
    assert len({p.scores.untyped_storage().data_ptr() for p in chunked.parts}) == 1
    full.forward()
    separate.forward()
    chunked.forward()
    assert torch.equal(full.output.sort(1).values, chunked.output.sort(1).values)
    assert torch.equal(
        full.candidates.sort(1).values, chunked.candidates.sort(1).values
    )
    assert torch.equal(full.maxima, separate.maxima)
    assert torch.equal(
        full.candidates.sort(1).values, separate.candidates.sort(1).values
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chunked.forward()
    lens.zero_()
    graph.replay()
    assert (chunked.output == -1).all().item()
    assert (chunked.candidates == -1).all().item()

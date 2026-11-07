# SPDX-License-Identifier: Apache-2.0
"""P3 full/source lifecycle checks and a real 1 GiB shared score arena."""

import argparse

from .common import check_vllm, save


def run_checks(*, include_budget=True, include_lifecycle=True):
    check_vllm()
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.grouped import GroupedIndexer
    from vllm_dsv41_indexer.module import (
        ChunkedIndexer,
        FullIndexer,
        IndexerInputs,
        NativeFullIndexer,
        NativeSourceIndexer,
    )
    from vllm_dsv41_indexer.reference import source_reference, stable_topk
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import make_inputs

    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for ratio in (1, 2) if include_lifecycle else ():
        for length in (1, 7, 517, 4099, 65536):
            case = dict(
                rows=6,
                compressed_length=length,
                rows_per_request=1,
                page_size=64,
                candidate_blocks=2048,
            )
            q, _, w, cache, table, req, lens, _ = make_inputs(case)
            lens.copy_(
                torch.tensor(
                    [
                        0,
                        min(1, length),
                        min(7, length),
                        min(511, length),
                        min(513, length),
                        length,
                    ],
                    device=q.device,
                    dtype=torch.int32,
                )
            )
            inputs = IndexerInputs(
                q, w, cache, table, req, lens, IndexerShape(6, length, ratio)
            )
            baseline = NativeFullIndexer(inputs)
            native_scores = baseline.score()
            ops = [
                FullIndexer(inputs, source=True),
                GroupedIndexer(inputs, group_rows=1, source=True),
                NativeSourceIndexer(inputs),
            ]
            max_error = 0.0
            for op in ops:
                op.forward()
                scores, positions = [], []
                for row, n in enumerate(lens.tolist()):
                    actual = op.scores[row, :n]
                    expected = native_scores[row, :n]
                    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
                    if n:
                        max_error = max(
                            max_error, (actual - expected).abs().max().item()
                        )
                    scores.append(actual)
                    positions.append(torch.arange(n, device=q.device))
                exact = stable_topk(scores, positions)
                assert torch.equal(op.output.sort(1).values, exact.sort(1).values)
                source = source_reference(op.scores, lens)
                assert torch.equal(op.candidates.sort(1).values, source.sort(1).values)

            source = ops[-1]
            state = CandidateState(6, q.device, compress_ratio=ratio)
            consumers, references = [], []
            for layer in range(4):
                cq = q.float().roll(layer * 7, -1).to(q.dtype)
                cw = w.roll(layer * 3, -1).contiguous()
                consumers.append(CompactIndexer(state, cq, cw, cache, table, req))
                references.append(
                    NativeFullIndexer(
                        IndexerInputs(cq, cw, cache, table, req, lens, inputs.shape)
                    )
                )

            def chain(source=source, state=state, consumers=consumers, lens=lens):
                # Native scheduler metadata and gathered local tables must update
                # when a graph replay changes lengths or request ordering.
                source.prepare()
                source.forward()
                state.publish(source.candidates, lens, source_unique=True)
                for consumer in consumers:
                    consumer.forward()

            chain()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain()
            for step in range(3):
                if step == 1:
                    req.copy_(req.flip(0))
                    lens.copy_(lens.roll(1))
                    for consumer in consumers:
                        consumer.weights.neg_()
                elif step == 2:
                    # Preserve all logical keys while changing physical page IDs.
                    permutation = torch.randperm(cache.shape[0], device=q.device)
                    reordered = torch.empty_like(cache)
                    reordered[permutation] = cache
                    cache.copy_(reordered)
                    table.copy_(permutation[table.long()].int())
                graph.replay()
                published = source_reference(source.scores, lens)
                assert torch.equal(
                    source.candidates.sort(1).values, published.sort(1).values
                )
                counts = state.counts.tolist()
                logical = (
                    state.blocks.long()[..., None] * 8
                    + torch.arange(8, device=q.device)
                ).flatten(1)
                for layer, (consumer, reference) in enumerate(
                    zip(consumers, references)
                ):
                    reference.prepare()
                    expected_full = reference.score()
                    compact, positions = [], []
                    for row, count in enumerate(counts):
                        pos = logical[row, :count]
                        values = consumer.scores[row, :count]
                        try:
                            torch.testing.assert_close(
                                values, expected_full[row, pos], rtol=1e-3, atol=1e-4
                            )
                        except AssertionError:
                            from vllm_dsv41_indexer.reference import scores_reference

                            saved = values.clone()
                            consumer.forward()
                            fp32, _ = scores_reference(
                                consumer.q,
                                consumer.weights,
                                cache,
                                table,
                                req,
                                lens,
                                source.candidates,
                            )
                            print(
                                dict(
                                    ratio=ratio,
                                    length=length,
                                    step=step,
                                    row=row,
                                    layer=layer,
                                    requests=req.tolist(),
                                    lengths=lens.tolist(),
                                    graph=saved[:8].tolist(),
                                    eager=consumer.scores[row, :count][:8].tolist(),
                                    native=expected_full[row, pos][:8].tolist(),
                                    fp32=fp32[row][:8].tolist(),
                                ),
                                flush=True,
                            )
                            raise
                        compact.append(values)
                        positions.append(pos)
                        selected = consumer.output[row]
                        valid = selected[selected >= 0]
                        assert len(valid) == len(valid.unique()) == min(512, count)
                        assert (valid < lens[row]).all().item()
                    exact = stable_topk(compact, positions)
                    assert torch.equal(
                        consumer.output.sort(1).values, exact.sort(1).values
                    )
            records.append(
                dict(
                    ratio=ratio,
                    index_keys=length,
                    logical_context=length * ratio,
                    max_abs_vs_native=max_error,
                    source_token_and_block_topk="exact_for_fixed_scores",
                    four_consumer_graph_replays=3,
                    request_reorder_and_physical_pages="passed",
                )
            )

    budget = None
    if include_budget:
        rows, keys = 1025, 262144
        pages = keys // 64
        raw = torch.zeros((pages, 64 * 132), device="cuda", dtype=torch.uint8)
        scales = torch.ones((pages, 64), device="cuda", dtype=torch.float32)
        raw[:, 64 * 128 :] = scales.view(torch.uint8)
        cache = raw.view(pages, 64, 1, 132)
        q = torch.zeros((rows, 32, 128), device="cuda", dtype=torch.float8_e4m3fn)
        weights = torch.ones((rows, 32), device="cuda", dtype=torch.float32)
        table = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :]
        requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
        lengths = torch.full((rows,), 513, device="cuda", dtype=torch.int32)
        lengths[-1] = keys
        inputs = IndexerInputs(
            q, weights, cache, table, requests, lengths, IndexerShape(rows, keys)
        )
        torch.cuda.reset_peak_memory_stats()
        op = ChunkedIndexer(inputs, group_rows=8, source=True)
        assert op.ranges == ((0, 1024), (1024, 1025))
        assert op.arena.numel() * op.arena.element_size() == 1 << 30
        assert len({part.scores.untyped_storage().data_ptr() for part in op.parts}) == 1
        op.arena.fill_(float("nan"))
        op.forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.forward()
        for step in range(2):
            if step:
                lengths[:-1].zero_()
                lengths[-1] = 65
            graph.replay()
            token_rows = op.output.tolist()
            source_rows = op.candidates.tolist()
            for row, n in enumerate(lengths.tolist()):
                assert {x for x in token_rows[row] if x >= 0} == set(range(min(512, n)))
                blocks = (n + 7) // 8
                expected = (
                    set(range(blocks))
                    if blocks <= 2048
                    else set(range(2047)) | {blocks - 1}
                )
                assert {x for x in source_rows[row] if x >= 0} == expected
        budget = dict(
            arena_bytes=op.arena.numel() * 4,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            rows=rows,
            planned_index_keys=keys,
            ranges=op.ranges,
            visible_lengths_initial="1024 rows × 513 keys, last row × 262144",
            all_rows_full_256k_scoring=False,
            analytic_zero_query_reference="passed",
            graph_replays=2,
        )
    return dict(status="passed", checks=records, real_budget=budget)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-budget", action="store_true")
    parser.add_argument("--budget-only", action="store_true")
    parser.add_argument("--deep-gemm-package")
    args = parser.parse_args()
    if args.deep_gemm_package:
        check_vllm()
        from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init

        _lazy_init()
        _import_deep_gemm()._C.init(args.deep_gemm_package)
    result = run_checks(
        include_budget=not args.skip_budget, include_lifecycle=not args.budget_only
    )
    result["deep_gemm_package"] = args.deep_gemm_package
    save(args.output, result)
    print(
        f"Passed {len(result['checks'])} P3 ratio/lifecycle cases and budget check",
        flush=True,
    )


if __name__ == "__main__":
    main()

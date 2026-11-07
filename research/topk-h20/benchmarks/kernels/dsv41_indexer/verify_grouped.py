# SPDX-License-Identifier: Apache-2.0
"""Grouped FP8 verify: full/source, ragged rows, causal bounds and shared candidates."""

import argparse

from .common import check_vllm, save


def run_checks(paired=False):
    check_vllm()
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.grouped import GroupedConsumer, GroupedIndexer
    from vllm_dsv41_indexer.module import IndexerInputs, NativeFullIndexer
    from vllm_dsv41_indexer.reference import source_reference, stable_topk
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import make_inputs

    results = []
    for group in (1, 2, 4, 6, 8):
        for length, ratio in [
            (size, ratio) for size in (517, 4099, 65536) for ratio in (1, 2)
        ]:
            rows = group * 2
            case = dict(
                rows=rows,
                compressed_length=length,
                rows_per_request=group,
                page_size=64,
                candidate_blocks=2048,
            )
            q, _, w, cache, table, req, lens, candidates = make_inputs(case)
            lens.copy_(
                (
                    length * ratio
                    - group
                    + 1
                    + torch.arange(group, device=q.device).repeat(2)
                )
                // ratio
            )
            i = IndexerInputs(q, w, cache, table, req, lens, IndexerShape(rows, length))
            grouped = GroupedIndexer(
                i, group_rows=1 << (group - 1).bit_length(), source=True, paired=paired
            )
            native = NativeFullIndexer(i, next_n=group)
            grouped.scores.fill_(float("nan"))
            grouped.forward()
            baseline = native.score()
            refs, positions, max_abs = [], [], 0.0
            for row, n in enumerate(lens.tolist()):
                actual = grouped.scores[row, :n]
                torch.testing.assert_close(
                    actual, baseline[row, :n], rtol=1e-3, atol=1e-4
                )
                max_abs = max(max_abs, (actual - baseline[row, :n]).abs().max().item())
                refs.append(actual)
                positions.append(torch.arange(n, device=q.device))
            exact = stable_topk(refs, positions)
            assert torch.equal(exact.sort(1).values, grouped.output.sort(1).values)
            expected = source_reference(grouped.scores, lens)
            assert torch.equal(
                expected.sort(1).values, grouped.candidates.sort(1).values
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                grouped.forward()
            lens[0] = 0
            if rows > 1:
                lens[1] = 1
            graph.replay()
            assert (grouped.output[0] == -1).all().item()
            if rows > 1:
                assert set(grouped.output[1].tolist()) <= {-1, 0}
            q.zero_()
            graph.replay()
            for row, n in enumerate(lens.tolist()):
                assert {x for x in grouped.output[row].tolist() if x >= 0} == set(
                    range(min(n, 512))
                )
            state = CandidateState(rows, q.device, compress_ratio=ratio)
            # Shared raw candidate layout; different visible tails are prefixes.
            for start in range(0, rows, group):
                candidates[start : start + group].copy_(
                    candidates[start : start + 1].expand(group, -1).clone()
                )
            state.publish(candidates, lens)
            shared = GroupedConsumer(
                state,
                q,
                w,
                cache,
                table,
                req,
                group_rows=1 << (group - 1).bit_length(),
                paired=paired,
            )
            separate = CompactIndexer(state, q, w, cache, table, req)
            shared.forward()
            separate.forward()
            assert torch.equal(
                shared.output.sort(1).values, separate.output.sort(1).values
            )

            def shared_call(
                state=state, candidates=candidates, lens=lens, shared=shared
            ):
                state.publish(candidates, lens)
                shared.forward()

            shared_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(shared_graph):
                shared_call()
            candidates[:, 0] = -1
            shared_graph.replay()
            separate.forward()
            assert torch.equal(
                shared.output.sort(1).values, separate.output.sort(1).values
            )
            # Distinct layouts must fall back, not form a union.
            if group > 1 and length > 16384:
                candidates[group + 1, 1] = -1
                state.publish(candidates, lens)
                assert not shared.replan()
                shared.forward()
                separate.forward()
                assert torch.equal(
                    shared.output.sort(1).values, separate.output.sort(1).values
                )
            results.append(
                dict(
                    group=group,
                    ratio=ratio,
                    keys=length,
                    rows=rows,
                    native_mode=native.baseline_mode,
                    max_abs=max_abs,
                    status="passed",
                )
            )
    # Mixed group sizes and page-layout changes, same logical values.
    sizes = [1, 2, 4, 6, 8]
    rows, length = sum(sizes), 4099
    q, _, w, cache, table, req, lens, _ = make_inputs(
        dict(
            rows=rows,
            compressed_length=length,
            rows_per_request=1,
            page_size=64,
            candidate_blocks=2048,
        )
    )
    req.copy_(
        torch.tensor(
            [r for r, n in enumerate(sizes) for _ in range(n)],
            device=q.device,
            dtype=torch.int32,
        )
    )
    lens.sub_(torch.arange(rows, device=q.device, dtype=torch.int32))
    i = IndexerInputs(q, w, cache, table, req, lens, IndexerShape(rows, length))
    op = GroupedIndexer(i, group_rows=8, paired=paired)
    base = NativeFullIndexer(i)
    op.forward()
    scores = base.score()
    for row, n in enumerate(lens.tolist()):
        torch.testing.assert_close(
            op.scores[row, :n], scores[row, :n], rtol=1e-3, atol=1e-4
        )
    before = op.output.clone()
    permutation = torch.randperm(cache.shape[0], device=q.device)
    moved = torch.empty_like(cache)
    moved[permutation] = cache
    cache.copy_(moved)
    table.copy_(permutation[table.long()].int())
    op.forward()
    assert torch.equal(before.sort(1).values, op.output.sort(1).values)
    results.append(dict(group="ragged", keys=length, rows=rows, status="passed"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    checks = run_checks(args.paired)
    save(args.output, dict(status="passed", paired=args.paired, checks=checks))
    print(f"Passed {len(checks)} grouped full/source/consumer cases", flush=True)


if __name__ == "__main__":
    main()

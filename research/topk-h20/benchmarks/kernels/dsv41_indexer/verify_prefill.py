# SPDX-License-Identifier: Apache-2.0
"""Prefill correctness: causal windows, request aliases, candidates and graphs."""

import argparse

from .common import check_vllm, save


def make_case(rows, requests, length, ratio=1):
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.module import IndexerInputs

    from .inputs import make_inputs

    q, _, weights, cache, table, req, lengths, _ = make_inputs(
        dict(
            rows=rows,
            compressed_length=length,
            rows_per_request=rows // requests,
            page_size=64,
            candidate_blocks=2048,
        )
    )
    candidates = torch.full((requests, 2048), -1, device=q.device, dtype=torch.int32)
    generator = torch.Generator(device=q.device).manual_seed(20261106)
    for request in range(requests):
        n = (length + 7) // 8
        chosen = torch.randperm(n, device=q.device, generator=generator)[: min(2048, n)]
        candidates[request, : len(chosen)] = chosen.int()
    inputs = IndexerInputs(
        q, weights, cache, table, req, lengths, IndexerShape(rows, length, ratio=ratio)
    )
    return inputs, candidates


def run_checks():
    check_vllm()
    import torch

    from vllm_dsv41_indexer.prefill import PrefillIndexer
    from vllm_dsv41_indexer.reference import (
        scores_reference,
        source_reference,
        stable_topk,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for rows, requests, length, ratio in [
        (1, 1, 7, 1),
        (16, 2, 517, 2),
        (12, 3, 4099, 1),
    ]:
        for compact in (False, True):
            inputs, candidates = make_case(rows, requests, length, ratio)
            if rows > 1:
                inputs.lengths[0] = 0
                inputs.lengths[1] = 5
            if rows == 12:
                inputs.requests.copy_(
                    torch.tensor(
                        [2] * 2 + [0] * 5 + [1] * 5,
                        device=inputs.q.device,
                        dtype=torch.int32,
                    )
                )
            candidate_input = candidates if compact else None
            # Small budget forces request chunks while each request's K is
            # prepared once; independent chunks must reuse the score allocation.
            budget = max(16384, (length + 63) // 64 * 64) * 4 * 3
            ops = {
                backend: PrefillIndexer(
                    inputs,
                    backend=backend,
                    candidate_blocks=candidate_input,
                    score_budget=budget,
                    source=not compact,
                )
                for backend in (
                    ("deepgemm", "row_tiled", "keys_major", "indirect")
                    if compact
                    else ("deepgemm", "row_tiled", "keys_major", "indirect", "gluon")
                )
            }
            for op in ops.values():
                op.forward()
            for backend, op in ops.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    op.forward()
                graph.replay()
                max_error, max_native_ref, max_ref_error, square, norm = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                native = ops["deepgemm"]
                native.prepare()
                for start, end, request in op.chunks:
                    got = op.score_chunk(start, end, request)
                    expected = native.score_chunk(start, end, request)
                    lengths = op.lengths[start:end].tolist()
                    scores, positions = [], []
                    ref, reference_positions = scores_reference(
                        inputs.q[start:end],
                        inputs.weights[start:end],
                        inputs.cache,
                        inputs.table,
                        inputs.requests[start:end],
                        inputs.lengths[start:end],
                        candidates[inputs.requests[start:end].long()]
                        if compact
                        else None,
                    )
                    for local, n in enumerate(lengths):
                        actual, baseline = got[local, :n], expected[local, :n]
                        torch.testing.assert_close(
                            actual, baseline, rtol=1e-3, atol=1e-4
                        )
                        if n:
                            max_error = max(
                                max_error, (actual - baseline).abs().max().item()
                            )
                            max_native_ref = max(
                                max_native_ref,
                                (baseline - ref[local]).abs().max().item(),
                            )
                            max_ref_error = max(
                                max_ref_error, (actual - ref[local]).abs().max().item()
                            )
                            square += (actual - baseline).square().sum().item()
                            norm += baseline.square().sum().item()
                        scores.append(actual)
                        positions.append(reference_positions[local])
                    op.select_chunk(got, start, end, request)
                    exact = stable_topk(scores, positions)
                    assert torch.equal(
                        exact.sort(1).values, op.output[start:end].sort(1).values
                    )
                    if not compact:
                        expected_source = source_reference(got, op.lengths[start:end])
                        assert torch.equal(
                            expected_source.sort(1).values,
                            op.candidates[start:end].sort(1).values,
                        )
                records.append(
                    dict(
                        rows=rows,
                        requests=requests,
                        keys=length,
                        ratio=ratio,
                        compact=compact,
                        backend=backend,
                        chunks=len(op.chunks),
                        max_abs_vs_native=max_error,
                        nrmse_vs_native=(square / max(norm, 1e-30)) ** 0.5,
                        native_max_abs_vs_fp32=max_native_ref,
                        max_abs_vs_fp32=max_ref_error,
                        fixed_scores_topk="exact",
                        source_candidates="exact" if not compact else None,
                        preparation_bytes=op.preparation_bytes,
                        score_budget=budget,
                    )
                )
            # Same captured inputs can change causal lengths and candidates.
            op = ops["keys_major"]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.forward()
            inputs.q.zero_()
            inputs.lengths.zero_()
            if compact:
                candidates.fill_(-1)
            graph.replay()
            assert (op.output == -1).all().item()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = run_checks()
    save(args.output, dict(status="passed", checks=records))
    print(f"Passed {len(records)} prefill backend/shape cases", flush=True)


if __name__ == "__main__":
    main()

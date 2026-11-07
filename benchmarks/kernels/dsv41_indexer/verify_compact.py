# SPDX-License-Identifier: Apache-2.0
"""Compact publication lifecycle, FP32 oracle and fused mapping regressions."""

import argparse

from .common import check_vllm, save


def run_checks(backend="wgmma"):
    check_vllm()
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.module import IndexerInputs, NativeFullIndexer
    from vllm_dsv41_indexer.reference import scores_reference, stable_topk
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import make_inputs

    torch.backends.cuda.matmul.allow_tf32 = False
    checks = []
    for ratio in (1, 2):
        for length in (7, 517, 4099, 16391):
            case = dict(
                rows=8,
                compressed_length=length,
                rows_per_request=2,
                page_size=64,
                candidate_blocks=2048,
            )
            q, _, w, cache, table, req, lens, blocks = make_inputs(case)
            state = CandidateState(8, q.device, compress_ratio=ratio)
            qs = [q] + [q.float().roll(i * 7, -1).to(q.dtype) for i in range(1, 4)]
            ws = [w] + [w.roll(i * 3, -1).contiguous() for i in range(1, 4)]
            ops = [
                CompactIndexer(
                    state, qs[i], ws[i], cache, table, req, score_backend=backend
                )
                for i in range(4)
            ]
            separate = CompactIndexer(
                state,
                q,
                w,
                cache,
                table,
                req,
                fused_mapping=False,
                score_backend=backend,
            )

            def call(state=state, blocks=blocks, lens=lens, ops=ops, separate=separate):
                state.publish(blocks, lens)
                for op in ops:
                    op.forward()
                separate.forward()

            call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            # Unique source fast publisher must agree with arbitrary-input path.
            canonical = state.blocks.clone()
            counts = state.counts.clone()
            state.publish(blocks, lens, source_unique=True)
            assert torch.equal(state.blocks, canonical)
            assert torch.equal(state.counts, counts)
            for step in range(3):
                if step == 1:
                    req.copy_(req.flip(0))
                    lens[0] = 0
                    lens[1] = min(length, 5)
                    blocks[:, 1] = blocks[:, 0]
                    blocks[:, 2] = -1
                    blocks[:, 3] = 2**30
                if step == 2:
                    for query in qs:
                        query.zero_()
                    blocks.copy_(blocks.roll(1, 0))
                    lens.copy_(lens.roll(1, 0))
                for op in ops:
                    op.scores.fill_(float("nan"))
                graph.replay()
                assert torch.equal(
                    ops[0].output.sort(1).values, separate.output.sort(1).values
                )
                max_abs, nrmse, overlap = 0.0, 0.0, 1.0
                for layer, op in enumerate(ops):
                    ref, positions = scores_reference(
                        qs[layer], ws[layer], cache, table, req, lens, blocks
                    )
                    native_scores = None
                    if backend == "wgmma":
                        native = NativeFullIndexer(
                            IndexerInputs(
                                qs[layer],
                                ws[layer],
                                cache,
                                table,
                                req,
                                lens,
                                IndexerShape(8, length, ratio=ratio),
                            )
                        )
                        native_scores = native.score()
                    fixed = []
                    for row, expected in enumerate(ref):
                        n = len(expected)
                        assert state.counts[row].item() == n
                        actual = op.scores[row, :n]
                        target = (
                            expected
                            if native_scores is None
                            else native_scores[row, positions[row]]
                        )
                        torch.testing.assert_close(actual, target, rtol=1e-3, atol=1e-4)
                        fixed.append(actual)
                        if n:
                            error = actual - expected
                            max_abs = max(max_abs, error.abs().max().item())
                            nrmse = max(
                                nrmse,
                                (
                                    error.square().mean()
                                    / expected.square().mean().clamp_min(1e-30)
                                )
                                .sqrt()
                                .item(),
                            )
                    exact = stable_topk(fixed, positions)
                    assert torch.equal(op.output.sort(1).values, exact.sort(1).values)
                    oracle = stable_topk(ref, positions)
                    for row in range(8):
                        a = {p for p in op.output[row].tolist() if p >= 0}
                        b = {p for p in oracle[row].tolist() if p >= 0}
                        assert len(a) == min(512, len(positions[row]))
                        overlap = min(
                            overlap, len(a & b) / max(1, len(a)) if a else 1.0
                        )
                        if a != b:
                            # Changed selection must lie within measured score error
                            # of the FP32 boundary; unexplained changes are failures.
                            values = dict(
                                zip(positions[row].tolist(), ref[row].tolist())
                            )
                            boundary = min(values[p] for p in b)
                            assert all(
                                abs(values[p] - boundary) <= 2 * max_abs for p in a ^ b
                            )
                checks.append(
                    dict(
                        ratio=ratio,
                        keys=length,
                        step=step,
                        max_abs=max_abs,
                        nrmse=nrmse,
                        overlap=overlap,
                    )
                )
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("wgmma", "precise"), default="wgmma")
    args = parser.parse_args()
    checks = run_checks(args.backend)
    save(args.output, dict(status="passed", backend=args.backend, checks=checks))
    print(f"Passed {len(checks)} compact lifecycle cases × four consumers", flush=True)


if __name__ == "__main__":
    main()

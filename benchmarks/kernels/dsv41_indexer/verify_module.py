# SPDX-License-Identifier: Apache-2.0
"""Small FP32 oracle and fixed-score selection checks for module experiments."""

import argparse

from .common import check_vllm, save


def run_checks():
    check_vllm()
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.module import FullIndexer, IndexerInputs, NativeFullIndexer
    from vllm_dsv41_indexer.reference import (
        scores_reference,
        source_reference,
        stable_topk,
    )

    from .inputs import make_inputs

    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for length in (517, 4099, 16391):
        for group in (1, 2, 4, 6, 8):
            rows = group * 2
            case = dict(
                rows=rows,
                compressed_length=length,
                rows_per_request=group,
                page_size=64,
                candidate_blocks=2048,
            )
            q, _, w, cache, table, req, lens, _ = make_inputs(case)
            lens[0] = 0
            if rows > 1:
                lens[1] = min(511, length)
            inputs = IndexerInputs(
                q, w, cache, table, req, lens, IndexerShape(rows, length, ratio=2)
            )
            op = FullIndexer(
                inputs,
                group_rows=1 << (group - 1).bit_length(),
                source=True,
                precise=False,
            )
            op.scores.fill_(float("nan"))
            op.maxima.fill_(float("nan"))
            ref, positions = scores_reference(q, w, cache, table, req, lens)
            op.forward()
            native = NativeFullIndexer(inputs, next_n=group)
            baseline = native.score()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.forward()
            graph.replay()
            absolute, square, ref_square = 0.0, 0.0, 0.0
            native_failures, native_max_abs = 0, 0.0
            lengths = lens.tolist()
            for row, n in enumerate(lengths):
                actual = op.scores[row, :n]
                torch.testing.assert_close(
                    actual, baseline[row, :n], rtol=1e-3, atol=1e-4
                )
                # Audit the unmodified baseline too; do not weaken optimized gates
                # to reproduce its limited-precision FP8 accumulation error.
                native_error = (baseline[row, :n] - ref[row]).abs()
                native_failures += (
                    (native_error > 1e-4 + 1e-3 * ref[row].abs()).sum().item()
                )
                if n:
                    native_max_abs = max(native_max_abs, native_error.max().item())
                if n:
                    error = actual - ref[row]
                    absolute = max(absolute, error.abs().max().item())
                    square += error.square().sum().item()
                    ref_square += ref[row].square().sum().item()
                    for block in range((n + 7) // 8):
                        expected = (
                            float("inf")
                            if block == (n - 1) // 8
                            else actual[block * 8 : min(block * 8 + 8, n)].max().item()
                        )
                        assert op.maxima[row, block].item() == expected
            fixed = [op.scores[row, :n] for row, n in enumerate(lengths)]
            exact = stable_topk(fixed, positions)
            assert torch.equal(op.output.sort(1).values, exact.sort(1).values)
            expected_candidates = source_reference(op.scores, lens)
            assert torch.equal(
                op.candidates.sort(1).values, expected_candidates.sort(1).values
            )
            # The cached graph must obey changed lengths and changed input values.
            lens.zero_()
            q.zero_()
            graph.replay()
            assert (op.output == -1).all().item()
            results.append(
                dict(
                    length=length,
                    group_rows=group,
                    rows=rows,
                    max_abs=absolute,
                    nrmse=(square / max(ref_square, 1e-30)) ** 0.5,
                    baseline_mode=native.baseline_mode,
                    native_strict_mismatches=native_failures,
                    native_max_abs=native_max_abs,
                    fixed_score_topk="exact",
                    graph_changes="passed",
                )
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = run_checks()
    save(args.output, dict(status="passed", checks=results))
    print(
        f"Passed {len(results)} full/grouped native-parity, FP32-audit and graph cases",
        flush=True,
    )


if __name__ == "__main__":
    main()

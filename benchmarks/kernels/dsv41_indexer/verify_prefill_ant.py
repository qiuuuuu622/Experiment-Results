# SPDX-License-Identifier: Apache-2.0
"""Dedicated Ant prefill scheduler regression, including actual FP8 semantics."""

import argparse
import json

from .common import check_vllm, save
from .verify_prefill import make_case


def run_checks():
    check_vllm()
    import torch

    from vllm_dsv41_indexer.prefill import PrefillIndexer
    from vllm_dsv41_indexer.prefill_keys_major_ant import AntPrefillIndexer
    from vllm_dsv41_indexer.reference import (
        scores_reference,
        source_reference,
        stable_topk,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    cases = [
        (1, 1, 7, 1),
        (16, 2, 517, 2),
        (33, 1, 517, 1),
        (65, 1, 4099, 2),
        (97, 1, 129, 1),
        (64, 2, 4099, 1),
    ]
    for rows, requests, keys, ratio in cases:
        inputs, candidates = make_case(rows, requests, keys, ratio)
        if rows > 1:
            inputs.lengths[0] = 0
            inputs.lengths[1] = 1
        if rows == 97:
            inputs.lengths[:32] = 0
            inputs.lengths[64:] = 0
        for compact in (False, True):
            kw = dict(
                candidate_blocks=candidates if compact else None, source=not compact
            )
            native = PrefillIndexer(inputs, backend="deepgemm", **kw)
            current = PrefillIndexer(inputs, backend="keys_major", **kw)
            ant = AntPrefillIndexer(inputs, **kw)
            for op in (native, current, ant):
                op.forward()
            maximum = square = norm = fp32_max = native_fp32_max = 0.0
            minimum_overlap = 1.0
            for start, end, request in ant.chunks:
                got = ant.score_chunk(start, end, request)
                ref = native.score_chunk(start, end, request)
                old = current.score_chunk(start, end, request)
                values, positions = [], []
                for row, n in enumerate(ant.lengths[start:end].tolist()):
                    pos = (
                        (
                            ant.state.blocks[request].long()[:, None] * 8
                            + torch.arange(8, device="cuda")
                        ).flatten()[:n]
                        if compact
                        else torch.arange(n, device="cuda")
                    )
                    a, b = got[row, :n], ref[row, :n]
                    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
                    torch.testing.assert_close(a, old[row, :n], rtol=1e-3, atol=1e-4)
                    if n:
                        error = a - b
                        maximum = max(maximum, error.abs().max().item())
                        square += error.square().sum().item()
                        norm += b.square().sum().item()
                    values.append(a.clone())
                    positions.append(pos)
                    want = set(
                        pos[
                            torch.argsort(b, descending=True, stable=True)[:512]
                        ].tolist()
                    )
                    actual = [p for p in ant.output[start + row].tolist() if p >= 0]
                    assert len(actual) == len(set(actual)) == min(n, 512)
                    assert all(p < inputs.lengths[start + row].item() for p in actual)
                    minimum_overlap = min(
                        minimum_overlap,
                        len(set(actual) & want) / len(want) if want else 1.0,
                    )
                    if set(actual) != want:
                        boundary = b[
                            torch.argsort(b, descending=True, stable=True)[
                                min(n, 512) - 1
                            ]
                        ]
                        lookup = dict(zip(pos.tolist(), b.tolist()))
                        assert (
                            max(
                                abs(lookup[p] - boundary.item())
                                for p in set(actual) ^ want
                            )
                            <= 2 * maximum + 1e-7
                        )
                exact = stable_topk(values, positions)
                assert torch.equal(
                    ant.output[start:end].sort(1).values, exact.sort(1).values
                )
                if not compact:
                    want_blocks = source_reference(got, inputs.lengths[start:end])
                    assert torch.equal(
                        ant.candidates[start:end].sort(1).values,
                        want_blocks.sort(1).values,
                    )
                    # FP32 diagnostic is deliberately separate from native parity.
                    diagnostic_rows = list(range(start, min(start + 3, end)))
                    fp32, _ = scores_reference(
                        inputs.q[diagnostic_rows],
                        inputs.weights[diagnostic_rows],
                        inputs.cache,
                        inputs.table,
                        inputs.requests[diagnostic_rows],
                        inputs.lengths[diagnostic_rows],
                    )
                    for local, exact_scores in enumerate(fp32):
                        n = len(exact_scores)
                        if n:
                            fp32_max = max(
                                fp32_max,
                                (got[local, :n] - exact_scores).abs().max().item(),
                            )
                            native_fp32_max = max(
                                native_fp32_max,
                                (ref[local, :n] - exact_scores).abs().max().item(),
                            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ant.forward()
            original_lens = inputs.lengths.clone()
            inputs.lengths.zero_()
            graph.replay()
            assert (ant.output == -1).all().item()
            if ant.source:
                assert (ant.candidates == -1).all().item()
            inputs.lengths.copy_(original_lens.flip(0))
            graph.replay()
            native.forward()
            for start, end, request in ant.chunks:
                actual_scores = ant.score_chunk(start, end, request)
                native_scores = native.score_chunk(start, end, request)
                replay_values, replay_positions = [], []
                for row, count in enumerate(ant.lengths[start:end].tolist()):
                    torch.testing.assert_close(
                        actual_scores[row, :count],
                        native_scores[row, :count],
                        rtol=1e-3,
                        atol=1e-4,
                    )
                    replay_values.append(actual_scores[row, :count].clone())
                    replay_positions.append(
                        (
                            ant.state.blocks[request].long()[:, None] * 8
                            + torch.arange(8, device="cuda")
                        ).flatten()[:count]
                        if compact
                        else torch.arange(count, device="cuda")
                    )
                expected = stable_topk(replay_values, replay_positions)
                assert torch.equal(
                    ant.output[start:end].sort(1).values, expected.sort(1).values
                )
                if ant.source:
                    expected_blocks = source_reference(
                        actual_scores, inputs.lengths[start:end]
                    )
                    assert torch.equal(
                        ant.candidates[start:end].sort(1).values,
                        expected_blocks.sort(1).values,
                    )
            # Restored causal metadata must not expose any future or stale rows.
            for row, n in enumerate(inputs.lengths.tolist()):
                valid = ant.output[row][ant.output[row] >= 0]
                assert (valid < n).all().item()
            inputs.lengths.copy_(original_lens)
            graph.replay()
            records.append(
                dict(
                    rows=rows,
                    requests=requests,
                    index_keys=keys,
                    ratio=ratio,
                    compact=compact,
                    max_abs_vs_native=maximum,
                    nrmse_vs_native=(square / max(norm, 1e-30)) ** 0.5,
                    min_topk_overlap=minimum_overlap,
                    fp32_diagnostic_first3_per_chunk=fp32_max,
                    native_fp32_diagnostic=native_fp32_max,
                    graph_metadata_updates="passed",
                    status="passed",
                )
            )
            print(json.dumps(records[-1]), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = dict(status="running", checks=[])
    try:
        report["checks"] = run_checks()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Additional non-random FP8, chunk and descriptor/PTX regression evidence."""

import argparse
import hashlib
from pathlib import Path

from .common import check_vllm, save
from .verify_prefill import make_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_vllm()
    import torch

    from vllm_dsv41_indexer import prefill_keys_major_ant as implementation
    from vllm_dsv41_indexer.prefill import PrefillIndexer
    from vllm_dsv41_indexer.reference import source_reference, stable_topk

    results = []
    for distribution in ("zero_query", "repeated_key", "mixed_scale"):
        inputs, candidates = make_case(12, 3, 517, 2)
        inputs.requests.copy_(torch.tensor([2] * 2 + [0] * 5 + [1] * 5, device="cuda"))
        inputs.lengths.copy_(
            torch.tensor(
                [0, 517, 3, 499, 1, 384, 512, 100, 517, 0, 130, 258], device="cuda"
            )
        )
        raw = inputs.cache.reshape(-1, 64 * 132)
        if distribution == "zero_query":
            inputs.q.zero_()
        elif distribution == "repeated_key":
            key = torch.linspace(-2, 2, 128, device="cuda").to(torch.float8_e4m3fn)
            raw[:, : 64 * 128] = key.repeat(64).view(torch.uint8)
            raw[:, 64 * 128 :] = torch.full((64,), 0.37, device="cuda").view(
                torch.uint8
            )
        else:
            scales = torch.linspace(0.031, 3.17, raw.shape[0] * 64, device="cuda")
            raw[:, 64 * 128 :] = scales.reshape(-1, 64).view(torch.uint8)
            inputs.weights.mul_(torch.linspace(-2, 2, 32, device="cuda"))
        for compact in (False, True):
            opts = dict(
                candidate_blocks=candidates if compact else None,
                source=not compact,
                selector="ant",
                score_budget=576 * 4 * 3,
            )
            ant = implementation.AntPrefillIndexer(inputs, **opts)
            native = PrefillIndexer(inputs, backend="deepgemm", **opts)
            assert len(ant.chunks) == 5
            ant.forward()
            native.forward()
            maximum = 0.0
            for start, end, request in ant.chunks:
                got = ant.score_chunk(start, end, request)
                ref = native.score_chunk(start, end, request)
                values, positions = [], []
                for row, count in enumerate(ant.lengths[start:end].tolist()):
                    a, b = got[row, :count], ref[row, :count]
                    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
                    if count:
                        maximum = max(maximum, (a - b).abs().max().item())
                    values.append(a.clone())
                    positions.append(
                        (
                            ant.state.blocks[request].long()[:, None] * 8
                            + torch.arange(8, device="cuda")
                        ).flatten()[:count]
                        if compact
                        else torch.arange(count, device="cuda")
                    )
                expected = stable_topk(values, positions)
                assert torch.equal(
                    ant.output[start:end].sort(1).values, expected.sort(1).values
                )
                if ant.source:
                    expected = source_reference(got, inputs.lengths[start:end])
                    assert torch.equal(
                        ant.candidates[start:end].sort(1).values,
                        expected.sort(1).values,
                    )
            results.append(
                dict(
                    distribution=distribution,
                    compact=compact,
                    chunks=len(ant.chunks),
                    max_abs=maximum,
                    passed=True,
                )
            )
    plan = next(iter(ant.ant_plans.values()))
    kernel = implementation._keys_major_scores_kernel[(plan.grid_size,)](
        plan.q_desc,
        plan.k_desc,
        plan.weights,
        plan.scales,
        plan.lengths,
        plan.prefix,
        plan.out,
        plan.rows,
        plan.width,
        plan.out.stride(0),
        plan.weights.stride(0),
        plan.n_row_blocks,
        plan.n_key_tiles,
        H=32,
        D=128,
        BLOCK_R=32,
        BLOCK_K=128,
        STAGES=3,
        num_warps=8,
        enable_fp_fusion=False,
    )
    ptx = kernel.asm["ptx"]
    assert "wgmma.mma_async" in ptx and ".e4m3.e4m3" in ptx
    assert "wgmma.wait_group.sync.aligned 1" in ptx
    assert "wgmma.wait_group.sync.aligned 0" in ptx
    assert ".f16.f16" not in ptx
    out = Path(args.output)
    out.with_suffix(".ptx").write_text(ptx)
    save(
        out,
        dict(
            status="passed",
            cases=results,
            ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
            shared_bytes=kernel.metadata.shared,
            fp8_wgmma=True,
            wait_groups=[1, 0],
            fp16_mma=False,
        ),
    )


if __name__ == "__main__":
    main()

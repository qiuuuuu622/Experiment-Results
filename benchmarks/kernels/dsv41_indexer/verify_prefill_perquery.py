# SPDX-License-Identifier: Apache-2.0
"""Real per-query compact contracts, bounded arenas and dynamic CUDA graphs."""

import argparse
import hashlib
import json
from pathlib import Path

from .common import check_vllm, device_info, save


def audit_outputs(inputs, candidates, paths, captured_outputs=None):
    import torch

    from vllm_dsv41_indexer.contract import candidate_positions
    from vllm_dsv41_indexer.prefill import PrefillIndexer
    from vllm_dsv41_indexer.reference import scores_reference, stable_topk

    native = PrefillIndexer(inputs, backend="deepgemm", selector="ant")
    native.prepare()
    full = torch.empty((inputs.shape.rows, native.width), device="cuda")
    for start, end, request in native.chunks:
        full[start:end].copy_(native.score_chunk(start, end, request))
    records = {}
    first_scores = {}
    for name, op in paths.items():
        maximum = square = norm = 0.0
        overlap = 1.0
        oracle_max = 0.0
        fp32_failures = native_fp32_failures = 0
        native_fp32_max = 0.0
        changed = 0
        for chunk, (start, end) in enumerate(op.chunks):
            op.gather_chunk(chunk)
            scores = op.score_chunk(chunk).clone()
            actual_output = (
                op.output if captured_outputs is None else captured_outputs[name]
            )[start:end].clone()
            row_scores, positions = [], []
            for row in range(start, end):
                n = int(op.state.counts[row])
                pos = torch.tensor(
                    candidate_positions(
                        candidates[row].tolist(), int(inputs.lengths[row])
                    ),
                    device="cuda",
                    dtype=torch.long,
                )
                assert len(pos) == n
                s = scores[row - start, :n]
                target = full[row, pos]
                torch.testing.assert_close(s, target, rtol=1e-3, atol=1e-4)
                error = (s - target).abs()
                e = float(error.max()) if n else 0.0
                maximum = max(maximum, e)
                square += float((error * error).sum())
                norm += float((target * target).sum())
                row_scores.append(s)
                positions.append(pos)
                reference = stable_topk([target], [pos])[0]
                got = actual_output[row - start]
                good = got[got >= 0]
                assert len(good) == min(n, 512) and len(torch.unique(good)) == len(good)
                assert bool(torch.isin(good, pos).all())
                a = set(good.tolist())
                b = set(reference[reference >= 0].tolist())
                overlap = min(overlap, len(a & b) / len(b) if b else 1.0)
                if a != b:
                    changed += 1
                    chosen = target[torch.isin(pos, good)]
                    excluded = target[~torch.isin(pos, good)]
                    assert float(excluded.max() - chosen.min()) <= 2 * e + 1e-7
                if name == "direct":
                    first_scores[row] = s.clone()
                else:
                    torch.testing.assert_close(s, first_scores[row], rtol=0, atol=0)
            expected = stable_topk(row_scores, positions)
            torch.testing.assert_close(
                actual_output.sort(1).values, expected.sort(1).values, rtol=0, atol=0
            )
        for row in sorted({0, inputs.shape.rows // 2, inputs.shape.rows - 1}):
            ref, _ = scores_reference(
                inputs.q[row : row + 1],
                inputs.weights[row : row + 1],
                inputs.cache,
                inputs.table,
                inputs.requests[row : row + 1],
                inputs.lengths[row : row + 1],
                candidates[row : row + 1],
            )
            if len(ref[0]):
                pos = torch.tensor(
                    candidate_positions(
                        candidates[row].tolist(), int(inputs.lengths[row])
                    ),
                    device="cuda",
                    dtype=torch.long,
                )
                opt_error = (first_scores[row] - ref[0]).abs()
                native_error = (full[row, pos] - ref[0]).abs()
                threshold = 1e-4 + 1e-3 * ref[0].abs()
                fp32_failures += int((opt_error > threshold).sum())
                native_fp32_failures += int((native_error > threshold).sum())
                native_fp32_max = max(native_fp32_max, float(native_error.max()))
                oracle_max = max(
                    oracle_max, float((first_scores[row] - ref[0]).abs().max())
                )
        records[name] = dict(
            max_abs_vs_native=maximum,
            nrmse_vs_native=(square / max(norm, 1e-30)) ** 0.5,
            min_topk_overlap=overlap,
            changed_rows=changed,
            fp32_diagnostic_max_abs=oracle_max,
            fp32_strict_failures=fp32_failures,
            native_fp32_strict_failures=native_fp32_failures,
            native_fp32_max_abs=native_fp32_max,
            score_bytes=op.score_bytes,
            gather_bytes=op.gather_bytes,
            chunks=len(op.chunks),
            fixed_logits_selection="exact",
            direct_gather_scores="bitwise_equal",
        )
    return records


def run(output_dir):
    check_vllm()
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.module import IndexerInputs
    from vllm_dsv41_indexer.perquery_prefill import PerQueryPrefillIndexer

    from .inputs import make_inputs

    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for rows, nreq, keys, source_unique in [
        (1, 1, 7, False),
        (10, 2, 517, False),
        (33, 3, 4099, False),
        (8, 2, 20003, False),
        (32, 2, 16396, True),
        (8, 2, 20003, True),
    ]:
        q, _, w, cache, table, req, lens, candidates = make_inputs(
            dict(
                rows=rows,
                compressed_length=keys,
                rows_per_request=rows // nreq,
                page_size=64,
                candidate_blocks=2048,
            )
        )
        # Exercise non-power-of-two FP32 scales without requantizing FP8 payload.
        cache.view(cache.shape[0], 8448)[:, 8192:].view(torch.float32).mul_(1.3)
        i = IndexerInputs(q, w, cache, table, req, lens, IndexerShape(rows, keys))
        width = min(16384, (keys + 63) // 64 * 64)
        # Force multi-chunk arenas even in small correctness cases.
        paths = {
            b: PerQueryPrefillIndexer(
                i,
                candidates,
                backend=b,
                source_unique=source_unique,
                score_budget=3 * width * 4,
                gather_budget=2 * width * 132,
                max_chunk_rows=3,
            )
            for b in ("direct", "gathered_wgmma")
        }
        saved = {
            k: t.clone()
            for k, t in dict(
                q=q,
                w=w,
                cache=cache,
                table=table,
                req=req,
                lens=lens,
                candidates=candidates,
            ).items()
        }
        graphs = {}
        for name, op in paths.items():
            op.forward()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                op.forward()
            graphs[name] = g
        for step in (
            "initial",
            "changed_candidates_lengths",
            "empty_candidates",
            "zero_query",
            "zero_lengths",
            "request_page_reorder",
            "restore",
        ):
            if step == "changed_candidates_lengths":
                candidates.copy_(candidates.roll(1, 0))
                if not source_unique:
                    candidates[:, 1].copy_(candidates[:, 0])
                candidates[:, 2] = -1
                lens.copy_((lens - 3).clamp_min(0))
                lens[0] = 0
            elif step == "empty_candidates":
                candidates.fill_(-1)
                lens.copy_(saved["lens"])
            elif step == "zero_query":
                candidates.copy_(saved["candidates"])
                lens.copy_(saved["lens"])
                q.zero_()
            elif step == "zero_lengths":
                lens.zero_()
                table.copy_(table.flip(1))
            elif step == "request_page_reorder":
                q.copy_(saved["q"])
                lens.copy_(saved["lens"])
                req.copy_(req.flip(0))
                table.copy_(table.flip(1))
                w.mul_(1.0001)
                cache.view(cache.shape[0], 8448)[:, 8192:].view(torch.float32).mul_(
                    1.0003
                )
            elif step == "restore":
                for k, t in dict(
                    q=q,
                    w=w,
                    cache=cache,
                    table=table,
                    req=req,
                    lens=lens,
                    candidates=candidates,
                ).items():
                    t.copy_(saved[k])
            for op in paths.values():
                op.state.blocks.fill_(2147483647)
                op.state.counts.fill_(-777)
                op.scores.fill_(float("nan"))
                op.output.fill_(-42)
                if op.keys is not None:
                    op.keys.fill_(127)
                    op.scales.fill_(float("nan"))
            for name, g in graphs.items():
                g.replay()
            torch.cuda.synchronize()
            snapshots = {name: op.output.clone() for name, op in paths.items()}
            result = audit_outputs(i, candidates, paths, snapshots)
            for name, op in paths.items():
                torch.testing.assert_close(
                    op.forward().sort(1).values,
                    snapshots[name].sort(1).values,
                    rtol=0,
                    atol=0,
                )
            result["graph_protocol"] = (
                "poison buffers, replay, clone all outputs, then oracle and eager"
            )
            records.append(
                dict(
                    rows=rows,
                    requests=nreq,
                    index_keys=keys,
                    step=step,
                    source_unique=source_unique,
                    checks=result,
                )
            )
            print(
                json.dumps(dict(rows=rows, keys=keys, step=step, status="passed")),
                flush=True,
            )
    compiled = {}
    for name, op in paths.items():
        kernel = op.compiled_kernel
        ptx = kernel.asm["ptx"]
        (output_dir / (name + ".ptx")).write_text(ptx)
        compiled[name] = dict(
            ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
            n_regs=kernel.n_regs,
            n_spills=kernel.n_spills,
            shared_bytes=kernel.metadata.shared,
            fp8_wgmma=("wgmma.mma_async" in ptx and "e4m3.e4m3" in ptx),
            fp16_mma="f16.f16" in ptx,
        )
        assert compiled[name]["fp8_wgmma"] and not compiled[name]["fp16_mma"]
    return dict(status="passed", device=device_info(), cases=records, compiled=compiled)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output_dir = Path(a.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save(a.output, run(output_dir))


if __name__ == "__main__":
    main()

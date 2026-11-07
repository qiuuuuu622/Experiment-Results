# SPDX-License-Identifier: Apache-2.0
"""Per-query compact vs existing full-score+mask, including all preparation."""

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from .benchmark_prefill_ant import sampled
from .benchmark_prefill_model import CASES, ModelPath, model_inputs
from .common import check_vllm, device_info, save


def source_fixture(inputs):
    import torch

    from vllm_dsv41_indexer.prefill import PrefillIndexer

    g = torch.Generator(device="cuda").manual_seed(2026110702)
    source_inputs = replace(
        inputs,
        q=torch.randn(inputs.q.shape, device="cuda", generator=g).to(inputs.q.dtype),
        weights=torch.randn(inputs.weights.shape, device="cuda", generator=g)
        / (128 * 32) ** 0.5,
    )
    source = PrefillIndexer(
        source_inputs, backend="keys_major", source=True, selector="ant"
    )
    source.forward()
    torch.cuda.synchronize()
    assert not torch.equal(source_inputs.q, inputs.q)
    assert not torch.equal(source_inputs.weights, inputs.weights)
    return source, source.candidates.clone()


def validate(inputs, candidates, paths, full_paths):
    """All visible candidate scores vs DeepGEMM, in bounded 32-query chunks."""
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.perquery_prefill import PerQueryPrefillIndexer
    from vllm_dsv41_indexer.prefill import PrefillIndexer
    from vllm_dsv41_indexer.reference import scores_reference, stable_topk

    width = min(16384, (inputs.shape.max_keys + 63) // 64 * 64)
    native = PrefillIndexer(
        inputs,
        backend="deepgemm",
        score_budget=32 * ((inputs.shape.max_keys + 63) // 64 * 64) * 4,
    )
    native.prepare()
    baseline_checks = {
        name: dict(max_abs_vs_native=0.0, checked_rows=0) for name in full_paths
    }
    stats = dict(
        rows=0,
        max_abs_vs_native=0.0,
        nrmse_numerator=0.0,
        nrmse_denominator=0.0,
        min_topk_overlap=1.0,
        fp32_strict_failures=0,
        native_fp32_strict_failures=0,
        fp32_max_abs=0.0,
        native_fp32_max_abs=0.0,
    )
    for start, end, request in native.chunks:
        reference = native.score_chunk(start, end, request)
        sub = replace(
            inputs,
            q=inputs.q[start:end],
            weights=inputs.weights[start:end],
            requests=inputs.requests[start:end],
            lengths=inputs.lengths[start:end],
            shape=IndexerShape(end - start, inputs.shape.max_keys),
        )
        a = PerQueryPrefillIndexer(sub, candidates[start:end], source_unique=True)
        b = PerQueryPrefillIndexer(
            sub, candidates[start:end], backend="gathered_wgmma", source_unique=True
        )
        a.forward()
        b.forward()
        a_score = a.scores[: end - start].clone()
        b_score = b.scores[: end - start]
        counts = a.state.counts
        col = torch.arange(width, device="cuda")[None, :]
        valid = col < counts[:, None]
        positions = a.state.blocks[:, col[0] // 8].long() * 8 + col % 8
        positions = positions.clamp(0, reference.shape[1] - 1)
        target = reference.gather(1, positions)
        torch.testing.assert_close(a_score[valid], b_score[valid], rtol=0, atol=0)
        torch.testing.assert_close(a_score[valid], target[valid], rtol=1e-3, atol=1e-4)
        err = torch.where(valid, (a_score - target).abs(), 0.0)
        row_error = err.max(1).values
        stats["max_abs_vs_native"] = max(
            stats["max_abs_vs_native"], float(row_error.max())
        )
        stats["nrmse_numerator"] += float((err * err).sum())
        stats["nrmse_denominator"] += float(
            torch.where(valid, target * target, 0.0).sum()
        )
        expected = stable_topk(
            [a_score[r, : int(counts[r])] for r in range(end - start)],
            [positions[r, : int(counts[r])] for r in range(end - start)],
        )
        native_top = stable_topk(
            [target[r, : int(counts[r])] for r in range(end - start)],
            [positions[r, : int(counts[r])] for r in range(end - start)],
        )
        for name, path in full_paths.items():
            actual_scores = path.score_chunk(start, end, request).gather(1, positions)
            torch.testing.assert_close(
                actual_scores[valid], target[valid], rtol=1e-3, atol=1e-4
            )
            own_expected = stable_topk(
                [actual_scores[r, : int(counts[r])] for r in range(end - start)],
                [positions[r, : int(counts[r])] for r in range(end - start)],
            )
            torch.testing.assert_close(
                path.op.output[start:end].sort(1).values,
                own_expected.sort(1).values,
                rtol=0,
                atol=0,
            )
            baseline_checks[name]["max_abs_vs_native"] = max(
                baseline_checks[name]["max_abs_vs_native"],
                float((actual_scores[valid] - target[valid]).abs().max())
                if bool(valid.any())
                else 0.0,
            )
            baseline_checks[name]["checked_rows"] += end - start
        for name in ("direct", "gathered_wgmma"):
            got = paths[name].output[start:end]
            torch.testing.assert_close(
                got.sort(1).values, expected.sort(1).values, rtol=0, atol=0
            )
        for row in range(end - start):
            n = int(counts[row])
            got = paths["direct"].output[start + row]
            good = got[got >= 0]
            assert len(good) == min(n, 512) and len(torch.unique(good)) == len(good)
            pos = positions[row, :n]
            t = target[row, :n]
            assert bool(torch.isin(good, pos).all())
            chosen = torch.isin(pos, good)
            if n > 512:
                assert (
                    float(t[~chosen].max() - t[chosen].min())
                    <= 2 * float(row_error[row]) + 1e-7
                )
            nt = native_top[row]
            want = set(nt[nt >= 0].tolist())
            actual = set(good.tolist())
            stats["min_topk_overlap"] = min(
                stats["min_topk_overlap"],
                len(actual & want) / len(want) if want else 1.0,
            )
        if (
            start == 0
            or end == inputs.shape.rows
            or start <= inputs.shape.rows // 2 < end
        ):
            row = min(max(inputs.shape.rows // 2 - start, 0), end - start - 1)
            ref, _ = scores_reference(
                sub.q[row : row + 1],
                sub.weights[row : row + 1],
                sub.cache,
                sub.table,
                sub.requests[row : row + 1],
                sub.lengths[row : row + 1],
                candidates[start + row : start + row + 1],
            )
            n = int(counts[row])
            f = ref[0]
            diff = (a_score[row, :n] - f).abs()
            ndiff = (target[row, :n] - f).abs()
            threshold = 1e-4 + 1e-3 * f.abs()
            stats["fp32_strict_failures"] += int((diff > threshold).sum())
            stats["native_fp32_strict_failures"] += int((ndiff > threshold).sum())
            if n:
                stats["fp32_max_abs"] = max(stats["fp32_max_abs"], float(diff.max()))
                stats["native_fp32_max_abs"] = max(
                    stats["native_fp32_max_abs"], float(ndiff.max())
                )
        stats["rows"] += end - start
        del a, b, reference
    stats["nrmse_vs_native"] = (
        stats.pop("nrmse_numerator") / max(stats.pop("nrmse_denominator"), 1e-30)
    ) ** 0.5
    stats["compact_direct_gather_score_comparison"] = (
        "all candidate scores bitwise equal"
    )
    stats["fixed_logits_topk"] = "all rows exact logical sets"
    stats["full_baselines"] = baseline_checks
    return stats


def run_case(case, reverse, phases):
    import torch

    from vllm_dsv41_indexer.perquery_prefill import PerQueryPrefillIndexer

    label, queries, contexts = case
    inputs = model_inputs(queries, contexts, 1)
    source, candidates = source_fixture(inputs)
    full_paths = {
        "full_keys_major_mask": ModelPath(inputs, "keys_major", "consumer", candidates),
        "full_deepgemm_mask": ModelPath(inputs, "deepgemm", "consumer", candidates),
    }
    paths = {
        name: PerQueryPrefillIndexer(
            inputs, candidates, backend=name, source_unique=True
        )
        for name in ("direct", "gathered_wgmma")
    }
    for p in paths.values():
        p.forward()
    for baseline in full_paths.values():
        baseline.forward()
    correctness = validate(inputs, candidates, paths, full_paths)
    torch.cuda.synchronize()
    records = []
    variants = [
        "full_deepgemm_mask",
        "full_keys_major_mask",
        "direct",
        "gathered_wgmma",
    ]
    if reverse:
        variants.reverse()
    for graph in (False, True):
        for name in variants:
            op = full_paths[name] if name in full_paths else paths[name]
            if "pipeline" in phases:
                records.append(
                    dict(
                        variant=name,
                        phase="pipeline",
                        execution="graph" if graph else "eager",
                        measurement=sampled(op.forward, graph),
                    )
                )
            if "parts" in phases:
                if name in full_paths:
                    records.append(
                        dict(
                            variant=name,
                            phase="prepare",
                            execution="graph" if graph else "eager",
                            measurement=sampled(op.op.prepare, graph),
                        )
                    )
                    op.op.prepare()
                    records.append(
                        dict(
                            variant=name,
                            phase="score_only",
                            execution="graph" if graph else "eager",
                            measurement=sampled(op.score_only, graph),
                        )
                    )
                else:

                    def prep(op=op):
                        op.publish()
                        for c in range(len(op.chunks)):
                            op.gather_chunk(c)

                    records.append(
                        dict(
                            variant=name,
                            phase="publish_and_all_gather",
                            execution="graph" if graph else "eager",
                            measurement=sampled(prep, graph),
                        )
                    )
                    op.publish()
                    # Staging is reused: measure each correctly prepared chunk,
                    # never score all queries with the final chunk's stale K.
                    for c, (start, end) in enumerate(op.chunks):
                        op.gather_chunk(c)
                        records.append(
                            dict(
                                variant=name,
                                phase="score_only_chunk",
                                query_start=start,
                                query_end=end,
                                execution="graph" if graph else "eager",
                                measurement=sampled(
                                    lambda op=op, c=c: op.score_chunk(c), graph
                                ),
                            )
                        )
                        op.score_chunk(c)
                        records.append(
                            dict(
                                variant=name,
                                phase="topk_map_chunk",
                                query_start=start,
                                query_end=end,
                                execution="graph" if graph else "eager",
                                measurement=sampled(
                                    lambda op=op, c=c: op.select_chunk(c), graph
                                ),
                            )
                        )
        if "source" in phases:
            records.append(
                dict(
                    variant="source20_independent_q_weights",
                    phase="source_generation",
                    execution="graph" if graph else "eager",
                    measurement=sampled(source.forward, graph),
                )
            )
    valid = (candidates >= 0) & (candidates * 8 < inputs.lengths[:, None])
    counts = torch.where(
        valid, (inputs.lengths[:, None] - candidates * 8).clamp(0, 8), 0
    ).sum(1)
    return dict(
        case=label,
        query_lens=queries,
        contexts=contexts,
        rows=sum(queries),
        ratio=1,
        heads=32,
        head_dim=128,
        candidate_layout="[Q,2048] perquery",
        candidate_positions_min=int(counts.min()),
        candidate_positions_max=int(counts.max()),
        candidate_positions_sum=int(counts.sum()),
        different_adjacent_candidate_sets=int(
            (candidates.sort(1).values[1:] != candidates.sort(1).values[:-1])
            .any(1)
            .sum()
        ),
        candidate_sha256=hashlib.sha256(candidates.cpu().numpy().tobytes()).hexdigest(),
        correctness=correctness,
        records=records,
        workspaces={
            n: dict(
                score_bytes=p.score_bytes,
                gather_bytes=p.gather_bytes,
                capacity=p.capacity,
                chunks=p.chunks,
            )
            for n, p in paths.items()
        },
        baseline=(
            "DeepGEMM and existing keys_major fullscore + perquery mask, "
            "same Ant selector; numerical gate against DeepGEMM"
        ),
        scope=(
            "single consumer: publication/normalize, all gather/score chunks, "
            "fused map/topk; source generation separately measured"
        ),
        score_only_parts=(
            "each score/selector chunk measured after own preparation; "
            "sums are derived, not continuous invocation timings"
        ),
        cache="warm repeated calls, no L2 flush",
        peak_scope="all compared variants and independent source resident",
        source_generation_in_consumer_pipeline=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases",
        nargs="+",
        default=[c[0] for c in CASES],
    )
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--phases", nargs="+", default=["pipeline", "parts", "source"])
    p.add_argument("--output", required=True)
    p.add_argument("--identity", required=True)
    a = p.parse_args()
    assert a.cases and set(a.cases) <= {case[0] for case in CASES}
    assert set(a.phases) <= {"pipeline", "parts", "source"} and a.phases
    if a.validate_only:
        a.phases = []
    check_vllm()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    identity = Path(a.identity)
    assert identity.is_file()
    report = dict(
        status="running",
        purpose="validation_only" if a.validate_only else "performance",
        args=vars(a),
        device=device_info(),
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        identity_sha256=hashlib.sha256(identity.read_bytes()).hexdigest(),
        cases=[],
    )
    for case in CASES:
        if case[0] in a.cases:
            result = run_case(case, a.reverse, a.phases)
            report["cases"].append(result)
            save(a.output, report)
            print(
                json.dumps(
                    dict(
                        case=case[0], status="completed", records=len(result["records"])
                    )
                ),
                flush=True,
            )
    report["status"] = "completed"
    save(a.output, report)


if __name__ == "__main__":
    main()

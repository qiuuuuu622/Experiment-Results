# SPDX-License-Identifier: Apache-2.0
"""Architecture-aligned DSV4.1 prefill; independent scorer experiment, no model."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from .benchmark_prefill_ant import sampled
from .common import check_vllm, device_info, save
from .verify_prefill import make_case

CASES = [
    ("initial_2k", [2048], [2048]),
    ("initial_8k", [8192], [8192]),
    ("initial_32k", [32768], [32768]),
    ("steady_16k", [512], [16384]),
    ("steady_64k", [2048], [65536]),
    ("steady_256k", [4096], [262144]),
    ("tail_1m", [256], [1048576]),
    ("ragged", [512, 128], [65535, 16384]),
]


def model_inputs(query_lens, final_contexts, ratio):
    import torch

    rows, requests = sum(query_lens), len(query_lens)
    max_keys = max(final_contexts) // ratio
    inputs, _ = make_case(rows, requests, max_keys, ratio)
    visible, request_ids = [], []
    for request, (n, end) in enumerate(zip(query_lens, final_contexts)):
        assert 0 < n <= end <= 1048576
        visible.extend((end - n + j + 1) // ratio for j in range(n))
        request_ids.extend([request] * n)
    inputs.requests.copy_(torch.tensor(request_ids, device="cuda", dtype=torch.int32))
    inputs.lengths.copy_(torch.tensor(visible, device="cuda", dtype=torch.int32))
    return inputs


class ModelPath:
    def __init__(self, inputs, backend, role, candidates=None):
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
            apply_candidate_mask,
        )
        from vllm_dsv41_indexer.prefill import PrefillIndexer
        from vllm_dsv41_indexer.prefill_keys_major_ant import AntPrefillIndexer

        options = dict(
            source=role == "source",
            selector="ant",
            native_gather=False,
            score_budget=1 << 30,
        )
        self.op = (
            AntPrefillIndexer(inputs, **options)
            if backend == "ant"
            else PrefillIndexer(inputs, backend=backend, **options)
        )
        self.candidates = candidates
        self.apply_mask = apply_candidate_mask
        self.ant = backend == "ant"

    def metadata(self):
        if self.ant:
            self.op.metadata()

    def score_chunk(self, start, end, request, *, mask=True):
        scores = self.op.score_chunk(start, end, request)
        if mask and self.candidates is not None:
            self.apply_mask(
                scores,
                None,
                self.op.inputs.lengths[start:end],
                self.candidates[start:end],
                8,
            )
        return scores

    def forward(self):
        self.op.prepare()
        self.metadata()
        for start, end, request in self.op.chunks:
            self.op.select_chunk(
                self.score_chunk(start, end, request), start, end, request
            )
        return self.op.output

    def score_only(self):
        for start, end, request in self.op.chunks:
            self.score_chunk(start, end, request, mask=False)


def correctness(paths, inputs, candidates):
    import torch

    from vllm_dsv41_indexer.reference import source_reference, stable_topk

    native = paths["deepgemm"]
    checks = {}
    for name, path in paths.items():
        op = path.op
        maximum = squares = norm = max_boundary = 0.0
        changed_rows = changed_candidate_rows = 0
        candidate_overlap = 1.0
        candidate_boundary = fp32_max = native_fp32_max = 0.0
        fp32_squared = fp32_norm = 0.0
        fp32_strict_failures = native_fp32_strict_failures = 0
        min_overlap = 1.0
        valid = op.output >= 0
        assert ((op.output < inputs.lengths[:, None]) | ~valid).all().item()
        sorted_output = op.output.sort(1).values
        assert (
            not (
                (sorted_output[:, 1:] == sorted_output[:, :-1])
                & (sorted_output[:, 1:] >= 0)
            )
            .any()
            .item()
        )
        assert torch.equal(valid.sum(1), inputs.lengths.clamp(max=512).long())
        if op.source:
            selected_blocks = op.candidates
            present = selected_blocks >= 0
            block_lengths = (inputs.lengths + 7) // 8
            assert torch.equal(present.sum(1), block_lengths.clamp(max=2048).long())
            assert ((selected_blocks < block_lengths[:, None]) | ~present).all().item()
            ordered = selected_blocks.sort(1).values
            assert (
                not ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0))
                .any()
                .item()
            )
            assert (
                (
                    (selected_blocks == ((inputs.lengths - 1) // 8)[:, None]).any(1)
                    | (inputs.lengths == 0)
                )
                .all()
                .item()
            )
        if candidates is not None:
            flags = torch.zeros(
                (inputs.shape.rows, (op.width + 7) // 8 + 1),
                device="cuda",
                dtype=torch.bool,
            )
            flags.scatter_(1, candidates.long().clamp_min(0), candidates >= 0)
            # Padding's repeated slot 0 may race; overwrite all real candidate slots.
            rr, cc = torch.where(candidates >= 0)
            flags[rr, candidates[rr, cc].long()] = True
            selected = flags.gather(1, (op.output.long().clamp_min(0) // 8))
            assert (selected | ~valid).all().item()
        for start, end, request in op.chunks:
            ref = native.score_chunk(start, end, request)
            got = path.score_chunk(start, end, request)
            columns = torch.arange(op.width, device="cuda")
            for lo in range(0, end - start, 64):
                hi = min(lo + 64, end - start)
                visible = (
                    columns[None, :] < inputs.lengths[start + lo : start + hi, None]
                )
                a, b = got[lo:hi][visible], ref[lo:hi][visible]
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
                finite = torch.isfinite(b)
                error = a[finite] - b[finite]
                if error.numel():
                    maximum = max(maximum, error.abs().max().item())
                    squares += error.square().sum().item()
                    norm += b[finite].square().sum().item()
            # Stable selector exactness on every chunk boundary and middle row.
            for row in sorted({0, (end - start) // 2, end - start - 1}):
                count = int(inputs.lengths[start + row])
                values = got[row, :count]
                positions = torch.arange(count, device="cuda")
                finite = torch.isfinite(values)
                expected = stable_topk([values[finite]], [positions[finite]])
                assert torch.equal(
                    op.output[start + row].sort().values, expected[0].sort().values
                )
                if count:
                    positions = torch.linspace(
                        0, count - 1, min(count, 3072), device="cuda"
                    ).long()
                    positions = torch.unique(
                        torch.cat(
                            [
                                positions,
                                op.output[start + row][
                                    op.output[start + row] >= 0
                                ].long(),
                                native.op.output[start + row][
                                    native.op.output[start + row] >= 0
                                ].long(),
                            ]
                        )
                    )
                    positions = positions[torch.isfinite(got[row, positions])]
                    if positions.numel():
                        k = op.keys[request, positions].float()
                        q = inputs.q[start + row].float()
                        dot = torch.einsum("hd,kd->kh", q, k)
                        exact = (
                            (dot * op.scales[request, positions, None]).clamp_min(0)
                            * inputs.weights[start + row]
                        ).sum(-1)
                        a, b = got[row, positions], ref[row, positions]
                        fp32_max = max(fp32_max, (a - exact).abs().max().item())
                        native_fp32_max = max(
                            native_fp32_max, (b - exact).abs().max().item()
                        )
                        fp32_squared += (a - exact).square().sum().item()
                        fp32_norm += exact.square().sum().item()
                        fp32_strict_failures += int(
                            (~torch.isclose(a, exact, rtol=1e-3, atol=1e-4)).sum()
                        )
                        native_fp32_strict_failures += int(
                            (~torch.isclose(b, exact, rtol=1e-3, atol=1e-4)).sum()
                        )
                if op.source:
                    expected = source_reference(
                        got[row : row + 1],
                        inputs.lengths[start + row : start + row + 1],
                    )
                    assert torch.equal(
                        op.candidates[start + row].sort().values,
                        expected[0].sort().values,
                    )
            if op.source:
                different = torch.where(
                    (
                        op.candidates[start:end].sort(1).values
                        != native.op.candidates[start:end].sort(1).values
                    ).any(1)
                )[0].tolist()
                changed_candidate_rows += len(different)
                for row in different:
                    actual = {p for p in op.candidates[start + row].tolist() if p >= 0}
                    wanted = {
                        p for p in native.op.candidates[start + row].tolist() if p >= 0
                    }
                    candidate_overlap = min(
                        candidate_overlap, len(actual & wanted) / len(wanted)
                    )
                    count = int(inputs.lengths[start + row])
                    score = ref[row].clone()
                    score[count:] = -torch.inf
                    maxima = score.reshape(-1, 8).max(1).values
                    maxima[(count - 1) // 8] = torch.inf
                    cutoff = maxima[list(wanted)].min()
                    delta = (maxima[list(actual ^ wanted)] - cutoff).abs().max().item()
                    candidate_boundary = max(candidate_boundary, delta)
                    assert delta <= 2 * maximum + 1e-7
            native_sorted = native.op.output[start:end].sort(1).values
            actual_sorted = op.output[start:end].sort(1).values
            differences = torch.where((native_sorted != actual_sorted).any(1))[
                0
            ].tolist()
            changed_rows += len(differences)
            # Explain every changed set by the actual native boundary/error.
            for row in differences:
                wanted = {p for p in native.op.output[start + row].tolist() if p >= 0}
                actual = {p for p in op.output[start + row].tolist() if p >= 0}
                min_overlap = min(min_overlap, len(actual & wanted) / len(wanted))
                cutoff = ref[row, list(wanted)].min().item()
                delta = (ref[row, list(actual ^ wanted)] - cutoff).abs().max().item()
                max_boundary = max(max_boundary, delta)
                assert delta <= 2 * maximum + 1e-7
        checks[name] = dict(
            max_abs_vs_native=maximum,
            fp32_diagnostic_max_abs=fp32_max,
            native_fp32_diagnostic_max_abs=native_fp32_max,
            fp32_diagnostic_nrmse=(fp32_squared / max(fp32_norm, 1e-30)) ** 0.5,
            fp32_strict_failures=fp32_strict_failures,
            native_fp32_strict_failures=native_fp32_strict_failures,
            fp32_scope=(
                "first/middle/last per chunk; <=3072 uniform keys "
                "+ both top512 boundaries; diagnostic only"
            ),
            changed_candidate_rows=changed_candidate_rows,
            min_candidate_overlap=candidate_overlap,
            max_candidate_boundary_distance=candidate_boundary,
            nrmse_vs_native=(squares / max(norm, 1e-30)) ** 0.5,
            min_topk_overlap=min_overlap,
            changed_rows=changed_rows,
            max_changed_boundary_distance=max_boundary,
            score_scope="every visible score; all changed top-k sets explained",
            stable_reference_scope="first/middle/last row of every arena chunk",
            source_reference_scope="first/middle/last row of every arena chunk",
        )
    return checks


def run_case(case, role, reverse):
    import torch

    from vllm_dsv41_indexer.prefill import PrefillIndexer
    from vllm_dsv41_indexer.prefill_keys_major_ant import AntKeysMajorScores

    label, query_lens, contexts = case
    ratio = 2 if role == "full" else 1
    inputs = model_inputs(query_lens, contexts, ratio)
    assert max(contexts) // ratio > 512  # Native short-context bypass is excluded.
    candidates = None
    if role == "consumer":
        # Layer 20 and later layers use different learned head weights. Fixed
        # synthetic tensors preserve this distinction without loading a model.
        generator = torch.Generator(device="cuda").manual_seed(2026110702)
        source_inputs = replace(
            inputs,
            q=torch.randn(inputs.q.shape, device="cuda", generator=generator).to(
                inputs.q.dtype
            ),
            weights=torch.randn(
                inputs.weights.shape, device="cuda", generator=generator
            )
            / (128 * 32) ** 0.5,
        )
        source = PrefillIndexer(
            source_inputs,
            backend="keys_major",
            source=True,
            selector="ant",
            native_gather=False,
        )
        source.forward()
        candidates = source.candidates.clone()
        del source
    paths, planning = {}, {}
    for name, backend in [
        ("deepgemm", "deepgemm"),
        ("current_keys_major", "keys_major"),
        ("ant", "ant"),
    ]:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter_ns()
        paths[name] = ModelPath(inputs, backend, role, candidates)
        torch.cuda.synchronize()
        planning[name] = dict(
            wall_us=(time.perf_counter_ns() - begin) / 1000,
            allocated_before=before,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            additional_peak_bytes=torch.cuda.max_memory_allocated() - before,
        )
    for path in paths.values():
        path.forward()
    checks = correctness(paths, inputs, candidates)
    ant = paths["ant"]

    def rebuild():
        op = ant.op
        for start, end, request in op.chunks:
            op.ant_plans[(start, end)] = AntKeysMajorScores(
                inputs.q[start:end],
                inputs.weights[start:end],
                op.keys[request],
                op.scales[request],
                op.lengths[start:end],
                op.scores[: end - start],
            )

    def dynamic_pipeline():
        rebuild()
        ant.forward()

    variants = [(name + "_pipeline", path.forward) for name, path in paths.items()]
    variants += [
        (name + "_score_only", path.score_only) for name, path in paths.items()
    ]
    variants += [
        ("common_key_prepare", ant.op.prepare),
        ("ant_prefix_metadata", ant.metadata),
    ]
    if candidates is not None:

        def mask_only():
            for start, end, _ in ant.op.chunks:
                ant.apply_mask(
                    ant.op.scores[: end - start],
                    None,
                    inputs.lengths[start:end],
                    candidates[start:end],
                    8,
                )

        variants += [("common_perquery_mask", mask_only)]
    timings = {}
    for execution in ("eager", "graph"):
        selected = variants + (
            [
                ("ant_descriptor_plan_only", rebuild),
                ("ant_dynamic_descriptors_pipeline", dynamic_pipeline),
            ]
            if execution == "eager"
            else []
        )
        timings[execution] = {}
        for name, fn in reversed(selected) if reverse else selected:
            ant.op.prepare()
            ant.metadata()
            timings[execution][name] = sampled(fn, execution == "graph")
    lengths = inputs.lengths.tolist()
    candidate_stats = None
    if candidates is not None:
        valid_blocks = (candidates >= 0) & (candidates * 8 < inputs.lengths[:, None])
        position_counts = torch.where(
            valid_blocks, (inputs.lengths[:, None] - candidates * 8).clamp(0, 8), 0
        ).sum(1)
        sorted_candidates = candidates.sort(1).values
        candidate_stats = dict(
            valid_blocks_per_query=valid_blocks.sum(1).tolist(),
            candidate_positions_per_query=position_counts.tolist(),
            adjacent_rows_with_different_sets=int(
                (sorted_candidates[1:] != sorted_candidates[:-1]).any(1).sum()
            ),
            source_projection_fixture="independent seeded Q/weights; same FP8 cache",
        )
    return dict(
        case=dict(
            label=label,
            role=role,
            ratio=ratio,
            index_heads=32,
            head_dim=128,
            layers=[2, 8, 14]
            if role == "full"
            else [20]
            if role == "source"
            else [24, 28, 32, 36],
            query_lens=query_lens,
            contexts=contexts,
            prefixes=[c - q for q, c in zip(query_lens, contexts)],
            key_counts=[c // ratio for c in contexts],
            visible_lengths=lengths,
            effective_score_count=sum(lengths),
            allocated_scores=inputs.shape.rows * ant.op.width,
            chunks=ant.op.chunks,
            score_arena_bytes=ant.op.scores.numel() * 4,
            request_key_preparation_bytes=ant.op.preparation_bytes,
        ),
        checks=checks,
        planning=planning,
        candidate_stats=candidate_stats,
        timings=timings,
        candidate_layout="[query,2048], perquery-indexed; sets may coincide"
        if candidates is not None
        else None,
        candidate_sha256=hashlib.sha256(candidates.cpu().numpy().tobytes()).hexdigest()
        if candidates is not None
        else None,
        source_candidate_generation="source20 output provided outside consumer timing"
        if candidates is not None
        else None,
        full_cost=(
            "prepare + prefix(Ant) + full score + perquery mask(consumer) "
            "+ same stable selector"
        ),
        score_only="prepared prefix, excludes preparation/metadata/mask/selector",
        descriptor_contract="static planned once; dynamic plan+pipeline Eager separate",
        source_candidates="included in source pipeline",
        cache="warm repeated; no flush",
        peak_scope="three variants resident together",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["full", "source", "consumer"], required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--case", choices=[c[0] for c in CASES])
    args = parser.parse_args()
    check_vllm()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    config = Path("architecture/model-config.json")
    text = json.loads(config.read_text())["text_config"]
    assert (
        text["index_n_heads"],
        text["index_head_dim"],
        text["index_topk"],
        text["candidate_topk_blocks"],
        text["candidate_block_size"],
    ) == (32, 128, 512, 2048, 8)
    assert text["candidate_source_layer_id"] == 20
    assert text["index_source_layer_ids"] == [2, 8, 14, 20, 24, 28, 32, 36]
    assert all(text["compress_ratios"][i] == 2 for i in [2, 8, 14])
    assert all(text["compress_ratios"][i] == 1 for i in [20, 24, 28, 32, 36])
    identity = Path(args.identity)
    report = dict(
        status="running",
        args=vars(args),
        device=device_info(),
        physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        source_and_runtime_identity=dict(
            path=str(identity), sha256=hashlib.sha256(identity.read_bytes()).hexdigest()
        ),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=[],
    )
    try:
        for case in CASES:
            if args.role == "full" and case[0] == "initial_32k":
                continue  # Extra long initial prefill targets source/consumer cutoff.
            if args.case and case[0] != args.case:
                continue
            result = run_case(case, args.role, args.reverse)
            report["cases"].append(result)
            save(args.output, report)
            print(
                json.dumps(
                    dict(
                        case=case[0],
                        role=args.role,
                        graph={
                            k: v["p50_us"]
                            for k, v in result["timings"]["graph"].items()
                        },
                    )
                ),
                flush=True,
            )
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Full/source and grouped verify ablations, including source-to-consumer chain."""

import argparse
import json
import time

from .common import check_sources, check_vllm, device_info, measure, save


def run_case(requests, length, group, args):
    import torch

    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
        select_candidate_blocks,
    )
    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.grouped import GroupedIndexer
    from vllm_dsv41_indexer.module import (
        FullIndexer,
        IndexerInputs,
        NativeFullIndexer,
        NativeSourceIndexer,
    )
    from vllm_dsv41_indexer.reference import stable_topk
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import make_inputs

    logical_context = length
    length = logical_context // args.ratio
    rows = requests * group
    case = dict(
        rows=rows,
        requests=requests,
        compressed_length=length,
        rows_per_request=group,
        page_size=64,
        candidate_blocks=2048,
        ratio=args.ratio,
        logical_context_length=logical_context,
    )
    q, _, w, cache, table, req, lens, _ = make_inputs(case)
    inputs = IndexerInputs(
        q, w, cache, table, req, lens, IndexerShape(rows, length, args.ratio)
    )
    native = NativeFullIndexer(inputs, next_n=group)
    hybrid = NativeSourceIndexer(inputs, next_n=group)
    rowwise = FullIndexer(inputs, source=True)
    grouped = FullIndexer(inputs, group_rows=1 << (group - 1).bit_length(), source=True)
    scan = FullIndexer(
        inputs,
        group_rows=1 << (group - 1).bit_length(),
        source=True,
        fuse_source_max=False,
    )
    full = FullIndexer(inputs, group_rows=1 << (group - 1).bit_length())
    gluon = GroupedIndexer(
        inputs, group_rows=1 << (group - 1).bit_length(), source=True
    )
    gluon_scan = GroupedIndexer(
        inputs,
        group_rows=1 << (group - 1).bit_length(),
        source=True,
        fuse_source_max=False,
    )
    gluon_full = GroupedIndexer(inputs, group_rows=1 << (group - 1).bit_length())
    native_candidates = torch.empty_like(grouped.candidates)

    def native_source():
        native.forward()
        select_candidate_blocks(native.scores, None, lens, 2048, 8, native_candidates)

    state = CandidateState(rows, q.device, compress_ratio=args.ratio)
    consumers, baseline_consumers = [], []
    for layer in range(4):
        cq = q.float().roll(layer * 7, -1).to(q.dtype)
        cw = w.roll(layer * 3, -1).contiguous()
        consumers.append(CompactIndexer(state, cq, cw, cache, table, req))
        baseline_consumers.append(
            NativeFullIndexer(
                IndexerInputs(cq, cw, cache, table, req, lens, inputs.shape),
                next_n=group,
            )
        )

    def optimized_chain():
        grouped.forward()
        state.publish(grouped.candidates, lens, source_unique=True)
        for consumer in consumers:
            consumer.forward()

    def native_chain():
        native_source()
        for consumer in baseline_consumers:
            consumer.score()
            apply_candidate_mask(consumer.scores, None, lens, native_candidates, 8, 1)
            consumer.select()

    def hybrid_chain():
        hybrid.forward()
        state.publish(hybrid.candidates, lens, source_unique=True)
        for consumer in consumers:
            consumer.forward()

    def gluon_chain():
        gluon.forward()
        state.publish(gluon.candidates, lens, source_unique=True)
        for consumer in consumers:
            consumer.forward()

    def hybrid_full_consumer_chain():
        hybrid.forward()
        for consumer in baseline_consumers:
            consumer.score()
            apply_candidate_mask(consumer.scores, None, lens, hybrid.candidates, 8, 1)
            hybrid.topk.topk(
                consumer.scores,
                lens,
                None,
                consumer.output,
                1,
                hybrid.source_plan,
                None,
            )

    native_source()
    hybrid.forward()
    gluon.forward()
    gluon_scan.forward()
    assert torch.equal(
        gluon.candidates.sort(1).values, gluon_scan.candidates.sort(1).values
    )
    assert torch.equal(gluon.maxima, gluon_scan.maxima)
    grouped.forward()
    rowwise.forward()
    scan.forward()
    assert torch.equal(
        scan.candidates.sort(1).values, grouped.candidates.sort(1).values
    )
    assert torch.equal(scan.maxima, grouped.maxima)
    mask = torch.arange(length, device=q.device)[None, :] < lens[:, None]
    torch.testing.assert_close(
        grouped.scores[mask], native.scores[:, :length][mask], rtol=1e-3, atol=1e-4
    )
    torch.testing.assert_close(
        grouped.scores[mask], rowwise.scores[mask], rtol=1e-3, atol=1e-4
    )
    error = grouped.scores[mask] - native.scores[:, :length][mask]
    # Verify the complete source->publisher->consumer chain against the SAME
    # published candidates; native source may choose a different exact tie.
    chain_checks = []
    for chain_name, chain_fn in (
        ("triton", optimized_chain),
        ("gluon", gluon_chain),
        ("hybrid", hybrid_chain),
    ):
        chain_fn()
        positions = (
            state.blocks.long()[..., None] * 8 + torch.arange(8, device=q.device)
        ).flatten(1)
        counts = state.counts.tolist()
        for consumer, baseline in zip(consumers, baseline_consumers):
            baseline.score()
            refs, logical = [], []
            max_error = 0.0
            for row, n in enumerate(counts):
                p = positions[row, :n]
                expected = baseline.scores[row, p]
                actual = consumer.scores[row, :n]
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
                max_error = max(max_error, (actual - expected).abs().max().item())
                refs.append(actual)
                logical.append(p)
            exact = stable_topk(refs, logical)
            assert torch.equal(exact.sort(1).values, consumer.output.sort(1).values)
            chain_checks.append(
                dict(
                    variant=chain_name,
                    max_abs_vs_native=max_error,
                    fixed_scores_topk="exact",
                )
            )
    hybrid_full_consumer_chain()
    for consumer in baseline_consumers:
        valid_scores, valid_positions = [], []
        for row, n in enumerate(lens.tolist()):
            positions = torch.arange(n, device=q.device)
            values = consumer.scores[row, :n]
            finite = torch.isfinite(values)
            valid_scores.append(values[finite])
            valid_positions.append(positions[finite])
        exact = stable_topk(valid_scores, valid_positions)
        assert torch.equal(exact.sort(1).values, consumer.output.sort(1).values)
    planning_start = time.perf_counter_ns()
    grouped.replan()
    torch.cuda.synchronize()
    grouping_wall_us = (time.perf_counter_ns() - planning_start) / 1000
    variants = [
        ("native_full", native.forward),
        ("gluon_full", gluon_full.forward),
        ("gluon_source_fused_max", gluon.forward),
        ("gluon_source_separate_max", gluon_scan.forward),
        ("gluon_score_and_max", gluon.score),
        ("optimized_full", full.forward),
        ("native_source", native_source),
        ("native_score_ant_token_and_source_topk", hybrid.forward),
        ("rowwise_source", rowwise.forward),
        ("grouped_source_fused_max", grouped.forward),
        ("grouped_source_separate_max", scan.forward),
        ("native_score", native.score),
        ("grouped_score_and_max", grouped.score),
        ("token_topk", grouped.select),
        ("candidate_topk", grouped.select_source),
        ("native_prepare", native.prepare),
    ]
    if args.phase == "full":
        variants += [
            ("native_source_four_consumers", native_chain),
            ("optimized_source_four_consumers", optimized_chain),
            ("hybrid_source_four_consumers", hybrid_chain),
            ("hybrid_source_full_consumers", hybrid_full_consumer_chain),
            ("gluon_source_four_consumers", gluon_chain),
        ]
    if args.variants:
        known = {name for name, _ in variants}
        if set(args.variants) - known:
            raise ValueError(f"Unknown variants: {set(args.variants) - known}")
        variants = [(name, fn) for name, fn in variants if name in args.variants]
    if args.reverse_order:
        variants.reverse()
    result = dict(
        case=case,
        baseline_mode=native.baseline_mode,
        score_max_abs=error.abs().max().item(),
        score_nrmse=(error.norm() / native.scores[:, :length][mask].norm()).item(),
        chain_checks=chain_checks,
        grouping_plan_wall_us=grouping_wall_us,
        score_workspace_bytes=grouped.scores.numel() * 4,
        cache_condition="warm repeated calls, no L2 flush",
        timings={},
    )
    for mode in ("eager", "graph"):
        result["timings"][mode] = {}
        for name, fn in variants:
            metric = measure(fn, graph=mode == "graph")
            if metric["coefficient_of_variation"] > 0.1:
                metric["extra_rounds"] = [
                    measure(fn, graph=mode == "graph") for _ in range(3)
                ]
            result["timings"][mode][name] = metric
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("full", "verify"), required=True)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio", choices=(1, 2), type=int, default=1)
    parser.add_argument("--deep-gemm-package")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Explicit requests:logical_context:rows_per_request cases",
    )
    args = parser.parse_args()
    baseline = check_vllm()
    if args.deep_gemm_package:
        from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init

        _lazy_init()
        _import_deep_gemm()._C.init(args.deep_gemm_package)
    cases = (
        [(1, 65536, 1), (8, 262144, 1)]
        if args.phase == "full"
        else [(4, 65536, group) for group in (1, 2, 4, 6, 8)]
    )
    if args.cases:
        cases = [tuple(map(int, value.split(":"))) for value in args.cases]
        if any(len(case) != 3 or min(case) < 1 for case in cases):
            parser.error("Each case must be three positive integers")
    report = dict(
        status="running",
        args=vars(args),
        source=check_sources(),
        baseline=baseline,
        device=device_info(),
        cases=[],
    )
    try:
        for requests, length, group in cases:
            result = run_case(requests, length, group, args)
            report["cases"].append(result)
            save(args.output, report)
            print(
                json.dumps(
                    dict(
                        case=result["case"],
                        timings={
                            mode: {
                                name: value["p50_us"]
                                for name, value in variants.items()
                            }
                            for mode, variants in result["timings"].items()
                        },
                    )
                ),
                flush=True,
            )
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()

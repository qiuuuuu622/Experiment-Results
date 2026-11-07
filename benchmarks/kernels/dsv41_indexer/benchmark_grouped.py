# SPDX-License-Identifier: Apache-2.0
"""P4 grouped verify tuning and full module A/B with raw forward/reverse samples."""

import argparse
import json
import time

from .common import check_sources, check_vllm, device_info, measure, save


def sampled(fn, mode):
    result = measure(fn, graph=mode == "graph")
    if result["coefficient_of_variation"] > 0.1:
        result["extra_rounds"] = [measure(fn, graph=mode == "graph") for _ in range(3)]
    return result


def run_case(requests, length, group, args):
    import torch

    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,
    )
    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.grouped import GroupedConsumer, GroupedIndexer
    from vllm_dsv41_indexer.module import IndexerInputs, NativeFullIndexer
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import make_inputs

    rows = requests * group
    case = dict(
        rows=rows,
        requests=requests,
        compressed_length=length,
        rows_per_request=group,
        page_size=64,
        candidate_blocks=2048,
        candidate_positions_limit=16384,
        physical_gpu=3,
        cache_condition="warm repeated calls; no L2 flush",
    )
    q, _, w, cache, table, req, lens, blocks = make_inputs(case)
    i = IndexerInputs(q, w, cache, table, req, lens, IndexerShape(rows, length))
    nr = 1 << (group - 1).bit_length()
    native = NativeFullIndexer(i, next_n=group)
    row = GroupedIndexer(i, group_rows=1, paired=args.paired)
    op = GroupedIndexer(i, group_rows=nr, paired=args.paired)
    op.forward()
    row.forward()
    valid = torch.arange(length, device=q.device)[None, :] < lens[:, None]
    assert torch.equal(op.scores[valid], row.scores[valid])
    assert torch.equal(op.output.sort(1).values, row.output.sort(1).values)
    baseline_scores = native.score()
    torch.testing.assert_close(
        op.scores[valid], baseline_scores[:, :length][valid], rtol=1e-3, atol=1e-4
    )
    error = op.scores[valid] - baseline_scores[:, :length][valid]
    result = dict(
        case=case,
        baseline_mode=native.baseline_mode,
        rowwise_score="bitwise_equal",
        topk="exact_set",
        max_abs_vs_native=error.abs().max().item(),
        tuning={},
        timings={},
    )
    for order in ("forward", "reverse"):
        targets = [256, 512, 1024, 2048, 4096]
        if order == "reverse":
            targets.reverse()
        result["tuning"][order] = {}
        for ctas in targets:
            op.grid_ctas = ctas
            result["tuning"][order][str(ctas)] = sampled(op.score, "graph")
    chosen = min(
        (256, 512, 1024, 2048, 4096),
        key=lambda c: sum(
            result["tuning"][o][str(c)]["p50_us"] for o in ("forward", "reverse")
        ),
    )
    op.grid_ctas = chosen
    result["chosen_grid_ctas"] = chosen
    source = GroupedIndexer(
        i, group_rows=nr, source=True, grid_ctas=chosen, paired=args.paired
    )
    row_source = GroupedIndexer(i, group_rows=1, source=True, paired=args.paired)
    native_candidates = torch.empty_like(source.candidates)

    def native_source():
        native.forward()
        select_candidate_blocks(native.scores, None, lens, 2048, 8, native_candidates)

    source.forward()
    row_source.forward()
    assert torch.equal(
        source.candidates.sort(1).values, row_source.candidates.sort(1).values
    )
    for start in range(0, rows, group):
        blocks[start : start + group].copy_(
            blocks[start + group - 1 : start + group].expand(group, -1).clone()
        )
    state = CandidateState(rows, q.device)
    state.publish(blocks, lens, source_unique=True)
    planning_start = time.perf_counter_ns()
    consumer = GroupedConsumer(
        state,
        q,
        w,
        cache,
        table,
        req,
        group_rows=nr,
        grid_ctas=chosen,
        paired=args.paired,
    )
    torch.cuda.synchronize()
    result["consumer_plan_wall_us"] = (time.perf_counter_ns() - planning_start) / 1000
    plain_consumer = CompactIndexer(state, q, w, cache, table, req)

    def shared_consumer():
        state.publish(blocks, lens, source_unique=True)
        consumer.forward()

    def old_consumer():
        state.publish(blocks, lens, source_unique=True)
        plain_consumer.forward()

    shared_consumer()
    old_consumer()
    assert torch.equal(
        consumer.output.sort(1).values, plain_consumer.output.sort(1).values
    )
    result["consumer_shared_enabled"] = consumer.shared_enabled
    variants = [
        ("native_score", native.score),
        ("rowwise_gluon_score", row.score),
        ("grouped_gluon_score", op.score),
        ("native_full", native.forward),
        ("rowwise_gluon_full", row.forward),
        ("grouped_gluon_full", op.forward),
        ("native_source", native_source),
        ("rowwise_gluon_source", row_source.forward),
        ("grouped_gluon_source", source.forward),
        ("rowwise_consumer_publish_included", old_consumer),
        ("shared_consumer_publish_included", shared_consumer),
    ]
    for order in ("forward", "reverse"):
        result["timings"][order] = {}
        for mode in ("eager", "graph"):
            result["timings"][order][mode] = {}
            for name, fn in variants if order == "forward" else reversed(variants):
                result["timings"][order][mode][name] = sampled(fn, mode)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--paired", action="store_true")
    args = p.parse_args()
    baseline = check_vllm()
    cases = (
        [(4, 65536, 6)]
        if args.smoke
        else [(4, 65536, n) for n in (1, 2, 4, 6, 8)] + [(32, 65536, 6), (1, 262144, 6)]
    )
    report = dict(
        status="running",
        device=device_info(),
        source=check_sources(),
        baseline=baseline,
        physical_gpu=3,
        paired=args.paired,
        cases=[],
    )
    try:
        for requests, length, n in cases:
            item = run_case(requests, length, n, args)
            report["cases"].append(item)
            save(args.output, report)
            print(
                json.dumps(
                    dict(
                        case=item["case"],
                        ctas=item["chosen_grid_ctas"],
                        graph={
                            order: {name: v["p50_us"] for name, v in x["graph"].items()}
                            for order, x in item["timings"].items()
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

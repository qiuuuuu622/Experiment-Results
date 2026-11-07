# SPDX-License-Identifier: Apache-2.0
"""Expose the host eligibility-check cost; fixed shared layouts need no replan."""

import argparse

from .benchmark_grouped import sampled
from .common import check_vllm, save


def run():
    check_vllm()
    from vllm_dsv41_indexer.grouped import GroupedConsumer
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import make_inputs

    cases = []
    for requests, group in [(4, g) for g in (2, 4, 6, 8)] + [(32, 6), (1, 6)]:
        rows = requests * group
        case = dict(
            rows=rows,
            requests=requests,
            compressed_length=65536,
            rows_per_request=group,
            page_size=64,
            candidate_blocks=2048,
        )
        q, _, w, cache, table, req, lens, blocks = make_inputs(case)
        for start in range(0, rows, group):
            blocks[start : start + group].copy_(
                blocks[start + group - 1 : start + group].expand(group, -1).clone()
            )
        state = CandidateState(rows, q.device)
        state.publish(blocks, lens, source_unique=True)
        shared = GroupedConsumer(
            state,
            q,
            w,
            cache,
            table,
            req,
            group_rows=1 << (group - 1).bit_length(),
            paired=True,
            grid_ctas=2048 if requests == 32 else 512,
        )
        rowwise = CompactIndexer(state, q, w, cache, table, req)

        def publish_shared(state=state, shared=shared, blocks=blocks, lens=lens):
            state.publish(blocks, lens, source_unique=True)
            shared.forward()

        def plan_and_execute(state=state, shared=shared, blocks=blocks, lens=lens):
            state.publish(blocks, lens, source_unique=True)
            shared.replan()
            shared.forward()

        def publish_rowwise(state=state, rowwise=rowwise, blocks=blocks, lens=lens):
            state.publish(blocks, lens, source_unique=True)
            rowwise.forward()

        variants = [
            ("rowwise_publish_execute", publish_rowwise),
            ("shared_publish_execute_preplanned", publish_shared),
            ("shared_eligibility_plan_only", shared.replan),
            ("shared_publish_replan_execute", plan_and_execute),
        ]
        timings = {}
        for order in ("forward", "reverse"):
            timings[order] = {
                name: sampled(fn, "eager")
                for name, fn in (variants if order == "forward" else reversed(variants))
            }
        cases.append(
            dict(
                case=case,
                timings=timings,
                graph_planning=(
                    "unsupported: host eligibility reads must remain outside capture"
                ),
            )
        )
    return cases


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    save(args.output, dict(status="completed", cases=run()))

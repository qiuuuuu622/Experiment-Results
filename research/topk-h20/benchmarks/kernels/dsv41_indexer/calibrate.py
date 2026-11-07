# SPDX-License-Identifier: Apache-2.0
"""Audit FP8 arithmetic before deciding numerical acceptance thresholds."""

import argparse

from .common import check_vllm, save


def run():
    check_vllm()
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.module import IndexerInputs, NativeFullIndexer
    from vllm_dsv41_indexer.reference import scores_reference, stable_topk
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import make_inputs

    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for length in (4099, 65536):
        for distribution in ("signed_weights", "positive_weights", "zero_query"):
            case = dict(
                rows=4,
                compressed_length=length,
                rows_per_request=2,
                page_size=64,
                candidate_blocks=2048,
            )
            q, _, w, cache, table, req, lens, blocks = make_inputs(case)
            if distribution == "positive_weights":
                w.abs_()
            if distribution == "zero_query":
                q.zero_()
            inputs = IndexerInputs(
                q, w, cache, table, req, lens, IndexerShape(4, length)
            )
            native = NativeFullIndexer(inputs, next_n=2)
            full = native.score()
            state = CandidateState(4, q.device)
            state.publish(blocks, lens, source_unique=True)
            ops = {
                name: CompactIndexer(state, q, w, cache, table, req, score_backend=name)
                for name in ("wgmma", "precise")
            }
            reference, positions = scores_reference(
                q, w, cache, table, req, lens, blocks
            )
            native_scores = [full[row, pos] for row, pos in enumerate(positions)]
            native_selected = stable_topk(native_scores, positions)
            ref_selected = stable_topk(reference, positions)
            for op in ops.values():
                op.forward()
            for row, ref in enumerate(reference):

                def metrics(value, target):
                    error = value - target
                    return dict(
                        max_abs=error.abs().max().item(),
                        nrmse=(error.norm() / target.norm().clamp_min(1e-30)).item(),
                        strict_mismatches=(error.abs() > 1e-4 + 1e-3 * target.abs())
                        .sum()
                        .item(),
                    )

                record = dict(
                    keys=length,
                    distribution=distribution,
                    row=row,
                    native_vs_fp32=metrics(native_scores[row], ref),
                    variants={},
                )
                a = set(native_selected[row].tolist()) - {-1}
                b = set(ref_selected[row].tolist()) - {-1}
                record["native_fp32_overlap"] = len(a & b) / len(a)
                for name, op in ops.items():
                    values = op.scores[row, : len(ref)]
                    selected = set(op.output[row].tolist()) - {-1}
                    # Same logits must have exact stable selection, independently
                    # of the numerical comparison against another score kernel.
                    exact = stable_topk([values], [positions[row]])[0]
                    assert selected == set(exact.tolist()) - {-1}
                    vs_native = metrics(values, native_scores[row])
                    delta = selected ^ a
                    boundary = torch.sort(native_scores[row], descending=True).values[
                        511
                    ]
                    logical = positions[row].tolist()
                    score_by_position = dict(zip(logical, native_scores[row].tolist()))
                    changed_distance = max(
                        (abs(score_by_position[p] - boundary.item()) for p in delta),
                        default=0.0,
                    )
                    # This asserts the actual observed error bound, not a
                    # silently widened global tolerance.
                    assert changed_distance <= 2 * vs_native["max_abs"] + 1e-7
                    record["variants"][name] = dict(
                        vs_fp32=metrics(values, ref),
                        vs_native=vs_native,
                        native_overlap=len(selected & a) / len(a),
                        fp32_overlap=len(selected & b) / len(b),
                        changed_boundary_distance=changed_distance,
                    )
                results.append(record)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checks = run()
    save(
        args.output,
        dict(
            status="completed",
            scope="numerical calibration, no performance claim",
            checks=checks,
        ),
    )
    print(f"Completed {len(checks)} row-level FP8 calibration records", flush=True)


if __name__ == "__main__":
    main()

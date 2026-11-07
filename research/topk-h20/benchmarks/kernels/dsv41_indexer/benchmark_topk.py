# SPDX-License-Identifier: Apache-2.0
"""Selector-only A/B: native, streaming, cluster, and update-plan costs."""

import argparse
import importlib
import json

from .common import check_sources, check_vllm, device_info, measure, save
from .verify_topk import expected, make_scores


def run_case(rows, width, k, args):
    import torch

    from vllm_dsv41_indexer.topk import StableTopK

    from .inputs import VllmSelector

    scores = make_scores(rows, width, "random", seed=6116)
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    stream = StableTopK(scores, lengths, k=k, backend="streaming")
    cluster = StableTopK(scores, lengths, k=k, backend="cluster")
    reference = expected(scores, lengths, k)
    for selector in (stream, cluster):
        selector.execute()
        assert torch.equal(selector.output.sort(1).values, reference.sort(1).values)
    native_output = torch.empty_like(reference)
    if k == 512:
        original = VllmSelector(scores.device, rows)

        def baseline():
            original(scores, lengths[:, None], native_output, width)
            return native_output

        baseline_name = f"vllm_{original.backend}"
    else:
        # vLLM source block selection uses torch.topk, including its intrinsic
        # output allocation. Here every block is valid; masking is upstream.
        def baseline():
            return torch.topk(scores, k, dim=-1).indices

        baseline_name = "vllm_source_torch_topk"
    actual = baseline()
    boundary = torch.sort(scores, descending=True).values[:, k - 1]
    chosen_scores = scores.gather(1, actual.long())
    assert (chosen_scores >= boundary[:, None]).all().item()
    native_exact = torch.equal(actual.sort(1).values.int(), reference.sort(1).values)
    variants = [
        (baseline_name, baseline),
        ("streaming", stream.execute),
        ("cluster_preplanned", cluster.execute),
        ("plan", cluster.prepare),
        ("cluster_replan_each_call", cluster.forward),
    ]
    if args.reverse_order:
        variants.reverse()
    record = dict(
        rows=rows,
        width=width,
        k=k,
        cluster_capacity=cluster.cluster_capacity,
        needs_plan=cluster.needs_plan,
        plan=cluster.metadata[0].tolist(),
        correctness="stable_exact",
        native_stable_exact=native_exact,
        score_bytes=scores.numel() * 4,
        cache="hot repeated calls, no L2 flush",
        metadata_mode="static input lengths; plan recomputed in combined variant",
        timings={},
    )
    for mode in ("eager", "graph"):
        record["timings"][mode] = {}
        for name, fn in variants:
            if name == "plan" and not cluster.needs_plan:
                record["timings"][mode][name] = dict(status="no_gpu_plan_required")
                continue
            metric = measure(fn, graph=mode == "graph")
            if metric["coefficient_of_variation"] > 0.1:
                metric["extra_rounds"] = [
                    measure(fn, graph=mode == "graph") for _ in range(3)
                ]
            record["timings"][mode][name] = metric
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reverse-order", action="store_true")
    args = parser.parse_args()
    baseline = check_vllm()
    importlib.import_module("vllm._custom_ops")
    cases = [(1, width, 512) for width in (4096, 16384, 65536, 131072, 262144)]
    cases += [
        (rows, width, 512)
        for rows in (8, 32, 64, 128)
        for width in (16384, 65536, 262144)
    ]
    cases += [
        (rows, width, 2048)
        for rows in (1, 8, 32, 128)
        for width in (4096, 16384, 32768)
    ]
    report = dict(
        status="running",
        args=vars(args),
        baseline=baseline,
        source=check_sources(),
        device=device_info(),
        cases=[],
    )
    try:
        for rows, width, k in cases:
            case = run_case(rows, width, k, args)
            report["cases"].append(case)
            save(args.output, report)
            print(
                json.dumps(
                    dict(
                        rows=rows,
                        width=width,
                        k=k,
                        timings={
                            mode: {
                                name: value.get("p50_us", value.get("status"))
                                for name, value in timings.items()
                            }
                            for mode, timings in case["timings"].items()
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

# SPDX-License-Identifier: Apache-2.0
"""P5 preparation-inclusive FP8 prefill comparisons on one explicitly bound GPU."""

import argparse
import json

from .common import check_sources, check_vllm, device_info, measure, save


def run_case(case, reverse=False):
    import torch

    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
    )
    from vllm_dsv41_indexer.prefill import PrefillIndexer

    from .verify_prefill import make_case

    rows, requests, keys, ratio, mode = (
        case[k] for k in ("rows", "requests", "index_keys", "ratio", "mode")
    )
    inputs, candidates = make_case(rows, requests, keys, ratio)
    compact = mode == "consumer"
    baseline = PrefillIndexer(
        inputs,
        backend="deepgemm",
        selector="native",
        native_gather=True,
        source=mode == "source",
    )
    ops = {
        backend: PrefillIndexer(
            inputs,
            backend=backend,
            candidate_blocks=candidates if compact else None,
            source=mode == "source",
        )
        for backend in (
            ("deepgemm", "row_tiled", "keys_major", "indirect")
            if compact
            else ("deepgemm", "row_tiled", "keys_major", "indirect", "gluon")
        )
    }

    def native_forward():
        baseline.prepare()
        for start, end, request in baseline.chunks:
            scores = baseline.score_chunk(start, end, request)
            if compact:
                blocks = candidates[request : request + 1].expand(end - start, -1)
                apply_candidate_mask(
                    scores, None, inputs.lengths[start:end], blocks, 8, 1
                )
            baseline.select_chunk(scores, start, end, request)
            del scores
        return baseline.output

    native_forward()
    for op in ops.values():
        op.forward()
    # Compare matching visible scores; compact coordinates map back to the
    # original request before comparing with the native full-scan baseline.
    checks = {}
    for name, op in ops.items():
        op.prepare()
        baseline.prepare()
        maximum, square, norm, min_overlap, max_boundary = 0.0, 0.0, 0.0, 1.0, 0.0
        for start, end, request in op.chunks:
            got = op.score_chunk(start, end, request)
            expected = baseline.score_chunk(start, end, request)
            op.select_chunk(got, start, end, request)
            for local, n in enumerate(op.lengths[start:end].tolist()):
                positions = (
                    (
                        op.state.blocks[request].long()[:, None] * 8
                        + torch.arange(8, device=got.device)
                    ).flatten()[:n]
                    if compact
                    else torch.arange(n, device=got.device)
                )
                a, b = got[local, :n], expected[local, positions]
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
                if n:
                    error = a - b
                    maximum = max(maximum, error.abs().max().item())
                    square += error.square().sum().item()
                    norm += b.square().sum().item()
                stable = torch.argsort(b, descending=True, stable=True)[:512]
                want = set(positions[stable].tolist())
                actual = {x for x in op.output[start + local].tolist() if x >= 0}
                assert len(actual) == len(set(actual)) == min(512, n)
                assert actual <= set(positions.tolist())
                min_overlap = min(min_overlap, len(actual & want) / max(len(want), 1))
                if actual != want:
                    values = dict(zip(positions.tolist(), b.tolist()))
                    boundary = b[stable[-1]].item()
                    delta = max(abs(values[p] - boundary) for p in actual ^ want)
                    max_boundary = max(max_boundary, delta)
                    assert delta <= 2 * maximum + 1e-7
        checks[name] = dict(
            max_abs_vs_native=maximum,
            nrmse_vs_native=(square / max(norm, 1e-30)) ** 0.5,
            min_topk_overlap=min_overlap,
            changed_boundary_distance=max_boundary,
            preparation_bytes=op.preparation_bytes,
            score_workspace_upper_bound_bytes=min(op.score_budget, rows * op.width * 4),
        )
    variants = [("native_pipeline", native_forward)] + [
        (name + "_pipeline", op.forward) for name, op in ops.items()
    ]
    # Score-only is kept separate; the primary figures always include prepare.
    variants += [
        ("native_prepare", baseline.prepare),
        ("optimized_prepare", ops["deepgemm"].prepare),
    ]

    def deepgemm_score():
        for start, end, request in ops["deepgemm"].chunks:
            scores = ops["deepgemm"].score_chunk(start, end, request)
            del scores

    variants.append(("deepgemm_score_only", deepgemm_score))
    if reverse:
        variants.reverse()
    timings = {}
    for execution in ("eager", "graph"):
        timings[execution] = {}
        for name, fn in variants:
            metric = measure(fn, graph=execution == "graph")
            if metric["coefficient_of_variation"] > 0.1:
                metric["extra_rounds"] = [
                    measure(fn, graph=execution == "graph") for _ in range(3)
                ]
            timings[execution][name] = metric
    return dict(
        case=case,
        checks=checks,
        timings=timings,
        cache_condition="warm repeated calls; no explicit L2 flush",
        candidate_layout="request-shared" if compact else None,
        includes_preparation=True,
        native_selector="top_k_per_row_prefill",
        key_preparation="once per request per module call, reused by query chunks",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("smoke", "matrix", "gluon"), default="smoke"
    )
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_vllm()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = (
        [(64, 1, 65536, 1, mode) for mode in ("full", "source", "consumer")]
        if args.suite == "smoke"
        else [
            (1, 1, 4096, 1, "full"),
            (16, 1, 262144, 1, "full"),
            (64, 1, 16384, 1, "full"),
            (64, 1, 32768, 2, "full"),
            (64, 1, 131072, 2, "full"),
            (256, 1, 16384, 1, "full"),
            (1024, 1, 16384, 1, "full"),
            (4096, 1, 4096, 1, "full"),
            (64, 2, 65536, 1, "source"),
            (256, 1, 16384, 2, "source"),
            (16, 1, 262144, 1, "consumer"),
            (256, 1, 65536, 1, "consumer"),
            (1024, 1, 16384, 2, "consumer"),
        ]
    )
    if args.suite == "gluon":
        shapes = [
            (64, 1, 65536, 1, "full"),
            (64, 1, 65536, 1, "source"),
            (16, 1, 262144, 1, "full"),
            (4096, 1, 4096, 1, "full"),
        ]
    report = dict(
        status="running",
        source=check_sources(),
        device=device_info(),
        args=vars(args),
        physical_gpu=1,
        cases=[],
        scope="function/module only",
    )
    try:
        for rows, requests, keys, ratio, mode in shapes:
            case = dict(
                rows=rows,
                requests=requests,
                index_keys=keys,
                ratio=ratio,
                logical_context=keys * ratio,
                mode=mode,
                scores=rows * keys,
                candidates_per_request=2048 if mode == "consumer" else None,
            )
            result = run_case(case, args.reverse)
            report["cases"].append(result)
            save(args.output, report)
            print(
                json.dumps(
                    dict(
                        case=case,
                        graph_p50_us={
                            k: v["p50_us"]
                            for k, v in result["timings"]["graph"].items()
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

# SPDX-License-Identifier: Apache-2.0
"""Direct three-way prefill comparison; common FP8 input, prep and selector."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .common import check_vllm, device_info, measure, save
from .verify_prefill import make_case

SUITES = {
    "small": [
        (1, 1, 4096, 1, "full"),
        (16, 1, 4096, 1, "full"),
        (16, 1, 65536, 1, "full"),
        (32, 1, 65536, 1, "full"),
        (64, 1, 16384, 1, "full"),
        (64, 1, 32768, 1, "full"),
        (64, 1, 65536, 1, "full"),
        (16, 1, 262144, 1, "full"),
    ],
    "large": [
        (128, 1, 16384, 1, "full"),
        (256, 1, 16384, 1, "full"),
        (1024, 1, 4096, 1, "full"),
        (1024, 1, 16384, 1, "full"),
        (4096, 1, 4096, 1, "full"),
        (256, 1, 65536, 1, "full"),
        (64, 1, 262144, 1, "full"),
    ],
    "mixed": [
        (64, 2, 65536, 1, "full"),
        (256, 2, 16384, 2, "full"),
        (64, 1, 65536, 1, "source"),
        (256, 1, 65536, 1, "consumer"),
        (1024, 1, 16384, 2, "consumer"),
        (64, 1, 32768, 2, "full"),
        (1024, 1, 65536, 1, "short_prefix"),
        (256, 2, 16384, 2, "reversed_visibility"),
    ],
}


def sampled(fn, graph):
    result = measure(fn, graph=graph)
    if result["coefficient_of_variation"] > 0.1:
        result["extra_rounds"] = [measure(fn, graph=graph) for _ in range(3)]
    return result


def run_case(shape, reverse):
    import torch

    from vllm_dsv41_indexer.prefill import PrefillIndexer
    from vllm_dsv41_indexer.prefill_keys_major_ant import (
        AntKeysMajorScores,
        AntPrefillIndexer,
    )
    from vllm_dsv41_indexer.reference import source_reference

    rows, requests, keys, ratio, mode = shape
    inputs, candidates = make_case(rows, requests, keys, ratio)
    if mode == "short_prefix":
        inputs.lengths.copy_(
            torch.arange(1, rows + 1, device="cuda", dtype=torch.int32)
        )
    if mode == "reversed_visibility":
        inputs.lengths.copy_(inputs.lengths.flip(0))
    options = dict(
        candidate_blocks=candidates if mode == "consumer" else None,
        source=mode == "source",
        selector="ant",
        native_gather=False,
    )
    ops = {
        "deepgemm": PrefillIndexer(inputs, backend="deepgemm", **options),
        "current_keys_major": PrefillIndexer(inputs, backend="keys_major", **options),
        "ant_keys_major": AntPrefillIndexer(inputs, **options),
    }
    for op in ops.values():
        op.forward()
    native = ops["deepgemm"]
    checks = {}
    for name, op in ops.items():
        maximum = square = norm = boundary_distance = 0.0
        overlap = 1.0
        for start, end, request in op.chunks:
            got = op.score_chunk(start, end, request)
            ref = native.score_chunk(start, end, request)
            if op.source:
                expected = source_reference(got, inputs.lengths[start:end])
                assert torch.equal(
                    op.candidates[start:end].sort(1).values, expected.sort(1).values
                )
            for row, n in enumerate(op.lengths[start:end].tolist()):
                a, b = got[row, :n], ref[row, :n]
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
                if n:
                    err = a - b
                    maximum = max(maximum, err.abs().max().item())
                    square += err.square().sum().item()
                    norm += b.square().sum().item()
                positions = (
                    (
                        op.state.blocks[request].long()[:, None] * 8
                        + torch.arange(8, device="cuda")
                    ).flatten()[:n]
                    if op.compact
                    else torch.arange(n, device="cuda")
                )
                order = torch.argsort(b, descending=True, stable=True)[:512]
                want = set(positions[order].tolist())
                actual = [p for p in op.output[start + row].tolist() if p >= 0]
                assert len(actual) == len(set(actual)) == min(512, n)
                assert set(actual) <= set(positions.tolist())
                overlap = min(
                    overlap, len(set(actual) & want) / len(want) if want else 1.0
                )
                if set(actual) != want:
                    lookup = dict(zip(positions.tolist(), b.tolist()))
                    cutoff = b[order[-1]].item()
                    delta = max(abs(lookup[p] - cutoff) for p in set(actual) ^ want)
                    boundary_distance = max(boundary_distance, delta)
                    assert delta <= 2 * maximum + 1e-7
        checks[name] = dict(
            max_abs_vs_native=maximum,
            nrmse_vs_native=(square / max(norm, 1e-30)) ** 0.5,
            min_topk_overlap=overlap,
            changed_boundary_distance=boundary_distance,
        )

    def score_call(op):
        def call():
            for start, end, request in op.chunks:
                scores = op.score_chunk(start, end, request)
                del scores

        return call

    ant = ops["ant_keys_major"]

    def rebuild_descriptors():
        for start, end, request in ant.chunks:
            ant.ant_plans[(start, end)] = AntKeysMajorScores(
                inputs.q[start:end],
                inputs.weights[start:end],
                ant.keys[request],
                ant.scales[request],
                ant.lengths[start:end],
                ant.scores[: end - start],
            )

    def dynamic_pipeline():
        rebuild_descriptors()
        return ant.forward()

    variants = [(name + "_pipeline", op.forward) for name, op in ops.items()]
    variants += [(name + "_score_only", score_call(op)) for name, op in ops.items()]
    variants += [
        ("common_key_prepare", ant.prepare),
        ("ant_prefix_metadata", ant.metadata),
    ]
    timings = {}
    for execution in ["eager", "graph"]:
        timings[execution] = {}
        selected = variants + (
            [
                ("ant_dynamic_descriptors_pipeline", dynamic_pipeline),
                ("ant_descriptor_plan_only", rebuild_descriptors),
            ]
            if execution == "eager"
            else []
        )
        for name, fn in reversed(selected) if reverse else selected:
            # Bare score uses prepared prefix; metadata is measured separately.
            ant.prepare()
            ant.metadata()
            timings[execution][name] = sampled(fn, execution == "graph")
    return dict(
        case=dict(
            rows=rows,
            requests=requests,
            index_keys=keys,
            ratio=ratio,
            logical_context=keys * ratio,
            mode=mode,
            scores=rows * keys,
            visible_score_count=int(inputs.lengths.sum().item()),
            ant_visible_key_tiles=int(
                sum(p.prefix[-1].item() for p in ant.ant_plans.values())
            ),
            total_output_bytes=rows * ant.width * 4,
            min_visible_keys=inputs.lengths.min().item(),
            max_visible_keys=inputs.lengths.max().item(),
            compact_width=ant.width if ant.compact else None,
        ),
        checks=checks,
        timings=timings,
        selector="same Ant stable top-k for all three",
        key_preparation="same FP8 gather and unchanged FP32 scales, once/request/call",
        prefix_metadata="Every Ant pipeline includes visible-prefix reconstruction",
        descriptor_contract=(
            "Static descriptors planned once for fixed buffers; "
            "dynamic replan Eager cost also measured"
        ),
        score_only=(
            "Prepared prefix, no metadata. DeepGEMM output allocation included; "
            "other paths use explicit workspace"
        ),
        invisible_tail="Ant fills -inf; DeepGEMM leaves invisible tails unspecified",
        cache="warm repeated calls, no L2 flush",
        preparation_bytes=ant.preparation_bytes,
        score_workspace_bytes=ant.scores.numel() * 4,
        prefix_workspace_bytes=sum(
            p.prefix.numel() * 4 for p in ant.ant_plans.values()
        ),
        peak_scope="all compared variants resident in same process",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_vllm()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    source = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path("vllm_dsv41_indexer/prefill_keys_major_ant.py"),
            Path("vllm_dsv41_indexer/prefill.py"),
            Path(__file__).relative_to(Path.cwd()),
        ]
    }
    report = dict(
        status="running",
        args=vars(args),
        device=device_info(),
        physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),
        source=source,
        cases=[],
    )
    try:
        for shape in SUITES[args.suite]:
            result = run_case(shape, args.reverse)
            report["cases"].append(result)
            save(args.output, report)
            print(
                json.dumps(
                    dict(
                        case=result["case"],
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

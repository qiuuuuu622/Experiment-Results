# SPDX-License-Identifier: Apache-2.0
"""One immutable P5b shape per GPU; independently sweep persistent grid sizes."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .common import check_vllm, device_info, measure, save
from .verify_prefill import make_case

SHAPES = ((16, 65536), (64, 65536), (256, 16384), (1024, 16384))


def source_hashes():
    root = Path.cwd()
    files = [
        p
        for p in (root / "vllm_dsv41_indexer").rglob("*")
        if p.is_file() and p.suffix in (".py", ".cuh", ".h", ".json")
    ]
    files += [
        Path(__file__),
        Path(__file__).with_name("common.py"),
        Path(__file__).with_name("verify_prefill.py"),
        Path(__file__).with_name("inputs.py"),
        Path(__file__).with_name("baseline.json"),
    ]
    return {
        str(p.resolve().relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def sampled(fn, graph):
    runs = [measure(fn, graph=graph, repeats=30, inner=10)]
    if runs[0]["coefficient_of_variation"] > 0.10:
        runs += [measure(fn, graph=graph, repeats=30, inner=10) for _ in range(3)]
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-index", type=int, choices=range(4), required=True)
    parser.add_argument("--grids", nargs="+", type=int, default=[39, 78, 156, 312])
    parser.add_argument("--expected-ant-sha256", required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_vllm()
    import torch

    from vllm_dsv41_indexer.prefill import PrefillIndexer
    from vllm_dsv41_indexer.prefill_keys_major_ant import AntPrefillIndexer

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = source_hashes()
    assert (
        frozen["vllm_dsv41_indexer/prefill_keys_major_ant.py"]
        == args.expected_ant_sha256
    )
    rows, keys = SHAPES[args.shape_index]
    inputs, _ = make_case(rows, 1, keys, 1)
    options = dict(selector="ant", native_gather=False, source=False)
    ops = {
        "deepgemm": PrefillIndexer(inputs, backend="deepgemm", **options),
        "current_keys_major": PrefillIndexer(inputs, backend="keys_major", **options),
    }
    ops.update(
        {
            f"ant_grid_{grid}": AntPrefillIndexer(inputs, grid_size=grid, **options)
            for grid in args.grids
        }
    )
    assert 78 in args.grids, "The default 78-SM control must be present"
    report = dict(
        status="running",
        arguments=vars(args),
        source_sha256=frozen,
        imported_from=str(Path.cwd()),
        input_snapshot_sha256=args.snapshot_sha256,
        physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),
        parallel_mode="independent_single_gpu_shape_per_device_not_TP4",
        device=device_info(),
        case=dict(
            rows=rows,
            requests=1,
            index_keys=keys,
            logical_context=keys,
            ratio=1,
            score_count=rows * keys,
            visible_score_count=int(inputs.lengths.sum().item()),
            min_visible_keys=int(inputs.lengths.min()),
            max_visible_keys=int(inputs.lengths.max()),
            selector="same Ant stable top-k",
            cache="same deterministic FP8 input, warm repeated calls, no L2 flush",
            score_only_contract=(
                "same already-prepared keys/scales/visibility; prefix excluded"
            ),
            full_contract=(
                "key prepare + Ant prefix when applicable + score + same selector"
            ),
        ),
        checks={},
        records=[],
    )
    save(args.output, report)
    try:
        for op in ops.values():
            op.forward()
        assert all(op.chunks == ops["deepgemm"].chunks for op in ops.values())
        assert len(ops["deepgemm"].chunks) == 1
        chunk = ops["deepgemm"].chunks[0]
        ref = ops["deepgemm"].score_chunk(*chunk).clone()
        default = ops["ant_grid_78"].score_chunk(*chunk).clone()
        default_topk = ops["ant_grid_78"].output.sort(1).values.clone()
        valid = torch.arange(keys, device="cuda")[None, :] < inputs.lengths[:, None]
        ref_topk = ops["deepgemm"].output.cpu().tolist()
        ref_values = ref.cpu()
        prefix_reference = ops["ant_grid_78"].ant_plans[chunk[:2]].prefix.clone()
        for name, op in ops.items():
            got = op.score_chunk(*chunk)
            a, b = got[valid], ref[valid]
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
            errors = torch.where(valid, (got - ref).abs(), 0.0)
            maximum = float(errors.max())
            row_errors = errors.max(1).values.cpu().tolist()
            minimum_overlap, boundary_distance = 1.0, 0.0
            for row, output in enumerate(op.output.cpu().tolist()):
                actual = [p for p in output if p >= 0]
                wanted = set(p for p in ref_topk[row] if p >= 0)
                assert len(actual) == len(set(actual)) == 512
                assert max(actual) < int(inputs.lengths[row])
                minimum_overlap = min(minimum_overlap, len(set(actual) & wanted) / 512)
                changed = set(actual) ^ wanted
                if changed:
                    cutoff = float(ref_values[row, list(wanted)].min())
                    delta = max(
                        abs(float(ref_values[row, p]) - cutoff) for p in changed
                    )
                    boundary_distance = max(boundary_distance, delta)
                    assert delta <= 2 * row_errors[row] + 1e-7
            check = dict(
                max_abs_vs_deepgemm=maximum,
                nrmse_vs_deepgemm=float(
                    (a - b).square().sum().sqrt()
                    / b.square().sum().sqrt().clamp_min(1e-30)
                ),
                min_topk_overlap_vs_deepgemm=minimum_overlap,
                changed_boundary_distance=boundary_distance,
            )
            if name.startswith("ant_grid_"):
                assert torch.equal(got, default), "Grid changed FP32 score bits"
                assert torch.equal(op.output.sort(1).values, default_topk)
                plan = op.ant_plans[chunk[:2]]
                assert torch.equal(plan.prefix, prefix_reference)
                check.update(
                    scores_bitwise_vs_grid78=True,
                    topk_sets_exact_vs_grid78=True,
                    effective_grid=plan.grid_size,
                    visible_tiles=int(plan.prefix[-1]),
                    prefix_sha256=hashlib.sha256(
                        plan.prefix.cpu().numpy().tobytes()
                    ).hexdigest(),
                )
            graph = torch.cuda.CUDAGraph()
            expected = op.output.sort(1).values.clone()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                op.forward()
            for _ in range(3):
                graph.replay()
                assert torch.equal(op.output.sort(1).values, expected)
            graph.reset()
            check["graph_replays"] = 3
            report["checks"][name] = check
        save(args.output, report)

        # Explicitly preplan every path before any score-only timing.
        def prepare(op):
            op.prepare()
            if hasattr(op, "metadata"):
                op.metadata()

        def score_only(op):
            def call():
                for start, end, request in op.chunks:
                    scores = op.score_chunk(start, end, request)
                    del scores

            return call

        variants = [(name, "score_only", score_only(op)) for name, op in ops.items()]
        variants += [(name, "full", op.forward) for name, op in ops.items()]
        for execution in ("eager", "graph"):
            for order, sequence in (("forward", variants), ("reverse", variants[::-1])):
                for name, phase, fn in sequence:
                    prepare(ops[name])
                    runs = sampled(fn, execution == "graph")
                    record = dict(
                        variant=name,
                        phase=phase,
                        execution=execution,
                        order=order,
                        runs=runs,
                    )
                    report["records"].append(record)
                    save(args.output, report)
                    print(
                        json.dumps(
                            dict(
                                shape=[rows, keys],
                                variant=name,
                                phase=phase,
                                execution=execution,
                                order=order,
                                p50_us=runs[0]["p50_us"],
                                cv=runs[0]["coefficient_of_variation"],
                            )
                        ),
                        flush=True,
                    )
        assert source_hashes() == frozen, "Source changed during the experiment"
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()

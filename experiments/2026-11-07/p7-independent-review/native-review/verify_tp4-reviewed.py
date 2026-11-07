# SPDX-License-Identifier: Apache-2.0
"""Actual four-rank ownership correctness; --allow-single-rank is diagnostic only."""

import argparse
import json

import torch
import torch.distributed as dist

from .common import save
from .tp4_cases import (
    build_variants,
    canonical,
    digest,
    initialize,
    make_case,
    provenance,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-single-rank", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--full-backend", choices=["portable", "native_stable"], default="portable"
    )
    parser.add_argument("--patched-deep-gemm")
    args = parser.parse_args()
    if args.full_backend == "native_stable" and not args.patched_deep_gemm:
        parser.error("native_stable requires explicit --patched-deep-gemm")
    world = initialize(args.allow_single_rank)
    if args.patched_deep_gemm:
        from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init

        _lazy_init()
        _import_deep_gemm()._C.init(args.patched_deep_gemm)
    cases = [
        dict(requests=1, verify=1, context=4096),
        dict(requests=1, verify=6, context=16384),
        dict(requests=5, verify=1, context=4096, ragged=True, reorder=True),
        dict(requests=5, verify=1, context=32768, ratio=2),
        dict(requests=3, verify=2, context=4096, workload="chain"),
        dict(requests=5, verify=1, context=4096, workload="full"),
        dict(requests=5, verify=1, context=4096, projection=True),
        dict(requests=128, verify=1, context=4096, workload="full"),
    ]
    if args.quick:
        cases = cases[:2] + cases[-1:]
    records = []
    for case in cases:
        fields, factory, spec, table = make_case(**case, full_backend=args.full_backend)
        variants = build_variants(
            fields, factory, input_modes=("replicated", "root"), poison_root=True
        )
        original = {k: v.clone() for k, v in fields.items()}
        graphs = {}
        for name, variant in variants.items():
            for _ in range(3):
                variant.forward()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                variant.forward()
            graphs[name] = graph
        for step in range(4):
            for k, value in fields.items():
                value.copy_(original[k])
            if step == 1:
                fields["lengths"][::3] = 0
                fields["lengths"][1::3] = 517
                if "blocks" in fields:
                    fields["blocks"][:, 1] = fields["blocks"][:, 0]
                    fields["blocks"][:, 2] = -1
            if step == 2:
                if "hidden" in fields:
                    fields["hidden"].zero_()
                else:
                    for k, value in fields.items():
                        if k.startswith("q"):
                            value.zero_()
            if step == 3:
                logical = factory.cache[table.flatten().long()].clone()
                permutation = torch.randperm(
                    factory.cache.shape[0],
                    device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(
                        9101 + dist.get_rank()
                    ),
                )
                factory.cache[permutation] = logical
                table.copy_(permutation.int().view_as(table))
                fields["requests"].copy_((original["requests"] + 1) % spec["requests"])
                fields["lengths"].sub_(13).clamp_min_(0)
                if "blocks" in fields:
                    fields["blocks"][:, ::2] = -1
            for execution in ("eager", "graph"):
                expected = None
                for name, variant in variants.items():
                    if execution == "graph":
                        graphs[name].replay()
                    else:
                        variant.forward()
                    actual = canonical(variant.output)
                    if expected is None:
                        expected = actual.clone()
                    assert torch.equal(actual, expected), (spec, step, execution, name)
                    if step == 2:
                        from vllm_dsv41_indexer.contract import candidate_positions

                        for row, length in enumerate(fields["lengths"].tolist()):
                            positions = (
                                candidate_positions(
                                    fields["blocks"][row].tolist(), length
                                )
                                if "blocks" in fields
                                else list(range(length))
                            )
                            wanted = set(positions[:512])
                            for selected in actual[row]:
                                assert set(selected[selected >= 0].tolist()) == wanted
                        if execution == "graph":
                            for _ in range(2):
                                graphs[name].replay()
                                assert torch.equal(canonical(variant.output), expected)
                    if spec["workload"] == "chain":
                        local = variant if name == "replicated" else variant.local
                        if local is not None:
                            ids = (
                                torch.arange(spec["rows"], device="cuda")
                                if name == "replicated"
                                else variant.row_ids[dist.get_rank()][: variant.count]
                            )
                            assert torch.equal(
                                local.source.candidates.sort(1).values,
                                variants["replicated"]
                                .source.candidates[ids]
                                .sort(1)
                                .values,
                            )
                    projection = []
                    if (
                        spec["projection"]
                        and name != "replicated"
                        and variant.local is not None
                    ):
                        ids = variant.row_ids[dist.get_rank()][: variant.count]
                        reference_module = variants["replicated"]
                        for stage, project in enumerate(variant.local.projections):
                            ref = reference_module.projections[stage]

                            def errors(a, b):
                                a, b = a.float(), b.float()
                                err = a - b
                                return dict(
                                    max_abs=float(err.abs().max()),
                                    nrmse=float(
                                        err.square().mean().sqrt()
                                        / b.square().mean().sqrt().clamp_min(1e-30)
                                    ),
                                    unequal=int((a != b).sum()),
                                )

                            projection.append(
                                dict(
                                    stage=stage,
                                    raw_q=errors(project.raw_q, ref.raw_q[ids]),
                                    raw_head_weights=errors(
                                        project.raw_w, ref.raw_w[ids]
                                    ),
                                    fp8_q=errors(project.q, ref.q[ids]),
                                    scaled_weights=errors(
                                        project.weights, ref.weights[ids]
                                    ),
                                )
                            )
                    for row, length in enumerate(fields["lengths"].tolist()):
                        for stage in actual[row]:
                            live = stage[stage >= 0]
                            assert live.numel() == live.unique().numel()
                            assert not live.numel() or int(live.max()) < length
                    record = dict(
                        case=spec,
                        step=step,
                        execution=execution,
                        variant=name,
                        sha256=digest(actual),
                        physical_table_sha256=digest(table),
                        projection_errors=projection,
                        root_input_nonroot_poisoned=name.endswith("root_input")
                        and dist.get_rank() != 0,
                    )
                    if name != "replicated":
                        record["communication"] = variant.communication()
                    records.append(record)
        torch.cuda.synchronize()
        for captured in graphs.values():
            captured.reset()
        del variants, graphs, fields, factory
        torch.cuda.empty_cache()
        if dist.get_rank() == 0:
            print(json.dumps(dict(case=spec, status="passed")), flush=True)
    gathered = [None] * world
    dist.all_gather_object(gathered, records)
    for idx in range(len(records)):
        assert len({r[idx]["sha256"] for r in gathered}) == 1
        if world > 1:
            assert len({r[idx]["physical_table_sha256"] for r in gathered}) == world
    if dist.get_rank() == 0:
        save(
            args.output,
            dict(
                status="passed",
                physical_world=world,
                single_rank_diagnostic=world == 1,
                source_sha256=provenance(),
                full_backend=args.full_backend,
                patched_deep_gemm=args.patched_deep_gemm,
                checks_per_rank=len(records),
                ranks=gathered,
            ),
        )
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

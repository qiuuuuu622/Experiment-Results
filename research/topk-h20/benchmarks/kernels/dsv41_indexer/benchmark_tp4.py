# SPDX-License-Identifier: Apache-2.0
"""Complete TP module cost, forward/reverse A/B and per-invocation rank maxima."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from .common import device_info, save
from .tp4_cases import (
    build_variants,
    canonical,
    initialize,
    make_case,
    measure,
    provenance,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-single-rank", action="store_true")
    parser.add_argument("--requests", nargs="+", type=int, default=[1, 8, 32, 64, 128])
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[4096, 16384, 65536, 131072, 262144]
    )
    parser.add_argument("--verify", nargs="+", type=int, default=[1])
    parser.add_argument("--ratios", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["compact", "full", "chain"],
        default=["compact"],
    )
    parser.add_argument("--projection", action="store_true")
    parser.add_argument("--patched-deep-gemm")
    parser.add_argument(
        "--full-backend", choices=["portable", "native_stable"], default="portable"
    )
    parser.add_argument(
        "--input-modes",
        nargs="+",
        choices=["replicated", "root"],
        default=["replicated", "root"],
    )
    parser.add_argument(
        "--execution", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    args = parser.parse_args()
    if args.full_backend == "native_stable" and not args.patched_deep_gemm:
        parser.error("native_stable requires explicit --patched-deep-gemm")
    world = initialize(args.allow_single_rank)
    patched = None
    if args.patched_deep_gemm:
        from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init

        _lazy_init()
        _import_deep_gemm()._C.init(args.patched_deep_gemm)
        package = Path(args.patched_deep_gemm)
        patched = dict(
            path=str(package),
            source_sha256={
                str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in package.rglob("*")
                if p.is_file() and p.suffix in (".h", ".cuh", ".so")
            },
        )
    frozen = provenance()
    payload = dict(
        status="running",
        physical_world=world,
        single_rank_diagnostic=world == 1,
        environment=device_info(),
        patched_deep_gemm=patched,
        source_sha256=frozen,
        command_args=vars(args),
        records=[],
        skips=[],
        dispatch="experimental; no automatic row-sharding enabled",
    )
    rank = dist.get_rank()
    for workload in args.workloads:
        for ratio in args.ratios:
            for context in args.contexts:
                for verify in args.verify:
                    for requests in args.requests:
                        fields, factory, spec, table = make_case(
                            requests,
                            verify,
                            context,
                            ratio,
                            workload=workload,
                            projection=args.projection,
                            full_backend=args.full_backend,
                        )
                        variants = build_variants(
                            fields,
                            factory,
                            input_modes=args.input_modes,
                            split=verify > 1 and requests < world,
                        )
                        reference = canonical(variants["replicated"].forward()).clone()
                        for name, variant in variants.items():
                            assert torch.equal(
                                canonical(variant.forward()), reference
                            ), (spec, name)
                        for execution in args.execution:
                            names = list(variants)
                            for order, sequence in (
                                ("forward", names),
                                ("reverse", names[::-1]),
                            ):
                                for name in sequence:
                                    measurement = measure(
                                        variants[name].forward,
                                        graph=execution == "graph",
                                    )
                                    measurements = [measurement]
                                    if measurement["coefficient_of_variation"] > 0.10:
                                        measurements += [
                                            measure(
                                                variants[name].forward,
                                                graph=execution == "graph",
                                            )
                                            for _ in range(3)
                                        ]
                                    record = dict(
                                        case=spec,
                                        execution=execution,
                                        order=order,
                                        variant=name,
                                        runs=measurements,
                                        communication=variants[name].communication()
                                        if name != "replicated"
                                        else dict(
                                            input_mode="replicated",
                                            result_collective=False,
                                        ),
                                    )
                                    payload["records"].append(record)
                                    if rank == 0:
                                        print(
                                            json.dumps(
                                                dict(
                                                    case=spec,
                                                    execution=execution,
                                                    order=order,
                                                    variant=name,
                                                    p50_us=measurement["p50_us"],
                                                    cv=measurement[
                                                        "coefficient_of_variation"
                                                    ],
                                                )
                                            ),
                                            flush=True,
                                        )
                                        save(args.output, payload)
                        del variants, fields, factory, reference, table
                        gc.collect()
                        torch.cuda.empty_cache()
    assert frozen == provenance(), (
        "Source changed during benchmark; retain evidence but "
        "do not compare as one frozen version"
    )
    payload["status"] = "complete"
    if rank == 0:
        save(args.output, payload)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

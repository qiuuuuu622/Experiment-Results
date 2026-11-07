# SPDX-License-Identifier: Apache-2.0
"""P7 pack/scatter, local compute and logical all-gather cost, measured separately.

Separate phases are diagnostics and are not added to infer complete latency.
"""

import argparse

import torch
import torch.distributed as dist

from .common import save
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
    parser.add_argument(
        "--workload", choices=["compact", "full", "chain"], default="compact"
    )
    parser.add_argument(
        "--full-backend", choices=["portable", "native_stable"], default="portable"
    )
    parser.add_argument("--patched-deep-gemm")
    parser.add_argument("--requests", nargs="+", type=int, default=[1, 32, 128])
    args = parser.parse_args()
    world = initialize()
    if args.full_backend == "native_stable":
        if not args.patched_deep_gemm:
            parser.error("native_stable requires --patched-deep-gemm")
        from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init

        _lazy_init()
        _import_deep_gemm()._C.init(args.patched_deep_gemm)
    payload = dict(
        status="running", physical_world=world, source_sha256=provenance(), records=[]
    )
    for requests in args.requests:
        fields, factory, spec, _ = make_case(
            requests=requests, workload=args.workload, full_backend=args.full_backend
        )
        variants = build_variants(
            fields, factory, input_modes=("replicated", "root"), split=False
        )
        reference = canonical(variants["replicated"].forward()).clone()
        for variant in variants.values():
            assert torch.equal(canonical(variant.forward()), reference)
        for execution in ("eager", "graph"):
            names = list(variants)
            for order, sequence in (("forward", names), ("reverse", names[::-1])):
                for name in sequence:
                    op = variants[name]
                    phases = (
                        ("forward",)
                        if name == "replicated"
                        else ("prepare", "compute", "collect", "forward")
                    )
                    for phase in phases:
                        runs = [measure(getattr(op, phase), graph=execution == "graph")]
                        if runs[0]["coefficient_of_variation"] > 0.10:
                            runs += [
                                measure(getattr(op, phase), graph=execution == "graph")
                                for _ in range(3)
                            ]
                        payload["records"].append(
                            dict(
                                case=spec,
                                execution=execution,
                                order=order,
                                variant=name,
                                phase=phase,
                                runs=runs,
                            )
                        )
                        if dist.get_rank() == 0:
                            save(args.output, payload)
        del fields, factory, variants
        torch.cuda.empty_cache()
    assert payload["source_sha256"] == provenance()
    payload["status"] = "complete"
    if dist.get_rank() == 0:
        save(args.output, payload)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

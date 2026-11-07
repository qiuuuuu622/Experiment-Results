# SPDX-License-Identifier: Apache-2.0
"""Projection M-dimension diagnostics; report FP32/FP8/weights independently."""

import argparse

import torch
import torch.distributed as dist

from .common import save
from .tp4_cases import build_variants, canonical, initialize, make_case, provenance


def errors(a, b):
    a, b = a.float(), b.float()
    delta = a - b
    return dict(
        max_abs=float(delta.abs().max()),
        nrmse=float(
            delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)
        ),
        unequal=int((a != b).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    world = initialize()
    records = []
    for requests in (1, 8, 32, 128):
        fields, factory, spec, _ = make_case(requests=requests, projection=True)
        variants = build_variants(
            fields,
            factory,
            input_modes=("replicated", "root"),
            split=False,
            poison_root=True,
        )
        reference_module = variants["replicated"]
        expected = canonical(reference_module.forward()).clone()
        for name, variant in variants.items():
            for execution in ("eager", "graph"):
                if execution == "graph":
                    for _ in range(3):
                        variant.forward()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        variant.forward()
                    graph.replay()
                else:
                    variant.forward()
                actual = canonical(variant.output)
                exact = torch.equal(actual, expected)
                record = dict(
                    case=spec,
                    variant=name,
                    execution=execution,
                    topk_sets_exact=exact,
                    projection=[],
                )
                if name != "replicated" and variant.local is not None:
                    ids = variant.row_ids[dist.get_rank()][: variant.count]
                    for stage, project in enumerate(variant.local.projections):
                        ref = reference_module.projections[stage]
                        record["projection"].append(
                            dict(
                                raw_q=errors(project.raw_q, ref.raw_q[ids]),
                                raw_head_weights=errors(project.raw_w, ref.raw_w[ids]),
                                fp8_q=errors(project.q, ref.q[ids]),
                                scaled_weights=errors(
                                    project.weights, ref.weights[ids]
                                ),
                            )
                        )
                records.append(record)
                assert exact, record
                if execution == "graph":
                    graph.reset()
        del fields, factory, variants
        torch.cuda.empty_cache()
    gathered = [None] * world
    dist.all_gather_object(gathered, records)
    if dist.get_rank() == 0:
        save(
            args.output,
            dict(
                status="passed",
                physical_world=world,
                source_sha256=provenance(),
                ranks=gathered,
            ),
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Four physical ranks: per-query consumers with replicated or row-owned work."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from .common import save
from .tp4_cases import initialize, provenance

EXPECTED_CORE = "77a7ffbf34ebe4495e560612c9b6943b17214b5fd01a9cdfe613d5887dbe9b1c"


def digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def make_fixture(sizes, keys):
    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.module import IndexerInputs

    from .inputs import make_inputs

    requests, rows = len(sizes), sum(sizes)
    _, _, _, cache, table, _, _, _ = make_inputs(
        dict(
            rows=requests,
            compressed_length=keys,
            rows_per_request=1,
            page_size=64,
            candidate_blocks=2048,
        )
    )
    logical = cache[table.flatten().long()].clone()
    permutation = torch.randperm(
        cache.shape[0],
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(55107 + dist.get_rank()),
    )
    local_cache = torch.empty_like(cache)
    local_cache[permutation] = logical
    table = permutation.int().view_as(table)
    # Actual FP8 bytes are unchanged; non-power-of-two scales exercise epilogue.
    local_cache.view(local_cache.shape[0], 8448)[:, 8192:].view(torch.float32).mul_(1.3)
    generator = torch.Generator(device="cuda").manual_seed(20261107)
    q = torch.randn((rows, 32, 128), generator=generator, device="cuda").to(
        torch.float8_e4m3fn
    )
    weights = torch.randn((rows, 32), generator=generator, device="cuda") / 64
    req = torch.tensor(
        [r for r, n in enumerate(sizes) for _ in range(n)],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor(
        [keys - n + j + 1 for n in sizes for j in range(n)],
        dtype=torch.int32,
        device="cuda",
    )
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32, device="cuda")
    for row, length in enumerate(lengths.tolist()):
        count = (length + 7) // 8
        chosen = torch.randperm(count - 1, device="cuda", generator=generator)
        chosen = torch.cat(
            (chosen[: min(2047, count - 1)], torch.tensor([count - 1], device="cuda"))
        )
        candidates[row, : chosen.numel()] = chosen.int()
    inputs = IndexerInputs(
        q, weights, local_cache, table, req, lengths, IndexerShape(rows, keys)
    )
    return inputs, candidates


def poison(op):
    op.state.blocks.fill_(-1)
    op.state.counts.zero_()
    op.scores.fill_(float("nan"))
    op.output.fill_(-42)
    if op.keys is not None:
        op.keys.fill_(127)
        op.scales.fill_(float("nan"))


def run_case(sizes, keys, source_unique, output_dir):
    from vllm_dsv41_indexer.contract import IndexerShape, candidate_positions
    from vllm_dsv41_indexer.module import IndexerInputs
    from vllm_dsv41_indexer.perquery_prefill import PerQueryPrefillIndexer
    from vllm_dsv41_indexer.row_shard import RowExchange, RowPlan

    from .verify_prefill_perquery import audit_outputs

    inputs, candidates = make_fixture(sizes, keys)
    fields = dict(
        q=inputs.q,
        weights=inputs.weights,
        requests=inputs.requests,
        lengths=inputs.lengths,
        blocks=candidates,
    )
    originals = {name: value.clone() for name, value in fields.items()}
    original_cache, original_table = inputs.cache.clone(), inputs.table.clone()
    identity = dict(
        rank=dist.get_rank(),
        physical_table_sha256=digest(inputs.table),
        logical_cache_sha256=digest(inputs.cache[inputs.table.flatten().long()]),
        logical_fields={name: digest(value) for name, value in fields.items()},
    )
    identities = [None] * dist.get_world_size()
    dist.all_gather_object(identities, identity)
    assert len({item["physical_table_sha256"] for item in identities}) == 4
    assert len({item["logical_cache_sha256"] for item in identities}) == 1
    assert all(
        item["logical_fields"] == identities[0]["logical_fields"] for item in identities
    )
    variants, replicated = {}, {}
    plans = [
        RowPlan.make(inputs.requests.tolist(), 4, split_rows=split)
        for split in (False, True)
    ]
    width = min(16384, (keys + 63) // 64 * 64)
    for backend in ("direct", "gathered_wgmma"):

        def factory(local, backend=backend):
            local_inputs = IndexerInputs(
                local["q"],
                local["weights"],
                inputs.cache,
                inputs.table,
                local["requests"],
                local["lengths"],
                IndexerShape(local["q"].shape[0], keys),
            )
            return PerQueryPrefillIndexer(
                local_inputs,
                local["blocks"],
                backend=backend,
                source_unique=source_unique,
                score_budget=5 * width * 4,
                gather_budget=3 * width * 132,
                max_chunk_rows=5,
            )

        op = factory(fields)
        replicated[backend] = op
        variants[backend + "/replicated"] = op
        for plan in plans:
            variants[backend + "/" + plan.mode] = RowExchange(
                fields, plan, factory, 512, input_mode="replicated"
            )
    graphs = {}
    for name, variant in variants.items():
        for _ in range(3):
            variant.forward()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            variant.forward()
        graphs[name] = graph
    records, baseline_outputs = [], {}
    for phase_id, phase in enumerate(
        [
            "initial",
            "candidate_length_update",
            "empty_candidates",
            "zero_query",
            "request_and_physical_page_remap",
            "all_zero_and_page_remap",
            "restore",
        ]
    ):
        for name, value in fields.items():
            value.copy_(originals[name])
        inputs.cache.copy_(original_cache)
        inputs.table.copy_(original_table)
        if phase == "candidate_length_update":
            candidates.copy_(candidates.roll(1, 0))
            if not source_unique:
                candidates[:, 1].copy_(candidates[:, 0])
            candidates[:, 2] = -1
            inputs.lengths.sub_(13).clamp_min_(0)
            inputs.lengths[::7] = 0
        elif phase == "empty_candidates":
            candidates.fill_(-1)
        elif phase == "zero_query":
            inputs.q.view(torch.uint8).zero_()
        elif phase in ("request_and_physical_page_remap", "all_zero_and_page_remap"):
            logical = inputs.cache[inputs.table.flatten().long()].clone()
            permutation = torch.randperm(
                inputs.cache.shape[0],
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(
                    61107 + 13 * phase_id + dist.get_rank()
                ),
            )
            inputs.cache[permutation] = logical
            inputs.table.copy_(permutation.int().view_as(inputs.table))
            if phase.startswith("all_zero"):
                inputs.lengths.zero_()
            else:
                # Bijective request renaming preserves each ownership group's rows.
                inputs.requests.copy_((inputs.requests + 1) % len(sizes))
                inputs.weights.mul_(1.0001)
                candidates[:, ::3] = -1
        for variant in variants.values():
            if isinstance(variant, RowExchange):
                variant.receive.zero_()
                variant.local_output.fill_(-1)
                variant.output.fill_(-42)
                if variant.local is not None:
                    poison(variant.local)
            else:
                poison(variant)
        dist.barrier()
        snapshots = {}
        # All captured outputs are saved before any eager/oracle refresh.
        for name, variant in variants.items():
            graphs[name].replay()
            snapshots[name] = variant.output.clone()
        torch.cuda.synchronize()
        dist.barrier()
        eager = {}
        for name, variant in variants.items():
            eager[name] = variant.forward().clone()
        torch.cuda.synchronize()
        expected = torch.empty_like(replicated["direct"].output)
        oracle = None
        if dist.get_rank() == 0:
            oracle = audit_outputs(
                inputs,
                candidates,
                replicated,
                {name: snapshots[name + "/replicated"] for name in replicated},
            )
            expected.copy_(snapshots["direct/replicated"])
        # This broadcast, like RowExchange.collect, carries only logical top-k.
        dist.broadcast(expected, src=0)
        expected_sorted = expected.sort(1).values
        for execution, outputs in (("graph", snapshots), ("eager", eager)):
            for name, actual in outputs.items():
                canonical = actual.sort(1).values
                assert torch.equal(canonical, expected_sorted), (phase, execution, name)
                if phase == "initial":
                    baseline_outputs[execution, name] = canonical.clone()
                elif phase == "restore":
                    assert torch.equal(canonical, baseline_outputs[execution, name])
                if phase in ("empty_candidates", "all_zero_and_page_remap"):
                    assert (canonical == -1).all().item()
                if phase == "zero_query":
                    for row, length in enumerate(inputs.lengths.tolist()):
                        positions = candidate_positions(
                            candidates[row].tolist(), length
                        )
                        selected = actual[row][actual[row] >= 0].tolist()
                        assert set(selected) == set(positions[:512])
        records.append(
            dict(
                phase=phase,
                rank=dist.get_rank(),
                logical_output_sha256=digest(expected_sorted),
                graph_and_eager_variants=len(variants) * 2,
                exact_rows_per_variant=sum(sizes),
                oracle_rank0=oracle,
                local_physical_table_sha256=digest(inputs.table),
            )
        )
        save(
            output_dir / f"rank{dist.get_rank()}-{sum(sizes)}-{keys}.json",
            dict(
                status="running" if phase != "restore" else "completed", records=records
            ),
        )
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    dict(rows=sum(sizes), keys=keys, phase=phase, status="passed")
                ),
                flush=True,
            )
    return dict(
        sizes=sizes,
        keys=keys,
        ratio=1,
        source_unique=source_unique,
        identities=identities,
        phases=records,
        plans=[plan.describe() for plan in plans],
        local_workspaces={
            name: dict(
                rows=op.inputs.shape.rows,
                chunks=op.chunks,
                score_bytes=op.score_bytes,
                gather_bytes=op.gather_bytes,
            )
            for name, variant in variants.items()
            if (op := variant.local if isinstance(variant, RowExchange) else variant)
            is not None
        },
        rank_local_cache=True,
        collectives_send_physical_pages=False,
        input_mode="replicated inputs; local row packing is captured",
        semantics="unordered logical index-key top-k; exact sorted-set comparison",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    world = initialize()
    import vllm_dsv41_indexer.perquery_prefill as core

    core_hash = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    assert core_hash == EXPECTED_CORE, core_hash
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    physical = visible[int(os.environ["LOCAL_RANK"])]
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={physical}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    devices = [None] * world
    dist.all_gather_object(devices, dict(rank=rank, physical_gpu=physical, uuid=uuid))
    assert len({x["uuid"] for x in devices}) == 4
    report = dict(
        status="running",
        rank=rank,
        world_size=world,
        devices=devices,
        core_sha256=core_hash,
        provenance_before=provenance(),
        cases=[],
        scope="real four-GPU correctness only; no speedup measurement",
    )
    try:
        for sizes, keys, unique in [
            ([17, 9, 25, 13], 16384, False),
            ([7, 57], 65536, True),
            ([33, 31, 64], 65536, True),
        ]:
            report["cases"].append(run_case(sizes, keys, unique, out))
            save(out / f"rank{rank}.json", report)
        report["provenance_after"] = provenance()
        assert report["provenance_before"] == report["provenance_after"]
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save(out / f"rank{rank}.json", report)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

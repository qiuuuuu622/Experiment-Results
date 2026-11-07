# SPDX-License-Identifier: Apache-2.0
"""Fixtures/measurement for real process-group indexer module experiments."""

import hashlib
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_dsv41_indexer.row_shard import IndexerRows, RowExchange, RowPlan

from .common import check_vllm
from .inputs import make_inputs


def initialize(allow_single_rank=False):
    check_vllm()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world != 4 and not (allow_single_rank and world == 1):
        raise RuntimeError(
            "Requires four physical ranks (or explicit single-rank diagnostic)"
        )
    devices = [None] * world
    dist.all_gather_object(
        devices,
        dict(host=os.uname().nodename, local_rank=int(os.environ["LOCAL_RANK"])),
    )
    if len({(d["host"], d["local_rank"]) for d in devices}) != world:
        raise RuntimeError("Ranks do not have distinct local device assignments")
    return world


def make_case(
    requests=8,
    verify=1,
    context=65536,
    ratio=1,
    *,
    workload="compact",
    projection=False,
    ragged=False,
    reorder=False,
    full_backend="portable",
):
    max_keys = (context + ratio - 1) // ratio
    sizes = [1, 2, 4, 6, 8][:requests] if ragged else [verify] * requests
    if ragged and requests > 5:
        sizes += [verify] * (requests - 5)
    r = sum(sizes)
    _, _, _, cache, table, _, _, base_blocks = make_inputs(
        dict(
            rows=requests,
            compressed_length=max_keys,
            rows_per_request=1,
            page_size=64,
            candidate_blocks=2048,
        )
    )
    generator = torch.Generator(device="cuda").manual_seed(20261107)
    req = torch.tensor(
        [i for i, n in enumerate(sizes) for _ in range(n)],
        device="cuda",
        dtype=torch.int32,
    )
    lens = torch.tensor(
        [max(0, max_keys - n + 1 + j) for n in sizes for j in range(n)],
        device="cuda",
        dtype=torch.int32,
    )
    if reorder:
        order = torch.randperm(r, device="cuda", generator=generator)
        req, lens = req[order], lens[order]
    # Same logical keys, deliberately different physical layout on each rank.
    logical = cache[table.flatten().long()]
    permutation = torch.randperm(
        cache.shape[0],
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(3071 + dist.get_rank()),
    )
    local_cache = torch.empty_like(cache)
    local_cache[permutation] = logical
    local_table = permutation.int().view_as(table)
    fields = dict(requests=req.contiguous(), lengths=lens.contiguous())
    if workload == "compact":
        fields["blocks"] = base_blocks[req.long()].contiguous()
    stages = 5 if workload == "chain" else 1
    projection_weights = None
    if projection:
        hidden = 256
        fields["hidden"] = torch.randn((r, hidden), generator=generator, device="cuda")
        projection_weights = [
            (
                torch.randn((hidden, 4096), generator=generator, device="cuda")
                / hidden**0.5,
                torch.randn((hidden, 32), generator=generator, device="cuda")
                / hidden**0.5,
            )
            for _ in range(stages)
        ]
    else:
        for stage in range(stages):
            fields[f"q{stage}"] = torch.randn(
                (r, 32, 128), generator=generator, device="cuda"
            ).to(torch.float8_e4m3fn)
            fields[f"w{stage}"] = (
                torch.randn((r, 32), generator=generator, device="cuda")
                / (128 * 32) ** 0.5
            )
    spec = dict(
        requests=requests,
        verify=verify,
        rows=r,
        logical_context=context,
        index_keys=max_keys,
        ratio=ratio,
        workload=workload,
        projection=projection,
        ragged=ragged,
        reorder=reorder,
        candidate_capacity=2048 if workload != "full" else None,
        cache_replicated=True,
        local_score_backend="compact_wgmma"
        if workload == "compact"
        else (
            "Native_DeepGEMM_next_n1_prepare_each_step_Ant_stable"
            if full_backend == "native_stable"
            else "ChunkedIndexer_grouped_scores_group_rows1"
        ),
        baseline_scope=(
            "same_local_kernel_replicated_vs_sharded; not best_native_full_comparison"
        ),
        valid_index_keys_per_query=lens.tolist(),
        external_candidate_counts_per_query=(fields["blocks"] >= 0).sum(1).tolist()
        if "blocks" in fields
        else None,
        projection_description=(
            "synthetic FP32 linear; per-head maxabs/448 FP8 Q; "
            "scale folded into head weight"
        )
        if projection
        else "prequantized FP8 Q; head weights already contain scale",
    )

    def factory(values):
        return IndexerRows(
            values,
            local_cache,
            local_table,
            max_keys,
            workload=workload,
            ratio=ratio,
            projection_weights=projection_weights,
            full_backend=full_backend,
        )

    factory.cache = local_cache
    factory.table = local_table
    return fields, factory, spec, local_table


def build_variants(
    fields, factory, *, input_modes=("replicated",), split=True, poison_root=False
):
    result = {"replicated": factory(dict(fields))}
    world = dist.get_world_size()
    for mode in input_modes:
        for split_rows in (False, True) if split else (False,):
            plan = RowPlan.make(
                fields["requests"].tolist(), world, split_rows=split_rows
            )
            name = f"shard_{plan.mode}_{mode}_input"
            # Separate dictionary: projections add stage Q/weights locally.
            transmitted = dict(fields)
            if poison_root and mode == "root" and dist.get_rank() != 0:
                transmitted = {k: torch.zeros_like(v) for k, v in fields.items()}
                transmitted["requests"].fill_(-12345)
                transmitted["lengths"].fill_(-12345)
            result[name] = RowExchange(
                transmitted,
                plan,
                factory,
                result["replicated"].output.shape[1],
                input_mode=mode,
            )
    return result


def canonical(output):
    return output.view(output.shape[0], -1, 512).sort(-1).values


def digest(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[3]
    files = [
        p
        for p in (root / "vllm_dsv41_indexer").rglob("*")
        if p.is_file() and p.suffix in (".py", ".cu", ".cuh", ".h", ".hpp", ".json")
    ]
    files += list(Path(__file__).parent.glob("*tp4*.py"))
    files += [
        Path(__file__).with_name(name)
        for name in ("common.py", "inputs.py", "baseline.json")
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def measure(fn, *, graph, repeats=30, inner=10):
    """Every invocation is timed on every rank, then max across ranks per call.

    Graph captures one complete module invocation (including NCCL). Each sample
    contains ten event pairs; per-invocation maxima are averaged per sample.
    GPU staging barrier is excluded; all communication inside fn is included.
    Host wall includes module calls and completion synchronization, averaged
    over ten invocations, then max across ranks. Instrumentation is disclosed.
    """
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph_object = None
    before_graph_allocated = torch.cuda.memory_allocated()
    if graph:
        graph_object = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_object):
            fn()
        call = graph_object.replay
    else:
        call = fn
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(inner)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(inner)]
    all_iterations, samples, wall_samples = [], [], []
    peak_base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        dist.barrier()
        wall_start = time.perf_counter_ns()
        for start, end in zip(starts, ends):
            start.record()
            call()
            end.record()
        ends[-1].synchronize()
        wall = (time.perf_counter_ns() - wall_start) / 1000 / inner
        values = torch.tensor(
            [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
            + [wall],
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        maximums = values.tolist()
        all_iterations.append(maximums[:-1])
        samples.append(statistics.mean(maximums[:-1]))
        wall_samples.append(maximums[-1])
    peak = torch.tensor(
        torch.cuda.max_memory_allocated(), dtype=torch.int64, device="cuda"
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if graph_object is not None:
        graph_object.reset()
    return dict(
        p50_us=statistics.median(samples),
        p90_us=sorted(samples)[int((repeats - 1) * 0.9)],
        coefficient_of_variation=statistics.pstdev(samples) / statistics.mean(samples),
        samples_us=samples,
        iterations_rank_max_us=all_iterations,
        wall_p50_us=statistics.median(wall_samples),
        wall_p90_us=sorted(wall_samples)[int((repeats - 1) * 0.9)],
        wall_samples_us=wall_samples,
        peak_allocated_bytes=int(peak.item()),
        premeasure_allocated_bytes=peak_base,
        graph_allocated_delta_bytes=peak_base - before_graph_allocated,
        warmup=20,
        repeats=repeats,
        inner=inner,
        aggregation="max_rank_each_invocation_then_mean_within_sample",
        event_instrumentation=(
            "one CUDA event pair per invocation; graph captures one invocation"
        ),
        staging_barrier_timed=False,
        result_collective_timed=True,
    )

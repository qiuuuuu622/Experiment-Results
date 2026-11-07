# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measurement helpers. Imported only by explicitly requested GPU runs."""

import hashlib
import json
import statistics
import sys
import time
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def check_sources():
    return json.loads(Path(__file__).with_name("baseline.json").read_text())


def check_vllm():
    # A checkout's vllm/ would otherwise shadow the installed baseline wheel.
    # Import the actual installed package, never replace an already loaded one.
    installed = Path(distribution("vllm").locate_file("vllm")).resolve()
    sys.path.insert(0, str(installed.parent))
    try:
        import vllm
    finally:
        sys.path.pop(0)
    if Path(vllm.__file__).resolve().parent != installed:
        raise RuntimeError("Start a fresh process without a different vllm import")

    root = Path(vllm.__file__).resolve().parent.parent
    manifest = check_sources()
    # Verify the actual wrappers against CURRENT HEAD, independent of wheel name.
    # This does not imply the binary extensions were compiled from that HEAD.
    for item in manifest["files"]:
        actual = root / item["source"]
        if not actual.is_file():
            raise RuntimeError(f"Missing current-HEAD baseline wrapper: {actual}")
        if hashlib.sha256(actual.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Installed wrapper differs from HEAD: {actual}")
    return str(root)


def device_info():
    import subprocess

    import torch

    p = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    assert p.major == 9 and p.minor == 0 and "H20" in p.name, p
    try:
        topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    except (OSError, subprocess.CalledProcessError):
        topology = "unavailable"
    return dict(
        name=p.name,
        sm_count=p.multi_processor_count,
        topology=topology,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        memory_bytes=p.total_memory,
    )


def measure(fn, *, repeats=30, inner=10, graph=False, distributed=False):
    import torch
    import torch.distributed as dist

    for _ in range(20):
        fn()
    torch.accelerator.synchronize()
    if distributed:
        dist.barrier()
    call = fn
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
        call = g.replay
    samples, wall_samples = [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        if distributed:
            dist.barrier()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        wall_start = time.perf_counter_ns()
        start.record()
        if graph:
            call()
        else:
            for _ in range(inner):
                call()
        end.record()
        end.synchronize()
        gpu_us = start.elapsed_time(end) * 1000 / inner
        wall_us = (time.perf_counter_ns() - wall_start) / 1000 / inner
        if distributed:
            values = torch.tensor([gpu_us, wall_us], device="cuda", dtype=torch.float64)
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            gpu_us, wall_us = values.tolist()
        samples.append(gpu_us)
        wall_samples.append(wall_us)
    return dict(
        p50_us=statistics.median(samples),
        p90_us=sorted(samples)[int(0.90 * (len(samples) - 1))],
        p95_us=sorted(samples)[int(0.95 * (len(samples) - 1))],
        samples_us=samples,
        wall_p50_us=statistics.median(wall_samples),
        wall_samples_us=wall_samples,
        coefficient_of_variation=statistics.pstdev(samples) / statistics.mean(samples),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        warmup=20,
        repeats=repeats,
        inner=inner,
        rank_aggregation="max_per_sample" if distributed else "single_rank",
    )


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

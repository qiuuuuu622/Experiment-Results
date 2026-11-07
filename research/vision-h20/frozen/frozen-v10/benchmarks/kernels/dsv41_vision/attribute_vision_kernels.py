# SPDX-License-Identifier: Apache-2.0
"""Attribute recorded kernels by launch correlation and exact source frames.

Source sites below refer to the pinned 29be941d vision.py. This does not match
kernel-name families heuristically and deliberately leaves graph work unmapped.
"""

import argparse
import heapq
import json
from collections import defaultdict
from pathlib import Path


def innermost_scopes(runtimes, scopes):
    """Sweep time once; keep shortest currently containing scope per thread."""
    ordered = sorted(scopes, key=lambda e: e["ts"])
    heaps, result = defaultdict(list), {}
    cursor = 0
    for correlation, runtime in sorted(runtimes.items(), key=lambda p: p[1]["ts"]):
        now = runtime["ts"]
        while cursor < len(ordered) and ordered[cursor]["ts"] <= now:
            event = ordered[cursor]
            heapq.heappush(heaps[event["tid"]], (event["dur"], cursor, event))
            cursor += 1
        heap = heaps[runtime["tid"]]
        while heap and heap[0][2]["ts"] + heap[0][2]["dur"] <= now:
            heapq.heappop(heap)
        result[correlation] = heap[0][2] if heap else None
    return result


def analyze(path):
    trace = json.loads(path.read_text())
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    runtimes = {
        e.get("args", {}).get("correlation"): e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
    }
    frames = [
        e
        for e in events
        if e.get("cat") == "python_function"
        and "/deepseek_v4/common/vision.py(" in e.get("name", "")
    ]
    ops = [e for e in events if e.get("cat") == "cpu_op"]
    runtime_frames = innermost_scopes(runtimes, frames)
    runtime_ops = innermost_scopes(runtimes, ops)
    rows = defaultdict(lambda: {"count": 0, "gpu_us": 0.0})
    sites = defaultdict(set)
    total = 0.0
    for kernel in events:
        if kernel.get("cat") != "kernel":
            continue
        total += kernel["dur"]
        correlation = kernel.get("args", {}).get("correlation")
        runtime = runtimes.get(correlation)
        family, op = "unattributed", "unknown"
        if runtime:
            frame = runtime_frames[correlation]
            cpu_op = runtime_ops[correlation]
            op = cpu_op["name"] if cpu_op else "unknown"
            site = frame["name"] if frame else "unknown"
            if frame:
                if "vision.py(62): apply_rotary" in site:
                    family = "apply_rotary"
                elif "vision.py(74): forward" in site:
                    family = "rms_norm"
                elif "vision.py(190): forward" in site and op == "aten::add":
                    family = "block_residual_add"
                elif op in ("aten::mm", "aten::addmm", "aten::bmm"):
                    family = "gemm"
                elif "flash::FlashAttnFwd" in kernel["name"]:
                    family = "flash_attention"
                else:
                    family = "other_vision"
                sites[family].add(site)
        # Exact communication symbols observed in these traces. Retain custom
        # and NCCL separately, including kernels launched outside vision frames.
        if "multimem_all_reduce_kernel" in kernel["name"]:
            family = "symmetric_memory_multimem_all_reduce"
        elif kernel["name"].startswith("ncclDevKernel"):
            family = "nccl_collective"
        row = rows[(family, op)]
        row["count"] += 1
        row["gpu_us"] += kernel["dur"]
    summary = defaultdict(float)
    for (family, _), row in rows.items():
        summary[family] += row["gpu_us"]
    return {
        "trace": str(path),
        "kernel_sum_us": total,
        "family_gpu_us": dict(summary),
        "source_frames": {k: sorted(v) for k, v in sites.items()},
        "operations": [
            {"family": family, "cpu_op": op, **row}
            for (family, op), row in sorted(rows.items())
        ],
        "method": "GPU correlation -> runtime timestamp -> innermost source frame",
        "limitations": "Kernel sums are not critical-path latency; pinned source lines",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(
        json.dumps([analyze(t) for t in args.traces], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

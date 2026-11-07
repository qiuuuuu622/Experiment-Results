# SPDX-License-Identifier: Apache-2.0
"""Per-rank Chrome trace accounting, complementary to profiler-skill triage.

GPU duration sums are resource usage, not latency. Busy time is interval union.
CPU scopes are inclusive and may overlap; they must never be added together.
Only observed launch correlations can attribute GPU work to Python scopes.
"""

import argparse
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def union_duration(intervals):
    end, total = float("-inf"), 0.0
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def analyze(path):
    raw = path.read_bytes()
    data = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    scopes = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("name", "").startswith(
            (
                "vision::",
                "vision.",
                "aligner.",
                "span.",
                "dp.",
                "metadata.",
                "module.",
                "cpu.",
                "h2d.",
                "image_component.",
            )
        )
    ]
    runtimes = {}
    for event in events:
        if event.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = event.get("args", {}).get("correlation")
            if corr is not None:
                runtimes[corr] = event
    scope_gpu = defaultdict(float)
    categories = defaultdict(float)
    kernel_rows = defaultdict(lambda: {"count": 0, "sum_us": 0.0})
    for event in gpu:
        duration = event["dur"]
        name = event["name"]
        categories[event.get("cat")] += duration
        if "nccl" in name.lower():
            categories["nccl_kernel_name_matched"] += duration
        kernel_rows[name]["count"] += 1
        kernel_rows[name]["sum_us"] += duration
        runtime = runtimes.get(event.get("args", {}).get("correlation"))
        matches = []
        if runtime:
            matches = [
                s
                for s in scopes
                if s.get("tid") == runtime.get("tid")
                and s["ts"] <= runtime["ts"] < s["ts"] + s["dur"]
            ]
        scope = min(matches, key=lambda s: s["dur"]) if matches else None
        scope_gpu[scope["name"] if scope else "unattributed"] += duration
    cpu = defaultdict(list)
    for event in scopes:
        cpu[event["name"]].append(event["dur"])
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in gpu]
    gpu_sum = sum(e["dur"] for e in gpu)
    busy = union_duration(intervals)
    return {
        "trace": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "gpu_event_count": len(gpu),
        "gpu_duration_sum_us": gpu_sum,
        "gpu_busy_union_us": busy,
        "gpu_observed_window_us": (
            max(b for _, b in intervals) - min(a for a, _ in intervals)
            if intervals
            else 0
        ),
        "categories_sum_us": dict(categories),
        "nccl_share_of_gpu_sum": (
            categories["nccl_kernel_name_matched"] / gpu_sum if gpu_sum else None
        ),
        "launch_scope_gpu_sum_us": dict(scope_gpu),
        "cpu_inclusive_scopes": {
            name: {"samples_us": vals, "p50_us": statistics.median(vals)}
            for name, vals in cpu.items()
        },
        "kernels": sorted(
            ({"name": name, **row} for name, row in kernel_rows.items()),
            key=lambda row: row["sum_us"],
            reverse=True,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ranks = [analyze(path) for path in args.traces]
    medians = [
        scope["p50_us"]
        for r in ranks
        for name, scope in r["cpu_inclusive_scopes"].items()
        if name == "vision::module" or name.startswith("module.")
    ]
    gpu_windows = [r["gpu_observed_window_us"] for r in ranks]
    result = {
        "notes": [
            "Compare traces of the same mode, shape, variant and iteration count",
            "Kernel sums and communication shares are resource time, not latency",
            "GPU busy time merges overlapping GPU intervals within each rank",
            "CPU scopes are inclusive and not additive",
            "Graph kernels cannot be assigned to internal phases without evidence",
            "Rank timestamps are not cross-rank aligned by this script",
            "No phase durations are inferred from call counts",
            "CPU module scope can be asynchronous enqueue; not end-to-end latency",
        ],
        "rank_module_p50_max_over_min": (
            max(medians) / min(medians) if medians and min(medians) > 0 else None
        ),
        "rank_gpu_window_max_over_min": (
            max(gpu_windows) / min(gpu_windows)
            if gpu_windows and min(gpu_windows) > 0
            else None
        ),
        "ranks": ranks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

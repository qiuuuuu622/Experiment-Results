# SPDX-License-Identifier: Apache-2.0
"""Audit behavior against missing ranks and incorrect aggregate statistics."""

import importlib.util
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/kernels/dsv41_vision/audit_formal_results.py"
)
SPEC = importlib.util.spec_from_file_location("vision_formal_audit", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def measurement():
    gpu = [100.0 + i for i in range(30)]
    wall = [200.0 + i for i in range(30)]
    rank_arrays = [
        {"gpu_us": [x - rank for x in gpu], "wall_us": [x - rank for x in wall]}
        for rank in range(4)
    ]
    statistics = MODULE.stats(gpu)
    return {
        "warmup": 20,
        "samples": 30,
        "inner": 10,
        "aggregation": "max_rank_each_invocation_then_mean_within_sample",
        "samples_us": gpu,
        "wall_samples_us": wall,
        "p50_us": statistics["p50_us"],
        "p90_us": statistics["p90_us"],
        "coefficient_of_variation": statistics["cv"],
        "wall_p50_us": 214.5,
        "per_rank_samples": rank_arrays,
        "peak_allocated_bytes": 1024,
        "extra_rounds": [],
    }


def test_recomputes_statistics_and_keeps_rank0_peak_label():
    audit = MODULE.Audit()
    row = audit.measurement(measurement(), "valid")
    assert not audit.errors
    assert row["gpu"]["p50_us"] == 114.5
    assert row["gpu"]["p90_us"] == 126
    assert row["peak_allocated_bytes_rank0_only"] == 1024
    assert len(row["per_rank"]) == 4


def test_rejects_missing_rank_and_incorrect_reported_percentile():
    item = measurement()
    item["per_rank_samples"].pop()
    item["p90_us"] = 999
    audit = MODULE.Audit()
    audit.measurement(item, "broken")
    assert any("four rank" in e for e in audit.errors)
    assert any("p90_us differs" in e for e in audit.errors)


def test_rejects_fastest_rank_wall_and_gpu_below_rank_mean():
    item = measurement()
    item["per_rank_samples"][3]["wall_us"][0] = 250
    item["per_rank_samples"][3]["gpu_us"][0] = 120
    audit = MODULE.Audit()
    audit.measurement(item, "broken")
    assert any("wall is not max rank" in e for e in audit.errors)
    assert any("GPU below max rank" in e for e in audit.errors)


def test_missing_file_stays_pending(tmp_path):
    audit = MODULE.Audit()
    audit.run_file(tmp_path / "future.json", "weights", {}, "frozen-v10")
    assert audit.pending
    assert not audit.errors
    assert not audit.rows


def test_progress_case_missing_correctness_is_not_a_kernel_failure():
    case = {"count": 8, "measurements": []}
    audit = MODULE.Audit()
    audit.case_correctness(case, "weights B8", running=True)
    assert audit.pending == ["weights B8: bitwise correctness not yet written"]
    assert not audit.errors

    completed = MODULE.Audit()
    completed.case_correctness(case, "weights B8", running=False)
    assert completed.errors
    assert not completed.pending

    case["eager_rope_vs_original_all_ranks"] = [
        [{"exact": rank != 0, "finite": True} for _ in range(8)] for rank in range(4)
    ]
    failed = MODULE.Audit()
    failed.case_correctness(case, "weights B8", running=True)
    assert failed.errors
    assert not failed.pending

    # Measurement records without prior correctness are not a valid progress case.
    case.pop("eager_rope_vs_original_all_ranks")
    case["measurements"] = [{"variant": "eager"}]
    invalid = MODULE.Audit()
    invalid.case_correctness(case, "weights B8", running=True)
    assert invalid.errors

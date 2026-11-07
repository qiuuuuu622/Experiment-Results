# SPDX-License-Identifier: Apache-2.0
"""Validate final P5c raw evidence and build an exact-contract experiment table."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from dataclasses import asdict
from pathlib import Path

from vllm_dsv41_indexer.dispatch import _rounds
from vllm_dsv41_indexer.perquery_dispatch import (
    COMPACT,
    FULL,
    PerQueryContract,
    digest,
    select_measurements,
)

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_evidence(directory, before, groups):
    """Audit same-object two-order replication; never reconstruct old tensors."""
    directory = Path(directory).resolve()
    artifacts = [directory / "paired_pipeline.py"]
    long_shapes = {"steady_64k", "steady_256k", "tail_1m", "ragged"}
    fields = {
        "q",
        "weights",
        "cache",
        "table",
        "requests",
        "lengths",
        "source_q",
        "source_weights",
        "candidate_raw",
        "candidate_canonical",
    }
    cases, rounds_checked = {}, 0
    for group in ("a", "b"):
        before_path = directory / f"runtime-before-{group}.json"
        after_path = directory / f"runtime-after-{group}.json"
        for path in (before_path, after_path):
            identity = json.loads(path.read_text())
            assert all(
                identity[k] == before[k]
                for k in before
                if k != "native_identity_capture"
            )
            artifacts.append(path)
        path = directory / f"group-{group}.json"
        raw = json.loads(path.read_text())
        assert raw["status"] == "completed"
        assert raw["identity_sha256"] == sha(before_path)
        assert raw["runner_sha256"] == sha(directory / "paired_pipeline.py")
        assert raw["old_raw_canonical_hash_reconstructed"] is False
        assert raw["host"] == "h20-4"
        assert raw["device"]["name"] == "NVIDIA H20"
        assert raw["device"]["sm_count"] == 78
        assert (
            raw["frozen_driver"]["sha256"]
            == before["source_files"][
                "benchmarks/kernels/dsv41_indexer/benchmark_prefill_perquery.py"
            ]
        )
        assert (
            raw["frozen_core"]["sha256"]
            == before["source_files"]["vllm_dsv41_indexer/perquery_prefill.py"]
        )
        artifacts.append(path)
        for case in raw["cases"]:
            name = case["case"]
            assert name not in cases and name in groups
            assert (case["ratio"], case["heads"], case["head_dim"]) == (1, 32, 128)
            assert case["input_bytes_unchanged"] is True
            assert set(case["input_before"]) == fields
            assert case["input_before"] == case["input_after"]
            assert case["source_canonical_sets_exact"] is True
            assert len(case["source_repeats"]) == 3
            assert all(
                repeat["canonical"] == case["input_before"]["candidate_canonical"]
                for repeat in case["source_repeats"]
            )
            for field in (
                "query_lens",
                "contexts",
                "candidate_positions_min",
                "candidate_positions_max",
                "candidate_positions_sum",
            ):
                assert case[field] == groups[name]["forward"][field]
            assert case["paired_pipeline"] == (name in long_shapes)
            assert case["source_generation_in_pipeline"] is False
            if case["paired_pipeline"]:
                expected = {
                    (order, mode, variant)
                    for order in ("forward", "reverse")
                    for mode in ("eager", "graph")
                    for variant in FULL + COMPACT
                }
                actual = [
                    (r["order"], r["execution"], r["variant"]) for r in case["records"]
                ]
                assert len(actual) == 16 and set(actual) == expected
                objects = case["path_object_identity"]
                assert set(objects) == set(FULL + COMPACT)
                assert len({v["python_id"] for v in objects.values()}) == 4
                for field in ("candidate_data_ptr", "q_data_ptr", "cache_data_ptr"):
                    assert len({v[field] for v in objects.values()}) == 1
                checks = case["correctness"]
                assert checks["fixed_logits_topk"] == "all rows exact logical sets"
                assert checks["compact_direct_gather_score_comparison"] == (
                    "all candidate scores bitwise equal"
                )
                assert all(
                    checks["full_baselines"][k]["checked_rows"] == checks["rows"]
                    for k in FULL
                )
                for item in case["records"]:
                    assert item["phase"] == "pipeline"
                    for measurement in _rounds(item["measurement"]):
                        assert math.isclose(
                            statistics.median(measurement["wall_samples_us"]),
                            measurement["wall_p50_us"],
                            rel_tol=1e-9,
                        )
                        rounds_checked += 1
            else:
                assert not case["records"]
            cases[name] = case
    assert set(cases) == set(groups)
    return cases, artifacts, rounds_checked


def collect(directory, paired_directory=None):
    directory = Path(directory).resolve()
    before_path = directory / "runtime-before.json"
    before = json.loads(before_path.read_text())
    artifacts = [before_path]
    for group in ("a", "b"):
        path = directory / f"runtime-after-{group}.json"
        after = json.loads(path.read_text())
        if any(before[k] != after[k] for k in before if k != "native_identity_capture"):
            raise ValueError("Source/native identity changed during performance run")
        if not after["source_archive_matches_post_run"]:
            raise ValueError("Post-run sources differ from archive")
        artifacts.append(path)
    archive = directory / "frozen/source.tar.gz"
    assert sha(archive) == before["source_archive_sha256"]
    artifacts.append(archive)
    with tarfile.open(archive) as tar:
        members = {m.name.removeprefix("./"): m for m in tar.getmembers() if m.isfile()}
        for name, value in before["source_files"].items():
            assert (
                hashlib.sha256(tar.extractfile(members[name]).read()).hexdigest()
                == value
            )

    verify_path, preflight_path = (
        directory / "verify.json",
        directory / "preflight.json",
    )
    verify, preflight = (
        json.loads(p.read_text()) for p in (verify_path, preflight_path)
    )
    assert verify["status"] == "passed" and len(verify["cases"]) == 42
    assert preflight["status"] == "completed" and len(preflight["cases"]) == 2
    for case in verify["cases"]:
        assert case["checks"]["graph_protocol"].startswith("poison buffers, replay")
        for name in COMPACT:
            assert case["checks"][name]["fixed_logits_selection"] == "exact"
            assert case["checks"][name]["direct_gather_scores"] == "bitwise_equal"
    artifacts += [verify_path, preflight_path]
    environment = digest(
        {
            "native_files": before["native_files"],
            "versions": before["versions"],
            "hardware": "h20-4:NVIDIA H20:sm90:78",
        }
    )
    groups, rounds_checked = {}, 0
    for group in ("a", "b"):
        for order in ("forward", "reverse"):
            path = directory / f"group-{group}-{order}.json"
            raw = json.loads(path.read_text())
            assert raw["status"] == "completed" and len(raw["cases"]) == 4
            assert raw["purpose"] == "performance"
            assert raw["identity_sha256"] == sha(before_path)
            assert raw["args"]["reverse"] == (order == "reverse")
            assert (
                raw["device"]["name"] == "NVIDIA H20"
                and raw["device"]["sm_count"] == 78
            )
            artifacts.append(path)
            for case in raw["cases"]:
                key = case["case"]
                assert (case["ratio"], case["heads"], case["head_dim"]) == (1, 32, 128)
                check = case["correctness"]
                assert check["rows"] == case["rows"]
                assert check["fixed_logits_topk"] == "all rows exact logical sets"
                assert (
                    check["compact_direct_gather_score_comparison"]
                    == "all candidate scores bitwise equal"
                )
                assert all(
                    check["full_baselines"][name]["checked_rows"] == case["rows"]
                    for name in FULL
                )
                assert all(
                    w["score_bytes"] <= 1 << 30 and w["gather_bytes"] <= 256 << 20
                    for w in case["workspaces"].values()
                )
                for item in case["records"]:
                    for measurement in _rounds(item["measurement"]):
                        assert math.isclose(
                            statistics.median(measurement["wall_samples_us"]),
                            measurement["wall_p50_us"],
                            rel_tol=1e-9,
                        )
                        rounds_checked += 1
                groups.setdefault(key, {})[order] = case
    assert len(groups) == 8
    paired, paired_rounds = {}, 0
    if paired_directory:
        paired, paired_artifacts, paired_rounds = paired_evidence(
            paired_directory, before, groups
        )
        artifacts.extend(paired_artifacts)
    records = []
    for name, cases in groups.items():
        a, b = cases["forward"], cases["reverse"]
        for field in (
            "query_lens",
            "contexts",
            "workspaces",
            "candidate_positions_min",
            "candidate_positions_max",
            "candidate_positions_sum",
        ):
            assert a[field] == b[field]
        for execution in ("eager", "graph"):
            contract = PerQueryContract(
                tuple(a["query_lens"]), tuple(a["contexts"]), execution, environment
            )
            orders = {}
            evidence = paired.get(name)
            same_object = evidence is not None and evidence["paired_pipeline"]
            for order, case in cases.items():
                orders[order] = {
                    item["variant"]: item["measurement"]
                    for item in (
                        evidence["records"] if same_object else case["records"]
                    )
                    if item["phase"] == "pipeline"
                    and item["execution"] == execution
                    and (not same_object or item["order"] == order)
                }
            decision = select_measurements(orders)
            if paired and not same_object:
                # Only full baselines remain in short regions. Do not admit an
                # optimized candidate from independent candidate-generation runs.
                assert decision["strategy"] in FULL
                if decision["strategy"] != FULL[0]:
                    assert a["candidate_sha256"] == b["candidate_sha256"]
            records.append(
                dict(
                    label=name,
                    contract=asdict(contract),
                    orders=orders,
                    **decision,
                    correctness=cases["forward"]["correctness"],
                    timing_evidence=(
                        "paired_same_object" if same_object else "original"
                    ),
                    paired_input_fingerprint=(
                        evidence["input_before"] if evidence else None
                    ),
                    unsorted_candidate_sha256={
                        order: case["candidate_sha256"] for order, case in cases.items()
                    },
                )
            )
    return dict(
        status="final_verified",
        admission_status=("admitted" if paired else "pending_paired_same_input_AB"),
        scope="single-GPU P5c module experiment only",
        correctness_scope=(
            "Native-compatible scores and exact stable logits selection; "
            "initial FP32 deviations analyzed separately, not waived"
        ),
        environment_digest=environment,
        runtime_identity=before,
        execution_sources=before["source_files"],
        planner_source_sha256=sha(ROOT / "vllm_dsv41_indexer/perquery_dispatch.py"),
        artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in artifacts},
        correctness_artifacts=[
            str(p.relative_to(ROOT)) for p in (verify_path, preflight_path)
        ],
        sample_rounds_checked=rounds_checked,
        paired_sample_rounds_checked=paired_rounds,
        historical_candidate_hashes_reconstructed=False,
        records=records,
        performance_contract=(
            "Changing lengths/grouping/budgets/execution requires replanning; "
            "no TP4 speed claim"
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-directory")
    args = parser.parse_args()
    report = collect(args.directory, args.paired_directory)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                records=len(report["records"]),
                rounds_checked=report["sample_rounds_checked"],
                choices=[
                    (r["label"], r["contract"]["execution"], r["strategy"])
                    for r in report["records"]
                ],
            )
        )
    )


if __name__ == "__main__":
    main()

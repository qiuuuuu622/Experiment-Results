# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for exact model schedules and conservative two-order choice."""

import hashlib
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from vllm_dsv41_indexer.perquery_dispatch import (
    PerQueryContract,
    PerQueryPlanner,
    select_measurements,
)


def metric(us):
    return dict(
        p50_us=us,
        wall_p50_us=us + 2,
        samples_us=[us] * 30,
        wall_samples_us=[us + 2] * 30,
        coefficient_of_variation=0.0,
        warmup=20,
        inner=10,
        peak_allocated_bytes=1024,
    )


def orders(dg=100, current=90, direct=60, gathered=80):
    values = dict(
        full_deepgemm_mask=dg,
        full_keys_major_mask=current,
        direct=direct,
        gathered_wgmma=gathered,
    )
    return {
        order: {k: metric(v) for k, v in values.items()}
        for order in ("forward", "reverse")
    }


def contract():
    return PerQueryContract((2048,), (65536,), "graph", "runtime")


def manifest(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = root / "vllm_dsv41_indexer/perquery_dispatch.py"
    data = orders()
    record = dict(
        contract=asdict(contract()),
        orders=data,
        strategy=select_measurements(data)["strategy"],
    )
    artifact = tmp_path / "raw.json"
    artifact.write_text("verified artifact")
    return dict(
        status="final_verified",
        admission_status="admitted",
        environment_digest="runtime",
        planner_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        execution_sources={},
        artifact_sha256={"raw.json": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        correctness_artifacts=["raw.json"],
        records=[record],
    )


def test_direct_beats_both_full_paths_and_each_order():
    assert select_measurements(orders())["strategy"] == "direct"
    # Winning only against the slower baseline is insufficient.
    assert select_measurements(orders(current=55))["strategy"] == "full_keys_major_mask"
    two = orders()
    two["reverse"]["direct"] = metric(95)
    assert select_measurements(two)["strategy"] == "gathered_wgmma"


def test_below_five_percent_keeps_full_baseline():
    assert (
        select_measurements(orders(current=98, direct=97, gathered=98))["strategy"]
        == "full_deepgemm_mask"
    )


def test_slowest_retained_round_controls_winner():
    data = orders(direct=60, gathered=70)
    data["forward"]["direct"]["extra_rounds"] = [metric(80)] * 3
    assert select_measurements(data)["strategy"] == "gathered_wgmma"


@pytest.mark.parametrize(
    "mutation",
    ["missing_order", "missing_baseline", "false_median", "false_cv", "wall_count"],
)
def test_incomplete_or_inconsistent_samples_rejected(mutation):
    data = orders()
    if mutation == "missing_order":
        del data["reverse"]
    elif mutation == "missing_baseline":
        del data["forward"]["full_deepgemm_mask"]
    elif mutation == "false_median":
        data["forward"]["direct"]["p50_us"] = 1
    elif mutation == "false_cv":
        data["forward"]["direct"]["coefficient_of_variation"] = 0.01
    else:
        data["forward"]["direct"]["wall_samples_us"].pop()
    with pytest.raises(ValueError):
        select_measurements(data)


@pytest.mark.parametrize(
    "changes",
    [
        {"ratio": 2},
        {"index_heads": 8},
        {"head_dim": 64},
        {"source_unique": False},
        {"candidate_layout": "request_shared"},
        {"score_budget": 512 << 20},
        {"gather_budget": 128 << 20},
        {"world_size": 4},
        {"execution": "unknown"},
    ],
)
def test_unmeasured_abi_budget_and_parallel_contracts_rejected(changes):
    with pytest.raises(ValueError):
        replace(contract(), **changes)


def test_full_query_schedule_and_mode_are_not_inferred_from_total_qk(tmp_path):
    c = PerQueryContract((512, 128), (65535, 16384), "graph", "runtime")
    d = replace(c, contexts=(65535, 32768))
    assert c.rows == d.rows and max(c.contexts) == max(d.contexts)
    assert c.identity != d.identity and c.visible_lengths != d.visible_lengths
    assert c.visible_lengths[0] == 65024
    assert c.visible_lengths[512] == 16257
    table = PerQueryPlanner(manifest(tmp_path), root=tmp_path)
    assert table.choose(contract())["strategy"] == "direct"
    assert table.choose(replace(contract(), execution="eager"))["evidence"] is False
    assert table.choose(replace(contract(), query_lens=(2047,)))["evidence"] is False
    with pytest.raises(ValueError, match="Runtime differs"):
        table.choose(replace(contract(), environment_digest="different"))


def test_artifact_tampering_and_unverified_decisions_fail_closed(tmp_path):
    original = manifest(tmp_path)
    changed = deepcopy(original)
    changed["records"][0]["strategy"] = "gathered_wgmma"
    with pytest.raises(ValueError, match="Stored decision"):
        PerQueryPlanner(changed, root=tmp_path)
    (tmp_path / "raw.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        PerQueryPlanner(original, root=tmp_path)


@pytest.mark.parametrize("status", [None, "pending_paired_same_input_AB"])
def test_pending_same_input_proof_requires_explicit_provisional_mode(tmp_path, status):
    data = manifest(tmp_path)
    if status is None:
        del data["admission_status"]
    else:
        data["admission_status"] = status
    with pytest.raises(ValueError, match="Paired same-input"):
        PerQueryPlanner(data, root=tmp_path)
    provisional = PerQueryPlanner(data, root=tmp_path, allow_provisional=True)
    assert provisional.choose(contract())["admission_status"] == status

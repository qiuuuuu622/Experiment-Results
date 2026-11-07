# SPDX-License-Identifier: Apache-2.0
"""CPU checks guard dispatch safety, evidence freshness and expensive replanning.

Input is explicit shape/layout plus final-source sample evidence; output is a
preconstructable strategy name. Unit tests catch accidental generalization or
semantic fallback without needing CUDA kernels or model loading.
"""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from vllm_dsv41_indexer.dispatch import Dispatcher, DispatchKey, admit


def metric(us):
    return dict(
        p50_us=us,
        p90_us=us,
        samples_us=[us] * 30,
        wall_samples_us=[us + 2] * 30,
        coefficient_of_variation=0.0,
        warmup=20,
        inner=10,
        peak_allocated_bytes=1024,
    )


def evidence(key=None, candidate="grouped_gluon_full", gain=0.25):
    key = key or DispatchKey("verify", "full", 192, 32, 65536, group_rows=6)
    return dict(
        key=asdict(key),
        candidate=candidate,
        baseline="native_full",
        stage="final",
        status="passed",
        source_digest="source-final",
        environment_digest="env-final",
        includes_total_cost=True,
        correctness_artifacts=["verify.json:sha256"],
        orders={
            order: {"baseline": metric(100), "candidate": metric(100 * (1 - gain))}
            for order in ("forward", "reverse")
        },
    )


def planner(*records):
    return Dispatcher(
        source_digest="source-final", environment_digest="env-final", records=records
    )


def test_only_exact_final_shape_is_admitted_and_callable_bound_once():
    record = evidence()
    key = DispatchKey(**record["key"])
    dispatch = planner(record)
    chosen = dispatch.choose(key)
    assert chosen.strategy == "grouped_gluon_full"
    assert chosen.bind({chosen.strategy: lambda: "executed"})() == "executed"
    for field, value in (
        ("rows", 193),
        ("ratio", 2),
        ("execution", "eager"),
        ("hardware", "other"),
    ):
        assert (
            dispatch.choose(replace(key, **{field: value})).strategy == "optimized_full"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "environment",
        "historical",
        "one_order",
        "noise",
        "incomplete",
        "fabricated_p50",
    ],
)
def test_invalid_evidence_cannot_enable_optimization(mutation):
    record = evidence()
    if mutation == "source":
        record["source_digest"] = "old-source"
    elif mutation == "environment":
        record["environment_digest"] = "other-machine"
    elif mutation == "historical":
        record["stage"] = "screening"
    elif mutation == "one_order":
        del record["orders"]["reverse"]
    elif mutation == "noise":
        record["orders"]["forward"]["candidate"]["coefficient_of_variation"] = 0.2
    elif mutation == "incomplete":
        record["includes_total_cost"] = False
    else:
        record["orders"]["forward"]["candidate"]["p50_us"] = 1
    dispatch = planner(record)
    assert dispatch.rejections
    assert dispatch.choose(DispatchKey(**record["key"])).strategy == "optimized_full"


def test_less_than_five_percent_keeps_baseline_and_noise_rounds_are_retained():
    record = evidence(gain=0.03)
    assert planner(record).rejections
    record = evidence(gain=0.5)
    item = record["orders"]["forward"]["candidate"]
    item["samples_us"] = [42.5, 57.5] * 15
    item["coefficient_of_variation"] = 0.15
    item["extra_rounds"] = [metric(52), metric(49), metric(51)]
    assert (
        admit(record, source_digest="source-final", environment_digest="env-final")
        > 1.9
    )
    item["extra_rounds"][0] = metric(99)
    assert planner(record).rejections


def test_deterministic_long_row_fallback_never_uses_faster_native_selector():
    key = DispatchKey("selector", "selector", 128, 128, 262144)
    assert planner().choose(key).strategy == "streaming"
    assert planner().choose(key, deterministic=False).strategy == "native"


def test_shared_consumer_requires_explicit_layout_guarantee_without_host_probe():
    key = DispatchKey("verify", "consumer", 24, 4, 65536, group_rows=6)
    record = evidence(key, "shared_consumer_publish_included")
    assert planner(record).rejections
    shared = replace(key, candidate_layout="shared_prefix")
    record["key"] = asdict(shared)
    assert planner(record).choose(shared).strategy == "shared_consumer_publish_included"
    assert planner(record).choose(key).strategy == "one_sort_ant_topk_fused_map"


def test_mutating_lengths_cannot_use_preplanned_cluster():
    key = DispatchKey(
        "selector", "selector", 32, 32, 262144, lengths_mode="updated_each_call"
    )
    record = evidence(key, "cluster_preplanned")
    assert planner(record).rejections
    record["candidate"] = "cluster_replan_each_call"
    assert not planner(record).rejections


def test_known_native_zero_length_bug_is_not_a_silent_fallback():
    key = DispatchKey("decode", "full", 8, 8, 65536)
    assert planner().choose(key, deterministic=False).strategy == "optimized_full"
    for safe in (
        replace(key, may_have_zero_lengths=False),
        replace(key, native_zero_prefetch_patched=True),
    ):
        assert planner().choose(safe).strategy == "native_score_stable_topk"
        assert planner().choose(safe, deterministic=False).strategy == "native_full"


def test_single_gpu_evidence_never_claims_actual_tp4_or_untested_gluon():
    key = DispatchKey("prefill", "full", 64, 1, 65536)
    assert planner(evidence(key, "gluon_pipeline")).rejections
    parallel = replace(key, parallel="replicated", world_size=4)
    with pytest.raises(ValueError, match="group sizes"):
        planner().choose(parallel)
    record = deepcopy(evidence())
    record["key"]["world_size"] = 2
    assert planner(record).rejections


def test_worst_retained_round_selects_stable_candidate_not_fast_initial_outlier():
    a = evidence(candidate="grouped_source_fused_max")
    key = DispatchKey("decode", "source", 8, 8, 65536)
    a["key"] = asdict(key)
    a["orders"]["forward"]["candidate"]["extra_rounds"] = [metric(90)]
    b = evidence(key, candidate="grouped_source_separate_max", gain=0.20)
    assert planner(a, b).choose(key).strategy == "grouped_source_separate_max"


def test_fabricated_cv_or_truncated_wall_samples_cannot_hide_noise():
    record = evidence()
    record["orders"]["forward"]["candidate"]["samples_us"] = [50, 100] * 15
    assert planner(record).rejections
    record = evidence()
    record["orders"]["reverse"]["baseline"]["wall_samples_us"].pop()
    assert planner(record).rejections


def test_tp4_requires_rank_max_and_keeps_projection_input_and_group_shape_separate():
    key = DispatchKey(
        "verify",
        "consumer",
        48,
        8,
        65536,
        group_rows=6,
        group_sizes=(6,) * 8,
        world_size=4,
        parallel="replicated",
        local_backend="compact_wgmma",
    )
    record = evidence(key, "tp4:shard_request_groups_replicated_input")
    record["communication_included"] = True
    assert planner(record).rejections
    for pair in record["orders"].values():
        for metric_ in pair.values():
            metric_["aggregation"] = "max_rank_each_invocation_then_mean_within_sample"
    assert planner(record).choose(key).strategy == record["candidate"]
    assert (
        planner(record).choose(replace(key, projection=True)).strategy
        == "tp4:replicated"
    )
    root_input = replace(key, input_mode="root")
    assert (
        planner(record).choose(root_input).strategy
        == "tp4:shard_request_groups_root_input"
    )

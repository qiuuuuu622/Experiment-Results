"""Recompute final exact-contract review from original and paired raw evidence."""
import hashlib
import json
from pathlib import Path

from benchmarks.kernels.dsv41_indexer.collect_perquery_dispatch import collect
from vllm_dsv41_indexer.perquery_dispatch import PerQueryPlanner

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest_path = HERE / "admission.json"
manifest = json.loads(manifest_path.read_text())
rebuilt = collect(BASE / "p5c-perquery/final-v1", BASE / "p5c-perquery/paired-v1")
assert manifest == json.loads(json.dumps(rebuilt))
assert manifest["admission_status"] == "admitted"
PerQueryPlanner(manifest, root=ROOT)
bind_path = HERE / "bind-v2/verify.json"
bind = json.loads(bind_path.read_text())
assert bind["status"] == "passed" and bind["admission_status"] == "admitted"
assert bind["manifest_sha256"] == sha(manifest_path)
assert bind["source_before"] == bind["source_after"]
assert bind["measured_sources_and_artifacts_rechecked_after_run"]
assert len(bind["cases"]) == 8
known_native = {x["path"]: x["sha256"] for x in manifest["runtime_identity"]["native_files"]}
for case in bind["cases"]:
    assert case["decision"]["admission_status"] == "admitted"
    assert case["decision"]["evidence"]
    assert case["changed_lengths_rejected"] and case["unknown_schedule_falls_back"]
    assert case["graph_capture_has_no_host_sync"]
    assert len(case["phases"]) == 3
    assert all(p["graph_vs_eager"] == "exact logical set" for p in case["phases"])
    assert len(set(case["repeated_source_canonical_sha256"])) == 1
    for item in case["decision"]["actual_native_identity"]["actually_loaded"]:
        assert known_native[item["path"]] == item["sha256"]

numerical = []
for group in "ab":
    raw = json.loads((BASE / f"p5c-perquery/final-v1/group-{group}-forward.json").read_text())
    numerical += [dict(label=c["case"], **c["correctness"]) for c in raw["cases"]]
assert len(numerical) == 8
parallel = BASE / "p5-perquery-tp4/final-v1"
ranks = [json.loads((parallel / f"rank{r}.json").read_text()) for r in range(4)]
for rank, result in enumerate(ranks):
    assert result["status"] == "completed" and result["world_size"] == 4
    assert result["rank"] == rank and len(result["cases"]) == 3
    assert result["provenance_before"] == result["provenance_after"]
    for index, case in enumerate(result["cases"]):
        assert len(case["phases"]) == 7
        identities = case["identities"]
        assert len({i["physical_table_sha256"] for i in identities}) == 4
        assert len({i["logical_cache_sha256"] for i in identities}) == 1
        assert all(i["logical_fields"] == identities[0]["logical_fields"] for i in identities)
        for step, phase in enumerate(case["phases"]):
            assert phase["graph_and_eager_variants"] == 12
            assert phase["logical_output_sha256"] == ranks[0]["cases"][index]["phases"][step]["logical_output_sha256"]

report = dict(
    status="review_complete_native_compatible_module_scope",
    manifest_sha256=sha(manifest_path), bind_sha256=sha(bind_path),
    exact_contract_records=len(manifest["records"]),
    choices=[dict(label=r["label"], execution=r["contract"]["execution"], strategy=r["strategy"], evidence=r["timing_evidence"]) for r in manifest["records"]],
    original_raw_rounds=manifest["sample_rounds_checked"],
    paired_raw_rounds=manifest["paired_sample_rounds_checked"],
    artifact_files_checked=len(manifest["artifact_sha256"]),
    measured_source_files_checked=len(manifest["execution_sources"]),
    native_identity_files=len(known_native),
    real_default_bind_cases=8, poison_first_graph_phases=24,
    numerical_forward_once=dict(
        rows=sum(x["rows"] for x in numerical),
        max_abs_vs_native=max(x["max_abs_vs_native"] for x in numerical),
        max_nrmse_vs_native=max(x["nrmse_vs_native"] for x in numerical),
        minimum_topk_overlap=min(x["min_topk_overlap"] for x in numerical),
        fp32_sample_score_failures=sum(x["fp32_strict_failures"] for x in numerical),
        native_fp32_sample_score_failures=sum(x["native_fp32_strict_failures"] for x in numerical),
        fp32_max_abs=max(x["fp32_max_abs"] for x in numerical),
        fp32_initial_gate="not fully passed; diagnostic locations, not failing rows",
    ),
    tp4_correctness=dict(ranks=4, shapes=3, phases_per_shape=7, variants_per_phase=12, speed_evidence=False),
    immutable_old_p8="old 400 admissions unchanged",
    limitations=["no service or model accuracy", "no automatic TP4 performance policy", "replan if length/grouping/mode/budgets change", "original candidate canonical hashes not retrospectively reconstructed"],
)
(HERE / "FINAL_RAW_REVIEW.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k:v for k,v in report.items() if k not in {"choices"}}, indent=2))

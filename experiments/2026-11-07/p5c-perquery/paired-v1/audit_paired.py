"""Verify paired inputs/identities/samples without reconstructing old raw hashes."""

import hashlib
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent


def read(name):
    return json.loads((BASE / name).read_text())


def sha(name):
    return hashlib.sha256((BASE / name).read_bytes()).hexdigest()


results = {}
record_count = round_count = extras = 0
for group in ("a", "b"):
    data = read(f"group-{group}.json")
    assert (
        data["status"] == "completed"
        and not data["old_raw_canonical_hash_reconstructed"]
    )
    assert data["runner_sha256"] == sha("paired_pipeline.py")
    before = read(f"runtime-before-{group}.json")
    after = read(f"runtime-after-{group}.json")
    for key in ("source_files", "native_files", "versions", "source_archive_sha256"):
        assert before[key] == after[key], key
    assert data["identity_sha256"] == sha(f"runtime-before-{group}.json")
    assert before["source_archive_sha256"] == sha("frozen/source.tar.gz")
    for case in data["cases"]:
        assert (
            case["input_before"] == case["input_after"]
            and case["input_bytes_unchanged"]
        )
        expected = case["input_before"]["candidate_canonical"]["sha256"]
        assert len(case["source_repeats"]) == 3 and case["source_canonical_sets_exact"]
        for repeat in case["source_repeats"]:
            assert repeat["canonical"]["sha256"] == expected
        metrics = {}
        assert len(case["records"]) == (16 if case["paired_pipeline"] else 0)
        for r in case["records"]:
            record_count += 1
            m = r["measurement"]
            rounds = [m] + m.get("extra_rounds", [])
            assert m["coefficient_of_variation"] <= 0.1 or len(rounds) == 4
            for x in rounds:
                assert (
                    x["warmup"],
                    x["repeats"],
                    x["inner"],
                    len(x["samples_us"]),
                    len(x["wall_samples_us"]),
                ) == (20, 30, 10, 30, 30)
            round_count += len(rounds)
            extras += len(rounds) - 1
            samples = [x for one in rounds for x in one["samples_us"]]
            key = f"{r['order']}:{r['execution']}:{r['variant']}"
            assert key not in metrics
            metrics[key] = dict(
                p50_us=statistics.median(samples),
                p90_us=sorted(samples)[int(0.9 * (len(samples) - 1))],
                wall_p50_us=statistics.median(
                    [x for one in rounds for x in one["wall_samples_us"]]
                ),
                min_round_p50_us=min(x["p50_us"] for x in rounds),
                max_round_p50_us=max(x["p50_us"] for x in rounds),
                max_round_cv=max(x["coefficient_of_variation"] for x in rounds),
                rounds=len(rounds),
                peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in rounds),
            )
        speedups = {}
        if case["paired_pipeline"]:
            for mode in ("eager", "graph"):
                base = min(
                    metrics[f"{order}:{mode}:{variant}"]["min_round_p50_us"]
                    for order in ("forward", "reverse")
                    for variant in ("full_deepgemm_mask", "full_keys_major_mask")
                )
                speedups[mode] = {
                    variant: base
                    / max(
                        metrics[f"{order}:{mode}:{variant}"]["max_round_p50_us"]
                        for order in ("forward", "reverse")
                    )
                    for variant in ("direct", "gathered_wgmma")
                }
        results[case["case"]] = dict(
            raw_file=f"group-{group}.json",
            physical_gpu=data["physical_gpu"],
            paired_pipeline=case["paired_pipeline"],
            input_bytes_unchanged=True,
            source_canonical_sets_exact=True,
            source_raw_order_changed=case["source_raw_order_changed"],
            candidate_raw_sha256=case["input_before"]["candidate_raw"]["sha256"],
            candidate_canonical_sha256=expected,
            metrics=metrics,
            conservative_speedup=speedups,
        )
assert len(results) == 8 and record_count == 64
summary = dict(
    status="passed",
    source_canonical_cases=8,
    paired_pipeline_cases=4,
    records=record_count,
    rounds=round_count,
    extra_rounds=extras,
    samples=round_count * 30,
    old_raw_hashes_reconstructed=False,
    cases=results,
)
(BASE / "AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "cases"}))
for name, row in results.items():
    print(
        name,
        "raw order changed:",
        row["source_raw_order_changed"],
        "speedup",
        row["conservative_speedup"],
    )

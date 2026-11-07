"""Audit final frozen P5c samples and derive order-preserving summaries."""

import hashlib
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent / "final-v1"


def read(name):
    return json.loads((BASE / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rounds):
    samples = [v for r in rounds for v in r["samples_us"]]
    walls = [v for r in rounds for v in r["wall_samples_us"]]
    ordered = sorted(samples)
    return dict(
        p50_us=statistics.median(samples),
        p90_us=ordered[int(0.9 * (len(ordered) - 1))],
        wall_p50_us=statistics.median(walls),
        max_round_cv=max(r["coefficient_of_variation"] for r in rounds),
        min_round_p50_us=min(r["p50_us"] for r in rounds),
        max_round_p50_us=max(r["p50_us"] for r in rounds),
        peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rounds),
        rounds=len(rounds),
        samples=len(samples),
    )


identity = read("runtime-before.json")
assert sha(BASE / "frozen/source.tar.gz") == identity["source_archive_sha256"]
for group in ("a", "b"):
    after = read(f"runtime-after-{group}.json")
    for key in ("source_files", "native_files", "source_archive_sha256", "versions"):
        assert identity[key] == after[key], key
cases = {}
records = round_count = extra_count = 0
for group in ("a", "b"):
    for order in ("forward", "reverse"):
        name = f"group-{group}-{order}.json"
        data = read(name)
        assert data["status"] == "completed" and data["purpose"] == "performance"
        assert data["args"]["reverse"] == (order == "reverse")
        assert data["physical_gpu"] == ("1" if group == "a" else "2")
        assert data["identity_sha256"] == sha(BASE / "runtime-before.json")
        assert len(data["cases"]) == 4
        for case in data["cases"]:
            entry = cases.setdefault(
                case["case"],
                dict(
                    orders={},
                    raw_files=[],
                    metadata={
                        k: v
                        for k, v in case.items()
                        if k not in ("records", "correctness")
                    },
                    correctness=[],
                ),
            )
            assert order not in entry["orders"]
            entry["raw_files"].append(name)
            entry["correctness"].append(case["correctness"])
            entry["orders"][order] = {}
            for record in case["records"]:
                records += 1
                m = record["measurement"]
                rounds = [m] + m.get("extra_rounds", [])
                assert m["coefficient_of_variation"] <= 0.1 or len(rounds) == 4
                for r in rounds:
                    assert (r["warmup"], r["repeats"], r["inner"]) == (20, 30, 10)
                    assert len(r["samples_us"]) == len(r["wall_samples_us"]) == 30
                round_count += len(rounds)
                extra_count += len(rounds) - 1
                key = ":".join(
                    str(record.get(k, ""))
                    for k in (
                        "variant",
                        "phase",
                        "execution",
                        "query_start",
                        "query_end",
                    )
                )
                entry["orders"][order][key] = metrics(rounds)
assert len(cases) == 8 and records == 3360
for label, entry in cases.items():
    entry["pipeline"] = {}
    for mode in ("eager", "graph"):
        variants = {}
        for variant in (
            "full_deepgemm_mask",
            "full_keys_major_mask",
            "direct",
            "gathered_wgmma",
        ):
            key = f"{variant}:pipeline:{mode}::"
            variants[variant] = {
                order: entry["orders"][order][key] for order in ("forward", "reverse")
            }
        native_fastest_round = min(
            x["min_round_p50_us"]
            for v in ("full_deepgemm_mask", "full_keys_major_mask")
            for x in variants[v].values()
        )
        entry["pipeline"][mode] = dict(
            variants=variants,
            conservative_speedup={
                v: native_fastest_round
                / max(x["max_round_p50_us"] for x in variants[v].values())
                for v in ("direct", "gathered_wgmma")
            },
            automatic_dispatch_changed=False,
        )
result = dict(
    status="completed",
    scope="8 architecture-aligned consumer shapes; single-consumer module only",
    source_archive_sha256=identity["source_archive_sha256"],
    measurement_records=records,
    total_rounds=round_count,
    extra_noise_rounds=extra_count,
    samples=round_count * 30,
    identity_files=len(identity["native_files"]),
    aggregation=(
        "Per-order rounds pooled for P50/P90; conservative speedup is "
        "fastest full-baseline round divided by slowest candidate round. "
        "Chunk score sums are not standalone invocation latency."
    ),
    cases=cases,
)
(BASE / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
for name, c in cases.items():
    print(name, "graph conservative", c["pipeline"]["graph"]["conservative_speedup"])

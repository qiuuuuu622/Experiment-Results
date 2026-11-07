# SPDX-License-Identifier: Apache-2.0
"""CPU final-run planning, source validation and raw-sample dispatch admission.

GPU work uses the existing phase drivers printed by ``plan``. ``collect`` only
accepts explicitly supplied frozen final bundles, never scans historical JSON
and silently treats it as a new run. No models or serving hooks are loaded.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from vllm_dsv41_indexer.dispatch import Dispatcher, DispatchKey, source_identity

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/indexer/20261107-resume/p8"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime_sources(root, *, parallel=False):
    """Kernel/module execution dependencies, distinct from evolving report tools.

    The dispatch planner and the independent P7 module are not imported by P2-P6
    GPU drivers. Their final files are frozen separately, never substituted for
    the phase driver's own original source archive.
    """
    complete = source_identity(root)
    modules = {
        "__init__.py",
        "score.py",
        "prepare.py",
        "grouped.py",
        "contract.py",
        "topk.py",
        "parallel.py",
        "workspace.py",
        "module.py",
        "reference.py",
        "prefill.py",
        "grouped_score.py",
        "native.py",
        "full_score.py",
    }
    if parallel:
        modules.add("row_shard.py")
    files = {
        path: sha
        for path, sha in complete["files"].items()
        if path.startswith("vllm_dsv41_indexer/")
        and (
            Path(path).name in modules
            or path.startswith(
                ("vllm_dsv41_indexer/csrc/", "vllm_dsv41_indexer/third_party/")
            )
        )
    }
    return {"digest": digest(files), "files": files}


def estimate_memory(rows, requests, keys, *, variants=16):
    cache = requests * ((keys + 63) // 64) * 8448
    score = rows * ((keys + 63) // 64) * 64 * 4
    return dict(
        cache_bytes_per_rank=cache,
        full_score_bytes=score,
        score_arena_limit_bytes=1 << 30,
        estimated_simultaneous_bytes=cache
        + variants * min(score, 1 << 30)
        + rows * 4224,
        requires_score_chunking=score > 1 << 30,
        estimate_not_measured_peak=True,
    )


def matrix():
    cases = []

    def add(family, rows, requests, keys, ratio=1, group=1, role="all", **extra):
        memory = estimate_memory(rows, requests, keys)
        status = "scheduled"
        reason = None
        if memory["estimated_simultaneous_bytes"] > 80 << 30:
            status, reason = (
                "excluded_memory_budget",
                "Estimated resident buffers exceed 80 GiB per rank",
            )
        if family in ("full", "grouped") and memory["requires_score_chunking"]:
            status, reason = (
                "excluded_driver_limit",
                (
                    "Use explicit ChunkedIndexer test; monolithic phase driver "
                    "exceeds 1 GiB score budget"
                ),
            )
        cases.append(
            dict(
                family=family,
                rows=rows,
                requests=requests,
                index_keys=keys,
                ratio=ratio,
                logical_context=keys * ratio,
                group_rows=group,
                role=role,
                execution=["eager", "graph"],
                order=["forward", "reverse"],
                status=status,
                exclusion_reason=reason,
                memory=memory,
                **extra,
            )
        )

    for rows in (1, 8, 32, 64, 128):
        for keys in (4096, 16384, 65536, 131072, 262144):
            add("compact", rows, rows, keys, role="consumer_and_four_consumers")
    for ratio in (1, 2):
        for requests, context in (
            (1, 4096),
            (8, 16384),
            (32, 65536),
            (64, 131072),
            (128, 262144),
        ):
            add(
                "full",
                requests,
                requests,
                context // ratio,
                ratio,
                role="full_source_chain",
            )
    for group in (1, 2, 4, 6, 8):
        add("grouped", 4 * group, 4, 65536, group=group)
    add("grouped", 192, 32, 65536, group=6)
    add("grouped", 6, 1, 262144, group=6)
    prefill = [(64, 1, 65536, 1, m) for m in ("full", "source", "consumer")]
    prefill += [
        (1, 1, 4096, 1, "full"),
        (16, 1, 262144, 1, "full"),
        (64, 1, 16384, 1, "full"),
        (64, 1, 32768, 2, "full"),
        (64, 1, 131072, 2, "full"),
        (256, 1, 16384, 1, "full"),
        (1024, 1, 16384, 1, "full"),
        (4096, 1, 4096, 1, "full"),
        (64, 2, 65536, 1, "source"),
        (256, 1, 16384, 2, "source"),
        (16, 1, 262144, 1, "consumer"),
        (256, 1, 65536, 1, "consumer"),
        (1024, 1, 16384, 2, "consumer"),
    ]
    for rows, requests, keys, ratio, role in prefill:
        add("prefill", rows, requests, keys, ratio, role=role)
    topk = [(1, k, 512) for k in (4096, 16384, 65536, 131072, 262144)]
    topk += [(r, k, 512) for r in (8, 32, 64, 128) for k in (16384, 65536, 262144)]
    topk += [(r, k, 2048) for r in (1, 8, 32, 128) for k in (4096, 16384, 32768)]
    for rows, keys, k in topk:
        add("topk", rows, rows, keys, role="selector", topk=k)
    # This shape belongs in the explicit budget validation, not an OOM-prone
    # monolithic benchmark. Keep it visible rather than silently omitting it.
    add("full", 1025, 1025, 262144, role="source_budget_boundary")
    return cases


def commands():
    prefix = ".venv/bin/python -m benchmarks.kernels.dsv41_indexer."
    result = [
        prefix
        + (
            "verify_compact --output p2/verify.json  "
            "# initialize patched DeepGEMM for mixed-zero oracle"
        ),
        prefix + "verify_topk --output p6/verify.json",
        prefix + "verify_grouped --paired --output p4/verify.json",
        prefix + "verify_prefill --output p5/verify.json",
        prefix + "benchmark_grouped --paired --output p4/final.json",
        prefix + "benchmark_grouped_planning --output p4/planning.json",
    ]
    for reverse in (False, True):
        order = "reverse" if reverse else "forward"
        flag = " --reverse-order" if reverse else ""
        result.append(
            prefix
            + "benchmark --rows 1 8 32 64 128 --context 4096 16384 65536 131072 262144"
            + flag
            + f" --output p2/{order}.json"
        )
        result.append(prefix + "benchmark_topk" + flag + f" --output p6/{order}.json")
        for ratio in (1, 2):
            result.append(
                prefix
                + (
                    "benchmark_full --phase full --cases 1:4096:1 8:16384:1 "
                    "32:65536:1 64:131072:1 128:262144:1"
                )
                + flag
                + f" --ratio {ratio} --output p3/r{ratio}-{order}.json"
            )
        for suite in ("smoke", "matrix"):
            flag_prefill = " --reverse" if reverse else ""
            result.append(
                prefix
                + f"benchmark_prefill --suite {suite}"
                + flag_prefill
                + f" --output p5/{suite}-{order}.json"
            )
    return result


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def frozen_files(manifest):
    files = manifest.get(
        "files", manifest.get("source_files", manifest.get("source_sha256", {}))
    )
    if isinstance(files, list):
        return {item["path"]: item["sha256"] for item in files}
    return files


def validate_bundle(bundle, root):
    """Verify final module bytes and explicit correctness artifacts before admission."""
    if bundle.get("stage") != "final":
        raise ValueError("Bundle must be explicitly designated a new final run")
    frozen = json.loads(Path(bundle["source_manifest"]).read_text())
    recorded = frozen_files(frozen)
    expected = runtime_sources(root, parallel=bundle["family"] == "tp4")
    mismatch = [
        path for path, sha in expected["files"].items() if recorded.get(path) != sha
    ]
    if mismatch:
        raise ValueError(f"Frozen runtime dependency differs/missing: {mismatch}")
    correctness = []
    for path in bundle["correctness"]:
        report = json.loads(Path(path).read_text())
        if report.get("status") not in ("passed", "completed"):
            raise ValueError(f"Correctness artifact not passed: {path}")
        correctness.append(artifact(path))
    if not correctness:
        raise ValueError("Final correctness artifact required")
    native = bundle.get("native_identity", {})
    if not native.get("path") or not native.get("sha256"):
        raise ValueError("Native package/header/binary identity required")
    if artifact(native["path"]) != native:
        raise ValueError("Native identity artifact content changed")
    # Timed closures and fixture preparation live in benchmark code as well.
    # Keep their original hashes, and require a recorded review of any delta.
    driver = {
        "compact": "benchmark.py",
        "topk": "benchmark_topk.py",
        "full": "benchmark_full.py",
        "grouped": "benchmark_grouped.py",
        "prefill": "benchmark_prefill.py",
        "tp4": "benchmark_tp4.py",
    }[bundle["family"]]
    names = [driver, "common.py", "inputs.py", "baseline.json"]
    names += {
        "compact": ["legacy.py"],
        "topk": ["verify_topk.py"],
        "prefill": ["verify_prefill.py"],
        "tp4": ["tp4_cases.py"],
    }.get(bundle["family"], [])
    execution_sources = {}
    for name in names:
        path = "benchmarks/kernels/dsv41_indexer/" + name
        current = hashlib.sha256((Path(root) / path).read_bytes()).hexdigest()
        if path not in recorded:
            raise ValueError(f"Missing frozen execution driver: {path}")
        execution_sources[path] = dict(frozen=recorded[path], current=current)
        if current != recorded[path]:
            review = bundle.get("reviewed_driver_differences", {}).get(path, {})
            if (
                review.get("frozen") != recorded[path]
                or review.get("current") != current
                or not review.get("reason")
            ):
                raise ValueError(f"Unreviewed execution-driver difference: {path}")
            execution_sources[path]["review"] = review["reason"]
    return expected, correctness, execution_sources


def normalize(family, report, execution):
    """Yield exact shape + complete-pipeline comparisons, never component timings."""
    hardware = report.get("device", {})
    device_name = hardware.get("name", "NVIDIA H20")
    hw = f"{device_name}:sm90:{hardware.get('sm_count', 78)}"
    for entry in report["cases"]:
        case = entry.get("case", entry)
        if family == "topk":
            key = DispatchKey(
                "selector",
                "selector",
                case["rows"],
                case["rows"],
                case["width"],
                topk=case["k"],
                execution=execution,
                hardware=hw,
            )
            timings = entry["timings"][execution]
            native = next(name for name in timings if name.startswith("vllm_"))
            for candidate in (
                "streaming",
                "cluster_preplanned",
                "cluster_replan_each_call",
            ):
                yield key, native, candidate, timings[native], timings[candidate]
        elif family == "compact":
            group = case["rows_per_request"]
            key = DispatchKey(
                "decode" if group == 1 else "verify",
                "consumer",
                case["rows"],
                case["rows"] // group,
                case["compressed_length"],
                group_rows=group,
                execution=execution,
                hardware=hw,
                may_have_zero_lengths=False,
            )
            timings = entry["timings"][execution]
            for candidate in (
                "one_sort_ant_topk_fused_map",
                "one_sort_ant_topk_separate_map",
            ):
                yield (
                    key,
                    "vllm_original",
                    candidate,
                    timings["vllm_original"],
                    timings[candidate],
                )
        elif family == "prefill":
            key = DispatchKey(
                "prefill",
                case["mode"],
                case["rows"],
                case["requests"],
                case["index_keys"],
                ratio=case["ratio"],
                execution=execution,
                hardware=hw,
                candidate_layout="request_shared",
            )
            timings = entry["timings"][execution]
            for candidate in (
                "deepgemm_pipeline",
                "row_tiled_pipeline",
                "keys_major_pipeline",
                "indirect_pipeline",
            ):
                yield (
                    key,
                    "native_pipeline",
                    candidate,
                    timings["native_pipeline"],
                    timings[candidate],
                )
        elif family == "full":
            group = case["rows_per_request"]
            timings = entry["timings"][execution]
            for role, baseline, candidates in (
                ("full", "native_full", ("optimized_full",)),
                (
                    "source",
                    "native_source",
                    (
                        "native_score_ant_token_and_source_topk",
                        "grouped_source_fused_max",
                        "grouped_source_separate_max",
                    ),
                ),
                (
                    "chain",
                    "native_source_four_consumers",
                    ("hybrid_source_four_consumers", "optimized_source_four_consumers"),
                ),
            ):
                key = DispatchKey(
                    "decode" if group == 1 else "verify",
                    role,
                    case["rows"],
                    case["requests"],
                    case["compressed_length"],
                    ratio=case["ratio"],
                    group_rows=group,
                    execution=execution,
                    hardware=hw,
                    may_have_zero_lengths=False,
                )
                if role == "chain":
                    candidates = (*candidates, "hybrid_source_full_consumers")
                for candidate in candidates:
                    if baseline not in timings or candidate not in timings:
                        continue
                    yield (
                        key,
                        baseline,
                        candidate,
                        timings[baseline],
                        timings[candidate],
                    )
        elif family == "grouped":
            # The driver stores both orders in each case. A synthetic report
            # is created by collect() selecting one order before normalization.
            group = case["rows_per_request"]
            timings = entry["timings"][execution]
            for role, baseline, candidate in (
                ("full", "native_full", "grouped_gluon_full"),
                ("source", "native_source", "grouped_gluon_source"),
                (
                    "consumer",
                    "rowwise_consumer_publish_included",
                    "shared_consumer_publish_included",
                ),
            ):
                key = DispatchKey(
                    "verify",
                    role,
                    case["rows"],
                    case["requests"],
                    case["compressed_length"],
                    group_rows=group,
                    execution=execution,
                    hardware=hw,
                    candidate_layout="shared_prefix",
                    may_have_zero_lengths=False,
                )
                yield key, baseline, candidate, timings[baseline], timings[candidate]
        else:
            raise ValueError(f"Unsupported final family {family}")


def normalize_tp4(bundle, source, correctness):
    report = json.loads(Path(bundle["forward"]).read_text())
    if (
        report.get("status") not in ("completed", "complete")
        or report.get("physical_world") != 4
        or report.get("single_rank_diagnostic")
    ):
        raise ValueError("TP4 admission requires a completed four-physical-rank run")
    environment = dict(
        device=report["environment"],
        native_identity=bundle["native_identity"],
        native_identity_contents=json.loads(
            Path(bundle["native_identity"]["path"]).read_text()
        ),
        performance_native_package=bundle.get(
            "performance_native_package", "installed"
        ),
    )
    env_digest = digest(environment)
    entries = report["records"]
    baselines = {
        (digest(r["case"]), r["execution"], r["order"]): r
        for r in entries
        if r["variant"] == "replicated"
    }
    pending = {}
    hardware = report["environment"]
    for entry in entries:
        name = entry["variant"]
        if name == "replicated":
            continue
        case = entry["case"]
        if case["ragged"] or case["reorder"]:
            raise ValueError(
                "Ragged/reordered TP4 performance requires explicit group IDs"
            )
        role = {"compact": "consumer", "full": "full", "chain": "chain"}[
            case["workload"]
        ]
        key = DispatchKey(
            "decode" if case["verify"] == 1 else "verify",
            role,
            case["rows"],
            case["requests"],
            case["index_keys"],
            ratio=case["ratio"],
            group_rows=case["verify"],
            group_sizes=(case["verify"],) * case["requests"],
            execution=entry["execution"],
            parallel="replicated",
            world_size=4,
            input_mode=entry["communication"]["input_mode"],
            projection=case["projection"],
            local_backend=case["local_score_backend"],
            may_have_zero_lengths=False,
            native_zero_prefetch_patched=bool(report.get("patched_deep_gemm")),
            hardware=f"{hardware['name']}:sm90:{hardware['sm_count']}",
        )
        if role in ("full", "chain") and "stable" not in key.local_backend.lower():
            # Portable same-kernel speedups do not establish a fastest-backend
            # policy when DeepGEMM hybrid is the real competing module.
            continue
        candidate = "tp4:" + name
        index = (key.identity, candidate)
        record = pending.setdefault(
            index,
            dict(
                key=asdict(key),
                baseline="tp4:replicated",
                candidate=candidate,
                stage="final",
                status="passed",
                source_scope="tp4",
                source_digest=source["digest"],
                environment_digest=env_digest,
                includes_total_cost=True,
                communication_included=True,
                baseline_scope=case["baseline_scope"],
                correctness_artifacts=[
                    f"{a['path']}:{a['sha256']}" for a in correctness
                ],
                performance_artifacts=[artifact(bundle["forward"])],
                source_manifest=artifact(bundle["source_manifest"]),
                orders={},
            ),
        )
        baseline = baselines[(digest(case), entry["execution"], entry["order"])]

        def metric(runs):
            return dict(runs[0], extra_rounds=runs[1:])

        record["orders"][entry["order"]] = dict(
            baseline=metric(baseline["runs"]), candidate=metric(entry["runs"])
        )
    return list(pending.values()), env_digest, environment


def collect(bundles, root, output):
    records, artifacts, environments, failures = [], [], {}, []
    for bundle_path in bundles:
        bundle = json.loads(Path(bundle_path).read_text())
        try:
            source, correctness, execution_sources = validate_bundle(bundle, root)
            if bundle["family"] == "tp4":
                new_records, env_digest, environment = normalize_tp4(
                    bundle, source, correctness
                )
                records.extend(new_records)
                for record in new_records:
                    record["execution_sources"] = execution_sources
                environments[env_digest] = environment
                artifacts.extend(correctness)
                continue
            reports = {
                o: json.loads(Path(bundle[o]).read_text())
                for o in ("forward", "reverse")
            }
            if any(report.get("status") != "completed" for report in reports.values()):
                raise ValueError("Both final performance reports must be completed")
            environment = dict(
                device=reports["forward"]["device"],
                native_identity=bundle["native_identity"],
                native_identity_contents=json.loads(
                    Path(bundle["native_identity"]["path"]).read_text()
                ),
                performance_native_package=bundle.get(
                    "performance_native_package", "installed"
                ),
            )
            if reports["forward"]["device"] != reports["reverse"]["device"]:
                raise ValueError("A/B environment metadata changed")
            env_digest = digest(environment)
            environments[env_digest] = environment
            pending = {}
            for order, report in reports.items():
                if bundle["family"] == "grouped":
                    report = dict(
                        report,
                        cases=[
                            dict(c, timings=c["timings"][order])
                            for c in report["cases"]
                        ],
                    )
                for execution in ("eager", "graph"):
                    for key, baseline, candidate, base_metric, new_metric in normalize(
                        bundle["family"], report, execution
                    ):
                        index = (key.identity, candidate)
                        record = pending.setdefault(
                            index,
                            dict(
                                key=asdict(key),
                                baseline=baseline,
                                candidate=candidate,
                                stage="final",
                                status="passed",
                                source_digest=source["digest"],
                                source_scope="single",
                                environment_digest=env_digest,
                                includes_total_cost=True,
                                execution_sources=execution_sources,
                                native_identity=bundle["native_identity"],
                                correctness_artifacts=[
                                    f"{a['path']}:{a['sha256']}" for a in correctness
                                ],
                                performance_artifacts=[
                                    artifact(bundle[o]) for o in ("forward", "reverse")
                                ],
                                source_manifest=artifact(bundle["source_manifest"]),
                                orders={},
                            ),
                        )
                        record["orders"][order] = dict(
                            baseline=base_metric, candidate=new_metric
                        )
                        if bundle["family"] == "grouped":
                            matching = next(
                                c
                                for c in report["cases"]
                                if c["case"]["rows"] == key.rows
                                and c["case"]["compressed_length"] == key.index_keys
                                and c["case"]["rows_per_request"] == key.group_rows
                            )
                            record["parameters"] = dict(
                                grid_ctas=matching["chosen_grid_ctas"],
                                paired=report["paired"],
                            )
            records.extend(pending.values())
            artifacts.extend(correctness)
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(dict(bundle=str(bundle_path), error=str(exc)))
    source = {
        scope: runtime_sources(root, parallel=scope == "tp4")
        for scope in ("single", "tp4")
    }
    admitted, rejected = [], []
    for env in environments:
        group = [r for r in records if r["environment_digest"] == env]
        dispatcher = Dispatcher(
            source_digest={scope: value["digest"] for scope, value in source.items()},
            environment_digest=env,
            records=group,
        )
        for values in dispatcher.records.values():
            admitted.extend(record for _, record in values)
        rejected.extend(dispatcher.rejections)
    report = dict(
        status="completed" if not failures else "incomplete",
        runtime_source=source,
        final_tool_source=source_identity(root),
        environments=environments,
        records=admitted,
        rejected=rejected,
        bundle_failures=failures,
        all_comparisons=records,
        correctness_artifacts=artifacts,
        scope=(
            "Single-GPU and actual TP4 module dispatch; no performance interpolation"
        ),
    )
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "collect"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT / "matrix.json")
    parser.add_argument("--bundles", nargs="*", default=[])
    args = parser.parse_args()
    if args.action == "plan":
        cases = matrix()
        write(
            args.output,
            dict(
                status="planned_not_run",
                cases=cases,
                commands=commands(),
                correctness=(
                    "Phase verifiers cover ratio 1/2, group 1/2/4/6/8 and ragged, "
                    "ties, repeated/zero Q, invalid tails, graph mutation"
                ),
                staged_selection=(
                    "Not a Cartesian sweep; every requested dimension appears. "
                    "No prefill Gluon auto path or inferred TP4 benefit."
                ),
                estimated_runtime=(
                    "Record actual per-command wall clock. Sampling: "
                    "20 warmup + 30*10 calls per variant/mode/order; "
                    "noisy cases add 3 rounds."
                ),
            ),
        )
        print(f"{len(cases)} planned cases; commands in {args.output}")
    else:
        report = collect(args.bundles, args.root, args.output)
        print(
            f"admitted={len(report['records'])} rejected={len(report['rejected'])} "
            f"failed_bundles={len(report['bundle_failures'])}"
        )


if __name__ == "__main__":
    main()

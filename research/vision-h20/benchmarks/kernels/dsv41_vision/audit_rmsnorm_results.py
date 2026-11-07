# SPDX-License-Identifier: Apache-2.0
"""Independent DP4 four-variant RMSNorm audit; never alters v10 audit behavior."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_formal_results import Audit, sha

VARIANTS = ("eager", "eager_rope", "eager_rmsnorm", "eager_rope_rmsnorm")


def inspect_result(audit, path, freeze, manifest):
    if not path.exists():
        audit.pending.append(f"formal result not yet present: {path}")
        return None
    raw = path.read_bytes()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        audit.pending.append("result JSON incomplete during write")
        return None
    audit.inputs.append(
        {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "status": result.get("status"),
        }
    )
    running = result.get("status") == "running"
    if running:
        audit.pending.append("formal run still running")
    else:
        audit.check(result.get("status") == "completed", "run failed or interrupted")
    args = result.get("args", {})
    for actual, expected, label in (
        (args.get("mode"), "data", "DP mode"),
        (result.get("world_size"), 4, "world size"),
        (result.get("rank"), 0, "result rank"),
        (result.get("vision_tensor_parallel_size"), 1, "vision TP"),
        (result.get("vision_heads_per_rank"), 16, "vision heads per rank"),
        (args.get("batch_sizes"), [4, 8], "batches"),
        (args.get("variants"), list(VARIANTS), "variants"),
        (args.get("profile_dir"), None, "profiler disabled"),
        (args.get("profile_end_to_end"), False, "end-to-end profiler disabled"),
        (args.get("correctness_only"), False, "formal performance mode"),
        (Path(args.get("source_root", "")).name, freeze.name, "source freeze"),
    ):
        audit.check(actual == expected, f"unexpected {label}: {actual}")
    audit.check(
        result.get("benchmark_sha256")
        == manifest.get("benchmarks/kernels/dsv41_vision/bench_vision.py"),
        "benchmark freeze hash mismatch",
    )
    sources = result.get("sources", {})
    audit.check(len(sources) == 6, "six source overlay hashes required")
    for name, digest in sources.items():
        relative = name.split(f"/{freeze.name}/")[-1]
        audit.check(
            relative in manifest and manifest[relative] == digest,
            f"source hash mismatch: {name}",
        )
    for filename in ("rotary.py", "rmsnorm.py"):
        relative = f"vllm_dsv41_vision/{filename}"
        actual = [
            digest
            for name, digest in result.get("loaded_files", {}).items()
            if name.endswith(f"/{freeze.name}/{relative}")
        ]
        if not actual and running:
            audit.pending.append(f"final loaded manifest pending for {filename}")
        else:
            audit.check(
                actual == [manifest.get(relative)],
                f"loaded kernel hash mismatch: {filename}",
            )
    weights = result.get("weights", [])
    audit.check(
        len(weights) == 266 and all(w.get("shard_verified") for w in weights),
        "all 266 selected visual tensors must pass shard verification",
    )
    expected = {
        (b, v, o) for b in (4, 8) for v in VARIANTS for o in ("forward", "reverse")
    }
    seen = set()
    for case in result.get("cases", []):
        batch = case.get("count")
        audit.check(batch in (4, 8), f"unexpected batch {batch}")
        for variant in VARIANTS[1:]:
            key = variant + "_vs_original_all_ranks"
            if case.get(key):
                audit.check(
                    case.get("exact_comparison_references", {}).get(variant) == "eager",
                    f"B{batch} {variant}: wrong exact reference",
                )
            check = {
                "count": batch,
                "measurements": case.get("measurements", []),
                "eager_rope_vs_original_all_ranks": case.get(key, []),
            }
            audit.case_correctness(check, f"B{batch} {variant} vs eager", running)
        combo_key = "eager_rope_rmsnorm_vs_rope_all_ranks"
        check = {
            "count": batch,
            "measurements": case.get("measurements", []),
            "eager_rope_vs_original_all_ranks": case.get(combo_key, []),
        }
        audit.case_correctness(check, f"B{batch} combined vs RoPE", running)
        for item in case.get("measurements", []):
            key = (batch, item.get("variant"), item.get("order"))
            audit.check(key in expected, f"unexpected measurement {key}")
            audit.check(key not in seen, f"duplicate measurement {key}")
            seen.add(key)
            summary = audit.measurement(item, str(key))
            if summary:
                audit.rows.append(
                    {"batch": batch, "variant": key[1], "order": key[2], **summary}
                )
    missing = sorted(expected - seen)
    if missing:
        (audit.pending if running else audit.errors).append(
            f"missing measurements {missing}"
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    audit = Audit()
    manifest_path = args.freeze / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in manifest.items():
        p = args.freeze / relative
        audit.check(p.is_file() and sha(p) == digest, f"freeze changed: {relative}")
    result = inspect_result(audit, args.input, args.freeze, manifest)
    comparisons = []
    for batch in (4, 8):
        for order in ("forward", "reverse"):
            rows = {
                r["variant"]: r
                for r in audit.rows
                if (r["batch"], r["order"]) == (batch, order)
            }
            for baseline, optimized in (
                ("eager", "eager_rope"),
                ("eager", "eager_rmsnorm"),
                ("eager", "eager_rope_rmsnorm"),
                ("eager_rope", "eager_rope_rmsnorm"),
            ):
                if baseline not in rows or optimized not in rows:
                    continue
                a, b = rows[baseline]["gpu"]["p50_us"], rows[optimized]["gpu"]["p50_us"]
                comparisons.append(
                    {
                        "batch": batch,
                        "order": order,
                        "baseline": baseline,
                        "optimized": optimized,
                        "gpu_p50_decrease_pct": (1 - b / a) * 100,
                        "gpu_p50_speedup": a / b,
                    }
                )
    status = "failed" if audit.errors else "pending" if audit.pending else "verified"
    output = {
        "status": status,
        "expected_measurements": 16,
        "observed_measurements": len(audit.rows),
        "errors": audit.errors,
        "pending": audit.pending,
        "inputs": audit.inputs,
        "audit_driver_sha256": sha(Path(__file__)),
        "shared_statistics_auditor_sha256": sha(
            Path(__file__).with_name("audit_formal_results.py")
        ),
        "freeze_manifest_sha256": sha(manifest_path),
        "measurements": audit.rows,
        "comparisons": comparisons,
        "timing_boundary": result.get("timing") if result else None,
        "notes": [
            "DP4 only; compare within same batch/order, never with old v10 samples",
            "GPU-ready vision+aligner+span with communication; no CPU prep/H2D",
            "Peak allocated memory is rank0 only, not max-rank memory",
            "Four per-rank sample means cannot reconstruct per-invocation max-rank",
            "Raw samples check host wall max and GPU rank-mean lower bound",
            "Extra rounds remain separately recorded; no selective pooling",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        f"# DP4 RMSNorm 四路径正式审核：{status}",
        "",
        f"完整矩阵：{len(audit.rows)}/16 measurement。所有优化须逐字节一致。",
        "",
        "GPU就绪视觉组件，含通信；CPU预处理/初始H2D不在计时内。",
        "",
        (
            "| B | 顺序 | 路径 | GPU P50/P90 ms | 主机 P50/P90 ms | "
            "GPU/主机 CV% | rank0峰值MiB |"
        ),
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in audit.rows:
        gpu, wall = row["gpu"], row["host_wall"]
        lines.append(
            f"| {row['batch']} | {row['order']} | {row['variant']} "
            f"| {gpu['p50_us'] / 1000:.3f}/{gpu['p90_us'] / 1000:.3f} "
            f"| {wall['p50_us'] / 1000:.3f}/{wall['p90_us'] / 1000:.3f} "
            f"| {gpu['cv'] * 100:.2f}/{wall['cv'] * 100:.2f} "
            f"| {row['peak_allocated_bytes_rank0_only'] / 2**20:.1f} |"
        )
    lines += [
        "",
        "| B | 顺序 | 原路径 → 优化路径 | GPU P50降幅 | 加速比 |",
        "|---:|---|---|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['batch']} | {row['order']} "
            f"| {row['baseline']} → {row['optimized']} "
            f"| {row['gpu_p50_decrease_pct']:.2f}% "
            f"| {row['gpu_p50_speedup']:.3f}× |"
        )
    lines += [
        "",
        *[f"- ERROR: {s}" for s in audit.errors],
        *[f"- Pending: {s}" for s in audit.pending],
        "",
        "不与旧v10的样本混算；独立保留正反序、原始样本及补测。",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "measurements": len(audit.rows),
                "errors": audit.errors,
                "pending": audit.pending,
            }
        )
    )


if __name__ == "__main__":
    main()

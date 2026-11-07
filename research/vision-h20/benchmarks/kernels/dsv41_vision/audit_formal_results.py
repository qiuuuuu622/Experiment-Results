# SPDX-License-Identifier: Apache-2.0
"""Independent, CPU-light audit of frozen four-rank vision benchmark JSON.

No torch import, GPU operations, benchmark execution, or input-file mutation.
Missing or still-running results stay pending; only a complete valid matrix
can become verified. Every reported statistic is recomputed from raw samples.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {
        "p50_us": statistics.median(values),
        "p90_us": sorted(values)[math.ceil(0.9 * len(values)) - 1],
        "mean_us": statistics.mean(values),
        "cv": statistics.pstdev(values) / statistics.mean(values),
    }


class Audit:
    def __init__(self):
        self.errors, self.pending, self.rows, self.inputs = [], [], [], []
        self.checkpoint_identity = None
        self.image_identities = {}

    def check(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def samples(self, values, label):
        return self.check(
            isinstance(values, list)
            and len(values) == 30
            and all(
                isinstance(x, (float, int)) and math.isfinite(x) and x > 0
                for x in values
            ),
            f"{label}: require 30 finite positive samples",
        )

    def case_correctness(self, case, label, running):
        correctness = case.get("eager_rope_vs_original_all_ranks", [])
        if running and not correctness and not case.get("measurements"):
            self.pending.append(f"{label}: bitwise correctness not yet written")
            return
        self.check(
            len(correctness) == 4
            and all(
                len(rank) == case["count"]
                and all(
                    m.get("exact") is True and m.get("finite") is True for m in rank
                )
                for rank in correctness
            ),
            f"{label}: missing/failing four-rank bitwise correctness",
        )

    def measurement(self, item, label, include_row=True):
        self.check(
            (item.get("warmup"), item.get("samples"), item.get("inner"))
            == (20, 30, 10),
            f"{label}: expected warmup/samples/inner=20/30/10",
        )
        self.check(
            item.get("aggregation")
            == "max_rank_each_invocation_then_mean_within_sample",
            f"{label}: incorrect aggregation",
        )
        gpu, wall = item.get("samples_us"), item.get("wall_samples_us")
        if not self.samples(gpu, label + " GPU"):
            return None
        if not self.samples(wall, label + " wall"):
            return None
        gs, ws = stats(gpu), stats(wall)
        for field, actual in (
            ("p50_us", gs["p50_us"]),
            ("p90_us", gs["p90_us"]),
            ("coefficient_of_variation", gs["cv"]),
            ("wall_p50_us", ws["p50_us"]),
        ):
            reported = item.get(field)
            self.check(
                isinstance(reported, (float, int))
                and math.isclose(reported, actual, rel_tol=1e-10, abs_tol=1e-9),
                f"{label}: {field} differs from raw samples",
            )
        ranks = item.get("per_rank_samples", [])
        rank_stats = []
        if self.check(len(ranks) == 4, f"{label}: need four rank sample arrays"):
            valid = True
            for rank, arrays in enumerate(ranks):
                valid &= self.samples(arrays.get("gpu_us"), f"{label} rank{rank} GPU")
                valid &= self.samples(arrays.get("wall_us"), f"{label} rank{rank} wall")
            if valid:
                for i in range(30):
                    lower = max(r["gpu_us"][i] for r in ranks)
                    self.check(
                        gpu[i] + 1e-6 >= lower,
                        f"{label} sample{i}: GPU below max rank sample mean",
                    )
                    expected_wall = max(r["wall_us"][i] for r in ranks)
                    self.check(
                        math.isclose(
                            wall[i], expected_wall, rel_tol=1e-10, abs_tol=1e-6
                        ),
                        f"{label} sample{i}: wall is not max rank",
                    )
                rank_stats = [
                    {
                        "rank": rank,
                        "gpu": stats(r["gpu_us"]),
                        "host_wall": stats(r["wall_us"]),
                    }
                    for rank, r in enumerate(ranks)
                ]
        extra = item.get("extra_rounds", [])
        if gs["cv"] > 0.1 and include_row:
            self.check(len(extra) == 3, f"{label}: CV>10% requires 3 extra rounds")
        extra_summaries = [
            self.measurement(x, f"{label} extra{i}", False) for i, x in enumerate(extra)
        ]
        peak = item.get("peak_allocated_bytes")
        self.check(
            isinstance(peak, int) and peak > 0,
            f"{label}: missing rank0 allocated-memory peak",
        )
        return {
            "gpu": gs,
            "host_wall": ws,
            "per_rank": rank_stats,
            "peak_allocated_bytes_rank0_only": peak,
            "extra_rounds": extra_summaries,
        }

    def run_file(self, path, mode, manifest, freeze_name):
        if not path.exists():
            self.pending.append(f"{mode}: result file missing: {path}")
            return
        try:
            raw = path.read_bytes()
            result = json.loads(raw)
        except json.JSONDecodeError:
            self.pending.append(f"{mode}: JSON incomplete during write: {path}")
            return
        self.inputs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "status": result.get("status"),
                "mode": mode,
            }
        )
        complete = result.get("status") == "completed"
        if result.get("status") == "running":
            self.pending.append(f"{mode}: run still running")
        elif not complete:
            self.errors.append(f"{mode}: run status {result.get('status')}")
        args = result.get("args", {})
        self.check(args.get("mode") == mode, f"{mode}: result mode mismatch")
        self.check(
            result.get("world_size") == 4 and result.get("rank") == 0,
            f"{mode}: expected rank0 result from actual four ranks",
        )
        self.check(args.get("batch_sizes") == [4, 8], f"{mode}: expected B4/B8")
        self.check(
            args.get("variants") == ["eager", "eager_rope"],
            f"{mode}: unexpected baseline/optimization variants",
        )
        self.check(
            args.get("profile_dir") is None and not args.get("profile_end_to_end"),
            f"{mode}: formal timing must not include profiler",
        )
        self.check(
            args.get("disable_custom_all_reduce") is False,
            f"{mode}: default communication configuration changed",
        )
        self.check(
            Path(args.get("source_root", "")).name == freeze_name,
            f"{mode}: source root not expected freeze",
        )
        self.check(
            result.get("vision_tensor_parallel_size") == (1 if mode == "data" else 4),
            f"{mode}: unexpected actual vision TP size",
        )
        self.check(
            result.get("vision_heads_per_rank") == (16 if mode == "data" else 4),
            f"{mode}: unexpected vision heads per rank",
        )
        driver_key = "benchmarks/kernels/dsv41_vision/bench_vision.py"
        self.check(
            result.get("benchmark_sha256") == manifest.get(driver_key),
            f"{mode}: executed benchmark hash differs from freeze",
        )
        self.check(
            len(result.get("sources", {})) == 6,
            f"{mode}: expected six exact-source overlay modules",
        )
        for remote, digest in result.get("sources", {}).items():
            relative = remote.split(f"/{freeze_name}/")[-1]
            self.check(
                relative in manifest and manifest[relative] == digest,
                f"{mode}: source hash mismatch {remote}",
            )
        loaded = result.get("loaded_files", {})
        for key in ("vllm_dsv41_vision/rotary.py",):
            matches = [
                value
                for remote, value in loaded.items()
                if remote.endswith(f"/{freeze_name}/{key}")
            ]
            if not matches and not complete:
                self.pending.append(f"{mode}: final loaded-file manifest not yet saved")
            else:
                self.check(
                    matches == [manifest.get(key)],
                    f"{mode}: loaded {key} differs from freeze",
                )
        weights = result.get("weights", [])
        identity = (
            result.get("config_sha256"),
            sorted((w.get("name"), w.get("sha256")) for w in weights),
        )
        if self.checkpoint_identity is None:
            self.checkpoint_identity = identity
        else:
            self.check(
                identity == self.checkpoint_identity,
                f"{mode}: checkpoint/config differs between TP4 and DP4",
            )
        self.check(
            len(weights) == 266 and all(w.get("shard_verified") for w in weights),
            f"{mode}: 266 selected visual tensors must pass shard loading checks",
        )
        seen = set()
        for case in result.get("cases", []):
            batch = case.get("count")
            self.check(batch in (4, 8), f"{mode}: unexpected batch {batch}")
            image_identity = tuple(
                image.get("sha256") for image in case.get("images", [])
            )
            self.check(
                len(image_identity) == batch, f"{mode} B{batch}: image hashes missing"
            )
            if batch in self.image_identities:
                self.check(
                    image_identity == self.image_identities[batch],
                    f"{mode} B{batch}: TP4/DP4 image input mismatch",
                )
            else:
                self.image_identities[batch] = image_identity
            self.case_correctness(
                case, f"{mode} B{batch}", running=result.get("status") == "running"
            )
            for item in case.get("measurements", []):
                variant, order = item.get("variant"), item.get("order")
                key = (batch, variant, order)
                self.check(key not in seen, f"{mode}: duplicate measurement {key}")
                seen.add(key)
                self.check(
                    variant in ("eager", "eager_rope")
                    and order in ("forward", "reverse"),
                    f"{mode}: unexpected measurement {key}",
                )
                row = self.measurement(item, f"{mode} {key}")
                if row:
                    self.rows.append(
                        {
                            "mode": mode,
                            "batch": batch,
                            "variant": variant,
                            "order": order,
                            **row,
                        }
                    )
        expected = {
            (b, v, o)
            for b in (4, 8)
            for v in ("eager", "eager_rope")
            for o in ("forward", "reverse")
        }
        missing = sorted(expected - seen)
        if missing:
            (self.errors if complete else self.pending).append(
                f"{mode}: missing measurements {missing}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", type=Path, required=True)
    parser.add_argument("--tp", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    audit = Audit()
    manifest = json.loads((args.freeze / "manifest.json").read_text())
    for relative, digest in manifest.items():
        path = args.freeze / relative
        audit.check(
            path.is_file() and sha(path) == digest,
            f"freeze content differs from manifest: {relative}",
        )
    audit.run_file(args.dp, "data", manifest, args.freeze.name)
    audit.run_file(args.tp, "weights", manifest, args.freeze.name)
    pairs = []
    for mode in ("data", "weights"):
        for batch in (4, 8):
            for order in ("forward", "reverse"):
                rows = {
                    r["variant"]: r
                    for r in audit.rows
                    if (r["mode"], r["batch"], r["order"]) == (mode, batch, order)
                }
                if set(rows) != {"eager", "eager_rope"}:
                    continue
                base, opt = rows["eager"], rows["eager_rope"]
                a, b = base["gpu"]["p50_us"], opt["gpu"]["p50_us"]
                pairs.append(
                    {
                        "mode": mode,
                        "batch": batch,
                        "order": order,
                        "baseline": base,
                        "optimized": opt,
                        "gpu_p50_decrease_pct": (1 - b / a) * 100,
                        "gpu_p50_speedup": a / b,
                    }
                )
    status = "failed" if audit.errors else "pending" if audit.pending else "verified"
    result = {
        "status": status,
        "audit_driver_sha256": sha(Path(__file__)),
        "expected_measurements": 16,
        "observed_measurements": len(audit.rows),
        "inputs": audit.inputs,
        "freeze_manifest_sha256": sha(args.freeze / "manifest.json"),
        "errors": audit.errors,
        "pending": audit.pending,
        "measurements": audit.rows,
        "comparisons": pairs,
        "limitations": [
            "GPU-ready vision+aligner+span; excludes PNG decode/preprocess/initial H2D",
            "P90 nearest-rank; CV is population standard deviation divided by mean",
            "GPU raw sample: mean of per-call max rank over ten calls",
            "Per-rank raw arrays contain sample means, not each inner invocation",
            "GPU aggregation cannot be fully reconstructed from per-rank means",
            "Host wall sample is max rank; includes timed host submission and sync",
            "Peak allocated memory is rank0 only, includes live model/buffers",
            "Extra rounds are retained separately and never selectively pooled",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"# 四卡视觉正式实验审核：{status}",
        "",
        f"完整矩阵：{len(audit.rows)}/16 measurement。",
        "",
        "仅GPU就绪的视觉+aligner+span组件，含通信；不含CPU预处理和初始H2D。",
        "峰值内存仅rank0，不是四rank最大值。正反序分别报告，不跨模式合并。",
        "",
        (
            "| 模式 | B | 顺序 | GPU P50 ms 原→优 | GPU P90 ms 原→优 | "
            "主机P50/P90 ms 原→优 | GPU CV% 原→优 | P50降幅 | 加速比 |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        a, b = pair["baseline"], pair["optimized"]

        def ms(row, kind, field):
            return row[kind][field] / 1000

        lines.append(
            f"| {'DP4' if pair['mode'] == 'data' else 'TP4'} "
            f"| {pair['batch']} | {pair['order']} "
            f"| {ms(a, 'gpu', 'p50_us'):.3f}→{ms(b, 'gpu', 'p50_us'):.3f} "
            f"| {ms(a, 'gpu', 'p90_us'):.3f}→{ms(b, 'gpu', 'p90_us'):.3f} "
            f"| {ms(a, 'host_wall', 'p50_us'):.3f}/{ms(a, 'host_wall', 'p90_us'):.3f}→"
            f"{ms(b, 'host_wall', 'p50_us'):.3f}/{ms(b, 'host_wall', 'p90_us'):.3f} "
            f"| {a['gpu']['cv'] * 100:.2f}→{b['gpu']['cv'] * 100:.2f} "
            f"| {pair['gpu_p50_decrease_pct']:.2f}% | {pair['gpu_p50_speedup']:.3f}× |"
        )
    lines += [
        "",
        *[f"- ERROR: {x}" for x in audit.errors],
        *[f"- Pending: {x}" for x in audit.pending],
        "",
        "原始rank样本见输入JSON；审核JSON保留host wall CV与rank0内存。",
        "重复轮不选择性合并。",
        "per-rank只保存样本均值，无法完全复算逐调用max→平均；已验证必要不等式，",
        "host wall逐样本max精确核对，聚合实现以冻结driver为证。",
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

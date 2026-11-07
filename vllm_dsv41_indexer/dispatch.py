# SPDX-License-Identifier: Apache-2.0
"""Explicit, evidence-bound module dispatch; no GPU inspection or host copies.

The caller supplies shape and layout contracts while planning, then binds the
selected preconstructed callable outside the timed region. A measured point
does not authorize interpolation to another shape, execution mode, or machine.
"""

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DispatchKey:
    phase: str
    role: str
    rows: int
    requests: int
    index_keys: int
    ratio: int = 1
    group_rows: int = 1
    execution: str = "graph"
    candidate_layout: str = "per_row"
    candidate_blocks: int = 2048
    parallel: str = "single"
    world_size: int = 1
    topk: int = 512
    lengths_mode: str = "static"
    hardware: str = "NVIDIA H20:sm90:78"
    may_have_zero_lengths: bool = True
    native_zero_prefetch_patched: bool = False
    input_mode: str = "replicated"
    projection: bool = False
    group_sizes: tuple[int, ...] = ()
    local_backend: str = "fp8_native"

    def __post_init__(self):
        object.__setattr__(self, "group_sizes", tuple(self.group_sizes))
        if self.phase not in ("decode", "verify", "prefill", "selector"):
            raise ValueError("Unknown phase")
        if self.role not in ("full", "source", "consumer", "chain", "selector"):
            raise ValueError("Unknown role")
        if min(self.rows, self.requests, self.index_keys) < 1:
            raise ValueError("Positive rows, requests and index_keys required")
        if self.requests > self.rows or self.ratio not in (1, 2):
            raise ValueError("Invalid request count or compression ratio")
        if self.group_rows not in (0, 1, 2, 4, 6, 8):
            raise ValueError("Use group_rows=0 for explicit ragged groups")
        if self.execution not in ("eager", "graph"):
            raise ValueError("Execution must be eager or graph")
        if self.parallel not in ("single", "replicated", "row_sharded"):
            raise ValueError("Unknown parallel mode")
        if self.world_size not in (1, 4):
            raise ValueError("Final dispatch supports single GPU or actual TP4")
        if (self.parallel == "single") != (self.world_size == 1):
            raise ValueError("Parallel mode and physical world size disagree")
        if self.candidate_layout not in ("per_row", "shared_prefix", "request_shared"):
            raise ValueError("Unknown caller-guaranteed candidate layout")
        if not 0 <= self.candidate_blocks <= 2048 or self.topk not in (512, 2048):
            raise ValueError("Unsupported candidate capacity or top-k")
        if self.lengths_mode not in ("static", "updated_each_call"):
            raise ValueError("Unknown metadata lifetime")
        if self.input_mode not in ("replicated", "root"):
            raise ValueError("Unknown input distribution")
        if self.group_sizes and (
            sum(self.group_sizes) != self.rows
            or len(self.group_sizes) != self.requests
            or min(self.group_sizes) < 1
        ):
            raise ValueError("Request groups must partition query rows")

    @property
    def identity(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Strategy:
    name: str
    deterministic: bool
    phases: tuple[str, ...]
    roles: tuple[str, ...]
    shared_candidates: bool = False
    includes_communication: bool = False


_MODULE = ("decode", "verify")
STRATEGIES = {
    s.name: s
    for s in (
        Strategy("native_full", False, _MODULE, ("full",)),
        Strategy("native_source", False, _MODULE, ("source",)),
        Strategy("native_score_stable_topk", True, _MODULE, ("full",)),
        Strategy("native_score_ant_token_and_source_topk", True, _MODULE, ("source",)),
        Strategy("optimized_full", True, _MODULE, ("full",)),
        Strategy("grouped_gluon_full", True, _MODULE, ("full",)),
        Strategy("grouped_gluon_source", True, _MODULE, ("source",)),
        Strategy("grouped_source_fused_max", True, _MODULE, ("source",)),
        Strategy("grouped_source_separate_max", True, _MODULE, ("source",)),
        Strategy("vllm_original", False, _MODULE, ("consumer",)),
        Strategy("one_sort_ant_topk_fused_map", True, _MODULE, ("consumer",)),
        Strategy("one_sort_ant_topk_separate_map", True, _MODULE, ("consumer",)),
        Strategy(
            "shared_consumer_publish_included", True, _MODULE, ("consumer",), True
        ),
        Strategy("native_source_four_consumers", False, _MODULE, ("chain",)),
        Strategy("hybrid_source_four_consumers", True, _MODULE, ("chain",)),
        Strategy("hybrid_source_full_consumers", True, _MODULE, ("chain",)),
        Strategy("optimized_source_four_consumers", True, _MODULE, ("chain",)),
        Strategy(
            "native_pipeline", False, ("prefill",), ("full", "source", "consumer")
        ),
        *(
            Strategy(
                name + "_pipeline", True, ("prefill",), ("full", "source", "consumer")
            )
            for name in ("deepgemm", "row_tiled", "keys_major", "indirect")
        ),
        Strategy("native", False, ("selector",), ("selector",)),
        Strategy("streaming", True, ("selector",), ("selector",)),
        Strategy("cluster_preplanned", True, ("selector",), ("selector",)),
        Strategy("cluster_replan_each_call", True, ("selector",), ("selector",)),
        *(
            Strategy(
                "tp4:" + name,
                True,
                _MODULE,
                ("full", "chain", "consumer"),
                includes_communication=True,
            )
            for name in (
                "replicated",
                "shard_request_groups_replicated_input",
                "shard_request_groups_root_input",
                "shard_split_rows_replicated_input",
                "shard_split_rows_root_input",
            )
        ),
    )
}


def _compatible(name, key, deterministic):
    strategy = STRATEGIES.get(name)
    if (
        strategy is None
        or key.phase not in strategy.phases
        or key.role not in strategy.roles
    ):
        return False
    if deterministic and not strategy.deterministic:
        return False
    if strategy.shared_candidates and key.candidate_layout != "shared_prefix":
        return False
    if (
        key.phase == "prefill"
        and key.role == "consumer"
        and key.candidate_layout != "request_shared"
    ):
        return False
    if key.lengths_mode != "static" and name == "cluster_preplanned":
        return False
    if (
        key.phase in _MODULE
        and key.may_have_zero_lengths
        and not key.native_zero_prefetch_patched
        and (
            name.startswith("native")
            or name
            in (
                "vllm_original",
                "hybrid_source_four_consumers",
                "hybrid_source_full_consumers",
            )
        )
    ):
        return False
    if name.startswith("tp4:"):
        if key.world_size != 4 or key.parallel == "single" or not key.group_sizes:
            return False
        if (
            key.may_have_zero_lengths
            and "native" in key.local_backend.lower()
            and not key.native_zero_prefetch_patched
        ):
            return False
        if name == "tp4:replicated":
            return key.input_mode == "replicated"
        return name.endswith(f"_{key.input_mode}_input")
    return key.parallel == "single"


def fallback(key, deterministic):
    """Return a semantic baseline, without claiming it is the fastest path."""
    if key.parallel != "single":
        if not key.group_sizes:
            raise ValueError("TP4 requires explicit request group sizes")
        name = (
            "tp4:replicated"
            if key.input_mode == "replicated"
            else "tp4:shard_request_groups_root_input"
        )
        if not _compatible(name, key, deterministic):
            raise ValueError("TP4 backend lacks safe empty-request semantics")
        return name
    if key.phase == "selector":
        name = "streaming" if deterministic else "native"
    elif key.phase == "prefill":
        name = "deepgemm_pipeline" if deterministic else "native_pipeline"
    else:
        if key.may_have_zero_lengths and not key.native_zero_prefetch_patched:
            return {
                "full": "optimized_full",
                "source": "grouped_source_fused_max",
                "consumer": "one_sort_ant_topk_fused_map",
                "chain": "optimized_source_four_consumers",
            }[key.role]
        choices = {
            "full": ("native_full", "native_score_stable_topk"),
            "source": ("native_source", "native_score_ant_token_and_source_topk"),
            "consumer": ("vllm_original", "one_sort_ant_topk_fused_map"),
            "chain": ("native_source_four_consumers", "hybrid_source_four_consumers"),
        }
        name = choices[key.role][int(deterministic)]
    if not _compatible(name, key, deterministic):
        raise ValueError("No supported baseline for this explicit layout contract")
    return name


def _rounds(metric):
    rounds = [metric, *metric.get("extra_rounds", [])]
    for item in rounds:
        samples = item.get("samples_us", [])
        if len(samples) < 30 or item.get("warmup", 0) < 20 or item.get("inner", 0) < 10:
            raise ValueError("Requires 20 warmups and 30 samples of 10 calls")
        if any(not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError("Nonfinite or nonpositive GPU timing sample")
        if not math.isclose(statistics.median(samples), item["p50_us"], rel_tol=1e-6):
            raise ValueError("P50 is inconsistent with raw samples")
        cv = statistics.pstdev(samples) / statistics.mean(samples)
        if not math.isclose(
            cv, item["coefficient_of_variation"], rel_tol=1e-6, abs_tol=1e-9
        ):
            raise ValueError("CV is inconsistent with raw samples")
        wall = item.get("wall_samples_us", [])
        if (
            len(wall) != len(samples)
            or any(not math.isfinite(x) or x <= 0 for x in wall)
            or "peak_allocated_bytes" not in item
        ):
            raise ValueError("Missing module wall clock or peak memory evidence")
    if metric["coefficient_of_variation"] > 0.1 and len(rounds) < 4:
        raise ValueError("Noisy cases require all three additional rounds")
    return rounds


def admit(record, *, source_digest, environment_digest):
    """Validate a final-source A/B record and return its conservative speedup."""
    if record.get("stage") != "final" or record.get("status") != "passed":
        raise ValueError("Historical/screening or failed evidence cannot be admitted")
    expected_source = (
        source_digest.get(record.get("source_scope", "single"))
        if isinstance(source_digest, dict)
        else source_digest
    )
    if record.get("source_digest") != expected_source:
        raise ValueError("Source digest differs from final snapshot")
    if record.get("environment_digest") != environment_digest:
        raise ValueError("Environment digest differs from final machine")
    if not record.get("correctness_artifacts") or not record.get("includes_total_cost"):
        raise ValueError("Missing correctness evidence or complete module cost")
    key = DispatchKey(**record["key"])
    if not _compatible(record["candidate"], key, False):
        raise ValueError("Candidate incompatible with shape/layout contract")
    orders = record.get("orders", {})
    if set(orders) != {"forward", "reverse"}:
        raise ValueError("Both final-source A/B orders are required")
    speedups = []
    for pair in orders.values():
        base, new = _rounds(pair["baseline"]), _rounds(pair["candidate"])
        if key.world_size == 4:
            for measurement in base + new:
                if (
                    measurement.get("aggregation")
                    != "max_rank_each_invocation_then_mean_within_sample"
                ):
                    raise ValueError("TP4 requires per-invocation slowest-rank samples")
            if not record.get("communication_included"):
                raise ValueError("TP4 collective/input costs must be included")
        worst_new = max(m["p50_us"] for m in new)
        best_base = min(m["p50_us"] for m in base)
        speedup = best_base / worst_new
        noise = max(m["coefficient_of_variation"] for m in base + new)
        reduction = 1 - 1 / speedup
        if reduction < 0.05 or reduction <= noise:
            raise ValueError("Gain below 5% or not larger than measured noise")
        speedups.append(speedup)
    return min(speedups)


@dataclass(frozen=True)
class Decision:
    strategy: str
    reason: str
    evidence: tuple[str, ...] = ()
    parameters: dict | None = None

    def bind(self, callables):
        """Resolve a preconstructed module once, before graph capture or timing."""
        if self.strategy not in callables:
            raise ValueError(f"Preconstruct required strategy: {self.strategy}")
        return callables[self.strategy]


class Dispatcher:
    """CPU planner requiring final-source evidence for each optimized point."""

    def __init__(self, *, source_digest, environment_digest, records=()):
        self.records = {}
        self.rejections = []
        for record in records:
            try:
                speedup = admit(
                    record,
                    source_digest=source_digest,
                    environment_digest=environment_digest,
                )
            except (ValueError, KeyError, TypeError) as exc:
                self.rejections.append(
                    {
                        "candidate": record.get("candidate"),
                        "reason": str(exc),
                        "key": record.get("key"),
                        "baseline": record.get("baseline"),
                        "performance_artifacts": record.get(
                            "performance_artifacts", []
                        ),
                    }
                )
                continue
            key = DispatchKey(**record["key"])
            self.records.setdefault(key.identity, []).append((speedup, record))

    def choose(self, key, *, deterministic=True):
        baseline = fallback(key, deterministic)
        matching = [
            (speedup, record)
            for speedup, record in self.records.get(key.identity, [])
            if _compatible(record["candidate"], key, deterministic)
        ]
        if not matching:
            return Decision(
                baseline, "No compatible final-source evidence for this exact shape"
            )
        # Compare absolute candidate latency, not speedups against potentially
        # different baselines. Deterministic-only calls never pick native top-k.
        _, best = min(
            matching,
            key=lambda item: max(
                m["p50_us"]
                for p in item[1]["orders"].values()
                for m in _rounds(p["candidate"])
            ),
        )
        return Decision(
            best["candidate"],
            "Exact final-source shape passed correctness and both A/B orders",
            tuple(best["correctness_artifacts"]),
            best.get("parameters", {}),
        )


def source_identity(root):
    """Hash all module/driver/native sources; result artifacts never hash themselves."""
    root = Path(root)
    paths = []
    for relative in (
        "vllm_dsv41_indexer",
        "benchmarks/kernels/dsv41_indexer",
        "tests/kernels/dsv41_indexer",
    ):
        paths += [
            p
            for p in (root / relative).rglob("*")
            if p.is_file()
            and p.suffix
            in (
                ".py",
                ".cu",
                ".cuh",
                ".h",
                ".hpp",
                ".cpp",
                ".json",
                ".sh",
                ".ini",
                ".patch",
            )
            and "__pycache__" not in p.parts
        ]
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"digest": hashlib.sha256(payload).hexdigest(), "files": files}

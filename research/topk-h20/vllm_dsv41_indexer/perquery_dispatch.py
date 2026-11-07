# SPDX-License-Identifier: Apache-2.0
"""Exact-contract selection for the P5c module experiment, independent of P8.

Planning validates host metadata and runtime identity once. Execution is the
selected module's unchanged forward method: no copies, checks or host syncs.
Changing lengths, request grouping, budgets or execution mode requires a new
plan; dynamic kernel correctness alone does not extend performance evidence.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .dispatch import _rounds

FULL = ("full_deepgemm_mask", "full_keys_major_mask")
COMPACT = ("direct", "gathered_wgmma")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PerQueryContract:
    query_lens: tuple[int, ...]
    contexts: tuple[int, ...]
    execution: str
    environment_digest: str
    role: str = "consumer"
    ratio: int = 1
    candidate_layout: str = "per_query"
    source_unique: bool = True
    source_layer: int = 20
    candidate_capacity: int = 2048
    candidate_block_size: int = 8
    index_heads: int = 32
    head_dim: int = 128
    topk: int = 512
    cache_format: str = "vllm_e4m3_page64_payload8192_scale256"
    q_dtype: str = "float8_e4m3fn"
    weight_scale_dtype: str = "float32"
    selector: str = "ant_stable_logical"
    score_budget: int = 1 << 30
    gather_budget: int = 256 << 20
    hardware: str = "h20-4:NVIDIA H20:sm90:78"
    world_size: int = 1

    def __post_init__(self):
        object.__setattr__(self, "query_lens", tuple(self.query_lens))
        object.__setattr__(self, "contexts", tuple(self.contexts))
        if not self.query_lens or len(self.query_lens) != len(self.contexts):
            raise ValueError("Each request requires a query count and final context")
        if any(
            not 0 < q <= c <= 1048576 for q, c in zip(self.query_lens, self.contexts)
        ):
            raise ValueError("Invalid model prefill query interval")
        if self.execution not in ("eager", "graph") or not self.environment_digest:
            raise ValueError("An execution mode and verified environment are required")
        required = {
            "role": "consumer",
            "ratio": 1,
            "candidate_layout": "per_query",
            "source_unique": True,
            "source_layer": 20,
            "candidate_capacity": 2048,
            "candidate_block_size": 8,
            "index_heads": 32,
            "head_dim": 128,
            "topk": 512,
            "cache_format": "vllm_e4m3_page64_payload8192_scale256",
            "q_dtype": "float8_e4m3fn",
            "weight_scale_dtype": "float32",
            "selector": "ant_stable_logical",
            "score_budget": 1 << 30,
            "gather_budget": 256 << 20,
            "world_size": 1,
            "hardware": "h20-4:NVIDIA H20:sm90:78",
        }
        if any(getattr(self, k) != v for k, v in required.items()):
            raise ValueError("Unsupported P5c contract; use an explicit backend")

    @property
    def rows(self):
        return sum(self.query_lens)

    @property
    def visible_lengths(self):
        return tuple(
            c - q + j + 1
            for q, c in zip(self.query_lens, self.contexts)
            for j in range(q)
        )

    @property
    def request_ids(self):
        return tuple(r for r, q in enumerate(self.query_lens) for _ in range(q))

    @property
    def identity(self):
        return digest(asdict(self))


def select_measurements(orders):
    """Choose only stable two-order wins; optimized paths beat BOTH baselines."""
    if set(orders) != {"forward", "reverse"}:
        raise ValueError("Both forward and reverse measurements are required")
    parsed = {}
    for order, variants in orders.items():
        if set(variants) != set(FULL + COMPACT):
            raise ValueError("Both full baselines and both compact variants required")
        parsed[order] = {name: _rounds(v) for name, v in variants.items()}

    def beats(candidate, baselines):
        comparisons = []
        for order, variants in parsed.items():
            new = variants[candidate]
            old = [m for name in baselines for m in variants[name]]
            speedup = min(m["p50_us"] for m in old) / max(m["p50_us"] for m in new)
            reduction = 1 - 1 / speedup
            noise = max(m["coefficient_of_variation"] for m in old + new)
            comparisons.append(
                dict(
                    order=order,
                    speedup=speedup,
                    reduction=reduction,
                    noise=noise,
                    passed=reduction >= 0.05 and reduction > noise,
                )
            )
        return all(c["passed"] for c in comparisons), comparisons

    baseline_wins, baseline_checks = beats(FULL[1], (FULL[0],))
    fallback = FULL[1] if baseline_wins else FULL[0]
    eligible, checks = [], {FULL[1]: baseline_checks}
    for name in COMPACT:
        passed, checks[name] = beats(name, FULL)
        if passed:
            eligible.append(name)
    chosen = (
        min(
            eligible,
            key=lambda name: max(m["p50_us"] for v in parsed.values() for m in v[name]),
        )
        if eligible
        else fallback
    )
    return dict(strategy=chosen, baseline=fallback, comparisons=checks)


class PerQueryPlanner:
    """An immutable final-evidence table; no interpolation by Q*K or thresholds."""

    def __init__(self, manifest, *, root, allow_provisional=False):
        self.manifest = manifest
        self.root = Path(root)
        if manifest.get("status") != "final_verified":
            raise ValueError("Final verified evidence is required")
        if manifest.get("admission_status") != "admitted" and not allow_provisional:
            raise ValueError("Paired same-input A/B evidence is still pending")
        if not manifest.get("correctness_artifacts"):
            raise ValueError("Missing final correctness evidence")
        if (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            != manifest["planner_source_sha256"]
        ):
            raise ValueError("Planner differs from the reviewed manifest")
        for relative, sha in manifest["artifact_sha256"].items():
            if hashlib.sha256((self.root / relative).read_bytes()).hexdigest() != sha:
                raise ValueError(f"Evidence artifact changed: {relative}")
        for relative, sha in manifest["execution_sources"].items():
            if hashlib.sha256((self.root / relative).read_bytes()).hexdigest() != sha:
                raise ValueError(f"Measured implementation changed: {relative}")
        self.records = {}
        for row in manifest["records"]:
            contract = PerQueryContract(**row["contract"])
            if contract.environment_digest != manifest["environment_digest"]:
                raise ValueError("Evidence runtime identity differs")
            decision = select_measurements(row["orders"])
            if decision["strategy"] != row["strategy"]:
                raise ValueError("Stored decision does not match raw measurements")
            self.records[contract.identity] = row

    def choose(self, contract):
        if contract.environment_digest != self.manifest["environment_digest"]:
            raise ValueError("Runtime differs from the measured environment")
        record = self.records.get(contract.identity)
        if record is None:
            return dict(
                strategy=FULL[0],
                evidence=False,
                reason="Unmeasured exact contract; explicit stable DeepGEMM baseline",
            )
        return dict(
            strategy=record["strategy"],
            evidence=True,
            reason="Exact contract; both orders and all retained rounds checked",
            admission_status=self.manifest.get("admission_status"),
        )

    def bind(self, contract, inputs, candidates):
        """Validate real tensors once, then return the unchanged selected module."""
        import torch

        from benchmarks.kernels.dsv41_indexer.benchmark_prefill_model import ModelPath
        from benchmarks.kernels.dsv41_indexer.verify_dispatch import (
            verify_native_identity,
        )

        from .perquery_prefill import PerQueryPrefillIndexer

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Plan/bind before capture; never inspect metadata in graph"
            )
        inputs.validate()
        expected_shape = (contract.rows, contract.index_heads, contract.head_dim)
        if (
            tuple(inputs.q.shape) != expected_shape
            or inputs.q.dtype != torch.float8_e4m3fn
            or inputs.weights.dtype != torch.float32
            or inputs.shape.ratio != 1
            or inputs.shape.max_keys != max(contract.contexts)
            or inputs.table.shape[0] != len(contract.query_lens)
            or tuple(inputs.requests.tolist()) != contract.request_ids
            or tuple(inputs.lengths.tolist()) != contract.visible_lengths
        ):
            raise ValueError("Actual inputs differ from the static model contract")
        if (
            candidates.shape != (contract.rows, 2048)
            or candidates.dtype != torch.int32
            or candidates.device != inputs.q.device
            or not candidates.is_contiguous()
        ):
            raise ValueError("Expected actual per-query int32 candidate blocks")
        counts = (inputs.lengths + 7) // 8
        ordered = candidates.sort(1).values
        present = candidates >= 0
        valid = (
            torch.equal(present.sum(1), counts.clamp(max=2048).long())
            and bool(((candidates == -1) | present).all())
            and bool(((candidates < counts[:, None]) | ~present).all())
            and not bool(
                ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()
            )
            and bool((candidates == ((inputs.lengths - 1) // 8)[:, None]).any(1).all())
        )
        if not valid:
            raise ValueError("Candidates violate the source20 unique/latest contract")
        device = torch.cuda.get_device_properties(inputs.q.device)
        if (device.name, device.major, device.minor, device.multi_processor_count) != (
            "NVIDIA H20",
            9,
            0,
            78,
        ):
            raise ValueError("Hardware differs from the measured H20")
        identity = self.manifest["runtime_identity"]
        from importlib.metadata import version

        for name, expected in identity["versions"].items():
            if version(name) != expected:
                raise ValueError(f"Runtime package differs: {name}")
        if torch.version.cuda != "13.0":
            raise ValueError("CUDA runtime differs")
        actual_native = verify_native_identity(
            {"native_identity_contents": identity}, None
        )
        known_native = {
            item["path"]: item["sha256"] for item in identity["native_files"]
        }
        if any(
            known_native.get(item["path"]) != item["sha256"]
            for item in actual_native["actually_loaded"]
        ):
            raise ValueError("Actually imported native module differs from evidence")
        decision = self.choose(contract)
        name = decision["strategy"]
        if name in COMPACT:
            op = PerQueryPrefillIndexer(
                inputs,
                candidates,
                backend=name,
                source_unique=True,
                score_budget=contract.score_budget,
                gather_budget=contract.gather_budget,
            )
        elif name in FULL:
            op = ModelPath(
                inputs,
                "deepgemm" if name == FULL[0] else "keys_major",
                "consumer",
                candidates,
            )
        else:
            raise ValueError("Unknown measured implementation")
        return op, {**decision, "actual_native_identity": actual_native}

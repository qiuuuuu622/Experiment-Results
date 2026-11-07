# SPDX-License-Identifier: Apache-2.0
"""Real exact-contract planner binding and poison-first graph replay."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .common import check_vllm, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args()
    check_vllm()
    import torch

    from vllm_dsv41_indexer.perquery_dispatch import (
        COMPACT,
        FULL,
        PerQueryContract,
        PerQueryPlanner,
    )
    from vllm_dsv41_indexer.perquery_prefill import PerQueryPrefillIndexer

    from .benchmark_prefill_model import ModelPath, model_inputs
    from .benchmark_prefill_perquery import source_fixture, validate

    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads(Path(args.manifest).read_text())
    planner = PerQueryPlanner(
        manifest, root=root, allow_provisional=args.allow_provisional
    )
    selected = {"initial_2k", "steady_16k", "steady_64k", "ragged"}
    source_paths = [Path(__file__), root / "vllm_dsv41_indexer/perquery_dispatch.py"]
    source_before = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
    }
    report = dict(
        status="running",
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        admission_status=manifest["admission_status"],
        source_before=source_before,
        cases=[],
    )
    save(args.output, report)
    try:
        for record in manifest["records"]:
            if record["label"] not in selected:
                continue
            contract = PerQueryContract(**record["contract"])
            inputs = model_inputs(contract.query_lens, contract.contexts, 1)
            publisher, candidates = source_fixture(inputs)
            canonical = candidates.sort(1).values
            hashes = []
            for _ in range(3):
                publisher.forward()
                current = publisher.candidates.sort(1).values
                assert torch.equal(canonical, current)
                hashes.append(
                    hashlib.sha256(current.cpu().numpy().tobytes()).hexdigest()
                )
            op, decision = planner.bind(contract, inputs, candidates)
            compact = {
                name: PerQueryPrefillIndexer(
                    inputs, candidates, backend=name, source_unique=True
                )
                for name in COMPACT
            }
            full = {
                name: ModelPath(
                    inputs,
                    "deepgemm" if name == FULL[0] else "keys_major",
                    "consumer",
                    candidates,
                )
                for name in FULL
            }
            (compact if decision["strategy"] in COMPACT else full)[
                decision["strategy"]
            ] = op
            for path in [*compact.values(), *full.values()]:
                path.forward()
            numerical = validate(inputs, candidates, compact, full)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.forward()
            phases = []
            for phase in ("baseline", "changed_q_and_candidates", "restore"):
                if phase == "changed_q_and_candidates":
                    inputs.q.view(torch.uint8).copy_(
                        inputs.q.view(torch.uint8).roll(1, 0)
                    )
                    publisher.inputs.q.view(torch.uint8).copy_(
                        publisher.inputs.q.view(torch.uint8).roll(3, 0)
                    )
                    publisher.forward()
                    candidates.copy_(publisher.candidates)
                elif phase == "restore":
                    inputs.q.view(torch.uint8).copy_(
                        inputs.q.view(torch.uint8).roll(-1, 0)
                    )
                    candidates.copy_(canonical)
                inner = op if decision["strategy"] in COMPACT else op.op
                inner.output.fill_(-42)
                if inner.scores is not None:
                    inner.scores.fill_(torch.nan)
                if inner.keys is not None:
                    inner.keys.view(torch.uint8).fill_(127)
                    inner.scales.fill_(torch.nan)
                if hasattr(inner, "state") and inner.state is not None:
                    inner.state.blocks.fill_(2147483647)
                    inner.state.counts.fill_(-777)
                graph.replay()
                captured = inner.output.sort(1).values
                op.forward()
                assert torch.equal(captured, inner.output.sort(1).values)
                phases.append(dict(phase=phase, graph_vs_eager="exact logical set"))
            inputs.lengths[0] -= 1
            try:
                planner.bind(contract, inputs, candidates)
            except ValueError as error:
                assert "Actual inputs" in str(error)
            else:
                raise AssertionError("Changed lengths must require replanning")
            inputs.lengths[0] += 1
            unmeasured = replace(
                contract, contexts=tuple(c + 1 for c in contract.contexts)
            )
            assert planner.choose(unmeasured)["evidence"] is False
            report["cases"].append(
                dict(
                    label=record["label"],
                    execution=contract.execution,
                    decision=decision,
                    numerical=numerical,
                    phases=phases,
                    repeated_source_canonical_sha256=hashes,
                    changed_lengths_rejected=True,
                    unknown_schedule_falls_back=True,
                    graph_capture_has_no_host_sync=True,
                )
            )
            save(args.output, report)
            print(
                json.dumps(
                    dict(
                        label=record["label"],
                        execution=contract.execution,
                        strategy=decision["strategy"],
                        status="passed",
                    )
                ),
                flush=True,
            )
            del compact, full, op, graph, publisher, inputs, candidates
        assert len(report["cases"]) == 8
        source_after = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        }
        assert source_before == source_after
        PerQueryPlanner(manifest, root=root, allow_provisional=args.allow_provisional)
        report["source_after"] = source_after
        report["measured_sources_and_artifacts_rechecked_after_run"] = True
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()

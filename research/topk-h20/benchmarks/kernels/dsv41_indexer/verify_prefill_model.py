# SPDX-License-Identifier: Apache-2.0
"""Model-shaped prefill dynamic-input regression; no model or performance claim."""

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from .benchmark_prefill_model import ModelPath, model_inputs
from .common import check_vllm, device_info, save


def identity():
    from vllm_dsv41_indexer import prefill, prefill_keys_major_ant

    from . import benchmark_prefill_model

    files = [
        Path(__file__),
        Path(benchmark_prefill_model.__file__),
        Path(prefill.__file__),
        Path(prefill_keys_major_ant.__file__),
    ]
    config = Path("architecture/model-config.json")
    if not config.exists():
        config = Path("results/indexer/20261101/model-config.json")
    files.append(config)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def logical_checks():
    # This is the model host-branch predicate, not a shortcut inside ModelPath.
    bypass = []
    for end in (1024, 1025, 1026):
        for capturing in (False, True):
            actual = end // 2 <= 512 and not capturing
            expected = end < 1026 and not capturing
            assert actual == expected
            bypass.append(dict(batch_max=end, capturing=capturing, bypass=actual))
    assert [j // 2 for j in range(1, 6)] == [0, 1, 1, 2, 2]
    assert [(1 + j) // 2 for j in range(1, 6)] == [1, 1, 2, 2, 3]
    assert max([1024, 1026]) // 2 > 512
    return dict(
        bypass=bypass,
        model_branch="max_seq_len // ratio <= 512 and not stream_capturing",
        scope="CPU predicate only; module does not implement host bypass",
        floor_even_prefix=[0, 1, 1, 2, 2],
        floor_odd_prefix=[1, 1, 2, 2, 3],
    )


def assert_outputs(path, output, published, inputs, native):
    import torch

    from vllm_dsv41_indexer.reference import source_reference, stable_topk

    # Recompute from current inputs independently of the captured output. Thus a
    # graph that reused stale preparation/metadata cannot pass by self-comparison.
    path.op.prepare()
    path.metadata()
    native.op.prepare()
    native.metadata()
    maximum, rows = 0.0, 0
    for start, end, request in path.op.chunks:
        got = path.score_chunk(start, end, request)
        ref = native.score_chunk(start, end, request)
        values, positions = [], []
        for row, count in enumerate(inputs.lengths[start:end].tolist()):
            a, b = got[row, :count], ref[row, :count]
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
            finite = torch.isfinite(a)
            assert not torch.isnan(a).any().item()
            if finite.any().item():
                maximum = max(maximum, (a[finite] - b[finite]).abs().max().item())
            values.append(a[finite])
            positions.append(torch.arange(count, device="cuda")[finite])
        expected = stable_topk(values, positions)
        assert torch.equal(output[start:end].sort(1).values, expected.sort(1).values)
        if path.op.source:
            blocks = source_reference(got, inputs.lengths[start:end])
            assert torch.equal(
                published[start:end].sort(1).values, blocks.sort(1).values
            )
            for row, count in enumerate(inputs.lengths[start:end].tolist()):
                if count:
                    assert (count - 1) // 8 in published[start + row].tolist()
        valid = output[start:end] >= 0
        assert ((output[start:end] < inputs.lengths[start:end, None]) | ~valid).all()
        rows += end - start
    return dict(
        rows_exact_stable=rows,
        source_rows_exact=rows if published is not None else 0,
        max_abs_vs_deepgemm=maximum,
    )


def fixed_logits(paths, inputs):
    """Every selector sees identical tie-heavy scores with NaN invalid tails."""
    import torch

    from vllm_dsv41_indexer.reference import source_reference, stable_topk

    results = {}
    for name, path in paths.items():
        for start, end, request in path.op.chunks:
            width = path.op.width
            col = torch.arange(width, device="cuda")
            scores = ((col % 17).float() / 4).expand(end - start, -1).clone()
            valid = col[None, :] < inputs.lengths[start:end, None]
            scores.masked_fill_(~valid, float("nan"))
            if path.candidates is not None:
                path.apply_mask(
                    scores,
                    None,
                    inputs.lengths[start:end],
                    path.candidates[start:end],
                    8,
                )
            path.op.select_chunk(scores, start, end, request)
            values, positions = [], []
            for row, count in enumerate(inputs.lengths[start:end].tolist()):
                finite = torch.isfinite(scores[row, :count])
                values.append(scores[row, :count][finite])
                positions.append(torch.arange(count, device="cuda")[finite])
            expected = stable_topk(values, positions)
            assert torch.equal(
                path.op.output[start:end].sort(1).values, expected.sort(1).values
            )
            if path.op.source:
                expected_blocks = source_reference(scores, inputs.lengths[start:end])
                assert torch.equal(
                    path.op.candidates[start:end].sort(1).values,
                    expected_blocks.sort(1).values,
                )
        results[name] = path.op.output.clone().sort(1).values
    assert all(torch.equal(result, results["deepgemm"]) for result in results.values())
    return dict(
        rows=inputs.shape.rows,
        exact_across_all_selectors=True,
        invalid_score_tails="NaN",
        ties="17 repeating exact FP32 buckets",
    )


def run_fixture(label, role, query_lens, contexts, *, dynamic):
    import torch

    ratio = 2 if role == "full" else 1
    inputs = model_inputs(query_lens, contexts, ratio)
    baseline = {
        key: getattr(inputs, key).clone()
        for key in ("q", "weights", "cache", "table", "lengths")
    }
    candidates, publisher = None, None
    if role == "consumer":
        candidates = torch.empty(
            (inputs.shape.rows, 2048), device="cuda", dtype=torch.int32
        )
        source_inputs = replace(
            inputs,
            q=inputs.q.view(torch.uint8).roll(1, 0).contiguous().view(inputs.q.dtype),
            weights=inputs.weights.roll(1, 1).contiguous(),
        )
        publisher = ModelPath(source_inputs, "keys_major", "source")
    paths = {
        name: ModelPath(inputs, backend, role, candidates)
        for name, backend in [
            ("deepgemm", "deepgemm"),
            ("current_keys_major", "keys_major"),
            ("ant", "ant"),
        ]
    }

    def publish(phase):
        if publisher is None:
            return
        publisher.op.inputs.q.view(torch.uint8).copy_(
            inputs.q.view(torch.uint8).roll(phase + 1, 0)
        )
        publisher.op.inputs.weights.copy_(inputs.weights.roll(phase + 1, 1))
        publisher.forward()
        candidates.copy_(publisher.op.candidates)

    publish(0)
    for path in paths.values():
        for _ in range(3):
            path.forward()
    torch.cuda.synchronize()
    graphs = {}
    for name, path in paths.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            path.forward()
        graphs[name] = graph
    torch.cuda.synchronize()
    phases = ["base", "candidate_update"]
    if dynamic:
        phases += [
            "length_update",
            "physical_page_permutation",
            "request_reorder",
            "all_zero_with_page_permutation",
            "restore",
            "zero_query_ties",
            "restore_after_ties",
        ]
    records, original_outputs, original_candidates = [], {}, None
    previous_candidates = None
    for phase_id, phase in enumerate(phases):
        if phase == "length_update":
            cursor = 0
            for request, count in enumerate(query_lens):
                prefix = request * 510
                values = [(prefix + j + 1) // ratio for j in range(count)]
                inputs.lengths[cursor : cursor + count].copy_(
                    torch.tensor(values, dtype=torch.int32, device="cuda")
                )
                cursor += count
        elif phase in ("physical_page_permutation", "all_zero_with_page_permutation"):
            order = torch.arange(inputs.cache.shape[0] - 1, -1, -1, device="cuda")
            old_cache, old_table = inputs.cache.clone(), inputs.table.clone()
            inputs.cache[order] = old_cache
            inputs.table.copy_(order[old_table.long()].int())
            if phase.startswith("all_zero"):
                inputs.lengths.zero_()
        elif phase == "request_reorder":
            assert len(query_lens) == 2 and query_lens[0] == query_lens[1]
            n = query_lens[0]
            order = torch.cat(
                (torch.arange(n, 2 * n, device="cuda"), torch.arange(n, device="cuda"))
            )
            inputs.q.view(torch.uint8).copy_(
                inputs.q.view(torch.uint8).index_select(0, order)
            )
            inputs.weights.copy_(inputs.weights.index_select(0, order))
            inputs.lengths.copy_(inputs.lengths.index_select(0, order))
            inputs.table.copy_(inputs.table.flip(0))
        elif phase.startswith("restore"):
            for key, value in baseline.items():
                getattr(inputs, key).copy_(value)
        elif phase == "zero_query_ties":
            inputs.q.view(torch.uint8).zero_()
        publish(0 if phase.startswith("restore") else phase_id)
        changed_candidates = None
        if candidates is not None:
            changed_candidates = (
                int((candidates.sort(1).values != previous_candidates).any(1).sum())
                if previous_candidates is not None
                else None
            )
            if phase == "candidate_update":
                # Above 16K keys, this fixture must actually change row sets.
                assert changed_candidates > 0
            previous_candidates = candidates.clone().sort(1).values
            if phase == "base":
                original_candidates = candidates.sort(1).values
            elif phase.startswith("restore"):
                assert torch.equal(candidates.sort(1).values, original_candidates)
        snapshots = {}
        # Poison every captured path before replay. No eager call or oracle may
        # repair preparation for a path whose captured output is not saved yet.
        for path in paths.values():
            path.op.keys.view(torch.uint8).fill_(127)
            path.op.scales.fill_(float("nan"))
            path.op.request_lengths.zero_()
            if path.op.scores is not None:
                path.op.scores.fill_(float("nan"))
            path.op.output.fill_(-2)
            if path.op.source:
                path.op.candidates.fill_(-2)
            if path.ant:
                for plan in path.op.ant_plans.values():
                    plan.prefix.zero_()
        for name, path in paths.items():
            graphs[name].replay()
            snapshots[("graph", name)] = (
                path.op.output.sort(1).values,
                path.op.candidates.clone() if path.op.source else None,
            )
        torch.cuda.synchronize()
        for name, path in paths.items():
            path.forward()
            snapshots[("eager", name)] = (
                path.op.output.sort(1).values,
                path.op.candidates.clone() if path.op.source else None,
            )
        torch.cuda.synchronize()
        for execution in ("graph", "eager"):
            for name, path in paths.items():
                output, published = snapshots[(execution, name)]
                check = assert_outputs(
                    path, output, published, inputs, paths["deepgemm"]
                )
                assert torch.equal(output, snapshots[("eager", name)][0])
                if phase == "base":
                    original_outputs[(execution, name)] = output
                elif phase.startswith("restore"):
                    assert torch.equal(output, original_outputs[(execution, name)])
                if phase == "all_zero_with_page_permutation":
                    assert (output == -1).all().item()
                    if published is not None:
                        assert (published == -1).all().item()
                records.append(
                    dict(
                        phase=phase,
                        execution=execution,
                        backend=name,
                        candidate_changed_rows=changed_candidates,
                        **check,
                    )
                )
        if phase in ("base", "all_zero_with_page_permutation", "zero_query_ties"):
            fixed_logits(paths, inputs)
    return dict(
        label=label,
        role=role,
        ratio=ratio,
        query_lens=query_lens,
        contexts=contexts,
        phases=phases,
        checks=records,
        fixed_logits_exact=True,
        source_reference_scope="every query row",
        candidate_layout="per-query [Q,2048], source generated outside consumer call",
        request_reorder="swap full equal-size groups then normalize request IDs",
        graph_contract="fixed pointers, descriptor sizes, request groups and chunks",
        graph_validation=(
            "poison all prepared K/scales, prefix and output; save all graph results "
            "before any eager/oracle preparation"
        ),
        physical_permutation=(
            "cache storage and logical-to-physical table changed together"
        ),
        scope="module correctness; no timing or model accuracy claim",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=["dynamic", "cutoff", "all"], default="all")
    args = parser.parse_args()
    check_vllm()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(
        status="running",
        args=vars(args),
        identity_before=identity(),
        device_before=device_info(),
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_logic=logical_checks(),
        cases=[],
    )
    save(args.output, report)
    try:
        fixtures = []
        if args.case in ("dynamic", "all"):
            fixtures += [
                ("ratio2_dynamic", "full", [8, 8], [1026, 1025], True),
                ("source_dynamic", "source", [8, 8], [16400, 16409], True),
                ("consumer_dynamic", "consumer", [8, 8], [16400, 16409], True),
            ]
        if args.case in ("cutoff", "all"):
            fixtures += [
                ("candidate_cutoff", role, [16], [16396], False)
                for role in ("source", "consumer")
            ]
        for label, role, queries, contexts, dynamic in fixtures:
            report["cases"].append(
                run_fixture(label, role, queries, contexts, dynamic=dynamic)
            )
            save(args.output, report)
            print(json.dumps(dict(completed=label, role=role)), flush=True)
        report["identity_after"] = identity()
        report["device_after"] = device_info()
        assert report["identity_before"] == report["identity_after"]
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error), identity_after=identity())
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Execute real final-manifest decisions, including CUDA Graph and TP4 gathers."""

import argparse
import hashlib
import json
from pathlib import Path

from vllm_dsv41_indexer.dispatch import Dispatcher, DispatchKey

from .benchmark_final import runtime_sources
from .common import check_vllm, device_info, save


def verify_single(key, decision):
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.grouped import GroupedConsumer, GroupedIndexer
    from vllm_dsv41_indexer.module import (
        FullIndexer,
        IndexerInputs,
        NativeSourceIndexer,
    )
    from vllm_dsv41_indexer.prefill import PrefillIndexer
    from vllm_dsv41_indexer.reference import source_reference, stable_topk
    from vllm_dsv41_indexer.topk import StableTopK
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import make_inputs
    from .verify_prefill import make_case
    from .verify_topk import expected, make_scores

    name = decision.strategy
    source_want = None
    if key.role == "chain":
        return verify_chain(key, decision)
    if key.phase == "selector":
        scores = make_scores(key.rows, key.index_keys, "cutoff_ties")
        lengths = torch.full(
            (key.rows,), key.index_keys, device="cuda", dtype=torch.int32
        )
        op = StableTopK(
            scores,
            lengths,
            k=key.topk,
            backend="streaming" if name == "streaming" else "cluster",
        )
        fn = op.forward if name == "cluster_replan_each_call" else op.execute
        want = expected(scores, lengths, key.topk)
    elif key.phase == "prefill":
        inputs, blocks = make_case(key.rows, key.requests, key.index_keys, key.ratio)
        op = PrefillIndexer(
            inputs,
            backend=name.removesuffix("_pipeline"),
            source=key.role == "source",
            candidate_blocks=blocks if key.role == "consumer" else None,
        )
        fn = op.forward
        fn()
        refs, positions, source_chunks = [], [], []
        for start, end, request in op.chunks:
            scores = op.score_chunk(start, end, request)
            if op.source:
                source_chunks.append(source_reference(scores, op.lengths[start:end]))
            for local, n in enumerate(op.lengths[start:end].tolist()):
                refs.append(scores[local, :n].clone())
                positions.append(
                    (
                        op.state.blocks[request].long()[:, None] * 8
                        + torch.arange(8, device="cuda")
                    ).flatten()[:n]
                    if op.compact
                    else torch.arange(n, device="cuda")
                )
        want = stable_topk(refs, positions)
        if source_chunks:
            source_want = torch.cat(source_chunks)
    else:
        group = max(key.group_rows, 1)
        q, _, w, cache, table, requests, lengths, blocks = make_inputs(
            dict(
                rows=key.rows,
                compressed_length=key.index_keys,
                rows_per_request=group,
                page_size=64,
                candidate_blocks=2048,
            )
        )
        inputs = IndexerInputs(
            q,
            w,
            cache,
            table,
            requests,
            lengths,
            IndexerShape(key.rows, key.index_keys, key.ratio),
        )
        if key.role == "consumer":
            state = CandidateState(key.rows, q.device, compress_ratio=key.ratio)
            if name == "shared_consumer_publish_included":
                for start in range(0, key.rows, group):
                    blocks[start : start + group].copy_(
                        blocks[start + group - 1].expand(group, -1).clone()
                    )
            state.publish(blocks, lengths, source_unique=True)
            if name == "shared_consumer_publish_included":
                op = GroupedConsumer(
                    state,
                    q,
                    w,
                    cache,
                    table,
                    requests,
                    group_rows=1 << (group - 1).bit_length(),
                    **(decision.parameters or {"paired": True}),
                )
                assert op.shared_enabled or group == 1
            elif name in (
                "one_sort_ant_topk_fused_map",
                "one_sort_ant_topk_separate_map",
            ):
                op = CompactIndexer(
                    state,
                    q,
                    w,
                    cache,
                    table,
                    requests,
                    fused_mapping=name.endswith("fused_map"),
                )
            else:
                raise ValueError(f"Unknown compact implementation: {name}")

            def fn():
                state.publish(blocks, lengths, source_unique=True)
                return op.forward()

            fn()
            all_positions = (
                state.blocks.long()[..., None] * 8 + torch.arange(8, device="cuda")
            ).flatten(1)
            want = stable_topk(
                [op.scores[r, :n].clone() for r, n in enumerate(state.counts.tolist())],
                [all_positions[r, :n] for r, n in enumerate(state.counts.tolist())],
            )
        elif key.role in ("full", "source"):
            if name == "native_score_ant_token_and_source_topk":
                op = NativeSourceIndexer(inputs, next_n=group)
            elif name.startswith("grouped_gluon"):
                op = GroupedIndexer(
                    inputs,
                    group_rows=1 << (group - 1).bit_length(),
                    source=key.role == "source",
                    **(decision.parameters or {"paired": True}),
                )
            elif name in (
                "optimized_full",
                "grouped_source_fused_max",
                "grouped_source_separate_max",
            ):
                op = FullIndexer(
                    inputs,
                    group_rows=1 << (group - 1).bit_length(),
                    source=key.role == "source",
                    fuse_source_max=name != "grouped_source_separate_max",
                )
            else:
                raise ValueError(f"Unknown full/source implementation: {name}")
            fn = op.forward
            fn()
            if key.role == "source":
                source_want = source_reference(op.scores, lengths)
            want = stable_topk(
                [op.scores[r, :n].clone() for r, n in enumerate(lengths.tolist())],
                [torch.arange(n, device="cuda") for n in lengths.tolist()],
            )
        else:
            raise ValueError(
                "Chain composition is verified by the P3 complete-chain artifact"
            )
    bound = decision.bind({name: fn})
    actual = bound()
    torch.testing.assert_close(
        actual.sort(-1).values, want.sort(-1).values, rtol=0, atol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bound()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        op.output.sort(-1).values, want.sort(-1).values, rtol=0, atol=0
    )
    if source_want is not None:
        torch.testing.assert_close(
            op.candidates.sort(-1).values, source_want.sort(-1).values, rtol=0, atol=0
        )
    return dict(
        strategy=name,
        fixed_logits_topk="exact",
        eager_and_three_graph_replays="passed",
        source_candidates="exact_forced_latest" if source_want is not None else None,
    )


def verify_chain(key, decision):
    import torch

    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
    )
    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.module import (
        FullIndexer,
        IndexerInputs,
        NativeFullIndexer,
        NativeSourceIndexer,
    )
    from vllm_dsv41_indexer.reference import source_reference, stable_topk
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import make_inputs

    name = decision.strategy
    if name not in (
        "hybrid_source_four_consumers",
        "hybrid_source_full_consumers",
        "optimized_source_four_consumers",
    ):
        raise ValueError(f"Unknown chain implementation: {name}")
    q, _, w, cache, table, req, lens, _ = make_inputs(
        dict(
            rows=key.rows,
            compressed_length=key.index_keys,
            rows_per_request=key.group_rows,
            page_size=64,
            candidate_blocks=2048,
        )
    )
    inputs = IndexerInputs(
        q, w, cache, table, req, lens, IndexerShape(key.rows, key.index_keys, key.ratio)
    )
    source = (
        FullIndexer(inputs, source=True)
        if name == "optimized_source_four_consumers"
        else NativeSourceIndexer(inputs, next_n=key.group_rows)
    )
    state = CandidateState(key.rows, q.device, compress_ratio=key.ratio)
    consumers = []
    full = name == "hybrid_source_full_consumers"
    for layer in range(4):
        cq, cw = (
            q.float().roll(layer * 7, -1).to(q.dtype),
            w.roll(layer * 3, -1).contiguous(),
        )
        consumer = (
            NativeFullIndexer(
                IndexerInputs(cq, cw, cache, table, req, lens, inputs.shape),
                next_n=key.group_rows,
            )
            if full
            else CompactIndexer(state, cq, cw, cache, table, req)
        )
        consumers.append(consumer)

    def forward():
        source.forward()
        if not full:
            state.publish(source.candidates, lens, source_unique=True)
        for consumer in consumers:
            if full:
                consumer.score()
                apply_candidate_mask(
                    consumer.scores, None, lens, source.candidates, 8, 1
                )
                source.topk.topk(
                    consumer.scores,
                    lens,
                    None,
                    consumer.output,
                    1,
                    source.source_plan,
                    None,
                )
            else:
                consumer.forward()

    bound = decision.bind({name: forward})
    bound()
    source_want = source_reference(source.scores, lens)
    expected = []
    for op in [source, *consumers]:
        lengths = lens if op is source or full else state.counts
        mapping = (
            None
            if op is source or full
            else (
                state.blocks.long()[..., None] * 8 + torch.arange(8, device="cuda")
            ).flatten(1)
        )
        values, positions = [], []
        for row, n in enumerate(lengths.tolist()):
            score = op.scores[row, :n]
            pos = (
                torch.arange(n, device="cuda") if mapping is None else mapping[row, :n]
            )
            finite = torch.isfinite(score)
            values.append(score[finite].clone())
            positions.append(pos[finite])
        expected.append(stable_topk(values, positions))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bound()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for op, want in zip([source, *consumers], expected):
        torch.testing.assert_close(
            op.output.sort(-1).values, want.sort(-1).values, rtol=0, atol=0
        )
    torch.testing.assert_close(
        source.candidates.sort(-1).values, source_want.sort(-1).values, rtol=0, atol=0
    )
    return dict(
        strategy=name,
        modules=5,
        fixed_logits_topk="exact",
        eager_and_three_graph_replays="passed",
    )


def verify_tp4(key, decision):
    import torch

    from .tp4_cases import build_variants, canonical, make_case

    workload = {"consumer": "compact", "full": "full", "chain": "chain"}[key.role]
    fields, factory, _, _ = make_case(
        key.requests,
        key.group_rows,
        key.index_keys * key.ratio,
        key.ratio,
        workload=workload,
        projection=key.projection,
        full_backend="native_stable"
        if "native" in key.local_backend.lower()
        else "portable",
    )
    variants = build_variants(
        fields,
        factory,
        input_modes=(key.input_mode,),
        split="split_rows" in decision.strategy,
    )
    reference = canonical(variants["replicated"].forward()).clone()
    callables = {"tp4:" + name: op.forward for name, op in variants.items()}
    bound = decision.bind(callables)
    assert torch.equal(canonical(bound()), reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bound()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    chosen = variants[decision.strategy.removeprefix("tp4:")]
    assert torch.equal(canonical(chosen.output), reference)
    return dict(
        strategy=decision.strategy,
        physical_world=4,
        eager_and_three_graph_replays="exact_replicated_topk_set",
    )


def verify_native_identity(environment, override):
    """Check actual loaded-package files, separately for each process environment."""
    from importlib.metadata import distribution

    package = environment.get("performance_native_package", "installed")
    installed = Path(distribution("vllm").locate_file("vllm/third_party/deep_gemm"))
    actual = Path(override) if override else installed
    expected = installed if package == "installed" else Path(package)
    if actual.resolve() != expected.resolve():
        raise RuntimeError(
            "Native package differs; run each environment in a separate process"
        )
    identity = environment["native_identity_contents"]
    if "identity" in identity:
        identity = identity["identity"]
    files = identity.get("files", identity.get("native_files", []))
    if not files and identity.get("patched_deep_gemm"):
        files = [
            dict(path=p, sha256=s)
            for p, s in identity["patched_deep_gemm"]["source_sha256"].items()
        ]
    if not files:
        raise RuntimeError("Native artifact contains no verifiable file hashes")
    checked = []
    for item in files:
        path = Path(item["path"])
        path = path if path.is_absolute() else actual / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Actual native dependency changed: {path}")
        checked.append(dict(path=str(path), sha256=item["sha256"]))
    import torch

    from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init

    _lazy_init()
    imported = _import_deep_gemm()
    module_files = [Path(imported.__file__), Path(imported._C.__file__)]
    binaries = list(Path(imported._C.__file__).parent.glob("_C_extension*.so"))
    if len(binaries) != 1 or str(binaries[0]) not in torch.ops.loaded_libraries:
        raise RuntimeError("Cannot identify the actually loaded DeepGEMM binary")
    known = {item["sha256"] for item in checked}
    loaded = []
    for path in [*module_files, *binaries]:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".so" and sha not in known:
            raise RuntimeError("Loaded DeepGEMM binary is outside recorded identity")
        loaded.append(dict(path=str(path), sha256=sha))
    return dict(package=str(actual), files=checked, actually_loaded=loaded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=("single", "tp4"), default="single")
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--deep-gemm-package")
    parser.add_argument("--environment-digest", required=True)
    args = parser.parse_args()
    if args.scope == "tp4":
        from .tp4_cases import initialize

        initialize()
    else:
        check_vllm()
    if args.deep_gemm_package:
        from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init

        _lazy_init()
        _import_deep_gemm()._C.init(args.deep_gemm_package)
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(__file__).resolve().parents[3]
    digests = {
        s: runtime_sources(root, parallel=s == "tp4")["digest"]
        for s in ("single", "tp4")
    }
    records = [
        r
        for r in manifest["records"]
        if r.get("source_scope", "single") == args.scope
        and r["environment_digest"] == args.environment_digest
    ]
    if not records:
        raise RuntimeError("No admitted actual measurements for this execution scope")
    environment = manifest["environments"][args.environment_digest]
    actual_native = verify_native_identity(environment, args.deep_gemm_package)
    planners = {
        env: Dispatcher(source_digest=digests, environment_digest=env, records=records)
        for env in {r["environment_digest"] for r in records}
    }
    report = dict(
        status="running",
        device=device_info(),
        checks=[],
        skipped=[],
        source_digests=digests,
        actual_native_identity=actual_native,
    )
    seen = set()
    try:
        for record in records:
            key = DispatchKey(**record["key"])
            decision = planners[record["environment_digest"]].choose(key)
            tag = (
                key.identity
                if args.all_shapes
                else (
                    key.phase,
                    key.role,
                    decision.strategy,
                    key.projection,
                    key.input_mode,
                )
            )
            if tag in seen:
                continue
            seen.add(tag)
            if not decision.evidence:
                raise RuntimeError(
                    "Admitted evidence is stale; refuse synthetic fallback validation"
                )
            measured_device = manifest["environments"][record["environment_digest"]][
                "device"
            ]
            for field in ("name", "sm_count", "torch", "cuda"):
                if measured_device[field] != report["device"][field]:
                    raise RuntimeError(f"Runtime environment changed: {field}")
            check = (
                verify_tp4(key, decision)
                if args.scope == "tp4"
                else verify_single(key, decision)
            )
            report["checks"].append(dict(key=record["key"], **check))
            if args.scope == "single":
                save(args.output, report)
            print(json.dumps(report["checks"][-1]), flush=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        if args.scope == "single":
            save(args.output, report)
        else:
            import torch.distributed as dist

            if dist.get_rank() == 0:
                save(args.output, report)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

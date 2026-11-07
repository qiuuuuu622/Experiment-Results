"""CPU-only independent model-contract and memory audit; never imports CUDA."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DRIVER = ROOT / "benchmarks/kernels/dsv41_indexer/benchmark_prefill_model.py"
ARCH = OUT.parent / "p5-ant-keys-major/model-aligned/architecture"


def main():
    config = json.loads((ARCH / "model-config.json").read_text())["text_config"]
    expected = {
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "max_position_embeddings": 1048576,
    }
    assert {k: config[k] for k in expected} == expected
    tree = ast.parse(DRIVER.read_text())
    cases = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CASES" for t in node.targets)
    )
    records, excluded = [], []
    for role, ratio, layers in (
        ("full", 2, [2, 8, 14]),
        ("source", 1, [20]),
        ("consumer", 1, [24, 28, 32, 36]),
    ):
        assert all(config["compress_ratios"][layer] == ratio for layer in layers)
        assert set(layers) <= set(config["index_source_layer_ids"])
        for label, queries, contexts in cases:
            if role == "full" and label == "initial_32k":
                # Matches the explicit filter in the reviewed driver main().
                excluded.append(
                    dict(
                        label=label,
                        role=role,
                        reason=(
                            "scope_not_memory: extra initial32K targets ratio1 "
                            "source/consumer candidate cutoff"
                        ),
                    )
                )
                continue
            assert len(queries) == len(contexts)
            width = ((max(contexts) // ratio + 63) // 64) * 64
            capacity = (1 << 30) // (4 * width)
            lengths, chunks = [], []
            cursor = 0
            for request, (q, context) in enumerate(zip(queries, contexts)):
                assert 0 < q <= context <= expected["max_position_embeddings"]
                prefix = context - q
                visible = [(prefix + j + 1) // ratio for j in range(q)]
                # Count fully completed compressed groups independently.
                for j in (0, q // 2, q - 1):
                    assert visible[j] * ratio <= prefix + j + 1
                    assert (visible[j] + 1) * ratio > prefix + j + 1
                lengths.extend(visible)
                for start in range(cursor, cursor + q, capacity):
                    chunks.append([start, min(cursor + q, start + capacity), request])
                cursor += q
            max_chunk = max(end - start for start, end, _ in chunks)
            arena = max_chunk * width * 4
            assert arena <= 1 << 30
            counts = [(n + 7) // 8 for n in lengths]
            records.append(
                dict(
                    label=label,
                    role=role,
                    ratio=ratio,
                    layers=layers,
                    query_rows=sum(queries),
                    query_lens=queries,
                    contexts=contexts,
                    prefixes=[c - q for c, q in zip(contexts, queries)],
                    first_last_visible=[[lengths[0], lengths[-1]]],
                    zero_visible_rows=lengths.count(0),
                    rows_at_most_token_topk=sum(n <= 512 for n in lengths),
                    effective_score_count=sum(lengths),
                    rectangular_score_count=sum(queries) * width,
                    score_arena_bytes=arena,
                    chunks=chunks,
                    unfiltered_candidate_rows=sum(n <= 2048 for n in counts),
                    truncated_candidate_rows=sum(n > 2048 for n in counts),
                    eager_whole_batch_bypass=max(contexts) // ratio <= 512,
                    candidate_semantics_relevant=role != "full",
                )
            )
    boundaries = []
    for ratio, contexts in ((1, [511, 512, 513]), (2, [1023, 1024, 1025, 1026])):
        for context in contexts:
            for capturing in (False, True):
                boundaries.append(
                    dict(
                        ratio=ratio,
                        max_context=context,
                        capturing=capturing,
                        max_index_keys=context // ratio,
                        model_bypass=context // ratio <= 512 and not capturing,
                    )
                )
    sources = [
        DRIVER,
        ROOT / "vllm_dsv41_indexer/prefill.py",
        ROOT / "vllm_dsv41_indexer/prefill_keys_major_ant.py",
        ROOT / "benchmarks/kernels/dsv41_indexer/benchmark_prefill_ant.py",
        ROOT / "benchmarks/kernels/dsv41_indexer/common.py",
        *sorted(ARCH.iterdir()),
    ]
    payload = dict(
        status="CPU_contract_audit_passed_GPU_execution_not_performed",
        expected_config=expected,
        shape_count=len(records),
        declared_shape_count=len(cases) * 3,
        excluded=excluded,
        cases=records,
        bypass_boundaries=boundaries,
        coverage_gaps={
            "initial_prefill_crosses_candidate_cap": any(
                r["role"] == "source"
                and not any(r["prefixes"])
                and r["unfiltered_candidate_rows"]
                and r["truncated_candidate_rows"]
                for r in records
            ),
        },
        sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
            if p.is_file()
        },
    )
    (OUT / "CPU_CONTRACT_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "shape_count": len(records),
                "coverage_gaps": payload["coverage_gaps"],
            }
        )
    )


if __name__ == "__main__":
    main()

"""CPU-only architecture shape audit; does not import Torch or launch CUDA."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DRIVER = ROOT / "benchmarks/kernels/dsv41_indexer/benchmark_prefill_model.py"
CONFIG = ROOT / "results/indexer/20261101/model-config.json"


def main():
    config = json.loads(CONFIG.read_text())["text_config"]
    assert (config["index_n_heads"], config["index_head_dim"], config["index_topk"]) == (
        32, 128, 512
    )
    assert config["kv_source_layer_ids"] == [2, 8, 14, 20]
    assert config["index_source_layer_ids"] == [2, 8, 14, 20, 24, 28, 32, 36]
    assert config["candidate_source_layer_id"] == 20
    assert (config["candidate_topk_blocks"], config["candidate_block_size"]) == (
        2048, 8
    )
    module = ast.parse(DRIVER.read_text())
    cases = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CASES" for t in node.targets)
    )
    result = []
    for role, ratio, layers in [
        ("full", 2, [2, 8, 14]),
        ("source", 1, [20]),
        ("consumer", 1, [24, 28, 32, 36]),
    ]:
        assert all(config["compress_ratios"][i] == ratio for i in layers)
        for name, queries, contexts in cases:
            if role == "full" and name == "initial_32k":
                continue  # Matches the frozen model driver's explicit exclusion.
            assert len(queries) == len(contexts)
            assert sum(queries) % len(queries) == 0  # Current make_case prerequisite.
            width = ((max(contexts) // ratio + 63) // 64) * 64
            capacity = (1 << 30) // (4 * width)
            spans = []
            visible_total = 0
            for q, end in zip(queries, contexts):
                assert 0 < q <= end <= 1048576
                prefix = end - q
                visible = [(prefix + j + 1) // ratio for j in range(q)]
                assert visible[-1] == end // ratio
                assert all(v * ratio <= prefix + j + 1 for j, v in enumerate(visible))
                visible_total += sum(visible)
                spans.append(dict(
                    query_rows=q, final_original_tokens=end, prefix=prefix,
                    closed_index_keys=end // ratio,
                    first_eight=visible[:8], last_eight=visible[-8:],
                    score_chunks=(q + capacity - 1) // capacity,
                ))
            arena_rows = min(max(queries), capacity)
            result.append(dict(
                label=name, role=role, ratio=ratio, layers=layers,
                requests=spans, total_rows=sum(queries), padded_width=width,
                visible_score_count=visible_total,
                unchunked_score_bytes=4 * sum(queries) * width,
                arena_score_bytes=4 * arena_rows * width,
                arena_max_rows=capacity,
                allocated_cache_or_one_prepared_pool_bytes=132 * len(queries) * width,
                query_bytes=4096 * sum(queries),
                perquery_candidate_bytes=8192 * sum(queries) if role != "full" else 0,
                perquery_output_bytes=2048 * sum(queries),
                source_maxima_bytes=4 * arena_rows * (width // 8)
                if role == "source" else 0,
                allquery_mask_flags_bytes=sum(queries) * (width // 8 + 1)
                if role == "consumer" else 0,
            ))
    examples = []
    for prefix in [0, 1, 2, 1019, 1020, 32763]:
        values = [(prefix + j + 1) // 2 for j in range(5)]
        examples.append(dict(prefix=prefix, query_rows=5, ratio=2, visible=values))
    sources = [
        CONFIG, DRIVER,
        ROOT / "vllm/models/deepseek_v41/attention.py",
        ROOT / "vllm/models/deepseek_v41/nvidia/model.py",
        ROOT / "vllm/v1/attention/backends/mla/indexer.py",
        ROOT / "vllm/model_executor/layers/sparse_attn_indexer.py",
        ROOT / "vllm/model_executor/kernels/attention/dsa/candidate_blocks.py",
        ROOT / "vllm_dsv41_indexer/prefill.py",
        ROOT / "vllm_dsv41_indexer/prefill_keys_major_ant.py",
        ROOT / "benchmarks/kernels/dsv41_indexer/inputs.py",
        ROOT / "benchmarks/kernels/dsv41_indexer/verify_prefill.py",
    ]
    manifest = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    output = dict(
        scope="CPU arithmetic/configuration audit; no GPU results",
        explicit_exclusions=["initial_32k/full: added case targets candidate cutoff"],
        baseline_head="29be941d8d5746cd98135bdd5e710eeebd8d7014",
        source_hashes=manifest, examples=examples, cases=result,
    )
    (OUT / "shape-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    (OUT / "benchmark_prefill_model.reviewed.py").write_bytes(DRIVER.read_bytes())
    print(json.dumps(dict(status="passed", shapes=len(result), source_hashes=manifest)))


if __name__ == "__main__":
    main()

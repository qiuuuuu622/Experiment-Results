"""Same-object forward/reverse pipeline replication; frozen kernels unchanged."""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    blob = (
        tensor.detach()
        .contiguous()
        .view(__import__("torch").uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    return dict(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        sha256=hashlib.sha256(blob).hexdigest(),
    )


def fingerprint(inputs, source, candidates):
    fields = {
        name: getattr(inputs, name)
        for name in ("q", "weights", "cache", "table", "requests", "lengths")
    }
    fields.update(
        source_q=source.inputs.q,
        source_weights=source.inputs.weights,
        candidate_raw=candidates,
        candidate_canonical=candidates.sort(1).values,
    )
    return {name: tensor_sha(tensor) for name, tensor in fields.items()}


def run_case(module, case, paired):
    import torch

    from vllm_dsv41_indexer.perquery_prefill import PerQueryPrefillIndexer

    label, queries, contexts = case
    inputs = module.model_inputs(queries, contexts, 1)
    source, candidates = module.source_fixture(inputs)
    before = fingerprint(inputs, source, candidates)
    expected = candidates.sort(1).values
    repeats = []
    for _ in range(3):
        source.forward()
        canonical = source.candidates.sort(1).values
        assert torch.equal(canonical, expected)
        repeats.append(
            dict(raw=tensor_sha(source.candidates), canonical=tensor_sha(canonical))
        )
    del expected, canonical
    records = []
    correctness = None
    object_identity = None
    if paired:
        full = {
            "full_deepgemm_mask": module.ModelPath(
                inputs, "deepgemm", "consumer", candidates
            ),
            "full_keys_major_mask": module.ModelPath(
                inputs, "keys_major", "consumer", candidates
            ),
        }
        compact = {
            name: PerQueryPrefillIndexer(
                inputs, candidates, backend=name, source_unique=True
            )
            for name in ("direct", "gathered_wgmma")
        }
        paths = {**full, **compact}
        for path in paths.values():
            path.forward()
        correctness = module.validate(inputs, candidates, compact, full)
        object_identity = {
            name: dict(
                python_id=id(path),
                candidate_data_ptr=candidates.data_ptr(),
                q_data_ptr=inputs.q.data_ptr(),
                cache_data_ptr=inputs.cache.data_ptr(),
            )
            for name, path in paths.items()
        }
        variants = list(paths)
        for order in ("forward", "reverse"):
            for graph in (False, True):
                for name in (
                    variants if order == "forward" else list(reversed(variants))
                ):
                    assert id(paths[name]) == object_identity[name]["python_id"]
                    measurement = module.sampled(paths[name].forward, graph)
                    records.append(
                        dict(
                            order=order,
                            execution="graph" if graph else "eager",
                            variant=name,
                            phase="pipeline",
                            measurement=measurement,
                        )
                    )
            # Fail immediately if either order mutated any input bytes.
            assert fingerprint(inputs, source, candidates) == before
    after = fingerprint(inputs, source, candidates)
    assert before == after
    counts = torch.where(
        candidates >= 0, (inputs.lengths[:, None] - candidates * 8).clamp(0, 8), 0
    ).sum(1)
    return dict(
        case=label,
        query_lens=queries,
        contexts=contexts,
        ratio=1,
        heads=32,
        head_dim=128,
        paired_pipeline=paired,
        input_before=before,
        input_after=after,
        input_bytes_unchanged=True,
        source_repeats=repeats,
        source_canonical_sets_exact=True,
        source_raw_order_changed=any(
            r["raw"]["sha256"] != before["candidate_raw"]["sha256"] for r in repeats
        ),
        candidate_positions_min=int(counts.min()),
        candidate_positions_max=int(counts.max()),
        candidate_positions_sum=int(counts.sum()),
        path_object_identity=object_identity,
        correctness=correctness,
        records=records,
        source_generation_in_pipeline=False,
        scope=(
            "Same inputs/candidates/paths for both orders in one process; "
            "complete pipelines only; source repeats precede timing."
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--group", choices=["a", "b"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen = Path(args.frozen_root).resolve()
    sys.path.insert(0, str(frozen))
    module = importlib.import_module(
        "benchmarks.kernels.dsv41_indexer.benchmark_prefill_perquery"
    )
    module.check_vllm()
    import torch

    core = importlib.import_module("vllm_dsv41_indexer.perquery_prefill")
    assert Path(module.__file__).resolve().is_relative_to(frozen)
    assert Path(core.__file__).resolve().is_relative_to(frozen)
    assert (
        file_sha(module.__file__)
        == "40f50488bfa7213ae3354340c08786e954f7403f2c8d81ebfcd4a2fa8feb96d4"
    )
    assert (
        file_sha(core.__file__)
        == "77a7ffbf34ebe4495e560612c9b6943b17214b5fd01a9cdfe613d5887dbe9b1c"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    paired = {"steady_64k", "steady_256k", "tail_1m", "ragged"}
    selected = (
        {
            "initial_2k",
            "initial_8k",
            "initial_32k",
            "steady_16k",
            "steady_64k",
            "steady_256k",
        }
        if args.group == "a"
        else {"tail_1m", "ragged"}
    )
    report = dict(
        status="running",
        args=vars(args),
        host="h20-4",
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device=module.device_info(),
        runner_sha256=file_sha(__file__),
        frozen_driver=dict(path=module.__file__, sha256=file_sha(module.__file__)),
        frozen_core=dict(path=core.__file__, sha256=file_sha(core.__file__)),
        identity_sha256=file_sha(args.identity),
        cases=[],
        old_raw_canonical_hash_reconstructed=False,
    )
    for case in module.CASES:
        if case[0] in selected:
            report["cases"].append(run_case(module, case, case[0] in paired))
            module.save(args.output, report)
            print(
                json.dumps(
                    dict(case=case[0], paired=case[0] in paired, status="completed")
                ),
                flush=True,
            )
    report["status"] = "completed"
    module.save(args.output, report)


if __name__ == "__main__":
    main()

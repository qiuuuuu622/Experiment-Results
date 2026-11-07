# SPDX-License-Identifier: Apache-2.0
"""Compare saved CPU intermediates without loading a model or using CUDA."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("Intermediate shapes and dtypes must match")
    # Large flattened CPU FP32 cosine reductions can themselves lose enough
    # precision to exceed one. Metrics are diagnostic FP64, not model compute.
    a, b = actual.double(), reference.double()
    diff = a - b
    rms = lambda x: x.square().mean().sqrt().item()
    reference_rms = rms(b)
    error_rms = rms(diff)
    row_energy = diff.square().flatten(1).sum(1)
    row_reference_energy = b.square().flatten(1).sum(1)
    worst = torch.topk(row_energy, min(5, len(row_energy))).indices
    cosine = (a.flatten().dot(b.flatten()) / (a.norm() * b.norm())).item()
    return {
        "shape": list(a.shape),
        "dtype": str(actual.dtype),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "bitwise_equal": bool(
            torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
        ),
        "changed_fraction": (actual != reference).float().mean().item(),
        "max_abs_error": diff.abs().max().item(),
        "reference_max_abs": b.abs().max().item(),
        "actual_max_abs": a.abs().max().item(),
        "reference_rms": reference_rms,
        "actual_rms": rms(a),
        "error_rms": error_rms,
        "relative_l2": error_rms / max(reference_rms, 1e-30),
        "cosine": cosine,
        "largest_error_rows": [
            {
                "row": index.item(),
                "relative_l2": (
                    row_energy[index] / row_reference_energy[index].clamp_min(1e-60)
                )
                .sqrt()
                .item(),
                "error_energy_fraction": (
                    row_energy[index] / row_energy.sum().clamp_min(1e-60)
                ).item(),
            }
            for index in worst
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = torch.load(args.reference, weights_only=True, map_location="cpu")
    actual = torch.load(args.actual, weights_only=True, map_location="cpu")
    if set(reference) != set(actual):
        raise ValueError(f"Different layer keys: {set(reference) ^ set(actual)}")
    order = (
        ["patch_embed"]
        + [f"block_{i}" for i in (0, 1, 4, 8, 16, 24, 31)]
        + ["final_norm", "aligner"]
    )
    if set(order) != set(reference):
        raise ValueError("Unexpected capture points")
    rows = []
    for name in order:
        metric = metrics(actual[name], reference[name])
        rows.append({"stage": name, **metric})
    result = {
        "status": "analysis_complete_not_numerical_acceptance",
        "reference": {"path": str(args.reference), "sha256": sha(args.reference)},
        "actual": {"path": str(args.actual), "sha256": sha(args.actual)},
        "analyzer_sha256": sha(__file__),
        "torch": torch.__version__,
        "metrics_dtype": "float64",
        "cuda_initialized": torch.cuda.is_initialized(),
        "rows": rows,
        "first_nonidentical_checkpoint": next(
            (x["stage"] for x in rows if not x["bitwise_equal"]), None
        ),
        "interpretation_limit": (
            "Intermediate differences combine newly introduced error and propagated "
            "earlier differences. They do not isolate an operation's local error. "
            "DP4 BF16 is a cross-mode reference, not an FP32 oracle."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print("stage relative_l2 cosine max_abs_error changed_fraction")
    for row in rows:
        print(
            row["stage"],
            *(
                f"{row[k]:.9g}"
                for k in ("relative_l2", "cosine", "max_abs_error", "changed_fraction")
            ),
        )


if __name__ == "__main__":
    main()

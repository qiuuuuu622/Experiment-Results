# SPDX-License-Identifier: Apache-2.0
"""CPU-only image decode and current-HEAD image preprocessing measurements."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(fn, warmup, samples, inner):
    for _ in range(warmup):
        fn()
    raw = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        raw.append((time.perf_counter_ns() - start) / inner / 1e6)
    return {
        "samples_ms": raw,
        "p50_ms": statistics.median(raw),
        "p90_ms": float(np.percentile(raw, 90)),
        "mean_ms": statistics.mean(raw),
        "coefficient_of_variation": statistics.pstdev(raw) / statistics.mean(raw),
        "warmup": warmup,
        "samples": samples,
        "inner": inner,
    }


def decode(path):
    with Image.open(path) as image:
        image.load()
        return image.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--inner", type=int, default=1)
    args = parser.parse_args()
    print("Starting exact-source CPU preprocessing benchmark", flush=True)
    default_threads = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    source = args.source_root / "vllm/models/deepseek_v41/common/mm_preprocess.py"
    spec = importlib.util.spec_from_file_location("vision_bench_preprocess", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config_path = args.model / "config.json"
    config = json.loads(config_path.read_text())["vision_config"]
    kwargs = {
        "patch_size": config["patch_size"],
        "downsample_ratio": config["downsample_ratio"],
        "max_n_token": config["max_image_tokens"],
        "min_pixels": config["min_pixels"],
        "max_wh_ratio": config.get("max_wh_ratio"),
    }
    paths = sorted(args.images.glob("[0-9][0-9]-*.png"))
    if len(paths) != 8:
        raise ValueError(f"Expected exactly eight numbered PNG inputs, found {paths}")
    decoded = [decode(path) for path in paths]
    observations = []
    for path, image in zip(paths, decoded, strict=True):
        patches, vh, vw, lh, lw = module.load_image(image, **kwargs)
        patch_count = vh * vw
        span_tokens = module.num_image_tokens(lh, lw)
        r = kwargs["downsample_ratio"]
        assert patches.device.type == "cpu" and patches.dtype == torch.bfloat16
        assert tuple(patches.shape) == (
            patch_count,
            3,
            kwargs["patch_size"],
            kwargs["patch_size"],
        )
        assert lh == math.ceil(vh / r) and lw == math.ceil(vw / r)
        assert span_tokens == lh * (lw + 1) + 2
        assert span_tokens <= kwargs["max_n_token"]
        assert torch.isfinite(patches).all()
        assert patches.min() >= -1 and patches.max() <= 1
        observations.append(
            {
                "filename": path.name,
                "sha256": digest(path),
                "image_size_wh": list(image.size),
                "vit_grid_hw": [vh, vw],
                "patch_count": patch_count,
                "llm_grid_hw": [lh, lw],
                "aligner_tokens": lh * lw,
                "span_tokens": span_tokens,
                "merge_grid_remainder_hw": [vh % r, vw % r],
                "patch_bytes": patches.numel() * patches.element_size(),
                "geometry_passed": True,
            }
        )

    protocol = (args.warmup, args.samples, args.inner)
    records = []
    for order in ("forward", "reverse"):
        indices = list(range(len(paths)))
        if order == "reverse":
            indices.reverse()
        for i in indices:
            image = decoded[i]
            funcs = {
                "png_decode": lambda i=i: decode(paths[i]),
                "original_preprocess_decoded_pil": (
                    lambda image=image: module.load_image(image, **kwargs)
                ),
                "decode_plus_original_preprocess": (
                    lambda i=i: module.load_image(decode(paths[i]), **kwargs)
                ),
            }
            names = list(funcs)
            if order == "reverse":
                names.reverse()
            for name in names:
                print(f"{order}: {paths[i].name}: {name}", flush=True)
                records.append(
                    {
                        "order": order,
                        "image": paths[i].name,
                        "stage": name,
                        **measure(funcs[name], *protocol),
                    }
                )
        funcs = {
            "batch_original_preprocess_decoded_pil": lambda: [
                module.load_image(image, **kwargs) for image in decoded
            ],
            "batch_decode_plus_original_preprocess": lambda: [
                module.load_image(decode(path), **kwargs) for path in paths
            ],
        }
        for name in list(funcs) if order == "forward" else reversed(funcs):
            print(f"{order}: all_eight: {name}", flush=True)
            records.append(
                {
                    "order": order,
                    "image": "all_eight_in_filename_order",
                    "stage": name,
                    **measure(funcs[name], *protocol),
                }
            )

    result = {
        "status": "completed",
        "scope": "CPU-only PNG decoding and exact current-HEAD load_image",
        "limitations": [
            "Warm filesystem page cache; no physical-disk cold-read claim",
            "No tokenizer, H2D, vision compute, communication or model accuracy",
            "Unchanged source loaded by spec; dependencies use the installed wheel",
            "CPU uses one call per sample by default, unlike GPU 10-call windows",
            "Original paths measured only; not an optimization speedup report",
        ],
        "source": {"path": str(source), "sha256": digest(source)},
        "driver": {"path": __file__, "sha256": digest(__file__)},
        "config": {"path": str(config_path), "sha256": digest(config_path)},
        "preprocess_params": kwargs,
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "pillow": PIL.__version__,
            "numpy": np.__version__,
            "torch_threads_default": default_threads,
            "torch_threads_used": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_initialized": torch.cuda.is_initialized(),
        },
        "command": sys.argv,
        "images": observations,
        "measurements": records,
    }
    if result["environment"]["cuda_initialized"]:
        raise RuntimeError("CPU-only benchmark unexpectedly initialized CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "records": len(records),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""CPU-only ordered thread-pool image preprocessing against native vLLM."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(path):
    with Image.open(path) as image:
        image.load()
        return image.copy()


def measure(fn, warmup, samples):
    for _ in range(warmup):
        fn()
    wall, cpu = [], []
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    for _ in range(samples):
        cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
        fn()
        wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        cpu.append((time.process_time_ns() - cpu_start) / 1e6)
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "samples_ms": wall,
        "cpu_samples_ms": cpu,
        "p50_ms": statistics.median(wall),
        "p90_ms": sorted(wall)[int(0.9 * len(wall)) - 1],
        "coefficient_of_variation": statistics.pstdev(wall) / statistics.mean(wall),
        "mean_cpu_cores_used": sum(cpu) / sum(wall),
        "voluntary_context_switches": end_usage.ru_nvcsw - start_usage.ru_nvcsw,
        "involuntary_context_switches": end_usage.ru_nivcsw - start_usage.ru_nivcsw,
        "max_rss_kib": end_usage.ru_maxrss,
        "warmup": warmup,
        "samples": samples,
        "inner": 1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/model"))
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    p.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4])
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument(
        "--stages",
        nargs="+",
        choices=["decoded", "decode"],
        default=["decoded", "decode"],
    )
    p.add_argument("--correctness-only", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from vllm_dsv41_vision.preprocess import ParallelImagePreprocessor

    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[-4:])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = args.source_root / "vllm/models/deepseek_v41/common/mm_preprocess.py"
    spec = importlib.util.spec_from_file_location("vision_parallel_preprocess", source)
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
        raise ValueError("Expected eight fixed image fixtures")
    images = [decode(path) for path in paths]
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    native_processor = module.DeepseekV4VLProcessor(
        DeepseekV41Config.from_pretrained(args.model)
    )
    pools, pool_init = {}, {}
    for workers in sorted(set(args.workers) - {0}):
        start = time.perf_counter_ns()
        pools[workers] = ParallelImagePreprocessor(
            native_processor.image_processor, module.image_token_types, workers
        )
        constructor_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        pools[workers](images[:workers])
        pool_init[str(workers)] = {
            "constructor_ms": constructor_ms,
            "first_call_with_images_ms": (time.perf_counter_ns() - start) / 1e6,
            "first_call_images": workers,
        }
    result = {
        "status": "running",
        "driver": {"path": __file__, "sha256": digest(__file__)},
        "source": {"path": str(source), "sha256": digest(source)},
        "adapter_sha256": digest(
            Path(__file__).resolve().parents[3] / "vllm_dsv41_vision/preprocess.py"
        ),
        "config_sha256": digest(config_path),
        "preprocess_params": kwargs,
        "images": [{"name": path.name, "sha256": digest(path)} for path in paths],
        "environment": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "pool_initialization": pool_init,
        "command": sys.argv,
        "correctness": [],
        "measurements": [],
        "limitations": [
            "CPU only: no H2D, vision tower, collectives, tokenizer or serving",
            "Warm filesystem cache; no cold disk-read claim",
            "Pinned to last four available logical CPUs, not exclusive CPU ownership",
            "GPU smoke/profile may execute concurrently; not formal GPU timing",
            "One call per CPU sample; thread pools initialized outside hot timing",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def call(count, workers, include_decode, reverse=False):
        inputs = paths[:count] if include_decode else images[:count]
        if reverse:
            inputs = list(reversed(inputs))

        if workers == 0:
            return native_processor(
                images=[decode(item) for item in inputs] if include_decode else inputs
            )
        return pools[workers](inputs, decode=decode if include_decode else None)

    try:
        for count in args.batch_sizes:
            for reverse in (False, True):
                ordered = images[:count]
                if reverse:
                    ordered = list(reversed(ordered))
                reference = native_processor(images=ordered)
                for decode_flag in (False, True):
                    for workers in args.workers:
                        actual = call(count, workers, decode_flag, reverse)
                        exact = {
                            key: torch.equal(actual[key], reference[key])
                            for key in reference
                        }
                        if not all(exact.values()):
                            raise ValueError(f"Native mismatch: {exact}")
                        result["correctness"].append(
                            {
                                "count": count,
                                "workers": workers,
                                "decode": decode_flag,
                                "reverse_images": reverse,
                                "exact_vs_native_processor": exact,
                            }
                        )
            print(f"B{count} exact native checks passed", flush=True)
            save()
            if args.correctness_only:
                continue
            for round_index in range(args.rounds):
                for order, worker_order in (
                    ("forward", args.workers),
                    ("reverse", list(reversed(args.workers))),
                ):
                    for decode_flag in [stage == "decode" for stage in args.stages]:
                        for workers in worker_order:
                            print(
                                f"B{count} round={round_index} {order} "
                                f"decode={decode_flag} workers={workers}",
                                flush=True,
                            )
                            metric = measure(
                                partial(call, count, workers, decode_flag),
                                args.warmup,
                                args.samples,
                            )
                            metric["extra_rounds"] = []
                            if metric["coefficient_of_variation"] > 0.1:
                                print("CV > 10%: adding three rounds", flush=True)
                                metric["extra_rounds"] = [
                                    measure(
                                        partial(call, count, workers, decode_flag),
                                        args.warmup,
                                        args.samples,
                                    )
                                    for _ in range(3)
                                ]
                            result["measurements"].append(
                                {
                                    "count": count,
                                    "workers": workers,
                                    "decode": decode_flag,
                                    "order": order,
                                    "round": round_index,
                                    **metric,
                                }
                            )
                            save()
                            print(f"P50={metric['p50_ms']:.3f}ms", flush=True)
        result["environment"]["cuda_initialized"] = torch.cuda.is_initialized()
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU benchmark unexpectedly initialized CUDA")
        result["status"] = "completed"
    except Exception:
        result["status"] = "failed"
        raise
    finally:
        save()
        for pool in pools.values():
            pool.close()


if __name__ == "__main__":
    main()

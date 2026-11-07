# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint vision-only experiment. Never constructs the language model."""

import argparse
import contextlib
import faulthandler
import functools
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from safetensors import safe_open


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(name, root):
    path = root / (name.replace(".", "/") + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model")
    p.add_argument("--images", required=True)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--mode", choices=["weights", "data"], default="data")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--variants", nargs="+", default=["eager", "packed", "manager"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path)
    p.add_argument("--reference-label", default="single_gpu")
    p.add_argument("--save-reference", action="store_true")
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--profile-dir", type=Path)
    p.add_argument("--profile-end-to-end", action="store_true")
    p.add_argument("--disable-custom-all-reduce", action="store_true")
    p.add_argument("--layer-dump-dir", type=Path)
    p.add_argument("--verify-weight-shards", action="store_true")
    p.add_argument("--debug-stacks", action="store_true")
    p.add_argument(
        "--rmsnorm-backend", choices=["single", "torch_mean"], default="single"
    )
    p.add_argument(
        "--graph-budget-policy",
        choices=["global", "local", "local-exact"],
        default="local",
    )
    return p.parse_args()


def tensor_metrics(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {reference.shape}")
    a, b = actual.double(), reference.double()
    error = a - b
    return {
        "finite": bool(torch.isfinite(a).all()),
        "max_abs": error.abs().max().item(),
        "relative_l2": (error.norm() / b.norm().clamp_min(1e-30)).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item(),
        "numeric_exact": bool(torch.equal(actual, reference)),
        "exact": actual.dtype == reference.dtype
        and bool(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                reference.contiguous().view(torch.uint8),
            )
        ),
        "allclose_1e-2_1e-2": bool(torch.allclose(a, b, rtol=1e-2, atol=1e-2)),
    }


def compare_outputs(actual, reference):
    if len(actual) != len(reference):
        raise ValueError("Incorrect image count")
    return [tensor_metrics(a, b) for a, b in zip(actual, reference)]


def gather_metrics(value, world):
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def numerical_gate(metrics):
    return all(
        m["finite"] and m["cosine"] >= 0.999 and m["relative_l2"] <= 0.02
        for m in metrics
    )


def loaded_manifest():
    files = {
        m.__file__
        for name, m in list(sys.modules.items())
        if name.startswith("vllm")
        and getattr(m, "__file__", None)
        and Path(m.__file__).is_file()
    }
    files.update(p for p in torch.ops.loaded_libraries if Path(p).is_file())
    return {p: sha(p) for p in sorted(files)}


class RotaryCall:
    def __init__(self, vision_module, fn):
        self.vision_module, self.fn = vision_module, fn

    def __call__(self):
        from vllm_dsv41_vision.rotary import fused_rotary

        with fused_rotary(self.vision_module):
            return self.fn()


class RMSNormCall:
    def __init__(self, vision_module, fn, backend):
        self.vision_module, self.fn, self.backend = vision_module, fn, backend

    def __call__(self):
        from vllm_dsv41_vision.rmsnorm import fused_rmsnorm

        with fused_rmsnorm(self.vision_module, backend=self.backend):
            return self.fn()


class CachedManagerCall:
    def __init__(self, model, manager, kwargs, cache):
        self.model, self.manager, self.kwargs, self.cache = (
            model,
            manager,
            kwargs,
            cache,
        )

    def __call__(self):
        from vllm_dsv41_vision.replay_metadata import cached_replay_metadata

        with cached_replay_metadata(self.model, self.cache):
            return self.manager.execute(self.kwargs)


def measure(fn, args, world):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    raw, wall, local_raw, local_wall = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.samples):
        start = [torch.cuda.Event(enable_timing=True) for _ in range(args.inner)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(args.inner)]
        before = time.perf_counter()
        for i in range(args.inner):
            start[i].record()
            fn()
            end[i].record()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - before) * 1e6 / args.inner
        values = torch.tensor(
            [s.elapsed_time(e) * 1000 for s, e in zip(start, end)],
            device="cuda",
            dtype=torch.float64,
        )
        wall_value = torch.tensor(elapsed, device="cuda", dtype=torch.float64)
        local_raw.append(values.mean().item())
        local_wall.append(elapsed)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        dist.all_reduce(wall_value, op=dist.ReduceOp.MAX)
        raw.append(values.mean().item())
        wall.append(wall_value.item())
    return {
        "samples_us": raw,
        "per_rank_samples": gather_metrics(
            {"gpu_us": local_raw, "wall_us": local_wall}, world
        ),
        "wall_samples_us": wall,
        "p50_us": statistics.median(raw),
        "p90_us": sorted(raw)[math.ceil(0.9 * len(raw)) - 1],
        "coefficient_of_variation": statistics.pstdev(raw) / statistics.mean(raw),
        "wall_p50_us": statistics.median(wall),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "warmup": args.warmup,
        "samples": args.samples,
        "inner": args.inner,
        "aggregation": "max_rank_each_invocation_then_mean_within_sample",
    }


@contextlib.contextmanager
def profile_ranges(model, manager):
    entries = [
        (model.vision, "forward", "vision.eager"),
        (model.vision, "forward_packed", "vision.packed"),
        (model.aligner, "forward", "aligner.eager"),
        (model.aligner, "forward_packed", "aligner.packed"),
        (model, "_build_image_span", "span.eager"),
        (model, "postprocess_encoder_output", "span.packed"),
        (manager, "_dp_shard", "dp.pack"),
        (manager, "_dp_gather", "dp.gather"),
    ]
    saved = []
    for obj, attr, label in entries:
        if not hasattr(obj, attr):
            continue
        original = getattr(obj, attr)

        @functools.wraps(original)
        def wrapped(*args, _original=original, _label=label, **kwargs):
            with torch.profiler.record_function(_label):
                return _original(*args, **kwargs)

        saved.append((obj, attr, original))
        setattr(obj, attr, wrapped)
    try:
        yield
    finally:
        for obj, attr, original in reversed(saved):
            setattr(obj, attr, original)


def load_vision_weights(model, directory, verify=False):
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    index = json.loads((directory / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    selected = {
        k: v
        for k, v in index.items()
        if k.startswith(("vision.", "aligner."))
        or k in ("image_start", "image_end", "image_newline")
    }
    params = dict(model.named_parameters())
    if set(selected) != set(params):
        raise ValueError(
            f"checkpoint mismatch missing={set(params) - set(selected)} "
            f"extra={set(selected) - set(params)}"
        )
    loaded = []
    for filename in sorted(set(selected.values())):
        with safe_open(directory / filename, framework="pt", device="cpu") as f:
            for name in sorted(k for k, v in selected.items() if v == filename):
                tensor = f.get_tensor(name)
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                if verify:
                    expected = tensor
                    if tensor.shape != param.shape:
                        if tensor.shape[0] != param.shape[0]:
                            parts = (
                                3
                                if ".wqkv." in name
                                else (2 if ".mlp.w1." in name else 1)
                            )
                            expected = torch.cat(
                                [
                                    piece.chunk(dist.get_world_size(), dim=0)[
                                        dist.get_rank()
                                    ]
                                    for piece in tensor.chunk(parts, dim=0)
                                ],
                                dim=0,
                            )
                        elif tensor.ndim == 2 and tensor.shape[1] != param.shape[1]:
                            expected = tensor.chunk(dist.get_world_size(), dim=1)[
                                dist.get_rank()
                            ]
                        else:
                            raise ValueError(f"Unknown shard shape: {name}")
                    expected = expected.to(param.dtype).contiguous()
                    actual = param.detach().cpu().contiguous()
                    if not torch.equal(
                        actual.view(torch.uint8), expected.view(torch.uint8)
                    ):
                        raise ValueError(f"Weight shard check failed {name}")
                loaded.append(
                    {
                        "name": name,
                        "shard_verified": verify,
                        "file": filename,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "local_shape": list(param.shape),
                        "local_dtype": str(param.dtype),
                        "sha256": hashlib.sha256(
                            tensor.contiguous().view(torch.uint8).numpy().tobytes()
                        ).hexdigest(),
                    }
                )
    return loaded


def preprocess(paths, config, mm):
    patches, grids, llm_grids, types, descriptions = [], [], [], [], []
    for path in paths:
        before = time.perf_counter()
        with Image.open(path) as image:
            image.load()
            original = image.size
            patch, h, w, lh, lw = mm.load_image(
                image,
                patch_size=config.vision_patch_size,
                downsample_ratio=config.vision_downsample_ratio,
                max_n_token=config.vision_max_n_token,
                min_pixels=config.vision_min_pixels,
                max_wh_ratio=config.vision_max_wh_ratio,
            )
        elapsed = time.perf_counter() - before
        patches.append(patch)
        grids.append([h, w])
        llm_grids.append([lh, lw])
        types.extend(
            [mm.IMAGE_START]
            + ([mm.IMAGE] * lw + [mm.IMAGE_NEW_LINE]) * lh
            + [mm.IMAGE_END]
        )
        descriptions.append(
            {
                "path": str(path),
                "sha256": sha(path),
                "pixels": original,
                "vit_grid": [h, w],
                "llm_grid": [lh, lw],
                "decode_and_preprocess_ms": elapsed * 1000,
            }
        )
    return {
        "patches": torch.cat(patches).pin_memory(),
        "vit_grid": torch.tensor(grids),
        "llm_grid": torch.tensor(llm_grids),
        "types": torch.tensor(types),
    }, descriptions


class PackedCall:
    """Module experiment using stock packed tower and stock span postprocessor.

    Includes per-call local input packing, metadata preparation, span assembly,
    and DP output gather. It is compared separately to the actual graph manager.
    """

    def __init__(self, model, kwargs, manager, use_dp):
        self.model, self.kwargs, self.manager, self.use_dp = (
            model,
            kwargs,
            manager,
            use_dp,
        )

    def __call__(self):
        mm = self.kwargs
        counts = self.manager._get_per_item_out_tokens(mm)
        if self.use_dp:
            local, assignment, per_rank, max_rows = self.manager._dp_shard(mm, counts)
        else:
            local = mm
        local_counts = self.manager._get_per_item_out_tokens(local)
        if local_counts:
            bufs = self.model.prepare_encoder_cudagraph_replay_buffers(
                local, len(local_counts), 0
            )
            out = self.model.encoder_cudagraph_forward(bufs.values)
            local_outputs = [None] * len(local_counts)
            self.model.postprocess_encoder_output(
                {"default": out},
                list(range(len(local_counts))),
                local_counts,
                local_outputs,
                batch_mm_kwargs=local,
            )
        else:
            local_outputs = []
        if self.use_dp:
            return self.manager._dp_gather(
                local_outputs, counts, assignment, per_rank, max_rows
            )
        return local_outputs


@torch.inference_mode()
def main():
    faulthandler.enable()
    args = parse_args()
    if args.debug_stacks:
        faulthandler.dump_traceback_later(60, repeat=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29841")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    from vllm.config import (
        ModelConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import destroy_model_parallel, graph_capture

    print("phase=config", flush=True)

    mc = ModelConfig(
        model=args.model,
        skip_tokenizer_init=True,
        max_model_len=32768,
        mm_encoder_tp_mode=args.mode,
        mm_encoder_attn_backend="FLASH_ATTN",
    )
    vc = VllmConfig(
        model_config=mc,
        parallel_config=ParallelConfig(
            tensor_parallel_size=world,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
        ),
    )
    print("phase=distributed", flush=True)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
    with set_current_vllm_config(vc):
        initialize_model_parallel(tensor_model_parallel_size=world)
        names = [
            "vllm.v1.worker.encoder_cudagraph_defs",
            "vllm.models.deepseek_v4.common.vision",
            "vllm.models.deepseek_v41.common.mm_preprocess",
            "vllm.models.deepseek_v41.common.vl_cudagraph",
            "vllm.models.deepseek_v41.nvidia.vl_model",
            "vllm.v1.worker.encoder_cudagraph",
        ]
        loaded_sources = [load_source(name, args.source_root) for name in names]
        _, vision, mm, mixin, vl, manager_module = loaded_sources
        EncoderCudaGraphManager = manager_module.EncoderCudaGraphManager
        print("phase=construct_vision_only", flush=True)
        # __new__ deliberately bypasses the wrapper's language-model constructor.
        model = vl.DeepseekV41ForCausalLM.__new__(vl.DeepseekV41ForCausalLM)
        torch.nn.Module.__init__(model)
        model.config, model.multimodal_config = mc.hf_config, mc.multimodal_config
        model.use_data_parallel = args.mode == "data"
        model.vision = vision.DeepseekV4ViT(mc.hf_config).to(
            dtype=torch.bfloat16, device="cuda"
        )
        expected_tp = world if args.mode == "weights" else 1
        expected_heads = mc.hf_config.vision_n_heads // expected_tp
        if not all(
            block.attn.n_heads == expected_heads and block.attn.tp_size == expected_tp
            for block in model.vision.blocks
        ):
            raise ValueError("Unexpected vision head sharding")
        model.aligner = vision.DeepseekV4Aligner(mc.hf_config).to(
            dtype=torch.bfloat16, device="cuda"
        )
        for name in ("image_start", "image_end", "image_newline"):
            setattr(
                model,
                name,
                torch.nn.Parameter(
                    torch.empty(
                        mc.hf_config.hidden_size, device="cuda", dtype=torch.float32
                    )
                ),
            )
        print("phase=load_selected_weights", flush=True)
        weight_manifest = load_vision_weights(
            model, Path(args.model), args.verify_weight_shards
        )
        print(f"phase=weights_loaded count={len(weight_manifest)}", flush=True)
        model.eval()
        paths = sorted(Path(args.images).glob("*.png"))
        if len(paths) < max(args.batch_sizes):
            raise ValueError("Not enough PNG test images")
        result = {
            "status": "running",
            "args": vars(args).copy(),
            "world_size": world,
            "vision_heads_per_rank": expected_heads,
            "vision_tensor_parallel_size": expected_tp,
            "rank": rank,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "config_sha256": sha(Path(args.model) / "config.json"),
            "sources": {str(Path(m.__file__)): sha(m.__file__) for m in loaded_sources},
            "weights": weight_manifest,
            "selected_parameter_count": sum(
                math.prod(x["shape"]) for x in weight_manifest
            ),
            "local_parameter_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            "benchmark_sha256": sha(__file__),
            "cases": [],
            "metric_dtype": "FP64 (model remains BF16)",
            "numerical_gate": {
                "cosine_min": 0.999,
                "relative_l2_max": 0.02,
                "optimization_vs_same_path": "bitwise_exact",
                "stock_crosspath_failure": "record_only_not_accepted",
            },
            "scope": "vision+aligner+span; no LLM, no serve, no model accuracy",
            "timing": (
                "GPU patches already present. CPU preprocessing and H2D separately. "
                "Per-call metadata, span and communication included. "
                "Graph capture/JIT excluded; warm-cache repeated inputs."
            ),
        }
        result["args"] = {
            k: str(v) if isinstance(v, Path) else v for k, v in result["args"].items()
        }
        result["native_libraries"] = {
            p: sha(p) for p in torch.ops.loaded_libraries if Path(p).is_file()
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def save():
            if rank == 0:
                args.output.write_text(json.dumps(result, indent=2))

        save()
        for count in args.batch_sizes:
            cpu_kwargs, images = preprocess(paths[:count], mc.hf_config, mm)
            before = time.perf_counter()
            kwargs = dict(cpu_kwargs)
            kwargs["patches"] = kwargs["patches"].to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter() - before) * 1000
            eager = lambda kw=kwargs: model.embed_multimodal(**kw)
            hook_handles, layer_outputs = [], {}
            if args.layer_dump_dir and count == 1 and rank == 0:
                points = {
                    "patch_embed": model.vision.patch_embed,
                    "final_norm": model.vision.norm,
                    "aligner": model.aligner,
                }
                points.update(
                    {
                        f"block_{i}": model.vision.blocks[i]
                        for i in (0, 1, 4, 8, 16, 24, 31)
                    }
                )
                for label, module in points.items():

                    def capture(
                        _module, _inputs, output, _label=label, _outputs=layer_outputs
                    ):
                        _outputs[_label] = output.detach().cpu().clone()

                    hook_handles.append(module.register_forward_hook(capture))
            reference = [x.clone() for x in eager()]
            for handle in hook_handles:
                handle.remove()
            if layer_outputs:
                args.layer_dump_dir.mkdir(parents=True, exist_ok=True)
                torch.save(layer_outputs, args.layer_dump_dir / f"{args.mode}-b1.pt")
                print(f"phase=layer_dump_saved mode={args.mode}", flush=True)
            total_tokens = sum(x.shape[0] for x in reference)
            vc.compilation_config.encoder_cudagraph_token_budgets = [
                2 ** math.ceil(math.log2(total_tokens))
            ]
            vc.compilation_config.encoder_cudagraph_max_vision_items_per_batch = count
            manager = EncoderCudaGraphManager(
                vc, torch.device("cuda", local_rank), torch.bfloat16, model
            )
            per_item = manager._get_per_item_out_tokens(kwargs)
            if manager.use_dp:
                local_kw, assignment, counts, max_rows = manager._dp_shard(
                    kwargs, per_item
                )
                local_stats = {
                    "images": counts[rank],
                    "patches": local_kw["patches"].shape[0],
                    "assignment": assignment,
                    "counts_per_rank": counts,
                    "allgather_input_bytes": max_rows * mc.hf_config.hidden_size * 2,
                    "allgather_output_bytes": max_rows
                    * mc.hf_config.hidden_size
                    * 2
                    * world,
                }
            else:
                local_stats = {"images": count, "patches": kwargs["patches"].shape[0]}
            rank_work = gather_metrics(local_stats, world)
            if manager.use_dp and args.graph_budget_policy != "global":
                max_local_rows = max_rows
                budget = (
                    2 ** math.ceil(math.log2(max_local_rows))
                    if args.graph_budget_policy == "local"
                    else max_local_rows
                )
                vc.compilation_config.encoder_cudagraph_token_budgets = [budget]
                vc.compilation_config.encoder_cudagraph_max_vision_items_per_batch = (
                    max(counts)
                )
                manager = EncoderCudaGraphManager(
                    vc, torch.device("cuda", local_rank), torch.bfloat16, model
                )
            capture_rows = (
                math.ceil(manager.token_budgets[-1] / manager.max_batch_size)
                * manager.max_batch_size
                * mc.hf_config.vision_downsample_ratio**2
            )
            for stats in rank_work:
                stats["capture_patch_rows"] = capture_rows

            variants = {
                "eager": eager,
                "packed": PackedCall(
                    model, kwargs, manager, args.mode == "data" and world > 1
                ),
            }
            for original in ("eager", "packed"):
                if any(
                    name in args.variants
                    for name in (original + "_rope", original + "_rope_rmsnorm")
                ):
                    variants[original + "_rope"] = RotaryCall(
                        vision, variants[original]
                    )
            if "eager_rmsnorm" in args.variants:
                variants["eager_rmsnorm"] = RMSNormCall(
                    vision, variants["eager"], args.rmsnorm_backend
                )
            if "eager_rope_rmsnorm" in args.variants:
                variants["eager_rope_rmsnorm"] = RMSNormCall(
                    vision, variants["eager_rope"], args.rmsnorm_backend
                )
            if any(v.startswith("manager") for v in args.variants):
                with graph_capture(device=torch.device("cuda", local_rank)):
                    manager.capture(graph_pool=torch.cuda.graph_pool_handle())
                variants["manager"] = lambda mg=manager, kw=kwargs: mg.execute(kw)
            if "manager_cached" in args.variants:
                from vllm_dsv41_vision.metadata_cache import GeometryCache

                cache = GeometryCache(capacity=16)
                variants["manager_cached"] = CachedManagerCall(
                    model, manager, kwargs, cache
                )
            case = {
                "images": images,
                "count": count,
                "h2d_ms": h2d_ms,
                "output_tokens": total_tokens,
                "correctness": {},
                "measurements": [],
                "rank_work": rank_work,
                "graph_token_budgets": manager.token_budgets,
                "graph_budget_policy": args.graph_budget_policy,
            }
            result["cases"].append(case)
            save()
            outputs = {}
            for name in args.variants:
                outputs[name] = [x.clone() for x in variants[name]()]
                metrics = compare_outputs(outputs[name], reference)
                case["correctness"][name] = {
                    "vs_eager_all_ranks": gather_metrics(metrics, world),
                    "local_stock_crosspath_gate_pass": numerical_gate(metrics),
                }
            exact_pairs = {
                "eager_rope": "eager",
                "packed_rope": "packed",
                "eager_rmsnorm": "eager",
                "eager_rope_rmsnorm": "eager",
            }
            case["exact_comparison_references"] = {}
            for optimized, original in exact_pairs.items():
                if optimized in outputs:
                    baseline = outputs.get(original)
                    if baseline is None:
                        baseline = [x.clone() for x in variants[original]()]
                    gathered = gather_metrics(
                        compare_outputs(outputs[optimized], baseline), world
                    )
                    case["exact_comparison_references"][optimized] = original
                    case[optimized + "_vs_original_all_ranks"] = gathered
                    if not all(m["exact"] for row in gathered for m in row):
                        result["status"] = "failed_optimization_correctness"
                        save()
                        raise ValueError(f"{optimized} not exact vs {original}")
            if "eager_rope_rmsnorm" in outputs:
                rope_reference = outputs.get("eager_rope")
                if rope_reference is None:
                    rope_reference = [x.clone() for x in variants["eager_rope"]()]
                incremental = gather_metrics(
                    compare_outputs(outputs["eager_rope_rmsnorm"], rope_reference),
                    world,
                )
                case["eager_rope_rmsnorm_vs_rope_all_ranks"] = incremental
                if not all(m["exact"] for row in incremental for m in row):
                    result["status"] = "failed_optimization_correctness"
                    save()
                    raise ValueError("Combined path not exact vs eager_rope")
            if "manager_cached" in args.variants:
                original = outputs.get("manager")
                if original is None:
                    original = [x.clone() for x in variants["manager"]()]
                exact_metrics = compare_outputs(outputs["manager_cached"], original)
                gathered = gather_metrics(exact_metrics, world)
                case["cached_vs_original_manager_all_ranks"] = gathered
                if not all(
                    m["exact"] for rank_metrics in gathered for m in rank_metrics
                ):
                    result["status"] = "failed_optimization_correctness"
                    save()
                    raise ValueError("Cached manager does not exactly match original")
            refpath = (
                args.reference_dir / f"batch-{count}.pt" if args.reference_dir else None
            )
            if refpath and args.save_reference:
                if rank == 0:
                    refpath.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {k: [x.cpu() for x in v] for k, v in outputs.items()}, refpath
                    )
                dist.barrier()
            elif refpath:
                external = torch.load(refpath, weights_only=True, map_location="cuda")
                case["external_reference"] = {
                    "label": args.reference_label,
                    "path": str(refpath),
                    "sha256": sha(refpath),
                }
                if not isinstance(external, dict):
                    external = {"eager": external}
                for name, output in outputs.items():
                    metrics = compare_outputs(output, external["eager"])
                    case["correctness"][name]["vs_external_eager_all_ranks"] = (
                        gather_metrics(metrics, world)
                    )
                    matching = external.get(name)
                    if matching is not None:
                        metrics = compare_outputs(output, matching)
                        case["correctness"][name]["vs_external_same_path_all_ranks"] = (
                            gather_metrics(metrics, world)
                        )
            del outputs
            for order, order_names in (
                ("forward", args.variants),
                ("reverse", list(reversed(args.variants))),
            ):
                for name in [] if args.correctness_only else order_names:
                    metric = measure(variants[name], args, world)
                    item = {"variant": name, "order": order, **metric}
                    item["extra_rounds"] = []
                    if metric["coefficient_of_variation"] > 0.1:
                        item["extra_rounds"] = [
                            measure(variants[name], args, world) for _ in range(3)
                        ]
                    case["measurements"].append(item)
                    save()
                    if rank == 0:
                        print(
                            json.dumps(
                                {
                                    "count": count,
                                    "variant": name,
                                    "order": order,
                                    "p50_us": metric["p50_us"],
                                }
                            ),
                            flush=True,
                        )
            if "manager_cached" in args.variants:
                dynamic = []
                for change in ("reordered", "pixels_changed"):
                    changed, _ = preprocess(
                        list(reversed(paths[:count]))
                        if change == "reordered"
                        else paths[:count],
                        mc.hf_config,
                        mm,
                    )
                    changed["patches"] = changed["patches"].to(
                        "cuda", non_blocking=True
                    )
                    if change == "pixels_changed":
                        changed["patches"].add_(0.01)
                    expected = [x.clone() for x in manager.execute(changed)]
                    actual = CachedManagerCall(model, manager, changed, cache)()
                    checks = gather_metrics(compare_outputs(actual, expected), world)
                    dynamic.append({"change": change, "all_ranks": checks})
                    if not all(m["exact"] for values in checks for m in values):
                        raise ValueError(f"Metadata cache failed dynamic {change}")
                case["dynamic_cache_checks"] = dynamic
            if args.profile_dir:
                args.profile_dir.mkdir(parents=True, exist_ok=True)
                for name in args.variants:
                    torch.cuda.synchronize()
                    dist.barrier()
                    with (
                        profile_ranges(model, manager),
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                            with_stack=True,
                        ) as prof,
                    ):
                        with torch.profiler.record_function(f"module.{name}"):
                            variants[name]()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(
                        str(
                            args.profile_dir
                            / f"{args.mode}-b{count}-{name}-rank{rank}.json"
                        )
                    )
            if args.profile_dir and args.profile_end_to_end:
                dist.barrier()
                with (  # noqa: SIM117
                    profile_ranges(model, manager),
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                        with_stack=True,
                    ) as prof,
                ):
                    with torch.profiler.record_function("image_component.end_to_end"):
                        with torch.profiler.record_function("cpu.decode_preprocess"):
                            fresh, _ = preprocess(paths[:count], mc.hf_config, mm)
                        with torch.profiler.record_function("h2d.patches"):
                            fresh["patches"] = fresh["patches"].to(
                                "cuda", non_blocking=True
                            )
                        with torch.profiler.record_function("encoder.eager"):
                            model.embed_multimodal(**fresh)
                        torch.cuda.synchronize()
                prof.export_chrome_trace(
                    str(
                        args.profile_dir
                        / f"{args.mode}-b{count}-end_to_end-rank{rank}.json"
                    )
                )
            if "manager_cached" in args.variants:
                case["geometry_cache"] = {"hits": cache.hits, "misses": cache.misses}
            result["loaded_files"] = loaded_manifest()
            save()
            manager.clear()
            del manager, variants, reference
            torch.cuda.synchronize()
        result["status"] = "completed"
        save()
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

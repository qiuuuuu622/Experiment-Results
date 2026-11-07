# SPDX-License-Identifier: Apache-2.0
"""Read-only FP8 dot/epilogue decomposition on frozen model-shaped inputs."""

import argparse
import hashlib
import json
from pathlib import Path

from .common import check_vllm, device_info, save


def make_kernels():
    from triton.experimental.gluon.language.nvidia.hopper import (
        fence_async_shared,
        warpgroup_mma,
    )

    from vllm.triton_utils import gl, gluon, tl, triton

    @gluon.jit
    def dots(Q, K, OUT, N, D: gl.constexpr, PRECISE: gl.constexpr):
        tile, segment = gl.program_id(0), gl.program_id(1)
        load: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
        mma: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 32]
        )
        dim = segment * D + gl.arange(0, D, layout=gl.SliceLayout(0, load))
        key = tile * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, load))
        head = gl.arange(0, 32, layout=gl.SliceLayout(1, load))
        k = gl.load(K + key[:, None] * 128 + dim[None, :], key[:, None] < N, 0).to(
            gl.float8e4nv, bitcast=True
        )
        q = gl.load(Q + head[:, None] * 128 + dim[None, :]).to(
            gl.float8e4nv, bitcast=True
        )
        qs = gl.allocate_shared_memory(
            gl.float8e4nv,
            [32, D],
            gl.NVMMASharedLayout.get_default_for([32, D], gl.float8e4nv),
        )
        ks = gl.allocate_shared_memory(
            gl.float8e4nv,
            [64, D],
            gl.NVMMASharedLayout.get_default_for([64, D], gl.float8e4nv),
        )
        qs.store(q)
        ks.store(k)
        fence_async_shared()
        gl.barrier()
        acc = gl.zeros([64, 32], gl.float32, layout=mma)
        if PRECISE:
            acc = warpgroup_mma(
                ks, qs.permute((1, 0)), acc, use_acc=False, max_num_imprecise_acc=32
            )
        else:
            acc = warpgroup_mma(ks, qs.permute((1, 0)), acc, use_acc=False)
        key_out = tile * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
        head_out = gl.arange(0, 32, layout=gl.SliceLayout(0, mma))
        gl.store(
            OUT + (segment * N + key_out[:, None]) * 32 + head_out[None, :],
            acc,
            key_out[:, None] < N,
        )

    @triton.jit
    def epilogue(DOT, SCALE, W, OUT, N, FMA: tl.constexpr, BLOCK: tl.constexpr):
        key = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        scale = tl.load(SCALE + key, key < N, 1.0)
        acc = tl.full((BLOCK,), 0, tl.float32)
        for head in tl.static_range(32):
            dot = tl.load(DOT + key * 32 + head, key < N, 0.0)
            weight = tl.load(W + head)
            value = tl.maximum(dot * scale, 0.0)
            acc = tl.fma(value, weight, acc) if FMA else acc + value * weight
        tl.store(OUT + key, acc, key < N)

    return dots, epilogue


def metrics(value, reference):
    import torch

    error = value.double() - reference.double()
    return dict(
        max_abs=float(error.abs().max()) if error.numel() else 0.0,
        nrmse=float(
            (
                error.square().sum()
                / reference.double().square().sum().clamp_min(1e-30)
            ).sqrt()
        ),
        strict_failures=int(
            (
                ~torch.isclose(value.double(), reference.double(), rtol=1e-3, atol=1e-4)
            ).sum()
        ),
        elements=error.numel(),
    )


def topk_stats(value, reference):
    import torch

    k = min(512, value.numel())
    actual = torch.argsort(value, descending=True, stable=True)[:k]
    expected = torch.argsort(reference, descending=True, stable=True)[:k]
    return dict(
        k=k,
        overlap=float(torch.isin(actual, expected).sum()) / max(k, 1),
        selected_only=actual[~torch.isin(actual, expected)][:16].tolist(),
        reference_only=expected[~torch.isin(expected, actual)][:16].tolist(),
    )


def run_case(label, query_rows, context, output_dir, kernels, selected_rows=None):
    import torch

    from .benchmark_prefill_model import ModelPath, model_inputs

    dots, epilogue = kernels
    inputs = model_inputs([query_rows], [context], 2)
    native = ModelPath(inputs, "deepgemm", "full")
    ant = ModelPath(inputs, "ant", "full")
    native.forward()
    ant.forward()
    changed_rows = torch.where(
        (native.op.output.sort(1).values != ant.op.output.sort(1).values).any(1)
    )[0].tolist()
    chosen_rows = set(changed_rows)
    for start, end, _ in ant.op.chunks:
        chosen_rows.update([start, start + (end - start) // 2, end - 1])
    if selected_rows is not None:
        chosen_rows = set(selected_rows)
    sampled_scores = {}
    for start, end, request in ant.op.chunks:
        ant_scores = ant.score_chunk(start, end, request)
        native_scores = native.score_chunk(start, end, request)
        for row in sorted(chosen_rows):
            if start <= row < end:
                n = int(inputs.lengths[row])
                sampled_scores[row] = (
                    ant_scores[row - start, :n].clone(),
                    native_scores[row - start, :n].clone(),
                )
    records = []
    for row in sorted(chosen_rows):
        actual_ant, actual_native = sampled_scores[row]
        n = actual_ant.numel()
        if not n:
            continue
        q = inputs.q[row].contiguous()
        keys = ant.op.keys[0, :n].contiguous()
        scales = ant.op.scales[0, :n].contiguous()
        weights = inputs.weights[row].contiguous()
        dot32 = torch.einsum("hd,kd->kh", q.float(), keys.float())
        dot64 = torch.einsum("hd,kd->kh", q.double(), keys.double())
        outputs = {}
        for name, dim, precise in [
            ("d128_default", 128, False),
            ("d128_max_imprecise_32", 128, True),
            ("four_d32_default", 32, False),
        ]:
            storage = torch.empty((128 // dim, n, 32), device="cuda")
            compiled = dots[((n + 63) // 64, 128 // dim)](
                q.view(torch.uint8),
                keys.view(torch.uint8),
                storage,
                n,
                dim,
                precise,
                num_warps=4,
            )
            ptx = compiled.asm["ptx"]
            assert "wgmma.mma_async" in ptx and ".e4m3.e4m3" in ptx
            (output_dir / (name + ".ptx")).write_text(ptx)
            if dim == 32:
                outputs[name] = ((storage[0] + storage[1]) + storage[2]) + storage[3]
            else:
                outputs[name] = storage[0]
        outputs["fp32_dot"] = dot32
        outputs["fp64_dot_cast_fp32"] = dot64.float()
        score64 = (
            (dot64 * scales.double()[:, None]).clamp_min(0) * weights.double()
        ).sum(1)
        score32 = ((dot32 * scales[:, None]).clamp_min(0) * weights).sum(1)
        reconstructed = {}
        for name, dot in outputs.items():
            out = torch.empty(n, device="cuda")
            epilogue[((n + 127) // 128,)](
                dot, scales, weights, out, n, True, 128, enable_fp_fusion=False
            )
            reconstructed[name] = out
        no_fma = torch.empty(n, device="cuda")
        epilogue[((n + 127) // 128,)](
            outputs["d128_default"],
            scales,
            weights,
            no_fma,
            n,
            False,
            128,
            enable_fp_fusion=False,
        )
        positions = torch.linspace(0, n - 1, min(n, 3072), device="cuda").long()
        positions = torch.unique(
            torch.cat(
                (positions, ant.op.output[row].long(), native.op.output[row].long())
            )
        )
        positions = positions[positions >= 0]
        worst = torch.argsort((actual_ant.double() - score64).abs(), descending=True)[
            :8
        ]
        details = []
        for pos in worst.tolist():
            head_error = outputs["d128_default"][pos].double() - dot64[pos]
            head = int(head_error.abs().argmax())
            details.append(
                dict(
                    key=pos,
                    head_with_max_dot_error=head,
                    scale=float(scales[pos]),
                    head_weight=float(weights[head]),
                    dot64=float(dot64[pos, head]),
                    dot32=float(dot32[pos, head]),
                    dot128=float(outputs["d128_default"][pos, head]),
                    dot4x32=float(outputs["four_d32_default"][pos, head]),
                    ant=float(actual_ant[pos]),
                    native=float(actual_native[pos]),
                    reconstructed128=float(reconstructed["d128_default"][pos]),
                    reconstructed4x32=float(reconstructed["four_d32_default"][pos]),
                    score64=float(score64[pos]),
                    score32=float(score32[pos]),
                )
            )
        torch.save(
            dict(
                q=q.cpu(),
                keys=keys.view(torch.uint8)[worst].cpu(),
                scales=scales[worst].cpu(),
                weights=weights.cpu(),
                logical_positions=worst.cpu(),
            ),
            output_dir / f"{label}-row{row}-worst-inputs.pt",
        )
        record = dict(
            row=row,
            visible_keys=n,
            production_ant_vs_native=metrics(actual_ant, actual_native),
            original_diagnostic_positions=positions.numel(),
            original_sample_ant_vs_fp32=metrics(
                actual_ant[positions], score32[positions]
            ),
            original_sample_native_vs_fp32=metrics(
                actual_native[positions], score32[positions]
            ),
            dot_vs_fp64={name: metrics(dot, dot64) for name, dot in outputs.items()},
            score_vs_fp64={
                name: metrics(score, score64) for name, score in reconstructed.items()
            },
            production_ant_vs_fp64=metrics(actual_ant, score64),
            production_native_vs_fp64=metrics(actual_native, score64),
            fp32_reference_vs_fp64=metrics(score32, score64),
            ant_vs_reconstructed128=metrics(actual_ant, reconstructed["d128_default"]),
            native_vs_reconstructed128=metrics(
                actual_native, reconstructed["d128_default"]
            ),
            sequential_fma_vs_nonfma=metrics(reconstructed["d128_default"], no_fma),
            scale_values=torch.unique(scales).tolist(),
            scale_multiplication_rounding=metrics(
                outputs["d128_default"] * scales[:, None],
                outputs["d128_default"].double() * scales.double()[:, None],
            ),
            full_row_topk_vs_fp64={
                name: topk_stats(score, score64)
                for name, score in {
                    **reconstructed,
                    "ant": actual_ant,
                    "native": actual_native,
                }.items()
            },
            worst_keys=details,
        )
        score_variants = {**reconstructed, "ant": actual_ant, "native": actual_native}
        differing_positions = set()
        for item in record["full_row_topk_vs_fp64"].values():
            differing_positions.update(item["selected_only"])
            differing_positions.update(item["reference_only"])
        reference_order = torch.argsort(score64, descending=True, stable=True)
        record["fp64_boundary_gap"] = (
            float(score64[reference_order[511]] - score64[reference_order[512]])
            if n > 512
            else None
        )
        record["changed_topk_position_scores"] = [
            dict(
                logical_position=pos,
                fp64=float(score64[pos]),
                fp32=float(score32[pos]),
                **{name: float(score[pos]) for name, score in score_variants.items()},
            )
            for pos in sorted(differing_positions)
        ]
        record["d32_vs_max_imprecise32"] = metrics(
            outputs["four_d32_default"], outputs["d128_max_imprecise_32"]
        )
        records.append(record)
        save(output_dir / f"{label}-partial.json", dict(rows=records))
        print(
            json.dumps(
                dict(
                    case=label,
                    row=row,
                    keys=n,
                    max_ant_fp64=record["production_ant_vs_fp64"]["max_abs"],
                    residual=record["ant_vs_reconstructed128"]["max_abs"],
                    d32_score_fp64=record["score_vs_fp64"]["four_d32_default"][
                        "max_abs"
                    ],
                )
            ),
            flush=True,
        )
    return dict(
        case=label,
        query_rows=query_rows,
        original_context=context,
        ratio=2,
        ant_native_changed_topk_rows=changed_rows,
        rows=records,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-only", action="store_true")
    args = parser.parse_args()
    check_vllm()
    import torch

    from .verify_prefill_model import identity

    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="running",
        identity_before=identity(),
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device=device_info(),
        cases=[],
        scope="read-only dot/epilogue diagnosis; no default change or timing",
    )
    save(out / "diagnostic.json", report)
    try:
        kernels = make_kernels()
        cases = [
            ("initial_2k", 2048, 2048),
            ("steady_256k", 4096, 262144),
        ]
        if args.boundary_only:
            cases = cases[-1:]
        for label, rows, context in cases:
            report["cases"].append(
                run_case(
                    label,
                    rows,
                    context,
                    out,
                    kernels,
                    selected_rows=[1401, 3072] if args.boundary_only else None,
                )
            )
            save(out / "diagnostic.json", report)
        report["identity_after"] = identity()
        assert report["identity_before"] == report["identity_after"]
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save(out / "diagnostic.json", report)


if __name__ == "__main__":
    main()

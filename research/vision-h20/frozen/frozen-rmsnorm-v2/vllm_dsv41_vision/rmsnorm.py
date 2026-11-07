# SPDX-License-Identifier: Apache-2.0
"""Vision RMSNorm experiments preserving the original FP32 rounding points.

The single-kernel reduction mirrors PyTorch CUDA's contiguous FP32 mean for
1024 columns and at least 16 rows: four accumulators per lane, eight inputs
per accumulator, then a 32-lane descending reduction. Smaller row counts use
the native reduction because its block width changes with the row count.
"""

from contextlib import contextmanager

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_1024_kernel(
    X,
    W,
    Y,
    STATS,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    EPS: tl.constexpr,
    WRITE_STATS: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    a0 = tl.full((32,), 0, tl.float32)
    a1 = tl.full((32,), 0, tl.float32)
    a2 = tl.full((32,), 0, tl.float32)
    a3 = tl.full((32,), 0, tl.float32)
    for j in tl.static_range(8):
        offset = row * XS0 + (4 * lane + 128 * j) * XS1
        x0 = tl.load(X + offset).to(tl.float32)
        x1 = tl.load(X + offset + XS1).to(tl.float32)
        x2 = tl.load(X + offset + 2 * XS1).to(tl.float32)
        x3 = tl.load(X + offset + 3 * XS1).to(tl.float32)
        a0 = a0 + x0 * x0
        a1 = a1 + x1 * x1
        a2 = a2 + x2 * x2
        a3 = a3 + x3 * x3
    combined = ((a0 + a1) + a2) + a3
    mean = tl.sum(combined, 0) * (1.0 / 1024)
    if WRITE_STATS:
        tl.store(STATS + row, mean)
    inv_rms = tl.rsqrt(mean + EPS)
    column = tl.arange(0, 1024)
    x = tl.load(X + row * XS0 + column * XS1).to(tl.float32)
    weight = tl.load(W + column * WS).to(tl.float32)
    normalized = x * inv_rms
    output = weight * normalized
    tl.store(Y + row * 1024 + column, output)


@triton.jit
def _square_kernel(
    X,
    SQUARE,
    COUNT: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = i // 1024, i % 1024
    x = tl.load(X + row * XS0 + column * XS1, i < COUNT, other=0).to(tl.float32)
    tl.store(SQUARE + i, x * x, i < COUNT)


@triton.jit
def _epilogue_kernel(
    X,
    W,
    MEAN,
    Y,
    COUNT: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = i // 1024, i % 1024
    valid = i < COUNT
    x = tl.load(X + row * XS0 + column * XS1, valid, other=0).to(tl.float32)
    weight = tl.load(W + column * WS, valid, other=0).to(tl.float32)
    mean = tl.load(MEAN + row, valid, other=0)
    normalized = x * tl.rsqrt(mean + EPS)
    tl.store(Y + i, weight * normalized, valid)


def apply_rmsnorm(x, weight, eps=1e-6, *, backend="single", stats=None):
    """Normalize BF16 [rows, 1024] with contiguous columns and optional row gaps.

    Column-strided layouts can select a different native mean reduction tree
    and are unsupported. Optional mean stats must be a contiguous FP32 vector
    on the same CUDA device. The residual addition remains separate.
    """
    if x.ndim != 2 or x.shape[1] != 1024 or weight.shape != (1024,):
        raise ValueError("Expected [rows, 1024] input and [1024] weight")
    if x.stride(1) != 1:
        raise ValueError("RMSNorm requires contiguous columns: x.stride(1) == 1")
    if x.dtype != torch.bfloat16 or weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Expected BF16 input and BF16/FP32 weight")
    if not x.is_cuda or x.device != weight.device:
        raise ValueError("Input and weight must share a CUDA device")
    if backend not in ("single", "torch_mean"):
        raise ValueError(f"Unknown RMSNorm backend {backend}")
    rows = x.shape[0]
    if stats is not None and (
        stats.shape != (rows,)
        or stats.dtype != torch.float32
        or stats.device != x.device
        or not stats.is_contiguous()
    ):
        raise ValueError(
            "stats must be contiguous FP32 [rows] on the input CUDA device"
        )
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if not rows:
        return out
    if backend == "single" and rows >= 16:
        _rmsnorm_1024_kernel[(rows,)](
            x,
            weight,
            out,
            stats if stats is not None else out,
            *x.stride(),
            weight.stride(0),
            eps,
            stats is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        square = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        count, block = x.numel(), 1024
        _square_kernel[(triton.cdiv(count, block),)](
            x,
            square,
            count,
            *x.stride(),
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
        mean = square.mean(-1)
        if stats is not None:
            stats.copy_(mean)
        _epilogue_kernel[(triton.cdiv(count, block),)](
            x,
            weight,
            mean,
            out,
            count,
            *x.stride(),
            weight.stride(0),
            eps,
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@contextmanager
def fused_rmsnorm(vision_module, *, backend="single"):
    """Scope the experimental forward to one isolated benchmark process.

    Graph capture must take place inside this context. This modifies the class
    forward and must not overlap other callers in the same process.
    """
    cls = vision_module.DeepseekV4RMSNorm
    original = cls.forward

    def forward(self, x):
        return apply_rmsnorm(x, self.weight, self.eps, backend=backend)

    cls.forward = forward
    try:
        yield
    finally:
        cls.forward = original

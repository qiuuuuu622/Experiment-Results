# SPDX-License-Identifier: Apache-2.0
"""Experimental vision RoPE fusion with the original FP32 rounding points."""

from contextlib import contextmanager

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    X,
    COS,
    SIN,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    HALF: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    CS0: tl.constexpr,
    CS2: tl.constexpr,
    SS0: tl.constexpr,
    SS2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, channel = i // (H * HALF), i % HALF
    head = (i // HALF) % H
    valid = i < N * H * HALF
    offset = row * XS0 + head * XS1 + channel * XS2
    x1 = tl.load(X + offset, valid, other=0).to(tl.float32)
    x2 = tl.load(X + offset + HALF * XS2, valid, other=0).to(tl.float32)
    cos = tl.load(COS + row * CS0 + channel * CS2, valid, other=0)
    sin = tl.load(SIN + row * SS0 + channel * SS2, valid, other=0)
    # Launch disables FP fusion: each multiplication rounds before add/sub,
    # matching the separate PyTorch FP32 operators rather than an FMA.
    first = x1 * cos - x2 * sin
    second = x2 * cos + x1 * sin
    out_offset = row * H * HALF * 2 + head * HALF * 2 + channel
    tl.store(OUT + out_offset, first, valid)
    tl.store(OUT + out_offset + HALF, second, valid)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D vision RoPE to BF16 ``[patch, head, channel]`` tensors.

    FP32 ``cos`` and ``sin`` broadcast over heads. Input strides are retained,
    including Q/K views into a combined QKV projection. The output is fresh.
    """
    if x.ndim != 3 or x.shape[-1] % 2:
        raise ValueError("Expected [patch, head, even channel] input")
    n, heads, channels = x.shape
    expected = (n, 1, channels // 2)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError(f"Expected cos/sin shape {expected}")
    if x.dtype != torch.bfloat16 or cos.dtype != torch.float32:
        raise TypeError("Requires BF16 input and FP32 cos/sin")
    if sin.dtype != torch.float32:
        raise TypeError("Requires FP32 sin")
    if not x.is_cuda or x.device != cos.device or x.device != sin.device:
        raise ValueError("Input and rotary tables must share a CUDA device")
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if not n or not heads:
        return out
    block = 1024
    _rotary_kernel[(triton.cdiv(n * heads * (channels // 2), block),)](
        x,
        cos,
        sin,
        out,
        n,
        heads,
        channels // 2,
        *x.stride(),
        cos.stride(0),
        cos.stride(2),
        sin.stride(0),
        sin.stride(2),
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@contextmanager
def fused_rotary(vision_module):
    """Install only for an isolated experiment, restoring the original on exit.

    This modifies module state and therefore must not overlap other callers
    within the same process. Graph capture must happen inside the context.
    """
    original = vision_module.apply_rotary
    vision_module.apply_rotary = apply_rotary
    try:
        yield
    finally:
        vision_module.apply_rotary = original

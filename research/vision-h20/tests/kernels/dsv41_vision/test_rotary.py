# SPDX-License-Identifier: Apache-2.0
"""Guard original FP32 rounding, QKV strides and fresh graph replay inputs."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def original_rotary(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


@pytest.mark.parametrize("rows,heads", [(1, 4), (17, 16), (5980, 4), (8418, 16)])
@pytest.mark.parametrize("layout", ["contiguous", "qkv_view"])
def test_fused_rotary_matches_original_bits(rows, heads, layout):
    from vllm_dsv41_vision.rotary import apply_rotary

    torch.manual_seed(20261107)
    qkv = torch.randn(rows, heads * 64 * 3, device="cuda", dtype=torch.bfloat16)
    x = qkv.chunk(3, dim=-1)[1].view(rows, heads, 64)
    if layout == "contiguous":
        x = x.contiguous()
    angle = torch.randn(rows, 1, 32, device="cuda", dtype=torch.float32)
    cos, sin = angle.cos(), angle.sin()
    before = qkv.clone()
    for scale in (0.0, 1.0, 0.00001, 10000.0):
        x.mul_(0).copy_(before.chunk(3, dim=-1)[1].view(rows, heads, 64) * scale)
        result = apply_rotary(x, cos, sin)
        reference = original_rotary(x, cos, sin)
        assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
        assert result.is_contiguous()
        assert result.data_ptr() != x.data_ptr()


def test_graph_reads_new_pixels_and_rotary_tables():
    from vllm_dsv41_vision.rotary import apply_rotary

    x = torch.randn(17, 16, 64, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(17, 1, 32, device="cuda")
    sin = torch.randn_like(cos)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            apply_rotary(x, cos, sin)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = apply_rotary(x, cos, sin)
    for _ in range(3):
        x.normal_()
        cos.normal_()
        sin.normal_()
        graph.replay()
        reference = original_rotary(x, cos, sin)
        assert torch.equal(result.view(torch.int16), reference.view(torch.int16))

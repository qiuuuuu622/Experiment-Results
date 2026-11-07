# SPDX-License-Identifier: Apache-2.0
"""Prove the vision RMSNorm reduction and output rounding without a model."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def native(x, weight, eps):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (weight * value).to(x.dtype)


@pytest.mark.parametrize("rows", [1, 15, 16, 17, 5980, 8418, 16836])
@pytest.mark.parametrize("backend", ["single", "torch_mean"])
def test_norm_original_mean_and_output_bits(rows, backend):
    from vllm_dsv41_vision.rmsnorm import apply_rmsnorm

    torch.manual_seed(20261107)
    storage = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
    x = storage[:, :1024]
    weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    stats = torch.empty(rows, device="cuda", dtype=torch.float32)
    for scale in (0.0, 1.0, 0.00001, 10000.0):
        source = (x * scale).contiguous()
        before = source.clone()
        actual = apply_rmsnorm(source, weight, stats=stats, backend=backend)
        expected_mean = source.float().square().mean(-1)
        expected = native(source, weight, 1e-6)
        assert torch.equal(stats.view(torch.int32), expected_mean.view(torch.int32)), (
            "mean mismatch",
            rows,
            backend,
            (stats != expected_mean).sum().item(),
            (stats - expected_mean).abs().max().item(),
        )
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
            "output mismatch",
            rows,
            backend,
            (actual != expected).sum().item(),
            (actual.float() - expected.float()).abs().max().item(),
        )
        assert torch.equal(source, before)
    # A strided row view must read the current tensor, not adjacent storage.
    actual = apply_rmsnorm(x, weight, backend=backend)
    assert torch.equal(
        actual.view(torch.int16), native(x, weight, 1e-6).view(torch.int16)
    )


def test_norm_graph_reads_current_input_and_weight():
    from vllm_dsv41_vision.rmsnorm import apply_rmsnorm

    x = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            apply_rmsnorm(x, weight)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = apply_rmsnorm(x, weight)
    for _ in range(3):
        x.normal_()
        weight.normal_()
        graph.replay()
        assert torch.equal(
            result.view(torch.int16), native(x, weight, 1e-6).view(torch.int16)
        )


@pytest.mark.parametrize("backend", ["single", "torch_mean"])
def test_rejects_different_reduction_layout_and_unsafe_stats(backend):
    from vllm_dsv41_vision.rmsnorm import apply_rmsnorm

    x = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    transpose = torch.empty(1024, 17, device="cuda", dtype=x.dtype).t()
    column_gap = torch.empty(17, 2048, device="cuda", dtype=x.dtype)[:, ::2]
    for unsupported in (transpose, column_gap):
        with pytest.raises(ValueError, match="contiguous columns"):
            apply_rmsnorm(unsupported, weight, backend=backend)
    invalid_stats = (
        torch.empty(17, device="cuda", dtype=torch.bfloat16),
        torch.empty(18, device="cuda"),
        torch.empty(17, 1, device="cuda"),
        torch.empty(34, device="cuda")[::2],
        torch.empty(17, device="cpu"),
    )
    for stats in invalid_stats:
        with pytest.raises(ValueError, match="stats must be"):
            apply_rmsnorm(x, weight, stats=stats, backend=backend)


@pytest.mark.parametrize("backend", ["single", "torch_mean"])
def test_fp32_strided_weight_preserves_output_bits(backend):
    from vllm_dsv41_vision.rmsnorm import apply_rmsnorm

    torch.manual_seed(20261107)
    x = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, device="cuda", dtype=torch.float32)[::2]
    actual = apply_rmsnorm(x, weight, backend=backend)
    expected = native(x, weight, 1e-6)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

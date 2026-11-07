# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks: cached geometry must never retain previous pixels."""

from functools import wraps
from types import SimpleNamespace

import pytest

from vllm_dsv41_vision.metadata_cache import GeometryCache
from vllm_dsv41_vision.replay_metadata import cached_replay_metadata


def build_packed_vit_metadata(grids, **kwargs):
    return {"vit_cos": object(), "grids": tuple(map(tuple, grids))}


def build_packed_merge_metadata(grids, ratio, **kwargs):
    return {"merge_idx": object()}


class EncoderCudaGraphReplayBuffers:
    def __init__(self, values):
        self.values = values


class Pixels:
    device = "cpu"

    def __init__(self, identity, dtype="bf16"):
        self.identity, self.dtype = identity, dtype

    def to(self, dtype):
        return Pixels(self.identity, dtype)


class Model:
    vision = SimpleNamespace(rope_dim=32, rope_theta=10000)
    config = SimpleNamespace(vision_downsample_ratio=3)
    aligner = SimpleNamespace(w1=SimpleNamespace(weight=SimpleNamespace(dtype="bf16")))

    def _get_grid_list(self, kwargs, key):
        return kwargs[key]

    def prepare_encoder_cudagraph_replay_buffers(self, *args, **kwargs):
        return "original"


def test_current_pixels_and_grid_reorder_with_cached_geometry():
    model, cache = Model(), GeometryCache()
    with cached_replay_metadata(model, cache):
        first = model.prepare_encoder_cudagraph_replay_buffers(
            {"patches": Pixels("first"), "vit_grid": [[4, 6], [3, 9]]}, 8, 1
        ).values
        current_pixels = Pixels("changed", "fp32")
        second = model.prepare_encoder_cudagraph_replay_buffers(
            {"patches": current_pixels, "vit_grid": [[4, 6], [3, 9]]}, 8, 1
        ).values
        assert first["vit_cos"] is second["vit_cos"]
        assert second["patches"].identity == "changed"
        assert second["patches"].dtype == "bf16"
        assert current_pixels.dtype == "fp32"
        assert first["patches"].identity == "first"
        third = model.prepare_encoder_cudagraph_replay_buffers(
            {"patches": current_pixels, "vit_grid": [[3, 9], [4, 6]]}, 8, 1
        ).values
        assert third["vit_cos"] is not first["vit_cos"]
        assert third["grids"] == ((3, 9), (4, 6))
        assert (cache.hits, cache.misses) == (1, 2)
        assert all("patches" not in item for item in cache.entries.values())
    assert model.prepare_encoder_cudagraph_replay_buffers() == "original"
    assert "prepare_encoder_cudagraph_replay_buffers" not in vars(model)


def test_restore_on_exception():
    model = Model()
    old = model.prepare_encoder_cudagraph_replay_buffers
    # Preserve an existing instance-bound override as well as class methods.
    model.prepare_encoder_cudagraph_replay_buffers = old
    with (
        pytest.raises(RuntimeError, match="caller failure"),
        cached_replay_metadata(model, GeometryCache()),
    ):
        raise RuntimeError("caller failure")
    assert model.prepare_encoder_cudagraph_replay_buffers is old


def test_profiler_wrapped_original_source_is_preserved():
    model = Model()
    original = model.prepare_encoder_cudagraph_replay_buffers

    @wraps(original)
    def profiler_wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    model.prepare_encoder_cudagraph_replay_buffers = profiler_wrapper
    with cached_replay_metadata(model, GeometryCache()):
        result = model.prepare_encoder_cudagraph_replay_buffers(
            {"patches": Pixels("current"), "vit_grid": [[4, 6]]}, 8, 1
        )
        assert result.values["patches"].identity == "current"
    assert model.prepare_encoder_cudagraph_replay_buffers is profiler_wrapper

# SPDX-License-Identifier: Apache-2.0
"""Scoped geometry reuse for the native DeepSeek vision graph replay path.

Only the immutable, unpadded geometry is cached. The original graph manager
still prepares and pads its static replay buffers on every call. Pixel tensors
are obtained from the current call, including any required dtype conversion.
"""

import inspect
import sys
from contextlib import contextmanager
from types import MethodType


@contextmanager
def cached_replay_metadata(model, cache):
    """Temporarily replace one model instance's replay geometry preparation.

    Builders and the return type come from the original bound method's module,
    preserving the exact source overlay used by the benchmark. The cache must
    be instance-local and must not be shared across models with different
    metadata implementations. Cold cache creation is visible to the caller.
    """
    name = "prepare_encoder_cudagraph_replay_buffers"
    original = getattr(model, name)
    source = sys.modules[inspect.unwrap(original).__module__]
    had_override = name in vars(model)
    old_override = vars(model).get(name)

    def prepare(self, mm_kwargs, max_batch_size, max_frames_per_batch, path="default"):
        vit_grid = self._get_grid_list(mm_kwargs, "vit_grid")
        patches = mm_kwargs["patches"]
        dtype = self.aligner.w1.weight.dtype
        if patches.dtype != dtype:
            patches = patches.to(dtype)

        def build():
            metadata = source.build_packed_vit_metadata(
                vit_grid,
                rope_dim=self.vision.rope_dim,
                rope_theta=self.vision.rope_theta,
                device=patches.device,
            )
            merge = source.build_packed_merge_metadata(
                vit_grid,
                self.config.vision_downsample_ratio,
                device=patches.device,
                dtype=dtype,
            )
            return {**metadata, **merge}

        geometry = cache.get(
            grids=vit_grid,
            device=patches.device,
            dtype=dtype,
            rope_dim=self.vision.rope_dim,
            rope_theta=self.vision.rope_theta,
            downsample_ratio=self.config.vision_downsample_ratio,
            build=build,
        )
        return source.EncoderCudaGraphReplayBuffers(
            values={"patches": patches, **geometry}
        )

    setattr(model, name, MethodType(prepare, model))
    try:
        yield cache
    finally:
        if had_override:
            setattr(model, name, old_override)
        else:
            delattr(model, name)

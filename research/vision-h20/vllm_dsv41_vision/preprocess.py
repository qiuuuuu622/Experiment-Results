# SPDX-License-Identifier: Apache-2.0
"""Ordered, CPU-only batching around the unchanged native image transform."""

from concurrent.futures import ThreadPoolExecutor

import torch


class ParallelImagePreprocessor:
    """Own a bounded thread pool for independent image preprocessing.

    Args:
        image_processor: Native configured callable returning patches and the
            four grid dimensions for one decoded image.
        image_token_types: Native callable creating one image's token roles.
        max_workers: Positive number of image workers. Torch CPU thread counts
            remain the caller's responsibility; this class changes no globals.

    Calls return the native CPU tensor fields in input order, or an empty dict
    for no inputs. Worker errors propagate to the caller. Close the instance or
    use it as a context manager; calls after closing raise RuntimeError.

    """

    def __init__(self, image_processor, image_token_types, max_workers=4):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.image_processor = image_processor
        self.image_token_types = image_token_types
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.closed = False

    def __call__(self, inputs, *, decode=None):
        """Process decoded images, or inputs transformed by optional decode."""
        if self.closed:
            raise RuntimeError("Preprocessor is closed")
        items = list(inputs)
        if not items:
            return {}

        def transform(item):
            return self.image_processor(decode(item) if decode else item)

        values = list(self.pool.map(transform, items))
        return {
            "patches": torch.cat([item[0] for item in values]),
            "vit_grid": torch.tensor([item[1:3] for item in values], dtype=torch.int64),
            "llm_grid": torch.tensor([item[3:5] for item in values], dtype=torch.int64),
            "types": torch.cat([self.image_token_types(*item[3:5]) for item in values]),
        }

    def close(self):
        self.pool.shutdown(wait=True)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

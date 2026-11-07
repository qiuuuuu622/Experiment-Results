# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch

from vllm_dsv41_vision.preprocess import ParallelImagePreprocessor


def test_parallel_completion_keeps_input_order_and_current_pixels():
    def transform(value):
        if value == 1:
            time.sleep(0.01)
        return torch.full((1, 3), value), 1, 1, 1, 1

    def roles(h, w):
        return torch.tensor([0, h, w, 3])

    with ParallelImagePreprocessor(transform, roles, 2) as processor:
        assert processor([]) == {}
        out = processor(["1", "2", "3"], decode=int)
        assert out["patches"][:, 0].tolist() == [1, 2, 3]
        out = processor([3, 2, 4])
        assert out["patches"][:, 0].tolist() == [3, 2, 4]
    with pytest.raises(RuntimeError, match="closed"):
        processor([1])


def test_worker_errors_propagate_and_pool_can_be_closed():
    def transform(value):
        raise ValueError("bad image")

    with (
        ParallelImagePreprocessor(transform, lambda h, w: None, 2) as processor,
        pytest.raises(ValueError, match="bad image"),
    ):
        processor([1])
    processor.close()

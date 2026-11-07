# SPDX-License-Identifier: Apache-2.0
"""Deterministic full/source top-k with explicit planning and owned workspace."""

import torch

from .native import load_kernels


class StableTopK:
    """Select finite visible FP32 scores, breaking exact ties by smaller position.

    Results are unordered unique logical positions, padded with -1. Scores may
    contain NaN outside lengths. Update lengths in place and call prepare()
    before execute() when persistent cluster routing may change. Planning and
    execution may be captured together for changing-length CUDA Graph replay.
    """

    def __init__(self, scores, lengths, *, k=512, backend="cluster"):
        if scores.dtype != torch.float32 or scores.ndim != 2:
            raise ValueError("scores must be a rank-two FP32 CUDA tensor")
        if not scores.is_cuda or scores.stride(1) != 1 or scores.stride(0) % 4:
            raise ValueError("scores need CUDA and a 16-byte-aligned row stride")
        rows, width = scores.shape
        if rows < 1 or width < 1 or not 1 <= k <= 2048:
            raise ValueError("Positive rows/width and top-k in [1,2048] required")
        if lengths.shape != (rows,) or lengths.dtype != torch.int32:
            raise ValueError("lengths must be int32 [rows]")
        if lengths.device != scores.device or not lengths.is_contiguous():
            raise ValueError("lengths must be contiguous on the score device")
        if ((lengths < 0) | (lengths > width)).any().item():
            raise ValueError("Visible lengths outside scores")
        if backend not in ("cluster", "streaming"):
            raise ValueError("backend must be cluster or streaming")
        self.scores, self.lengths, self.backend = scores, lengths, backend
        self.native = load_kernels("full" if backend == "cluster" else "streaming")
        self.output = torch.empty((rows, k), device=scores.device, dtype=torch.int32)
        self.metadata = torch.zeros(
            (rows + 1, 2), device=scores.device, dtype=torch.int32
        )
        self.cluster_floor = self.native.cluster_floor(rows)
        self.cluster_capacity = 2 * (
            torch.cuda.get_device_properties(scores.device).multi_processor_count // 8
        )
        self.needs_plan = (
            backend == "cluster"
            and width > self.cluster_floor
            and self.cluster_capacity < rows <= 512
        )
        self.prepare()

    def prepare(self):
        if self.needs_plan:
            self.native.plan(self.lengths, self.metadata, self.cluster_floor)
        return self.metadata

    def execute(self):
        self.native.topk(
            self.scores, self.lengths, None, self.output, 1, self.metadata, None
        )
        return self.output

    def forward(self):
        self.prepare()
        return self.execute()


class RaggedTopK:
    """Prefill windows with logical rebasing; consumes the score workspace.

    A selected column c in [start,start+length) maps to c-start+offset. The
    vendored kernel may mask up to three columns before start for aligned reads.
    No later consumer may depend on those invalid score-prefix values.
    """

    def __init__(self, scores, lengths, starts, offsets, *, k=512):
        # Reuse the same bounds/layout checks, then check the packed windows.
        self.selector = StableTopK(scores, lengths, k=k, backend="streaming")
        rows = scores.shape[0]
        for name, tensor in (("starts", starts), ("offsets", offsets)):
            if tensor.shape != (rows,) or tensor.dtype != torch.int32:
                raise ValueError(f"{name} must be int32 [rows]")
            if tensor.device != scores.device or not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must share the score device and be contiguous"
                )
        if ((starts < 0) | (starts + lengths > scores.shape[1])).any().item():
            raise ValueError("Ragged window outside score allocation")
        self.starts, self.offsets = starts, offsets
        self.output = self.selector.output

    def execute(self):
        s = self.selector
        s.native.ragged(s.scores, s.lengths, self.starts, self.offsets, self.output)
        return self.output

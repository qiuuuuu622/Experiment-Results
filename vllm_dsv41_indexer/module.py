# SPDX-License-Identifier: Apache-2.0
"""Explicit module experiment API. Construction/planning are outside execution."""

from dataclasses import dataclass

import torch

from .contract import IndexerShape
from .full_score import grouped_scores
from .native import load_kernels
from .workspace import _tensor


@dataclass
class IndexerInputs:
    q: torch.Tensor
    weights: torch.Tensor
    cache: torch.Tensor
    table: torch.Tensor
    requests: torch.Tensor
    lengths: torch.Tensor
    shape: IndexerShape

    def validate(self):
        r, dev = self.shape.rows, self.q.device
        if dev.type != "cuda":
            raise ValueError("Module experiments require CUDA")
        _tensor(self.q, (r, 32, 128), torch.float8_e4m3fn, dev, "Q")
        _tensor(self.weights, (r, 32), torch.float32, dev, "weights")
        _tensor(
            self.cache, (self.cache.shape[0], 64, 1, 132), torch.uint8, dev, "cache"
        )
        _tensor(self.table, self.table.shape, torch.int32, dev, "page table")
        _tensor(self.requests, (r,), torch.int32, dev, "requests")
        _tensor(self.lengths, (r,), torch.int32, dev, "lengths")
        if self.table.ndim != 2 or self.table.shape[1] * 64 < self.shape.max_keys:
            raise ValueError("Page table cannot address max_keys")
        if ((self.requests < 0) | (self.requests >= self.table.shape[0])).any().item():
            raise ValueError("Request outside local page table")
        if ((self.lengths < 0) | (self.lengths > self.shape.max_keys)).any().item():
            raise ValueError("Visible key count outside planned width")
        if ((self.table < 0) | (self.table >= self.cache.shape[0])).any().item():
            raise ValueError("Physical page outside rank-local cache")


def plan_groups(request_ids, group_rows=1):
    """Consecutive same-request rows, split at selected tile size; ragged padded."""
    if group_rows not in (1, 2, 4, 8):
        raise ValueError("Tile group_rows must be 1, 2, 4, or 8")
    groups, current = [], []
    for row, request in enumerate(request_ids):
        if current and (
            request_ids[current[0]] != request or len(current) == group_rows
        ):
            groups.append(current + [-1] * (group_rows - len(current)))
            current = []
        current.append(row)
    if current:
        groups.append(current + [-1] * (group_rows - len(current)))
    return groups


class FullIndexer:
    """Full/source FP8 scores and deterministic top-k in local index coordinates.

    Row grouping can change via replan() outside capture. During graph replay,
    requests/lengths may change as long as each group still names one request.
    The score budget is enforced; use IndexerShape.chunks for larger modules.
    Source block maxima are optional and share the score kernel's key loads.
    """

    def __init__(
        self,
        inputs,
        *,
        group_rows=1,
        source=False,
        keys_major=False,
        precise=False,
        fuse_source_max=True,
        score_workspace=None,
    ):
        inputs.validate()
        self.inputs, self.group_rows = inputs, group_rows
        self.source, self.keys_major = source, keys_major
        self.precise = precise
        self.fuse_source_max = fuse_source_max
        s, dev = inputs.shape, inputs.q.device
        width = (s.max_keys + 63) // 64 * 64
        if s.rows * width * 4 > s.score_budget:
            raise ValueError("Use planned chunks to respect the 1 GiB score budget")
        self.native = load_kernels()
        self.scores = (
            torch.empty((s.rows, width), device=dev, dtype=torch.float32)
            if score_workspace is None
            else score_workspace[: s.rows]
        )
        _tensor(self.scores, (s.rows, width), torch.float32, dev, "score workspace")
        self.output = torch.empty((s.rows, s.topk), device=dev, dtype=torch.int32)
        self.plan = torch.zeros((s.rows + 1, 2), device=dev, dtype=torch.int32)
        self.maxima = torch.empty((s.rows, width // 8), device=dev) if source else None
        self.block_counts = torch.empty(s.rows, device=dev, dtype=torch.int32)
        self.candidates = torch.empty((s.rows, 2048), device=dev, dtype=torch.int32)
        self.replan()

    def replan(self):
        i = self.inputs
        self.groups = torch.tensor(
            plan_groups(i.requests.tolist(), self.group_rows),
            device=i.q.device,
            dtype=torch.int32,
        )

    def score(self):
        i = self.inputs
        grouped_scores(
            i.q,
            i.weights,
            i.cache,
            i.table,
            i.requests,
            i.lengths,
            self.groups,
            self.scores,
            maxima=self.maxima if self.fuse_source_max else None,
            keys_major=self.keys_major,
            precise=self.precise,
        )
        if self.source and not self.fuse_source_max:
            from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
                _block_scores_kernel,
            )

            blocks = self.maxima.shape[1]
            _block_scores_kernel[(i.shape.rows, (blocks + 127) // 128)](
                self.scores,
                None,
                i.lengths,
                self.maxima,
                *self.scores.stride(),
                0,
                1,
                self.scores.shape[1],
                blocks,
                8,
                False,
                1,
                128,
            )
        return self.scores

    def select(self):
        self.native.topk(
            self.scores, self.inputs.lengths, None, self.output, 1, self.plan, None
        )
        return self.output

    def forward(self):
        self.score()
        self.select()
        if self.source:
            self.select_source()
        return self.output

    def select_source(self):
        from .full_score import _block_lengths

        if self.maxima is None:
            raise RuntimeError("Enable source during initialization")
        _block_lengths[((self.inputs.shape.rows + 127) // 128,)](
            self.inputs.lengths, self.block_counts, self.inputs.shape.rows, 128
        )
        self.native.topk(
            self.maxima, self.block_counts, None, self.candidates, 1, self.plan, None
        )
        return self.candidates


class ChunkedIndexer:
    """Execute full/source groups serially using a shared <=1 GiB score arena."""

    def __init__(self, inputs, *, group_rows=1, source=False, **kwargs):
        inputs.validate()
        requests = inputs.requests.tolist()
        sizes = []
        previous = None
        for request in requests:
            if request == previous:
                sizes[-1] += 1
            else:
                sizes.append(1)
            previous = request
        self.ranges = inputs.shape.chunks(sizes)
        width = (inputs.shape.max_keys + 63) // 64 * 64
        capacity = max(end - start for start, end in self.ranges)
        self.arena = torch.empty(
            (capacity, width), device=inputs.q.device, dtype=torch.float32
        )
        self.output = torch.empty(
            (inputs.shape.rows, 512), device=inputs.q.device, dtype=torch.int32
        )
        self.candidates = (
            torch.empty(
                (inputs.shape.rows, 2048), device=inputs.q.device, dtype=torch.int32
            )
            if source
            else None
        )
        self.parts = []
        for start, end in self.ranges:
            shape = IndexerShape(
                end - start,
                inputs.shape.max_keys,
                inputs.shape.ratio,
                score_budget=inputs.shape.score_budget,
            )
            local = IndexerInputs(
                inputs.q[start:end],
                inputs.weights[start:end],
                inputs.cache,
                inputs.table,
                inputs.requests[start:end],
                inputs.lengths[start:end],
                shape,
            )
            part = FullIndexer(
                local,
                group_rows=group_rows,
                source=source,
                score_workspace=self.arena,
                **kwargs,
            )
            part.output = self.output[start:end]
            if source:
                part.candidates = self.candidates[start:end]
            self.parts.append(part)

    def forward(self):
        for part in self.parts:
            part.forward()
        return self.output


class NativeFullIndexer:
    """Current-HEAD DeepGEMM + native selector algorithm, explicit preparation.

    Regular groups of 1/2/4 use native next_n; other/ragged groups use next_n=1.
    DeepGEMM owns its returned logits allocation, an intrinsic baseline cost.
    Call prepare() whenever request mapping or lengths change. Include that cost
    when measuring changing metadata, separately from a preplanned graph replay.
    """

    def __init__(self, inputs, *, next_n=1):
        from vllm.utils.deep_gemm import native_next_n_supported

        inputs.validate()
        self.inputs = inputs
        self.next_n = next_n if native_next_n_supported(next_n) else 1
        self.baseline_mode = (
            "native_grouped" if self.next_n == next_n else "expanded_single_row"
        )
        r = inputs.shape.rows
        if r % self.next_n:
            raise ValueError("Native grouping must divide rows")
        host = inputs.requests.tolist()
        if any(
            len(set(host[s : s + self.next_n])) != 1 for s in range(0, r, self.next_n)
        ):
            raise ValueError("Native next_n groups must share a request")
        self.workspace = torch.empty(1 << 20, device=inputs.q.device, dtype=torch.uint8)
        self.output = torch.empty((r, 512), device=inputs.q.device, dtype=torch.int32)
        self.backend = "cooperative" if r <= 64 else "persistent"
        self.prepare()

    def prepare(self):
        from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

        i = self.inputs
        self.local_table = i.table[i.requests[:: self.next_n].long()].contiguous()
        self.lens = i.lengths.view(-1, self.next_n)
        self.metadata = get_paged_mqa_logits_metadata(
            self.lens,
            64,
            torch.cuda.get_device_properties(i.q.device).multi_processor_count,
        )

    def score(self):
        from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

        i = self.inputs
        self.scores = fp8_fp4_paged_mqa_logits(
            (i.q.view(-1, self.next_n, 32, 128), None),
            i.cache,
            i.weights,
            self.lens,
            self.local_table,
            self.metadata,
            i.shape.max_keys,
            clean_logits=False,
        )
        return self.scores

    def forward(self):
        self.score()
        return self.select()

    def select(self):
        getattr(torch.ops._C, self.backend + "_topk")(
            self.scores,
            self.lens,
            self.output,
            self.workspace,
            512,
            self.inputs.shape.max_keys,
        )
        return self.output


class NativeSourceIndexer(NativeFullIndexer):
    """Keep DeepGEMM scoring; use deterministic token and source block selection."""

    def __init__(self, inputs, *, next_n=1):
        super().__init__(inputs, next_n=next_n)
        rows = inputs.shape.rows
        width = (inputs.shape.max_keys + 31) // 32 * 4
        self.maxima = torch.empty((rows, width), device=inputs.q.device)
        self.candidates = torch.empty(
            (rows, 2048), device=inputs.q.device, dtype=torch.int32
        )
        self.block_counts = torch.empty(rows, device=inputs.q.device, dtype=torch.int32)
        self.source_plan = torch.zeros(
            (rows + 1, 2), device=inputs.q.device, dtype=torch.int32
        )
        self.topk = load_kernels()

    def select_source(self):
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
            _block_scores_kernel,
        )

        from .full_score import _block_lengths

        blocks = self.maxima.shape[1]
        rows = self.inputs.shape.rows
        _block_scores_kernel[(rows, (blocks + 127) // 128)](
            self.scores,
            None,
            self.inputs.lengths,
            self.maxima,
            *self.scores.stride(),
            0,
            1,
            self.scores.shape[1],
            blocks,
            8,
            False,
            1,
            128,
        )
        _block_lengths[((rows + 127) // 128,)](
            self.inputs.lengths, self.block_counts, rows, 128
        )
        self.topk.topk(
            self.maxima,
            self.block_counts,
            None,
            self.candidates,
            1,
            self.source_plan,
            None,
        )
        return self.candidates

    def select(self):
        self.topk.topk(
            self.scores,
            self.inputs.lengths,
            None,
            self.output,
            1,
            self.source_plan,
            None,
        )
        return self.output

    def forward(self):
        super().forward()
        self.select_source()
        return self.output

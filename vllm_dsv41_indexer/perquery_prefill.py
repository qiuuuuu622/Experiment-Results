# SPDX-License-Identifier: Apache-2.0
"""Bounded per-query FP8 compact prefill consumers; no request aliases.

The gathered backend is a custom batched WGMMA GEMM, not DeepGEMM. Both
backends retain the existing P2 dot/reduction and stable logical selector.
"""

import torch
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from vllm.triton_utils import gl, gluon, tl, triton

from .score import _candidate_wgmma_kernel, _row_head_sum
from .workspace import CandidateState


@triton.jit
def _gather_perquery(
    CACHE,
    TABLE,
    REQS,
    BLOCKS,
    COUNTS,
    KEYS,
    SCALES,
    TABLE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * 64 + tl.arange(0, 64)
    count = tl.load(COUNTS + row)
    valid = col < count
    request = tl.load(REQS + row).to(tl.int64)
    block = tl.load(BLOCKS + row * 2048 + col // 8, valid, 0).to(tl.int64)
    logical = block * 8 + col % 8
    page = tl.load(TABLE + request * TABLE_STRIDE + logical // 64, valid, 0).to(
        tl.int64
    )
    d = tl.arange(0, 128)
    payload = tl.load(
        CACHE + page[:, None] * 8448 + (logical % 64)[:, None] * 128 + d[None, :],
        valid[:, None],
        0,
    )
    scale = tl.load(
        (CACHE + page * 8448 + 8192 + logical % 64 * 4).to(tl.pointer_type(tl.float32)),
        valid,
        1.0,
    )
    # Fill the fixed staging extent so the GEMM never consumes stale payload.
    tl.store(KEYS + (row * WIDTH + col[:, None]) * 128 + d[None, :], payload)
    tl.store(SCALES + row * WIDTH + col, scale)


@gluon.jit
def _gathered_gemm(
    q_desc,
    W,
    KEYS,
    SCALES,
    COUNTS,
    OUT,
    WIDTH: gl.constexpr,
    WEIGHT_STRIDE: gl.constexpr,
):
    row = gl.program_id(0).to(gl.int64)
    lane = gl.program_id(1)
    step = gl.num_programs(1)
    count = gl.load(COUNTS + row)
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 32]
    )
    layout3: gl.constexpr = gl.BlockedLayout(
        [1, 1, 16], [8, 4, 1], [4, 1, 1], [2, 1, 0]
    )
    key_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout3))
    q_smem = gl.allocate_shared_memory(gl.float8e4nv, [32, 128], q_desc.layout)
    k_smem = gl.allocate_shared_memory(
        gl.float8e4nv,
        [64, 128],
        gl.NVMMASharedLayout.get_default_for([64, 128], gl.float8e4nv),
    )
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, 32 * 128)
    tma.async_copy_global_to_shared(q_desc, [(row * 32).to(gl.int32), 0], bar, q_smem)
    mbarrier.wait(bar, 0)
    mbarrier.invalidate(bar)
    heads = gl.arange(0, 32, layout=gl.SliceLayout(0, mma))
    weight = gl.load(W + row * WEIGHT_STRIDE + heads)
    acc = gl.zeros([64, 32], gl.float32, layout=mma)
    weight = gl.broadcast(gl.expand_dims(weight, 0), acc)[0]
    block = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout3)))
    byte = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout3)))
    dimension = gl.expand_dims(gl.expand_dims(block * 32, 0), 2) + gl.expand_dims(
        gl.expand_dims(byte, 0), 1
    )
    for tile in range(lane, gl.cdiv(count, 64), step):
        cols = tile * 64 + gl.arange(0, 64, layout=key_layout)
        base = gl.expand_dims(gl.expand_dims((row * WIDTH + cols) * 128, 1), 2)
        payload = gl.load(
            KEYS + base + dimension,
            gl.expand_dims(gl.expand_dims(cols < count, 1), 2),
            0,
        )
        scale = gl.load(SCALES + row * WIDTH + cols, cols < count, 1.0)
        k = gl.reshape(payload, [64, 128]).to(gl.float8e4nv, bitcast=True)
        gl.barrier()
        k_smem.store(k)
        fence_async_shared()
        gl.barrier()
        acc = warpgroup_mma(k_smem, q_smem.permute((1, 0)), acc, use_acc=False)
        scale = gl.convert_layout(scale, gl.SliceLayout(1, acc.type.layout))
        acc = acc * gl.expand_dims(scale, 1)
        logits = _row_head_sum(gl.maximum(acc, 0.0) * weight.to(gl.float32))
        outcols = tile * 64 + gl.arange(0, 64, layout=logits.type.layout)
        gl.store(
            OUT + row * WIDTH + outcols,
            gl.where(outcols < count, logits, float("-inf")),
        )


class PerQueryPrefillIndexer:
    """One publication per call; sequential chunks reuse bounded score/staging.

    Candidates are explicit [Q,2048] logical blocks and requests are real
    request IDs. Plans depend on shape/storage, not request ownership or length
    values, so in-place request/page/length/candidate updates are graph-safe.
    """

    def __init__(
        self,
        inputs,
        candidate_blocks,
        *,
        backend="direct",
        source_unique=False,
        score_budget=1 << 30,
        gather_budget=256 << 20,
        max_chunk_rows=None,
    ):
        inputs.validate()
        if backend not in ("direct", "gathered_wgmma"):
            raise ValueError("backend must be direct or gathered_wgmma")
        rows = inputs.shape.rows
        if (
            candidate_blocks.shape != (rows, 2048)
            or candidate_blocks.dtype != torch.int32
            or not candidate_blocks.is_contiguous()
            or candidate_blocks.device != inputs.q.device
        ):
            raise ValueError(
                "Candidates must be contiguous int32 [Q,2048] on query device"
            )
        self.inputs = inputs
        self.candidate_input = candidate_blocks
        self.backend = backend
        self.source_unique = source_unique
        self.width = min(16384, triton.cdiv(inputs.shape.max_keys, 64) * 64)
        capacity = score_budget // (4 * self.width)
        if backend == "gathered_wgmma":
            capacity = min(capacity, gather_budget // (132 * self.width))
        if max_chunk_rows is not None:
            capacity = min(capacity, max_chunk_rows)
        if capacity < 1:
            raise ValueError("Budgets must fit one compact row")
        self.capacity = min(rows, capacity)
        self.chunks = [
            (start, min(start + self.capacity, rows))
            for start in range(0, rows, self.capacity)
        ]
        device = inputs.q.device
        self.state = CandidateState(rows, device, compress_ratio=inputs.shape.ratio)
        self.scores = torch.empty(
            (self.capacity, self.width), device=device, dtype=torch.float32
        )
        self.output = torch.empty((rows, 512), device=device, dtype=torch.int32)
        self.keys = (
            torch.empty(
                (self.capacity, self.width, 128), device=device, dtype=torch.uint8
            )
            if backend == "gathered_wgmma"
            else None
        )
        self.scales = (
            torch.empty((self.capacity, self.width), device=device, dtype=torch.float32)
            if self.keys is not None
            else None
        )
        layout = gl.NVMMASharedLayout.get_default_for([32, 128], gl.float8e4nv)
        self.q_descriptors = [
            TensorDescriptor.from_tensor(
                inputs.q[start:end].view((end - start) * 32, 128), [32, 128], layout
            )
            for start, end in self.chunks
        ]
        self.score_bytes = self.scores.numel() * 4
        self.gather_bytes = (
            0 if self.keys is None else self.keys.numel() + self.scales.numel() * 4
        )
        assert self.score_bytes <= score_budget and self.gather_bytes <= gather_budget

    def publish(self):
        self.state.publish(
            self.candidate_input, self.inputs.lengths, source_unique=self.source_unique
        )

    def gather_chunk(self, chunk):
        if self.keys is not None:
            start, end = self.chunks[chunk]
            i = self.inputs
            _gather_perquery[(end - start, self.width // 64)](
                i.cache,
                i.table,
                i.requests[start:end],
                self.state.blocks[start:end],
                self.state.counts[start:end],
                self.keys,
                self.scales,
                i.table.stride(0),
                self.width,
            )

    def score_chunk(self, chunk):
        start, end = self.chunks[chunk]
        rows = end - start
        i = self.inputs
        lanes = min(triton.cdiv(self.width, 64), triton.cdiv(4096, rows))
        if self.backend == "direct":
            self.compiled_kernel = _candidate_wgmma_kernel[(rows, lanes)](
                self.q_descriptors[chunk],
                i.weights[start:end],
                i.table,
                i.requests[start:end],
                self.state.blocks[start:end],
                self.state.counts[start:end],
                i.cache,
                self.scores,
                self.width,
                64,
                i.cache.stride(0),
                i.weights.stride(0),
                i.table.stride(0),
                self.state.blocks.stride(0),
                self.width,
                CACHE_PAGE_SIZE=64,
                CANDIDATE_BLOCK_SIZE=8,
                num_warps=4,
            )
        else:
            self.compiled_kernel = _gathered_gemm[(rows, lanes)](
                self.q_descriptors[chunk],
                i.weights[start:end],
                self.keys,
                self.scales,
                self.state.counts[start:end],
                self.scores,
                self.width,
                i.weights.stride(0),
                num_warps=4,
            )
        return self.scores[:rows]

    def select_chunk(self, chunk):
        start, end = self.chunks[chunk]
        self.state.native.topk(
            self.scores[: end - start],
            self.state.counts[start:end],
            self.state.blocks[start:end],
            self.output[start:end],
            8,
            self.state.plan[: end - start + 1],
            None,
        )

    def forward(self):
        self.publish()
        for chunk in range(len(self.chunks)):
            self.gather_chunk(chunk)
            self.score_chunk(chunk)
            self.select_chunk(chunk)
        return self.output

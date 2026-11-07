# SPDX-License-Identifier: Apache-2.0
"""FP8 prefill module experiments with explicit per-request key preparation.

All positions/lengths are in index-key coordinates, including ratio=2. Keys stay
E4M3 and FP32 scales remain separate. Candidate sets are shared within a request;
independent per-query sets can be represented by request aliases with identical
physical page tables. No cache is reused implicitly across calls or steps.
"""

import torch
import triton
import triton.language as tl

from .module import IndexerInputs
from .native import load_kernels
from .workspace import CandidateState


@triton.jit
def _request_lengths(
    REQUESTS, LENGTHS, OUT, ROWS: tl.constexpr, NREQ: tl.constexpr, TILE: tl.constexpr
):
    req = tl.program_id(0)
    row = tl.arange(0, TILE)
    maximum = tl.full((), 0, tl.int32)
    for start in range(0, ROWS, TILE):
        r = start + row
        owner = tl.load(REQUESTS + r, r < ROWS, -1)
        length = tl.load(LENGTHS + r, r < ROWS, 0)
        maximum = tl.maximum(maximum, tl.max(tl.where(owner == req, length, 0), 0))
    tl.store(OUT + req, maximum)


@triton.jit
def _gather_keys(
    CACHE,
    TABLE,
    LENS,
    BLOCKS,
    COUNTS,
    KEYS,
    SCALES,
    WIDTH: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    COMPACT: tl.constexpr,
    TILE: tl.constexpr,
):
    req, tile = tl.program_id(1), tl.program_id(0)
    cols = tile * TILE + tl.arange(0, TILE)
    length = tl.load(COUNTS + req) if COMPACT else tl.load(LENS + req)
    valid = cols < length
    logical = cols
    if COMPACT:
        block = tl.load(BLOCKS + req * 2048 + cols // 8, valid, 0)
        logical = block * 8 + cols % 8
    pages = tl.load(TABLE + req * TABLE_STRIDE + logical // 64, valid, 0).to(tl.int64)
    d = tl.arange(0, 128)
    payload = tl.load(
        CACHE + pages[:, None] * 8448 + (logical % 64)[:, None] * 128 + d[None, :],
        valid[:, None],
        0,
    )
    scale = tl.load(
        (CACHE + pages * 8448 + 8192 + logical % 64 * 4).to(
            tl.pointer_type(tl.float32)
        ),
        valid,
        1.0,
    )
    tl.store(
        KEYS + (req * WIDTH + cols[:, None]) * 128 + d[None, :],
        payload,
        cols[:, None] < WIDTH,
    )
    tl.store(SCALES + req * WIDTH + cols, scale, cols < WIDTH)


@triton.jit
def _compact_counts(REQUESTS, LENGTHS, BLOCKS, OUT, ROWS: tl.constexpr):
    row = tl.program_id(0)
    req = tl.load(REQUESTS + row)
    length = tl.load(LENGTHS + row)
    col = tl.arange(0, 2048)
    block = tl.load(BLOCKS + req * 2048 + col).to(tl.int64)
    n = tl.minimum(tl.maximum(length - block * 8, 0), 8)
    tl.store(OUT + row, tl.sum(n, 0).to(tl.int32))


@triton.jit
def _score(
    Q,
    W,
    CACHE,
    TABLE,
    POOL,
    SCALES,
    BLOCKS,
    LENGTHS,
    OUT,
    REQUEST: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    GROUP: tl.constexpr,
    REUSE: tl.constexpr,
    BK: tl.constexpr,
    INDIRECT: tl.constexpr,
    COMPACT: tl.constexpr,
    KEYS_MAJOR: tl.constexpr,
):
    key_tile = tl.program_id(0) if KEYS_MAJOR else tl.program_id(1)
    query_tile = tl.program_id(1) if KEYS_MAJOR else tl.program_id(0)
    keys = key_tile * BK + tl.arange(0, BK)
    d = tl.arange(0, 128)
    # A keys-major CTA keeps this key tile for REUSE groups of queries.
    rr = query_tile * GROUP * REUSE + tl.arange(0, GROUP * REUSE)
    limits = tl.load(LENGTHS + rr, rr < ROWS, 0)
    maximum = tl.max(limits, 0)
    if key_tile * BK < maximum:
        valid = (keys < maximum) & (keys < WIDTH)
        if INDIRECT:
            logical = keys
            if COMPACT:
                block = tl.load(BLOCKS + REQUEST * 2048 + keys // 8, valid, 0)
                logical = block * 8 + keys % 8
            page = tl.load(TABLE + REQUEST * TABLE_STRIDE + logical // 64, valid, 0).to(
                tl.int64
            )
            payload = tl.load(
                CACHE
                + page[:, None] * 8448
                + (logical % 64)[:, None] * 128
                + d[None, :],
                valid[:, None],
                0,
            ).to(tl.float8e4nv, bitcast=True)
            scales = tl.load(
                (CACHE + page * 8448 + 8192 + logical % 64 * 4).to(
                    tl.pointer_type(tl.float32)
                ),
                valid,
                1.0,
            )
        else:
            payload = tl.load(
                POOL + (REQUEST * WIDTH + keys[:, None]) * 128 + d[None, :],
                valid[:, None],
                0.0,
            )
            scales = tl.load(SCALES + REQUEST * WIDTH + keys, valid, 1.0)
        for part in tl.static_range(REUSE):
            h = tl.arange(0, GROUP * 32)
            rows = query_tile * GROUP * REUSE + part * GROUP + h // 32
            query = tl.load(
                Q + rows[:, None] * 4096 + (h % 32)[:, None] * 128 + d[None, :],
                rows[:, None] < ROWS,
                0.0,
            )
            weights = tl.load(W + rows * 32 + h % 32, rows < ROWS, 0.0)
            dot = tl.dot(
                payload,
                tl.trans(query),
                out_dtype=tl.float32,
                max_num_imprecise_acc=128,
            )
            weighted = tl.maximum(dot * scales[:, None], 0.0) * weights[None, :]
            logits = tl.trans(tl.sum(tl.reshape(weighted, [BK, GROUP, 32]), 2))
            r = query_tile * GROUP * REUSE + part * GROUP + tl.arange(0, GROUP)
            lens = tl.load(LENGTHS + r, r < ROWS, 0)
            logits = tl.where(keys[None, :] < lens[:, None], logits, -float("inf"))
            tl.store(
                OUT + r[:, None] * WIDTH + keys[None, :],
                logits,
                (r[:, None] < ROWS) & (keys[None, :] < WIDTH),
            )


class PrefillIndexer:
    """Prepare keys once, score contiguous request chunks, return logical top-k.

    Backends: deepgemm (gather+native FP8 GEMM), row_tiled (gather+custom),
    keys_major (gather+K reuse), indirect (no gather, K reuse). Score allocation
    is at most score_budget. A request may have many chunks; its keys are still
    prepared once in each forward(). Constructor/plans/JIT are outside timing.
    """

    def __init__(
        self,
        inputs: IndexerInputs,
        *,
        backend="deepgemm",
        candidate_blocks=None,
        score_budget=1 << 30,
        group_rows=4,
        query_reuse=4,
        source=False,
        selector="ant",
        native_gather=False,
    ):
        inputs.validate()
        if backend not in ("deepgemm", "row_tiled", "keys_major", "indirect", "gluon"):
            raise ValueError("Unknown prefill backend")
        if group_rows not in (1, 2, 4, 8) or query_reuse not in (1, 2, 4, 8):
            raise ValueError("Unsupported query tile")
        if candidate_blocks is not None and backend == "gluon":
            raise ValueError(
                "Gluon prefill comparison currently covers full/source only"
            )
        if candidate_blocks is not None and source:
            raise ValueError("A source must score full keys")
        self.inputs, self.backend, self.source = inputs, backend, source
        if selector not in ("ant", "native"):
            raise ValueError("Unknown selector")
        if native_gather and (backend != "deepgemm" or candidate_blocks is not None):
            raise ValueError("Native gather requires full DeepGEMM mode")
        self.selector, self.native_gather = selector, native_gather
        self.group_rows, self.query_reuse = group_rows, query_reuse
        self.rows, self.nreq = inputs.shape.rows, inputs.table.shape[0]
        self.device = inputs.q.device
        self.compact = candidate_blocks is not None
        padded_keys = (inputs.shape.max_keys + 63) // 64 * 64
        self.width = min(16384, padded_keys) if self.compact else padded_keys
        self.candidate_input = candidate_blocks
        if self.compact and (
            candidate_blocks.shape != (self.nreq, 2048)
            or candidate_blocks.dtype != torch.int32
            or candidate_blocks.device != self.device
            or not candidate_blocks.is_contiguous()
        ):
            raise ValueError(
                "Candidates must be int32 [requests,2048] on the query device"
            )
        self.state = (
            CandidateState(self.nreq, self.device, compress_ratio=inputs.shape.ratio)
            if self.compact
            else None
        )
        self.request_lengths = torch.empty(
            self.nreq, device=self.device, dtype=torch.int32
        )
        self.lengths = (
            torch.empty(self.rows, device=self.device, dtype=torch.int32)
            if self.compact
            else inputs.lengths
        )
        self.keys = (
            torch.empty(
                (self.nreq, self.width, 128),
                device=self.device,
                dtype=torch.float8_e4m3fn,
            )
            if backend not in ("indirect", "gluon")
            else None
        )
        self.scales = (
            torch.empty((self.nreq, self.width), device=self.device)
            if self.keys is not None
            else None
        )
        self.native = load_kernels()
        self.output = torch.empty(
            (self.rows, 512), device=self.device, dtype=torch.int32
        )
        capacity = score_budget // (4 * self.width)
        if capacity < 1:
            raise ValueError("One score row exceeds score budget")
        self.chunks = []
        host = inputs.requests.tolist()
        start = 0
        while start < self.rows:
            end = start + 1
            while (
                end < self.rows and host[end] == host[start] and end - start < capacity
            ):
                end += 1
            self.chunks.append((start, end, host[start]))
            start = end
        chunk_rows = max(end - start for start, end, _ in self.chunks)
        self.scores = (
            torch.empty((chunk_rows, self.width), device=self.device)
            if backend != "deepgemm"
            else None
        )
        self.plan = torch.zeros(
            (chunk_rows + 1, 2), device=self.device, dtype=torch.int32
        )
        self.starts = torch.zeros(self.rows, device=self.device, dtype=torch.int32)
        self.candidates = (
            torch.empty((self.rows, 2048), device=self.device, dtype=torch.int32)
            if source
            else None
        )
        self.maxima = (
            torch.empty((chunk_rows, self.width // 8), device=self.device)
            if source
            else None
        )
        self.block_counts = (
            torch.empty(chunk_rows, device=self.device, dtype=torch.int32)
            if source
            else None
        )
        self.score_budget = score_budget
        self.group_plans = {}
        if backend == "gluon":
            for start, end, request in self.chunks:
                size = end - start
                group = min(self.group_rows, triton.next_power_of_2(size))
                offsets = torch.arange(
                    triton.cdiv(size, group) * group,
                    device=self.device,
                    dtype=torch.int32,
                )
                self.group_plans[(start, end)] = torch.where(
                    offsets < size, offsets, -1
                ).view(-1, group)
        self.cu_lengths = torch.zeros(
            (self.nreq, 2), device=self.device, dtype=torch.int32
        )

    @property
    def preparation_bytes(self):
        return 0 if self.keys is None else self.keys.numel() + self.scales.numel() * 4

    def prepare(self):
        i = self.inputs
        _request_lengths[(self.nreq,)](
            i.requests, i.lengths, self.request_lengths, self.rows, self.nreq, 256
        )
        if self.compact:
            self.state.publish(self.candidate_input, self.request_lengths)
            _compact_counts[(self.rows,)](
                i.requests, i.lengths, self.state.blocks, self.lengths, self.rows
            )
        if self.native_gather:
            from vllm import _custom_ops as ops

            self.cu_lengths[:, 1].copy_(self.request_lengths)
            for request in range(self.nreq):
                ops.cp_gather_indexer_k_quant_cache(
                    i.cache.view(-1, 64, 132),
                    self.keys[request].view(torch.uint8),
                    self.scales[request].view(torch.uint8).view(-1, 4),
                    i.table[request : request + 1],
                    self.cu_lengths[request],
                )
        elif self.keys is not None:
            _gather_keys[(triton.cdiv(self.width, 64), self.nreq)](
                i.cache,
                i.table,
                self.request_lengths,
                self.state.blocks if self.compact else None,
                self.state.counts if self.compact else None,
                self.keys.view(torch.uint8),
                self.scales,
                self.width,
                i.table.stride(0),
                self.compact,
                64,
            )

    def score_chunk(self, start, end, request):
        i = self.inputs
        q, w = i.q[start:end], i.weights[start:end]
        lengths = self.lengths[start:end]
        if self.backend == "deepgemm":
            from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

            return fp8_fp4_mqa_logits(
                (q, None),
                (self.keys[request], self.scales[request]),
                w,
                self.starts[start:end],
                lengths,
                clean_logits=False,
            )
        rows = end - start
        if self.backend == "gluon":
            from .grouped_score import grouped_logits

            out = self.scores[:rows]
            groups = self.group_plans[(start, end)]
            grouped_logits(
                q,
                w,
                i.cache,
                i.table,
                i.requests[start:end],
                lengths,
                groups,
                out,
                maxima=self.maxima[:rows] if self.source else None,
                paired=True,
                pair_rows=min(rows, groups.shape[1]),
            )
            return out
        reuse = self.query_reuse if self.backend in ("keys_major", "indirect") else 1
        group = min(self.group_rows, triton.next_power_of_2(rows))
        major = self.backend != "row_tiled"
        grid = (
            (triton.cdiv(self.width, 64), triton.cdiv(rows, group * reuse))
            if major
            else (triton.cdiv(rows, group * reuse), triton.cdiv(self.width, 64))
        )
        out = self.scores[:rows]
        _score[grid](
            q,
            w,
            i.cache,
            i.table,
            self.keys,
            self.scales,
            self.state.blocks if self.compact else None,
            lengths,
            out,
            request,
            i.table.stride(0),
            self.width,
            rows,
            group,
            reuse,
            64,
            self.backend == "indirect",
            self.compact,
            major,
            num_warps=8 if group >= 4 else 4,
        )
        return out

    def select_chunk(self, scores, start, end, request):
        rows = end - start
        mapping = (
            self.state.blocks[request : request + 1].expand(rows, -1)
            if self.compact
            else None
        )
        if self.selector == "native":
            if self.compact:
                raise ValueError(
                    "Native selector benchmark uses full score coordinates"
                )
            from vllm import _custom_ops as ops

            ops.top_k_per_row_prefill(
                scores,
                self.starts[start:end],
                self.lengths[start:end],
                self.output[start:end],
                rows,
                *scores.stride(),
                512,
            )
        else:
            self.native.topk(
                scores,
                self.lengths[start:end],
                mapping,
                self.output[start:end],
                8 if self.compact else 1,
                self.plan[: rows + 1],
                None,
            )
        if self.source and self.selector == "native":
            from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
                select_candidate_blocks,
            )

            select_candidate_blocks(
                scores,
                None,
                self.lengths[start:end],
                2048,
                8,
                self.candidates[start:end],
            )
            return
        if self.source:
            from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
                _block_scores_kernel,
            )

            from .full_score import _block_lengths

            blocks = self.maxima.shape[1]
            if self.backend != "gluon":
                _block_scores_kernel[(rows, triton.cdiv(blocks, 128))](
                    scores,
                    None,
                    self.lengths[start:end],
                    self.maxima[:rows],
                    *scores.stride(),
                    0,
                    1,
                    self.width,
                    blocks,
                    8,
                    False,
                    1,
                    128,
                )
            _block_lengths[(triton.cdiv(rows, 128),)](
                self.lengths[start:end], self.block_counts, rows, 128
            )
            self.native.topk(
                self.maxima[:rows],
                self.block_counts[:rows],
                None,
                self.candidates[start:end],
                1,
                self.plan[: rows + 1],
                None,
            )

    def forward(self):
        self.prepare()
        for start, end, request in self.chunks:
            scores = self.score_chunk(start, end, request)
            self.select_chunk(scores, start, end, request)
            del scores
        return self.output

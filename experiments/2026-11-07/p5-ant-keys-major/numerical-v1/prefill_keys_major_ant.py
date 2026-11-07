# SPDX-License-Identifier: Apache-2.0
# Adapted from antgroup/sglang commit 6e6d5d9e13915adb01db0856b15d6908eaaa2c48.
# Original: python/sglang/kernels/ops/attention/dsv4/sm90_fp8_prefill_keys_major.py
# Original SHA256: 0bdde39dd1ddb4805492badbd838aa40a51a7dc79f29c070d27b661fc3f66b3f
# Original and license archived in results/.../p5-ant-keys-major/upstream/.
"""Dedicated prefill scheduler adapted to vLLM FP8 scales and FP32 epilogue.

Keeps 32-query/all-head shared Q, Q/K TMA, three K stages, asynchronous
head double buffering, and visible-tile persistent scheduling. No FP4 exactness
assumption or BF16 rounding is retained. Descriptor construction is explicit.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .prefill import PrefillIndexer

INDEX_HEAD_DIM = 128

# wgmma N (query rows per CTA). All heads of the rows stay in shared memory:
# 32 heads * 32 rows * 128 B = 128 KiB next to STAGES * 16 KiB of keys.
BLOCK_ROWS = 32
# wgmma M over two warpgroups.
BLOCK_KEYS = 128
STAGES = 3
NUM_WARPS = 8
MAX_HEADS = 32


@triton.jit(do_not_specialize=["B", "L", "n_row_blocks"])
def _visible_tile_prefix_kernel(
    lens_ptr,
    prefix_ptr,
    B,
    L,
    n_row_blocks,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """prefix[rb] = key tiles visible to some row of the row blocks before rb."""
    offs_c = tl.arange(0, CHUNK)
    offs_r = tl.arange(0, BLOCK_R)
    tl.store(prefix_ptr, 0)
    carry = tl.zeros([], dtype=tl.int32)
    for c0 in range(0, n_row_blocks, CHUNK):
        rbs = c0 + offs_c
        rows = rbs[:, None] * BLOCK_R + offs_r[None, :]
        lens = tl.load(lens_ptr + rows, mask=rows < B, other=0)
        vis = tl.maximum(tl.minimum(tl.max(lens, axis=1), L), 0)
        tiles = ((vis + BLOCK_K - 1) // BLOCK_K).to(tl.int32)
        tiles = tl.where(rbs < n_row_blocks, tiles, 0)
        tl.store(
            prefix_ptr + 1 + rbs,
            tl.cumsum(tiles, axis=0) + carry,
            mask=rbs < n_row_blocks,
        )
        carry += tl.sum(tiles, axis=0)


@gluon.jit
def _last_row_block_at(
    prefix_ptr, n_row_blocks, v, n_key_tiles, INVISIBLE: gl.constexpr
):
    """Largest row block whose first tile index is <= v.

    Visible tiles of row block rb start at prefix[rb]; invisible ones at
    rb * n_key_tiles - prefix[rb]. Both are non-decreasing in rb.
    """
    lo = 0
    hi = n_row_blocks
    while hi - lo > 1:
        mid = (lo + hi) // 2
        start = gl.load(prefix_ptr + mid)
        if INVISIBLE:
            start = mid * n_key_tiles - start
        right = start <= v
        lo = gl.where(right, mid, lo)
        hi = gl.where(right, hi, mid)
    return lo


@gluon.jit
def _issue_head(k_tile, q_head, c0, c1, c2, c3):
    # Four independent 32-wide FP8 partial dots; not assumed exact.
    # use_acc=False on recycled registers: no non-wgmma instruction writes an
    # accumulator while other wgmmas are in flight (ptxas would serialize).
    p0 = warpgroup_mma(
        k_tile.slice(0, 32, dim=1),
        q_head.slice(0, 32, dim=1).permute((1, 0)),
        c0,
        use_acc=False,
        is_async=True,
    )
    p1 = warpgroup_mma(
        k_tile.slice(32, 32, dim=1),
        q_head.slice(32, 32, dim=1).permute((1, 0)),
        c1,
        use_acc=False,
        is_async=True,
    )
    p2 = warpgroup_mma(
        k_tile.slice(64, 32, dim=1),
        q_head.slice(64, 32, dim=1).permute((1, 0)),
        c2,
        use_acc=False,
        is_async=True,
    )
    p3 = warpgroup_mma(
        k_tile.slice(96, 32, dim=1),
        q_head.slice(96, 32, dim=1).permute((1, 0)),
        c3,
        use_acc=False,
        is_async=True,
    )
    return p0, p1, p2, p3


@gluon.jit
def _head_epilogue(acc, p0, p1, p2, p3, w, key_scale):
    # Arbitrary FP8 partial dots need numerical validation, not FP4 exactness.
    score = ((p0 + p1) + p2) + p3
    weighted = gl.maximum(score * key_scale[:, None], 0.0) * w[None, :]
    return acc + weighted


@gluon.jit
def _load_row_block(
    q_desc,
    q_bar,
    q_smem,
    w_ptr,
    w_smem,
    lens_ptr,
    B,
    L,
    stride_wb,
    rb,
    q_phase,
    offs_r0,
    offs_w,
    H: gl.constexpr,
    D: gl.constexpr,
    BLOCK_R: gl.constexpr,
):
    """Queries and weights of row block rb into smem; returns visible lengths."""
    offs_r = rb * BLOCK_R + offs_r0
    n_vis = gl.minimum(gl.load(lens_ptr + offs_r, mask=offs_r < B, other=0), L).to(
        gl.int64
    )
    offs_wr = rb * BLOCK_R + offs_w % BLOCK_R
    w_blk = gl.load(
        w_ptr + offs_wr.to(gl.int64) * stride_wb + offs_w // BLOCK_R,
        mask=offs_wr < B,
        other=0.0,
    )
    w_smem.store(w_blk)
    mbarrier.expect(q_bar, H * BLOCK_R * D)
    for h in gl.static_range(H):
        tma.async_copy_global_to_shared(
            q_desc, [rb * BLOCK_R, h * D], q_bar, q_smem.index(h)
        )
    mbarrier.wait(q_bar, q_phase)
    return n_vis


@gluon.jit
def _score_tile(
    k_tile,
    q_smem,
    w_smem,
    key_scale,
    hz,
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    b2,
    b3,
    H: gl.constexpr,
    BLOCK_R: gl.constexpr,
    row_layout: gl.constexpr,
):
    """Scores of one key tile summed over heads; a* / b* are recycled buffers."""
    acc = gl.zeros(a0.shape, gl.float32, layout=a0.type.layout)
    # Fully unrolled and double-buffered over heads: the epilogue of head h
    # overlaps the wgmmas of head h + 1. A runtime head loop copies in-flight
    # accumulators through phis and ptxas serializes every wgmma.
    a0, a1, a2, a3 = _issue_head(k_tile, q_smem.index(hz), a0, a1, a2, a3)
    for h in gl.static_range(0, H - 2, 2):
        b0, b1, b2, b3 = _issue_head(k_tile, q_smem.index(hz + h + 1), b0, b1, b2, b3)
        w = w_smem.slice(h * BLOCK_R, BLOCK_R).load(row_layout)
        a0, a1, a2, a3 = warpgroup_mma_wait(4, deps=[a0, a1, a2, a3])
        acc = _head_epilogue(acc, a0, a1, a2, a3, w, key_scale)
        a0, a1, a2, a3 = _issue_head(k_tile, q_smem.index(hz + h + 2), a0, a1, a2, a3)
        w = w_smem.slice((h + 1) * BLOCK_R, BLOCK_R).load(row_layout)
        b0, b1, b2, b3 = warpgroup_mma_wait(4, deps=[b0, b1, b2, b3])
        acc = _head_epilogue(acc, b0, b1, b2, b3, w, key_scale)
    b0, b1, b2, b3 = _issue_head(k_tile, q_smem.index(hz + H - 1), b0, b1, b2, b3)
    w = w_smem.slice((H - 2) * BLOCK_R, BLOCK_R).load(row_layout)
    a0, a1, a2, a3 = warpgroup_mma_wait(4, deps=[a0, a1, a2, a3])
    acc = _head_epilogue(acc, a0, a1, a2, a3, w, key_scale)
    w = w_smem.slice((H - 1) * BLOCK_R, BLOCK_R).load(row_layout)
    b0, b1, b2, b3 = warpgroup_mma_wait(0, deps=[b0, b1, b2, b3])
    acc = _head_epilogue(acc, b0, b1, b2, b3, w, key_scale)
    return acc, a0, a1, a2, a3, b0, b1, b2, b3


@gluon.jit
def _prefetch_keys(
    k_desc,
    k_bar,
    k_smem,
    prefix_ptr,
    n_row_blocks,
    n_key_tiles,
    v,
    stage,
    p_lo,
    p_hi,
    BLOCK_K: gl.constexpr,
    D: gl.constexpr,
):
    """TMA the key tile of visible tile v; (p_lo, p_hi) caches its row block."""
    if v >= p_hi:
        rb = _last_row_block_at(prefix_ptr, n_row_blocks, v, n_key_tiles, False)
        p_lo = gl.load(prefix_ptr + rb)
        p_hi = gl.load(prefix_ptr + rb + 1)
    mbarrier.expect(k_bar.index(stage), BLOCK_K * D)
    tma.async_copy_global_to_shared(
        k_desc, [(v - p_lo) * BLOCK_K, 0], k_bar.index(stage), k_smem.index(stage)
    )
    return p_lo, p_hi


@gluon.jit
def _fill_invisible_tiles(
    prefix_ptr,
    out_ptr,
    B,
    OUT_L,
    n_row_blocks,
    n_key_tiles,
    n_vis_tiles,
    BLOCK_R: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """-inf for the tiles no row of their block can see, split across CTAs."""
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [num_warps, 1], [1, 0])
    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    n_inv = n_row_blocks * n_key_tiles - n_vis_tiles
    u0 = (pid.to(gl.int64) * n_inv // grid).to(gl.int32)
    u1 = ((pid.to(gl.int64) + 1) * n_inv // grid).to(gl.int32)
    offs_r0 = gl.arange(0, BLOCK_R, layout=gl.SliceLayout(1, layout))
    offs_k0 = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, layout))
    neg_inf = gl.full([BLOCK_R, BLOCK_K], float("-inf"), gl.float32, layout=layout)
    rb = 0
    lo = 0
    hi = 0
    first_kt = 0
    for u in range(u0, u1):
        if u >= hi:
            rb = _last_row_block_at(prefix_ptr, n_row_blocks, u, n_key_tiles, True)
            vis_lo = gl.load(prefix_ptr + rb)
            first_kt = gl.load(prefix_ptr + rb + 1) - vis_lo
            lo = rb * n_key_tiles - vis_lo
            hi = lo + n_key_tiles - first_kt
        offs_r = rb * BLOCK_R + offs_r0
        offs_k = (first_kt + u - lo) * BLOCK_K + offs_k0
        ptrs = out_ptr + offs_r.to(gl.int64)[:, None] * OUT_L + offs_k[None, :]
        gl.store(ptrs, neg_inf, mask=(offs_r < B)[:, None] & (offs_k < OUT_L)[None, :])


# These sizes are runtime loop bounds, including 1-row CP tails. Specializing
# their value/alignment adds binaries without changing the tile layout. Keep
# OUT_L and pointer alignment specialization for vectorized loads/stores.
@gluon.jit(do_not_specialize=["B", "L", "n_row_blocks", "n_key_tiles"])
def _keys_major_scores_kernel(
    q_desc,  # [B, H * D] e4m3, block [BLOCK_R, D]
    k_desc,  # [L, D] e4m3, block [BLOCK_K, D]
    w_ptr,  # [B, H] FP32, includes Q scales
    scale_ptr,  # [L] FP32 K scales
    lens_ptr,  # [B] int64
    prefix_ptr,  # [n_row_blocks + 1] int32
    out_ptr,  # [B, OUT_L] fp32
    B,
    L,
    OUT_L,
    stride_wb,
    n_row_blocks,
    n_key_tiles,
    H: gl.constexpr,
    D: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_K: gl.constexpr,
    STAGES: gl.constexpr,
    num_warps: gl.constexpr,
):
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BLOCK_R, 32]
    )
    key_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    row_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    w_layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])

    q_smem = gl.allocate_shared_memory(gl.float8e4nv, [H, BLOCK_R, D], q_desc.layout)
    k_smem = gl.allocate_shared_memory(
        gl.float8e4nv, [STAGES, BLOCK_K, D], k_desc.layout
    )
    # [H, BLOCK_R] flattened: per-head slices need constexpr offsets.
    w_smem = gl.allocate_shared_memory(
        gl.float32, [H * BLOCK_R], gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    q_bar = mbarrier.allocate_mbarrier()
    k_bar = mbarrier.allocate_mbarrier(batch=STAGES)
    mbarrier.init(q_bar, count=1)
    for s in gl.static_range(STAGES):
        mbarrier.init(k_bar.index(s), count=1)

    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    n_vis_tiles = gl.load(prefix_ptr + n_row_blocks)
    v0 = (pid.to(gl.int64) * n_vis_tiles // grid).to(gl.int32)
    v1 = ((pid.to(gl.int64) + 1) * n_vis_tiles // grid).to(gl.int32)
    p_lo = 0
    p_hi = 0
    for s in gl.static_range(STAGES):
        if v0 + s < v1:
            p_lo, p_hi = _prefetch_keys(
                k_desc,
                k_bar,
                k_smem,
                prefix_ptr,
                n_row_blocks,
                n_key_tiles,
                v0 + s,
                s,
                p_lo,
                p_hi,
                BLOCK_K,
                D,
            )

    a0 = gl.zeros([BLOCK_K, BLOCK_R], gl.float32, layout=mma_layout)
    a1 = a0
    a2 = a0
    a3 = a0
    b0 = a0
    b1 = a0
    b2 = a0
    b3 = a0
    offs_r0 = gl.arange(0, BLOCK_R, layout=row_layout)
    offs_k0 = gl.arange(0, BLOCK_K, layout=key_layout)
    offs_w = gl.arange(0, H * BLOCK_R, layout=w_layout)
    rb = 0
    lo = 0
    hi = 0
    q_phase = 0
    n_vis = gl.zeros([BLOCK_R], gl.int64, layout=row_layout)
    for v in range(v0, v1):
        if v >= hi:
            # New row block: queries, weights and visible lengths. Every wgmma
            # reading q_smem / w_smem has retired (wait(0) + barrier per tile).
            rb = _last_row_block_at(prefix_ptr, n_row_blocks, v, n_key_tiles, False)
            lo = gl.load(prefix_ptr + rb)
            hi = gl.load(prefix_ptr + rb + 1)
            n_vis = _load_row_block(
                q_desc,
                q_bar,
                q_smem,
                w_ptr,
                w_smem,
                lens_ptr,
                B,
                L,
                stride_wb,
                rb,
                q_phase,
                offs_r0,
                offs_w,
                H,
                D,
                BLOCK_R,
            )
            q_phase ^= 1
            gl.barrier()

        j = v - v0
        stage = j % STAGES
        mbarrier.wait(k_bar.index(stage), (j // STAGES) & 1)
        k_tile = k_smem.index(stage)
        # Opaque per-tile zero: keeps the 4 * H q descriptors from being
        # hoisted out of the tile loop, where they spill.
        hz = gl.minimum(v, 0)
        logical_keys = (v - lo) * BLOCK_K + offs_k0
        key_scale = gl.load(scale_ptr + logical_keys, logical_keys < L, 1.0)
        acc, a0, a1, a2, a3, b0, b1, b2, b3 = _score_tile(
            k_tile,
            q_smem,
            w_smem,
            key_scale,
            hz,
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
            b2,
            b3,
            H,
            BLOCK_R,
            row_layout,
        )
        # Every warp is done with this K stage (and q / w) before refills.
        gl.barrier()
        if v + STAGES < v1:
            p_lo, p_hi = _prefetch_keys(
                k_desc,
                k_bar,
                k_smem,
                prefix_ptr,
                n_row_blocks,
                n_key_tiles,
                v + STAGES,
                stage,
                p_lo,
                p_hi,
                BLOCK_K,
                D,
            )
        offs_r = rb * BLOCK_R + offs_r0
        offs_k = (v - lo) * BLOCK_K + offs_k0
        logit = acc
        logit = gl.where(offs_k[:, None] < n_vis[None, :], logit, float("-inf"))
        ptrs = out_ptr + offs_r.to(gl.int64)[None, :] * OUT_L + offs_k[:, None]
        gl.store(ptrs, logit, mask=(offs_r < B)[None, :] & (offs_k < OUT_L)[:, None])

    mbarrier.invalidate(q_bar)
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(k_bar.index(s))
    _fill_invisible_tiles(
        prefix_ptr,
        out_ptr,
        B,
        OUT_L,
        n_row_blocks,
        n_key_tiles,
        n_vis_tiles,
        BLOCK_R,
        BLOCK_K,
        num_warps,
    )


class AntKeysMajorScores:
    """Preplan descriptors/workspace; refresh visibility metadata once per call."""

    def __init__(self, q, weights, keys, scales, lengths, out, *, grid_size=0):
        rows, heads, dim = q.shape
        if heads != 32 or dim != 128 or rows < 1:
            raise ValueError("Requires positive rows, 32 index heads, dimension 128")
        for value in (q, keys):
            if value.dtype != torch.float8_e4m3fn or not value.is_contiguous():
                raise ValueError("Q/K must be contiguous FP8 E4M3")
            if value.data_ptr() % 16:
                raise ValueError("TMA requires 16-byte base alignment")
        if (
            weights.dtype != torch.float32
            or scales.dtype != torch.float32
            or weights.shape != (rows, heads)
            or scales.shape != (keys.shape[0],)
        ):
            raise ValueError("FP32 head weights and per-key scales required")
        if lengths.dtype not in (torch.int32, torch.int64) or lengths.shape != (rows,):
            raise ValueError("One int32/int64 visible key count per query")
        if (
            out.dtype != torch.float32
            or out.shape[0] != rows
            or out.shape[1] < keys.shape[0]
        ):
            raise ValueError("FP32 output must hold every query/key score")
        if not all(x.is_contiguous() for x in (weights, scales, lengths, out)):
            raise ValueError("Weights/scales/lengths/output must be contiguous")
        if any(x.device != q.device for x in (keys, weights, scales, lengths, out)):
            raise ValueError("All operands must share one CUDA device")
        self.q, self.weights, self.keys, self.scales = q, weights, keys, scales
        self.lengths, self.out = lengths, out
        self.rows, self.width = rows, keys.shape[0]
        self.n_row_blocks = triton.cdiv(rows, BLOCK_ROWS)
        self.n_key_tiles = triton.cdiv(out.shape[1], BLOCK_KEYS)
        self.prefix = torch.empty(
            self.n_row_blocks + 1, dtype=torch.int32, device=q.device
        )
        q_layout = gl.NVMMASharedLayout.get_default_for(
            [BLOCK_ROWS, dim], gl.float8e4nv
        )
        k_layout = gl.NVMMASharedLayout.get_default_for(
            [BLOCK_KEYS, dim], gl.float8e4nv
        )
        self.q_desc = TensorDescriptor.from_tensor(
            q.view(rows, heads * dim), [BLOCK_ROWS, dim], q_layout
        )
        self.k_desc = TensorDescriptor.from_tensor(keys, [BLOCK_KEYS, dim], k_layout)
        if grid_size <= 0:
            grid_size = torch.cuda.get_device_properties(q.device).multi_processor_count
        self.grid_size = min(self.n_row_blocks * self.n_key_tiles, grid_size)

    def metadata(self):
        _visible_tile_prefix_kernel[(1,)](
            self.lengths,
            self.prefix,
            self.rows,
            self.width,
            self.n_row_blocks,
            BLOCK_R=BLOCK_ROWS,
            BLOCK_K=BLOCK_KEYS,
            CHUNK=256,
        )

    def score(self):
        _keys_major_scores_kernel[(self.grid_size,)](
            self.q_desc,
            self.k_desc,
            self.weights,
            self.scales,
            self.lengths,
            self.prefix,
            self.out,
            self.rows,
            self.width,
            self.out.stride(0),
            self.weights.stride(0),
            self.n_row_blocks,
            self.n_key_tiles,
            H=32,
            D=128,
            BLOCK_R=BLOCK_ROWS,
            BLOCK_K=BLOCK_KEYS,
            STAGES=STAGES,
            num_warps=NUM_WARPS,
            enable_fp_fusion=False,
        )
        return self.out

    def forward(self):
        self.metadata()
        return self.score()


class AntPrefillIndexer(PrefillIndexer):
    """Same preparation/selector contract as existing P5; only scorer differs."""

    def __init__(self, inputs, *, grid_size=0, **kwargs):
        super().__init__(inputs, backend="keys_major", **kwargs)
        self.ant_plans = {}
        for start, end, request in self.chunks:
            self.ant_plans[(start, end)] = AntKeysMajorScores(
                inputs.q[start:end],
                inputs.weights[start:end],
                self.keys[request],
                self.scales[request],
                self.lengths[start:end],
                self.scores[: end - start],
                grid_size=grid_size,
            )

    def metadata(self):
        for plan in self.ant_plans.values():
            plan.metadata()

    def score_chunk(self, start, end, request):
        return self.ant_plans[(start, end)].score()

    def forward(self):
        self.prepare()
        self.metadata()
        for start, end, request in self.chunks:
            self.select_chunk(
                self.score_chunk(start, end, request), start, end, request
            )
        return self.output

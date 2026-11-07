# SPDX-License-Identifier: Apache-2.0
# Adapted from antgroup/sglang 6e6d5d9e SM90 grouped indexer (MIT).
# See third_party/ant/LICENSE and NOTICE.md. FP8 cache, scales and FP32 epilogue
# follow vLLM; the FP4 unpack and BF16 rounding points are deliberately absent.
"""Gluon grouped FP8 scoring: TMA Q, one shared K tile, two-deep key prefetch."""

from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from vllm.triton_utils import gl, gluon, triton

from .score import _decode_tile, _row_head_sum, _tile_payload


@gluon.jit
def _head_sum_pair(s):
    x = gl.reshape(s, [64, 2, 2, 2, 4, 2])
    x = gl.permute(x, [0, 1, 2, 4, 5, 3])
    x0, x1 = gl.split(x)
    x0, x1 = gl.permute(x0, [0, 1, 3, 4, 2]), gl.permute(x1, [0, 1, 3, 4, 2])
    h0, h2 = gl.split(x0)
    h1, h3 = gl.split(x1)
    even, odd = gl.split(((h0 + h1) + h2) + h3)
    even = gl.sum(gl.sum(gl.reshape(even, [64, 2, 2, 2]), 2), 2)
    odd = gl.sum(gl.sum(gl.reshape(odd, [64, 2, 2, 2]), 2), 2)
    return even + odd


@gluon.jit
def _store_pair(
    acc,
    scale,
    W,
    LENS,
    GROUPS,
    OUT,
    MAXIMA,
    group,
    pair_id,
    tile,
    WIDTH,
    GROUP: gl.constexpr,
    WRITE_MAX: gl.constexpr,
):
    head = gl.arange(0, 64, layout=gl.SliceLayout(0, acc.type.layout))
    offset = pair_id * 2 + head // 32
    row = gl.load(GROUPS + group * GROUP + offset, offset < GROUP, -1)
    weights = gl.load(W + row * 32 + head % 32, row >= 0, 0.0)
    scale = gl.convert_layout(scale, gl.SliceLayout(1, acc.type.layout))
    logits = _head_sum_pair(
        gl.maximum(acc * gl.expand_dims(scale, 1), 0.0) * gl.expand_dims(weights, 0)
    )
    layout: gl.constexpr = logits.type.layout
    keys = tile * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    pair = gl.arange(0, 2, layout=gl.SliceLayout(0, layout))
    row = gl.load(
        GROUPS + group * GROUP + pair_id * 2 + pair, pair_id * 2 + pair < GROUP, -1
    )
    length = gl.load(LENS + row, row >= 0, 0)
    logits = gl.where(
        gl.expand_dims(keys, 1) < gl.expand_dims(length, 0), logits, -float("inf")
    )
    gl.store(
        OUT + gl.expand_dims(row.to(gl.int64), 0) * WIDTH + gl.expand_dims(keys, 1),
        logits,
        gl.expand_dims(row >= 0, 0) & gl.expand_dims(keys < WIDTH, 1),
    )
    if WRITE_MAX:
        maxima = gl.max(gl.reshape(logits, [8, 8, 2]), 1)
        bl: gl.constexpr = maxima.type.layout
        blocks = tile * 8 + gl.arange(0, 8, layout=gl.SliceLayout(1, bl))
        row_b = gl.convert_layout(row, gl.SliceLayout(0, bl))
        len_b = gl.convert_layout(length, gl.SliceLayout(0, bl))
        maxima = gl.where(
            gl.expand_dims(len_b > 0, 0)
            & (gl.expand_dims(blocks, 1) == gl.expand_dims((len_b - 1) // 8, 0)),
            float("inf"),
            maxima,
        )
        gl.store(
            MAXIMA
            + gl.expand_dims(row_b.to(gl.int64), 0) * (WIDTH // 8)
            + gl.expand_dims(blocks, 1),
            maxima,
            gl.expand_dims(row_b >= 0, 0) & gl.expand_dims(blocks < WIDTH // 8, 1),
        )


@gluon.jit
def _paired(
    Q,
    W,
    CACHE,
    TABLE,
    REQUESTS,
    LENS,
    GROUPS,
    OUT,
    MAXIMA,
    CANDIDATES,
    WIDTH,
    TABLE_STRIDE,
    CACHE_STRIDE,
    GROUP: gl.constexpr,
    WRITE_MAX: gl.constexpr,
    COMPACT: gl.constexpr,
    PAIR_ROWS: gl.constexpr,
):
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 32]
    )
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 16], [8, 4, 1], [4, 1, 1], [2, 1, 0])
    kl: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))
    group_l: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    ql: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    smem_l: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [64, 128], gl.float8e4nv
    )
    pairs: gl.constexpr = (PAIR_ROWS + 1) // 2
    q_smem = gl.allocate_shared_memory(gl.float8e4nv, [pairs, 64, 128], smem_l)
    k_smem = gl.allocate_shared_memory(gl.float8e4nv, [64, 128], smem_l)
    group, lane, stride = gl.program_id(0), gl.program_id(1), gl.num_programs(1)
    offsets = gl.arange(0, GROUP, layout=group_l)
    rows = gl.load(GROUPS + group * GROUP + offsets)
    visible = gl.load(LENS + rows, rows >= 0, 0)
    limit = gl.minimum(gl.max(visible, 0), WIDTH)
    first = gl.load(GROUPS + group * GROUP)
    req = gl.load(REQUESTS + first).to(gl.int64)
    h = gl.arange(0, 64, layout=gl.SliceLayout(1, ql))
    d = gl.arange(0, 128, layout=gl.SliceLayout(0, ql))
    for p in gl.static_range(pairs):
        offset = p * 2 + h // 32
        row = gl.load(GROUPS + group * GROUP + offset, offset < GROUP, -1)
        q = gl.load(
            Q + gl.expand_dims(row * 4096 + h % 32 * 128, 1) + gl.expand_dims(d, 0),
            gl.expand_dims(row >= 0, 1),
            0.0,
        )
        q_smem.index(p).store(q)
    fence_async_shared()
    gl.barrier()
    acc_a = gl.zeros([64, 64], gl.float32, layout=mma)
    acc_b = gl.zeros([64, 64], gl.float32, layout=mma)
    slot, valid = _slots(
        TABLE, CANDIDATES, req, first, lane, limit, TABLE_STRIDE, COMPACT, kl
    )
    payload, scales = _tile_payload(CACHE, slot, valid, 64, CACHE_STRIDE, L3)
    slot, valid = _slots(
        TABLE, CANDIDATES, req, first, lane + stride, limit, TABLE_STRIDE, COMPACT, kl
    )
    for tile in range(lane, gl.cdiv(limit, 64), stride):
        key_scale = scales
        k_tile = _decode_tile(payload, scales)
        payload, scales = _tile_payload(CACHE, slot, valid, 64, CACHE_STRIDE, L3)
        slot, valid = _slots(
            TABLE,
            CANDIDATES,
            req,
            first,
            tile + 2 * stride,
            limit,
            TABLE_STRIDE,
            COMPACT,
            kl,
        )
        gl.barrier()
        k_smem.store(k_tile)
        fence_async_shared()
        gl.barrier()
        acc_a = warpgroup_mma(
            k_smem, q_smem.index(0).permute((1, 0)), acc_a, use_acc=False, is_async=True
        )
        for p in gl.static_range(pairs):
            if p + 1 < pairs:
                acc_b = warpgroup_mma(
                    k_smem,
                    q_smem.index(p + 1).permute((1, 0)),
                    acc_b,
                    use_acc=False,
                    is_async=True,
                )
                acc_a = warpgroup_mma_wait(1, deps=[acc_a])
            else:
                acc_a = warpgroup_mma_wait(0, deps=[acc_a])
            _store_pair(
                acc_a,
                key_scale,
                W,
                LENS,
                GROUPS,
                OUT,
                MAXIMA,
                group,
                p,
                tile,
                WIDTH,
                GROUP,
                WRITE_MAX,
            )
            if p + 1 < pairs:
                acc_a, acc_b = acc_b, acc_a


@gluon.jit
def _slots(
    TABLE,
    CANDIDATES,
    req,
    first,
    tile,
    count,
    table_stride,
    COMPACT: gl.constexpr,
    KL: gl.constexpr,
):
    cols = tile * 64 + gl.arange(0, 64, layout=KL)
    valid = cols < count
    logical = cols
    if COMPACT:
        block = gl.load(CANDIDATES + first * 2048 + cols // 8, valid, 0)
        logical = block * 8 + cols % 8
    pages = gl.load(TABLE + req * table_stride + logical // 64, valid, 0).to(gl.int64)
    return pages * 64 + logical % 64, valid


@gluon.jit
def _grouped(
    q_desc,
    W,
    CACHE,
    TABLE,
    REQUESTS,
    LENS,
    GROUPS,
    OUT,
    MAXIMA,
    CANDIDATES,
    WIDTH,
    TABLE_STRIDE,
    CACHE_STRIDE,
    GROUP: gl.constexpr,
    WRITE_MAX: gl.constexpr,
    COMPACT: gl.constexpr,
):
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 32]
    )
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 16], [8, 4, 1], [4, 1, 1], [2, 1, 0])
    kl: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))
    group_l: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    k_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [64, 128], gl.float8e4nv
    )
    q_smem = gl.allocate_shared_memory(gl.float8e4nv, [GROUP, 32, 128], q_desc.layout)
    k_smem = gl.allocate_shared_memory(gl.float8e4nv, [64, 128], k_layout)
    group, lane, stride = gl.program_id(0), gl.program_id(1), gl.num_programs(1)
    offsets = gl.arange(0, GROUP, layout=group_l)
    rows = gl.load(GROUPS + group * GROUP + offsets)
    visible = gl.load(LENS + rows, rows >= 0, 0)
    limit = gl.minimum(gl.max(visible, 0), WIDTH)
    first = gl.load(GROUPS + group * GROUP)
    req = gl.load(REQUESTS + first).to(gl.int64)
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, GROUP * 32 * 128)
    for p in gl.static_range(GROUP):
        row = gl.load(GROUPS + group * GROUP + p)
        tma.async_copy_global_to_shared(
            q_desc, [gl.maximum(row, 0) * 32, 0], bar, q_smem.index(p)
        )
    mbarrier.wait(bar, 0)
    mbarrier.invalidate(bar)
    acc = gl.zeros([64, 32], gl.float32, layout=mma)
    slot, valid = _slots(
        TABLE, CANDIDATES, req, first, lane, limit, TABLE_STRIDE, COMPACT, kl
    )
    payload, scales = _tile_payload(CACHE, slot, valid, 64, CACHE_STRIDE, L3)
    slot, valid = _slots(
        TABLE, CANDIDATES, req, first, lane + stride, limit, TABLE_STRIDE, COMPACT, kl
    )
    for tile in range(lane, gl.cdiv(limit, 64), stride):
        key_scale = scales
        k_tile = _decode_tile(payload, scales)
        payload, scales = _tile_payload(CACHE, slot, valid, 64, CACHE_STRIDE, L3)
        slot, valid = _slots(
            TABLE,
            CANDIDATES,
            req,
            first,
            tile + 2 * stride,
            limit,
            TABLE_STRIDE,
            COMPACT,
            kl,
        )
        gl.barrier()
        k_smem.store(k_tile)
        fence_async_shared()
        gl.barrier()
        for p in gl.static_range(GROUP):
            row = gl.load(GROUPS + group * GROUP + p)
            if row >= 0:
                acc = warpgroup_mma(
                    k_smem, q_smem.index(p).permute((1, 0)), acc, use_acc=False
                )
                head = gl.arange(0, 32, layout=gl.SliceLayout(0, mma))
                weights = gl.load(W + row * 32 + head)
                scale = gl.convert_layout(key_scale, gl.SliceLayout(1, mma))
                values = gl.maximum(
                    acc * gl.expand_dims(scale, 1), 0.0
                ) * gl.expand_dims(weights, 0)
                logits = _row_head_sum(values)
                keys = tile * 64 + gl.arange(0, 64, layout=logits.type.layout)
                length = gl.load(LENS + row)
                logits = gl.where(keys < length, logits, -float("inf"))
                gl.store(OUT + row.to(gl.int64) * WIDTH + keys, logits, keys < WIDTH)
                if WRITE_MAX:
                    maxima = gl.max(gl.reshape(logits, [8, 8]), 1)
                    blocks = tile * 8 + gl.arange(0, 8, layout=maxima.type.layout)
                    maxima = gl.where(
                        (length > 0) & (blocks == (length - 1) // 8),
                        float("inf"),
                        maxima,
                    )
                    gl.store(
                        MAXIMA + row.to(gl.int64) * (WIDTH // 8) + blocks,
                        maxima,
                        blocks < WIDTH // 8,
                    )


def grouped_logits(
    q,
    weights,
    cache,
    table,
    requests,
    lengths,
    groups,
    out,
    *,
    maxima=None,
    candidates=None,
    grid_ctas=2048,
    paired=False,
    pair_rows=None,
):
    lanes = min(triton.cdiv(out.shape[1], 64), triton.cdiv(grid_ctas, groups.shape[0]))
    if paired:
        kernel, query_arg = _paired, q
    else:
        layout = gl.NVMMASharedLayout.get_default_for([32, 128], gl.float8e4nv)
        query_arg = TensorDescriptor.from_tensor(q.view(-1, 128), [32, 128], layout)
        kernel = _grouped
    kernel[(groups.shape[0], lanes)](
        query_arg,
        weights,
        cache,
        table,
        requests,
        lengths,
        groups,
        out,
        maxima,
        candidates,
        out.shape[1],
        table.stride(0),
        cache.stride(0),
        groups.shape[1],
        maxima is not None,
        candidates is not None,
        *([pair_rows or groups.shape[1]] if paired else []),
        num_warps=4,
        maxnreg=168,
    )

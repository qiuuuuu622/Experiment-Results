# SPDX-License-Identifier: Apache-2.0
"""FP8 paged full/source scoring with actual key-tile reuse across query rows."""

import triton
import triton.language as tl


@triton.jit
def _grouped_scores(
    Q,
    W,
    CACHE,
    TABLE,
    REQUESTS,
    LENGTHS,
    GROUPS,
    OUT,
    MAXIMA,
    CANDIDATES,
    TABLE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BK: tl.constexpr,
    FUSE_MAX: tl.constexpr,
    KEYS_MAJOR: tl.constexpr,
    COMPACT: tl.constexpr,
    PRECISE: tl.constexpr,
):
    group = tl.program_id(1) if KEYS_MAJOR else tl.program_id(0)
    lane = tl.program_id(0) if KEYS_MAJOR else tl.program_id(1)
    rr = tl.arange(0, GROUP_ROWS)
    rows = tl.load(GROUPS + group * GROUP_ROWS + rr)
    lens = tl.load(LENGTHS + rows, rows >= 0, 0)
    limit = tl.max(lens, 0)
    first = tl.load(GROUPS + group * GROUP_ROWS)
    req = tl.load(REQUESTS + first)
    d = tl.arange(0, 128)
    h = tl.arange(0, GROUP_ROWS * 32)
    qrows = tl.gather(rows, h // 32, 0)
    queries = tl.load(
        Q + qrows[:, None] * 4096 + (h % 32)[:, None] * 128 + d[None, :],
        qrows[:, None] >= 0,
        0.0,
    )
    weights = tl.load(W + qrows * 32 + h % 32, qrows >= 0, 0)
    lanes = tl.num_programs(0) if KEYS_MAJOR else tl.num_programs(1)
    for tile in range(lane, tl.cdiv(limit, BK), lanes):
        keys = tile * BK + tl.arange(0, BK)
        valid = (keys < limit) & (keys < WIDTH)
        logical = keys
        if COMPACT:
            block = tl.load(CANDIDATES + first * 2048 + keys // 8, valid, 0)
            logical = block * 8 + keys % 8
        pages = tl.load(TABLE + req * TABLE_STRIDE + logical // 64, valid, 0).to(
            tl.int64
        )
        payload = tl.load(
            CACHE
            + pages[:, None] * (64 * 132)
            + (logical % 64)[:, None] * 128
            + d[None, :],
            valid[:, None],
            0,
        ).to(tl.float8e4nv, bitcast=True)
        scales = tl.load(
            (CACHE + pages * (64 * 132) + 64 * 128 + logical % 64 * 4).to(
                tl.pointer_type(tl.float32)
            ),
            valid,
            1,
        )
        dots = tl.dot(
            payload,
            tl.trans(queries),
            out_dtype=tl.float32,
            max_num_imprecise_acc=0 if PRECISE else 128,
        )
        weighted = tl.maximum(dots * scales[:, None], 0) * weights[None, :]
        logits = tl.trans(tl.sum(tl.reshape(weighted, [BK, GROUP_ROWS, 32]), 2))
        logits = tl.where(keys[None, :] < lens[:, None], logits, -float("inf"))
        tl.store(
            OUT + rows[:, None] * WIDTH + keys[None, :],
            logits,
            (rows[:, None] >= 0) & (keys[None, :] < WIDTH),
        )
        if FUSE_MAX:
            block = tile * (BK // 8) + tl.arange(0, BK // 8)
            maxima = tl.max(tl.reshape(logits, [GROUP_ROWS, BK // 8, 8]), 2)
            maxima = tl.where(
                (lens[:, None] > 0) & (block[None, :] == (lens[:, None] - 1) // 8),
                float("inf"),
                maxima,
            )
            tl.store(
                MAXIMA + rows[:, None] * tl.cdiv(WIDTH, 8) + block[None, :],
                maxima,
                (rows[:, None] >= 0) & (block[None, :] < tl.cdiv(WIDTH, 8)),
            )


def grouped_scores(
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
    keys_major=False,
    tile=64,
    candidates=None,
    precise=False,
    grid_ctas=256,
):
    """Groups contain rows of ONE request, padded with -1; first row is valid.

    The invisible score suffix is deliberately untouched. Consumers must obey
    lengths. Source block selection must likewise obey ceil(length/8).
    """
    n, r = groups.shape
    lanes = min(triton.cdiv(out.shape[1], tile), max(1, triton.cdiv(grid_ctas, n)))
    grid = (lanes, n) if keys_major else (n, lanes)
    _grouped_scores[grid](
        q,
        weights,
        cache,
        table,
        requests,
        lengths,
        groups,
        out,
        maxima,
        candidates,
        table.stride(0),
        out.shape[1],
        r,
        tile,
        maxima is not None,
        keys_major,
        candidates is not None,
        precise,
        num_warps=8 if r >= 4 else 4,
    )


@triton.jit
def _block_lengths(LENGTHS, OUT, ROWS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    length = tl.load(LENGTHS + row, row < ROWS, 0)
    tl.store(OUT + row, tl.cdiv(length, 8), row < ROWS)

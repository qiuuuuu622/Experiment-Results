# SPDX-License-Identifier: Apache-2.0
"""Small FP32 oracle. Intentionally allocates; never a performance baseline."""

import torch

from .contract import candidate_positions


def decode_cache(cache, table, request, length):
    """Decode vLLM's page payload followed by scales (not interleaved tokens)."""
    raw = cache.reshape(cache.shape[0], 64 * 132)
    pages = raw[table[request, : (length + 63) // 64].long()]
    keys = pages[:, : 64 * 128].contiguous().view(torch.float8_e4m3fn)
    scales = pages[:, 64 * 128 :].contiguous().view(torch.float32)
    return keys.reshape(-1, 128)[:length].float(), scales.reshape(-1)[:length]


def scores_reference(q, weights, cache, table, requests, lengths, candidates=None):
    """Score only visible keys with FP32 dot, ReLU, key scales and head weights."""
    outputs, positions = [], []
    for row, (request, length) in enumerate(zip(requests.tolist(), lengths.tolist())):
        keys, scales = decode_cache(cache, table, request, length)
        pos = (
            list(range(length))
            if candidates is None
            else candidate_positions(candidates[row].tolist(), length)
        )
        idx = torch.tensor(pos, device=q.device, dtype=torch.long)
        dot = torch.einsum("hd,kd->kh", q[row].float(), keys[idx])
        score = ((dot * scales[idx, None]).clamp_min(0) * weights[row]).sum(-1)
        outputs.append(score)
        positions.append(idx)
    return outputs, positions


def stable_topk(scores, positions, k=512):
    """Exact FP32 ties ordered by logical ID; unordered output can compare sets."""
    result = torch.full(
        (len(scores), k), -1, device=scores[0].device, dtype=torch.int32
    )
    for row, (s, p) in enumerate(zip(scores, positions)):
        logical = torch.argsort(p, stable=True)
        order = logical[torch.argsort(s[logical], descending=True, stable=True)[:k]]
        result[row, : len(order)] = p[order].int()
    return result


def source_reference(scores, lengths, capacity=2048):
    """Block maxima with newest block pinned, matching vLLM source semantics."""
    out = torch.full(
        (scores.shape[0], capacity), -1, dtype=torch.int32, device=scores.device
    )
    for row, length in enumerate(lengths.tolist()):
        nblocks = (length + 7) // 8
        if not nblocks:
            continue
        values = torch.full((nblocks * 8,), -float("inf"), device=scores.device)
        values[:length] = scores[row, :length]
        maxima = values.view(-1, 8).max(-1).values
        maxima[-1] = float("inf")
        selected = torch.argsort(maxima, descending=True, stable=True)[:capacity]
        out[row, : len(selected)] = selected.int()
    return out

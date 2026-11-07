# SPDX-License-Identifier: Apache-2.0
"""Explicit grouped verify experiments; no implicit cross-step candidate cache."""

import torch

from .contract import shared_candidate_groups
from .grouped_score import grouped_logits
from .module import FullIndexer, plan_groups
from .workspace import CompactIndexer


class GroupedIndexer(FullIndexer):
    """Gluon full/source scorer sharing each key tile across same-request rows."""

    def __init__(
        self,
        inputs,
        *,
        group_rows=4,
        source=False,
        grid_ctas=2048,
        paired=False,
        fuse_source_max=True,
    ):
        self.grid_ctas = grid_ctas
        self.paired = paired
        super().__init__(
            inputs,
            group_rows=group_rows,
            source=source,
            fuse_source_max=fuse_source_max,
        )

    def replan(self):
        super().replan()
        self.pair_rows = max(
            sum(r >= 0 for r in group) for group in self.groups.tolist()
        )

    def score(self):
        i = self.inputs
        grouped_logits(
            i.q,
            i.weights,
            i.cache,
            i.table,
            i.requests,
            i.lengths,
            self.groups,
            self.scores,
            maxima=self.maxima if self.fuse_source_max else None,
            grid_ctas=self.grid_ctas,
            paired=self.paired,
            pair_rows=self.pair_rows,
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


class GroupedConsumer(CompactIndexer):
    """Share K only for compatible published candidate prefixes.

    Call replan after publication when request order or candidate layout changes,
    outside capture. A replay may update Q/weights and candidate values provided
    the declared shared-prefix layout and longest-row ordering remain true.
    Arbitrary changed layouts must replan; no candidate union is constructed.
    """

    def __init__(
        self,
        state,
        q,
        weights,
        cache,
        table,
        requests,
        *,
        group_rows=4,
        grid_ctas=2048,
        paired=False,
    ):
        super().__init__(state, q, weights, cache, table, requests)
        self.group_rows, self.grid_ctas = group_rows, grid_ctas
        self.paired = paired
        self.shared_enabled = False
        if state._published:
            self.replan()

    def replan(self):
        if not self.state._published:
            raise RuntimeError("Publish before planning shared candidates")
        proposed = plan_groups(self.requests.tolist(), self.group_rows)
        counts = self.state.counts.tolist()
        blocks = self.state.blocks.tolist()
        reordered = shared_candidate_groups(proposed, counts, blocks)
        if reordered is None:
            self.shared_enabled = False
            return False
        self.shared_groups = torch.tensor(
            reordered, device=self.q.device, dtype=torch.int32
        )
        self.pair_rows = max(sum(r >= 0 for r in group) for group in reordered)
        self.shared_enabled = any(sum(r >= 0 for r in g) > 1 for g in proposed)
        return self.shared_enabled

    def score(self):
        if not self.shared_enabled:
            return super().score()
        if not self.state._published:
            raise RuntimeError("Publish before scoring")
        grouped_logits(
            self.q,
            self.weights,
            self.cache,
            self.table,
            self.requests,
            self.state.counts,
            self.shared_groups,
            self.scores,
            candidates=self.state.blocks,
            grid_ctas=self.grid_ctas,
            paired=self.paired,
            pair_rows=self.pair_rows,
        )
        return self.scores

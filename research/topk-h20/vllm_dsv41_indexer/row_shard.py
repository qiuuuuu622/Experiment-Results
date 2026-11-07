# SPDX-License-Identifier: Apache-2.0
"""Explicit TP row experiments; caches and physical page IDs never move.

Plans are host metadata, constructed outside execution/capture. Input tensors
and workspaces have fixed shapes. Updating lengths, candidates or request IDs
in place is allowed; changing group membership requires a new plan/capture.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RowPlan:
    """Partition every original row once, preserving complete request groups."""

    rank_rows: tuple[tuple[int, ...], ...]
    rows: int
    mode: str

    @classmethod
    def make(cls, requests, world=4, *, split_rows=False):
        requests = tuple(int(x) for x in requests)
        if world < 1 or not requests or min(requests) < 0:
            raise ValueError(
                "Positive world and nonnegative nonempty requests required"
            )
        groups = {}
        for row, request in enumerate(requests):
            groups.setdefault(row if split_rows else request, []).append(row)
        assigned = [[] for _ in range(world)]
        # Largest request first balances ragged verify groups deterministically.
        for group in sorted(groups.values(), key=lambda g: (-len(g), g[0])):
            rank = min(range(world), key=lambda r: (len(assigned[r]), r))
            assigned[rank].extend(group)
        return cls(
            tuple(tuple(rows) for rows in assigned),
            len(requests),
            "split_rows" if split_rows else "request_groups",
        )

    @property
    def capacity(self):
        return max(map(len, self.rank_rows))

    @property
    def inverse(self):
        positions = [None] * self.rows
        for rank, rows in enumerate(self.rank_rows):
            for local, row in enumerate(rows):
                positions[row] = rank * self.capacity + local
        return tuple(positions)

    def describe(self):
        return dict(
            mode=self.mode,
            rows_per_rank=[len(r) for r in self.rank_rows],
            padded_rows_per_rank=self.capacity,
            rank_rows=self.rank_rows,
        )


class RowExchange:
    """Pack inputs, run a rank-local callable, collect only logical outputs.

    factory(local_fields) returns an object with forward()->int32 [local_rows,W].
    Empty ranks never construct/invoke a kernel; they still join collectives.
    replicated inputs exist on every rank and include the local pack cost.
    root inputs are meaningful only on rank 0: root packs all destinations then
    scatters one fused byte buffer, including metadata and external candidates.
    Page tables/cache are captured in the local factory, never in fields.
    """

    def __init__(self, fields, plan, factory, width, *, input_mode="replicated"):
        import torch
        import torch.distributed as dist

        if input_mode not in ("replicated", "root"):
            raise ValueError("input_mode must be replicated or root")
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if len(plan.rank_rows) != self.world:
            raise ValueError("Plan must match the actual process group")
        if any(k in fields for k in ("cache", "table", "page_table")):
            raise ValueError("Physical cache mappings must remain rank-local")
        self.fields, self.plan, self.input_mode = fields, plan, input_mode
        self.device = next(iter(fields.values())).device
        self.count = len(plan.rank_rows[self.rank])
        offsets, total = {}, 0
        for key, value in fields.items():
            if value.shape[0] != plan.rows or not value.is_contiguous():
                raise ValueError("All fields must be contiguous global-row tensors")
            if value.device != self.device:
                raise ValueError("Fields must have one local device")
            total = (total + 15) // 16 * 16
            size = value[0].numel() * value.element_size() * plan.capacity
            offsets[key] = (total, total + size)
            total += size
        self.byte_count = (total + 15) // 16 * 16
        self.receive = torch.zeros(
            self.byte_count, device=self.device, dtype=torch.uint8
        )
        self.send = (
            torch.zeros(
                (self.world, self.byte_count), device=self.device, dtype=torch.uint8
            )
            if input_mode == "root" and self.rank == 0
            else None
        )
        self.row_ids = [
            torch.tensor(
                tuple(rows) + (0,) * (plan.capacity - len(rows)),
                device=self.device,
                dtype=torch.int64,
            )
            for rows in plan.rank_rows
        ]
        self.inverse = torch.tensor(plan.inverse, device=self.device, dtype=torch.int64)
        self.offsets = offsets
        self.local_fields = self._views(self.receive, fields, offsets, plan.capacity)
        self.local_fields = {k: v[: self.count] for k, v in self.local_fields.items()}
        self._source_bytes = {
            k: value.reshape(plan.rows, -1).view(torch.uint8)
            for k, value in fields.items()
        }
        self._recv_bytes = {
            k: self.receive[a:b].view(plan.capacity, -1)
            for k, (a, b) in offsets.items()
        }
        self._send_bytes = (
            [
                {
                    k: self.send[r, a:b].view(plan.capacity, -1)
                    for k, (a, b) in offsets.items()
                }
                for r in range(self.world)
            ]
            if self.send is not None
            else None
        )
        self.send_list = list(self.send.unbind()) if self.send is not None else None
        # Initialize data before the local factory validates shapes/metadata.
        self.prepare()
        self.local = factory(self.local_fields) if self.count else None
        self.local_output = torch.full(
            (plan.capacity, width), -1, dtype=torch.int32, device=self.device
        )
        self.gathered = torch.empty(
            (self.world * plan.capacity, width), dtype=torch.int32, device=self.device
        )
        self.output = torch.empty(
            (plan.rows, width), dtype=torch.int32, device=self.device
        )

    @staticmethod
    def _views(buffer, fields, offsets, capacity):
        return {
            k: buffer[offsets[k][0] : offsets[k][1]]
            .view(v.dtype)
            .view(capacity, *v.shape[1:])
            for k, v in fields.items()
        }

    def prepare(self):
        import torch
        import torch.distributed as dist

        if self.input_mode == "root":
            if self.rank == 0:
                for rank in range(self.world):
                    for key, src in self._source_bytes.items():
                        torch.index_select(
                            src, 0, self.row_ids[rank], out=self._send_bytes[rank][key]
                        )
            dist.scatter(self.receive, scatter_list=self.send_list, src=0)
        elif self.count:
            ids = self.row_ids[self.rank][: self.count]
            for key, src in self._source_bytes.items():
                torch.index_select(src, 0, ids, out=self._recv_bytes[key][: self.count])

    def compute(self):
        if self.local is not None:
            self.local_output[: self.count].copy_(self.local.forward())

    def collect(self):
        import torch
        import torch.distributed as dist

        dist.all_gather_into_tensor(self.gathered, self.local_output)
        torch.index_select(self.gathered, 0, self.inverse, out=self.output)
        return self.output

    def forward(self):
        self.prepare()
        self.compute()
        return self.collect()

    def communication(self):
        width = self.output.shape[1]
        return dict(
            input_mode=self.input_mode,
            root_scatter_payload_bytes_per_rank=self.byte_count
            if self.input_mode == "root"
            else 0,
            topk_allgather_payload_bytes_per_rank=self.plan.capacity * width * 4,
            payload_contains_physical_pages=False,
            input_pack_in_timing=True,
            result_layout="original_query_order_logical_index_key_positions",
            **self.plan.describe(),
        )


class FixedProjection:
    """Synthetic linear Q/head-weight projection, with explicit FP8 semantics.

    This is not the model's complete projection/RoPE. FP32 GEMM uses fixed random
    weights. Each query head uses scale=max(abs(Q))/448 (zero -> 1), E4M3 Q,
    and FP32 head_weights*scale/sqrt(32*128). Existing cache scales are unchanged.
    GEMM/output buffers are preallocated. Torch quantization creates temporary
    tensors; complete forward cost is measured, including graph-pool storage.
    """

    def __init__(self, hidden, q_weight, head_weight, q, weights):
        import torch

        self.hidden, self.q_weight, self.head_weight = hidden, q_weight, head_weight
        self.q, self.weights = q, weights
        self.raw_q = torch.empty((q.shape[0], 4096), device=q.device)
        self.raw_w = torch.empty_like(weights)

    def forward(self):
        import math

        import torch

        torch.mm(self.hidden, self.q_weight, out=self.raw_q)
        torch.mm(self.hidden, self.head_weight, out=self.raw_w)
        values = self.raw_q.view_as(self.q)
        scale = values.abs().amax(-1, keepdim=True) / 448.0
        scale = torch.where(scale > 0, scale, 1.0)
        self.q.copy_((values / scale).clamp(-448, 448).to(torch.float8_e4m3fn))
        self.weights.copy_(self.raw_w * scale.squeeze(-1) / math.sqrt(32 * 128))


class IndexerRows:
    """Local full, compact, or source+four-consumer module for TP experiments."""

    def __init__(
        self,
        fields,
        cache,
        table,
        max_keys,
        *,
        workload="compact",
        ratio=1,
        projection_weights=None,
        full_backend="portable",
    ):
        import torch

        from .contract import IndexerShape
        from .module import (
            ChunkedIndexer,
            IndexerInputs,
            NativeSourceIndexer,
        )
        from .workspace import CandidateState, CompactIndexer

        self.fields, self.workload = fields, workload
        if workload not in ("compact", "full", "chain"):
            raise ValueError("Unknown workload")
        self.stages = 5 if workload == "chain" else 1
        rows, device = fields["lengths"].shape[0], cache.device
        self.projections = []
        if projection_weights is not None:
            for stage in range(self.stages):
                fields[f"q{stage}"] = torch.empty(
                    (rows, 32, 128), device=device, dtype=torch.float8_e4m3fn
                )
                fields[f"w{stage}"] = torch.empty((rows, 32), device=device)
                self.projections.append(
                    FixedProjection(
                        fields["hidden"],
                        *projection_weights[stage],
                        fields[f"q{stage}"],
                        fields[f"w{stage}"],
                    )
                )
            for project in self.projections:
                project.forward()
        self.source = None
        self.consumers = []
        if workload in ("full", "chain"):
            i = IndexerInputs(
                fields["q0"],
                fields["w0"],
                cache,
                table,
                fields["requests"],
                fields["lengths"],
                IndexerShape(rows, max_keys, ratio),
            )
            if full_backend == "portable":
                self.source = ChunkedIndexer(i, source=workload == "chain")
            elif full_backend == "native_stable":
                # Ant's stable selector is mandatory: native cooperative and
                # persistent selectors can choose different members of ties.
                self.source = NativeSourceIndexer(i)
            else:
                raise ValueError("full_backend must be portable or native_stable")
        self.full_backend = full_backend
        if workload != "full":
            self.state = CandidateState(rows, device, compress_ratio=ratio)
            for stage in range(1, 5) if workload == "chain" else (0,):
                self.consumers.append(
                    CompactIndexer(
                        self.state,
                        fields[f"q{stage}"],
                        fields[f"w{stage}"],
                        cache,
                        table,
                        fields["requests"],
                    )
                )
        self.output = torch.empty(
            (rows, self.stages * 512), device=device, dtype=torch.int32
        )

    def forward(self):
        for project in self.projections:
            project.forward()
        stage = 0
        if self.source is not None:
            if self.full_backend == "native_stable":
                # Rebuild the gathered table and scheduler metadata on-device
                # every step. Graph capture must include these dependencies.
                self.source.prepare()
                if self.workload == "full":
                    self.source.score()
                    selected = self.source.select()
                else:
                    selected = self.source.forward()
            else:
                selected = self.source.forward()
            self.output[:, :512].copy_(selected)
            stage = 1
        if self.consumers:
            self.state.publish(
                self.source.candidates
                if self.source is not None
                else self.fields["blocks"],
                self.fields["lengths"],
                source_unique=self.source is not None,
            )
            for consumer in self.consumers:
                self.output[:, stage * 512 : (stage + 1) * 512].copy_(
                    consumer.forward()
                )
                stage += 1
        return self.output

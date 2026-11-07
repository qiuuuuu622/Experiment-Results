# SPDX-License-Identifier: Apache-2.0
"""Fixed-score exact tests for register, streaming, cluster and ragged top-k."""

import argparse

from .common import check_vllm, save


def expected(scores, lengths, k, starts=None, offsets=None):
    import torch

    output = torch.full(
        (scores.shape[0], k), -1, device=scores.device, dtype=torch.int32
    )
    lengths = lengths.tolist()
    starts = [0] * len(lengths) if starts is None else starts.tolist()
    offsets = [0] * len(lengths) if offsets is None else offsets.tolist()
    for row, (n, start, offset) in enumerate(zip(lengths, starts, offsets)):
        chosen = torch.argsort(
            scores[row, start : start + n], descending=True, stable=True
        )[:k]
        output[row, : len(chosen)] = chosen.int() + offset
    return output


def make_scores(rows, width, distribution, *, seed=607):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    if distribution == "random":
        return torch.randn((rows, width), generator=generator, device="cuda")
    if distribution == "zero":
        return torch.zeros((rows, width), device="cuda")
    if distribution == "equal_nonzero":
        return torch.full((rows, width), 1.25, device="cuda")
    if distribution == "tie_bin_overflow":
        # >2048 values occupy the same coarse bin; exact value/position still
        # decides. Reversed positions prevent a prefix-only accidental success.
        values = 1 + (
            (torch.arange(width, device="cuda").flip(0) % 4096).float() * 2**-23
        )
        return values[None, :].repeat(rows, 1)
    if distribution == "cutoff_ties":
        scores = torch.full((rows, width), -2.0, device="cuda")
        scores[:, : min(257, width)] = 2.0
        scores[:, min(257, width) :] = 1.0
        return scores
    raise ValueError(distribution)


def check_case(rows, width, k, distribution, backend):
    import torch

    from vllm_dsv41_indexer.topk import StableTopK

    scores = make_scores(rows, width, distribution)
    possible = [width, 0, 1, k - 1, k, k + 1, max(0, width - 3)]
    lengths = torch.tensor(
        [min(width, possible[r % len(possible)]) for r in range(rows)],
        device="cuda",
        dtype=torch.int32,
    )
    columns = torch.arange(width, device="cuda")[None, :]
    scores.masked_fill_(columns >= lengths[:, None], float("nan"))
    selector = StableTopK(scores, lengths, k=k, backend=backend)
    selector.execute()
    reference = expected(scores, lengths, k)
    assert torch.equal(selector.output.sort(1).values, reference.sort(1).values), (
        rows,
        width,
        k,
        distribution,
        backend,
    )
    before = selector.metadata[0].tolist()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selector.forward()
    for step in (0, 1):
        if step == 1:
            lengths.fill_(min(width, k + 1))
            scores.fill_(0.0)
            scores.masked_fill_(columns >= lengths[:, None], float("nan"))
        graph.replay()
        reference = expected(scores, lengths, k)
        assert torch.equal(selector.output.sort(1).values, reference.sort(1).values), (
            "graph",
            rows,
            width,
            k,
            distribution,
            backend,
            step,
        )
    path = (
        "register"
        if width <= 16384
        else "small_cluster"
        if backend == "cluster"
        and width > selector.cluster_floor
        and rows <= selector.cluster_capacity
        else "persistent_cluster"
        if selector.needs_plan
        else "streaming"
    )
    return dict(
        rows=rows,
        width=width,
        k=k,
        distribution=distribution,
        backend=backend,
        dispatched_path=path,
        initial_plan=before,
        changed_plan=selector.metadata[0].tolist(),
        fixed_logits="exact",
        graph_replan="exact",
        poisoned_tail="passed",
    )


def check_ragged():
    import torch

    from vllm_dsv41_indexer.topk import RaggedTopK

    records = []
    for k in (512, 2048):
        for distribution in ("random", "equal_nonzero", "tie_bin_overflow"):
            scores = make_scores(8, 131072, distribution)
            starts = torch.tensor(
                [0, 1, 2, 3, 7, 8, 31, 33], device="cuda", dtype=torch.int32
            )
            lengths = torch.tensor(
                [0, 1, k - 1, k, k + 1, 16384, 65536, 131000],
                device="cuda",
                dtype=torch.int32,
            )
            offsets = torch.tensor(
                [0, 100, 2, 4, 1024, 32, 16, 8], device="cuda", dtype=torch.int32
            )
            columns = torch.arange(scores.shape[1], device="cuda")[None, :]
            scores.masked_fill_(
                (columns < starts[:, None]) | (columns >= (starts + lengths)[:, None]),
                float("nan"),
            )
            reference = expected(scores, lengths, k, starts, offsets)
            selector = RaggedTopK(scores, lengths, starts, offsets, k=k)
            selector.execute()
            assert torch.equal(selector.output.sort(1).values, reference.sort(1).values)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                selector.execute()
            graph.replay()
            assert torch.equal(selector.output.sort(1).values, reference.sort(1).values)
            records.append(dict(k=k, distribution=distribution, ragged="exact"))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_vllm()
    report = dict(status="running", checks=[], ragged=[])
    try:
        cases = [(1, width) for width in (4096, 16384, 65536, 131072, 262144)]
        cases += [
            (rows, width)
            for rows in (8, 32, 64, 128)
            for width in (16384, 65536, 262144)
        ]
        cases += [(513, 131072)]
        for rows, width in cases:
            for k in (512, 2048):
                for distribution in (
                    "random",
                    "zero",
                    "equal_nonzero",
                    "tie_bin_overflow",
                    "cutoff_ties",
                ):
                    for backend in ("streaming", "cluster"):
                        print(
                            f"Checking rows={rows} width={width} k={k} "
                            f"{distribution} {backend}",
                            flush=True,
                        )
                        report["checks"].append(
                            check_case(rows, width, k, distribution, backend)
                        )
                    save(args.output, report)
        report["ragged"] = check_ragged()
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save(args.output, report)
    print(
        f"Passed {len(report['checks'])} full/source "
        f"and {len(report['ragged'])} ragged cases",
        flush=True,
    )


if __name__ == "__main__":
    main()

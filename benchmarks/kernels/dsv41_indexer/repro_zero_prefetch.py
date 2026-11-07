# SPDX-License-Identifier: Apache-2.0
"""P3 regression: empty requests between non-empty native Q pipeline tasks."""

import argparse

from .common import check_vllm, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-gemm-package")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_vllm()
    if args.deep_gemm_package:
        from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init

        _lazy_init()
        _import_deep_gemm()._C.init(args.deep_gemm_package)
    import torch

    from vllm_dsv41_indexer.contract import IndexerShape
    from vllm_dsv41_indexer.module import FullIndexer, IndexerInputs, NativeFullIndexer

    from .inputs import make_inputs

    case = dict(
        rows=6,
        compressed_length=65536,
        rows_per_request=1,
        page_size=64,
        candidate_blocks=2048,
    )
    q, _, weights, cache, table, requests, lengths, _ = make_inputs(case)
    requests.copy_(requests.flip(0))
    weights.neg_()
    records = []
    for lens in (
        [65536, 0, 1, 7, 511, 513],
        [65536, 1, 1, 7, 511, 513],
        [65536, 0, 0, 7, 511, 513],
        [0, 0, 1, 0, 511, 513],
        [0] * 6,
    ):
        lengths.copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        inputs = IndexerInputs(
            q, weights, cache, table, requests, lengths, IndexerShape(6, 65536)
        )
        native = NativeFullIndexer(inputs)
        actual = native.score()
        oracle = FullIndexer(inputs)
        oracle.score()
        errors = []
        for row, n in enumerate(lens):
            errors.append(
                float((actual[row, :n] - oracle.scores[row, :n]).abs().max())
                if n
                else 0
            )
        record = dict(lengths=lens, max_abs_per_row=errors)
        records.append(record)
        print(record, flush=True)
    save(args.output, dict(deep_gemm_package=args.deep_gemm_package, records=records))


if __name__ == "__main__":
    main()

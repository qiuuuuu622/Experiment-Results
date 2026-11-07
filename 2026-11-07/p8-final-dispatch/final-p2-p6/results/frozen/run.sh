#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p8-final-20261107-1340
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.freeze_p3 results/frozen --command-file run.sh --native-package /root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1 > results/freeze.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_topk --output results/p6/verify.json > results/p6/verify.log 2>&1
.venv/bin/python -c 'from benchmarks.kernels.dsv41_indexer.common import check_vllm; check_vllm(); from vllm.utils.deep_gemm import _lazy_init, _import_deep_gemm; _lazy_init(); _import_deep_gemm()._C.init("/root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1"); import runpy, sys; sys.argv = ["verify_compact", "--output", "results/p2/verify.json"]; runpy.run_module("benchmarks.kernels.dsv41_indexer.verify_compact", run_name="__main__")' > results/p2/verify.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark --rows 1 8 32 64 128 --context 4096 16384 65536 131072 262144 --output results/p2/forward.json > results/p2/forward.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark --rows 1 8 32 64 128 --context 4096 16384 65536 131072 262144 --reverse-order --output results/p2/reverse.json > results/p2/reverse.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_topk --output results/p6/forward.json > results/p6/forward.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_topk --reverse-order --output results/p6/reverse.json > results/p6/reverse.log 2>&1
printf '%s\n' completed > results/STATUS

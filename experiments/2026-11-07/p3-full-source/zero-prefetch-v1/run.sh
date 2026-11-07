#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107
export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.repro_zero_prefetch --output results/indexer/20261107-resume/p3/zero-prefetch-v1/original.json > results/indexer/20261107-resume/p3/zero-prefetch-v1/original.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.repro_zero_prefetch --deep-gemm-package /root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1 --output results/indexer/20261107-resume/p3/zero-prefetch-v1/patched.json > results/indexer/20261107-resume/p3/zero-prefetch-v1/patched.log 2>&1

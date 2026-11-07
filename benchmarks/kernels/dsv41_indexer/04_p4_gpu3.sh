#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-port
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p results/indexer/20261106-tp4/p4
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_grouped --paired \
    --output results/indexer/20261106-tp4/p4/benchmark-paired.json \
    > results/indexer/20261106-tp4/p4/benchmark-paired.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_grouped \
    --output results/indexer/20261106-tp4/p4/benchmark.json \
    > results/indexer/20261106-tp4/p4/benchmark.log 2>&1

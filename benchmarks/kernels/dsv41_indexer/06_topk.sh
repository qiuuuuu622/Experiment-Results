#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-port
export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p results/indexer/20261106-tp4/p6
for order in forward reverse; do
    extra=()
    if [ "$order" = reverse ]; then extra=(--reverse-order); fi
    .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_topk \
        "${extra[@]}" --output "results/indexer/20261106-tp4/p6/$order.json" \
        > "results/indexer/20261106-tp4/p6/$order.log" 2>&1
done

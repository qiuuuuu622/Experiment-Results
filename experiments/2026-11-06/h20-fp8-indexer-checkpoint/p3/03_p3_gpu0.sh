#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-port
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p results/indexer/20261106-tp4/p3
for order in forward reverse; do
    args=()
    if [ "$order" = reverse ]; then args=(--reverse-order); fi
    .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_full --phase full \
        "${args[@]}" --output "results/indexer/20261106-tp4/p3/$order.json" \
        > "results/indexer/20261106-tp4/p3/$order.log" 2>&1
done

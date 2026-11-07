#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-port
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p results/indexer/20261106-tp4
for order in forward reverse; do
    extra=()
    if [ "$order" = reverse ]; then extra=(--reverse-order); fi
    .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark \
        --rows 1 8 --context 65536 262144 --verify 1 \
        --modes eager graph --repeats 30 --inner 10 "${extra[@]}" \
        --output "results/indexer/20261106-tp4/p2-$order.json" \
        > "results/indexer/20261106-tp4/p2-$order.log" 2>&1
done

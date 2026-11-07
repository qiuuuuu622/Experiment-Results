#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-port
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
for phase in full verify; do
    for order in forward reverse; do
        extra=()
        if [ "$order" = reverse ]; then extra=(--reverse-order); fi
        .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_full \
            --phase "$phase" "${extra[@]}" \
            --output "results/indexer/20261106-tp4/$phase-$order.json" \
            > "results/indexer/20261106-tp4/$phase-$order.log" 2>&1
    done
done

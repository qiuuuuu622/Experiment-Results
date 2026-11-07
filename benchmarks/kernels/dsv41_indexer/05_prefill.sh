#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-port
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
for suite in smoke matrix; do
    for order in forward reverse; do
        extra=()
        if [ "$order" = reverse ]; then extra=(--reverse); fi
        .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_prefill \
            --suite "$suite" "${extra[@]}" \
            --output "results/indexer/20261106-tp4/p5/$suite-$order.json" \
            > "results/indexer/20261106-tp4/p5/$suite-$order.log" 2>&1
    done
done

#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-port
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_prefill \
    --output results/indexer/20261106-tp4/p5/gluon-verify.json \
    > results/indexer/20261106-tp4/p5/gluon-verify.log 2>&1
for order in forward reverse; do
    extra=()
    if [ "$order" = reverse ]; then extra=(--reverse); fi
    .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_prefill \
        --suite gluon "${extra[@]}" \
        --output "results/indexer/20261106-tp4/p5/gluon-$order.json" \
        > "results/indexer/20261106-tp4/p5/gluon-$order.log" 2>&1
done

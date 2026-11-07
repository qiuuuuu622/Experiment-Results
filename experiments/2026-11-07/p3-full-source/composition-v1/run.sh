#!/usr/bin/env bash
set -euo pipefail
# Reuse the completed prior extraction location solely to reuse compilation cache.
cd /root/dsv41-indexer-resume-20261107/runs/p3-full-final-v1
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
TASK_PY=/root/dsv41-indexer-resume-20261107/.venv/bin/python
TASK_OUT=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p3/composition-v1
TASK_NATIVE=/root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1
for TASK_ORDER in forward reverse; do
 TASK_REVERSE=()
 if [ "$TASK_ORDER" = reverse ]; then TASK_REVERSE=(--reverse-order); fi
 "$TASK_PY" -m benchmarks.kernels.dsv41_indexer.benchmark_full --phase full --cases 32:4096:1 64:16384:1 1:65536:1 8:262144:1 --variants native_source_four_consumers hybrid_source_four_consumers hybrid_source_full_consumers --deep-gemm-package "$TASK_NATIVE" "${TASK_REVERSE[@]}" --output "$TASK_OUT/$TASK_ORDER.json" > "$TASK_OUT/$TASK_ORDER.log" 2>&1
done

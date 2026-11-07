#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p3-full-final-v1
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
TASK_PY=/root/dsv41-indexer-resume-20261107/.venv/bin/python
TASK_OUT=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p3/full-final-v1
TASK_NATIVE=/root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1
hostname > "$TASK_OUT/hostname.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used --format=csv > "$TASK_OUT/gpus.csv"
"$TASK_PY" -m benchmarks.kernels.dsv41_indexer.verify_full --deep-gemm-package "$TASK_NATIVE" --output "$TASK_OUT/verify.json" > "$TASK_OUT/verify.log" 2>&1
for TASK_ORDER in forward reverse; do
 TASK_REVERSE=()
 if [ "$TASK_ORDER" = reverse ]; then TASK_REVERSE=(--reverse-order); fi
 "$TASK_PY" -m benchmarks.kernels.dsv41_indexer.benchmark_full --phase full --ratio 1 --cases 1:65536:1 8:262144:1 32:4096:1 64:16384:1 128:131072:1 --deep-gemm-package "$TASK_NATIVE" "${TASK_REVERSE[@]}" --output "$TASK_OUT/ratio1-$TASK_ORDER.json" > "$TASK_OUT/ratio1-$TASK_ORDER.log" 2>&1
 "$TASK_PY" -m benchmarks.kernels.dsv41_indexer.benchmark_full --phase full --ratio 2 --cases 8:262144:1 --deep-gemm-package "$TASK_NATIVE" "${TASK_REVERSE[@]}" --output "$TASK_OUT/ratio2-$TASK_ORDER.json" > "$TASK_OUT/ratio2-$TASK_ORDER.log" 2>&1
 "$TASK_PY" -m benchmarks.kernels.dsv41_indexer.benchmark_full --phase full --cases 1:65536:1 8:262144:1 --variants native_full native_score native_source native_score_ant_token_and_source_topk hybrid_source_four_consumers "${TASK_REVERSE[@]}" --output "$TASK_OUT/unpatched-$TASK_ORDER.json" > "$TASK_OUT/unpatched-$TASK_ORDER.log" 2>&1
done

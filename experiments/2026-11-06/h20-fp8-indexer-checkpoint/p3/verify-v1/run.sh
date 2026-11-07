#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-port/runs/p3-verify-v1
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
/root/dsv41-indexer-port/.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_full --output /root/dsv41-indexer-port/results/indexer/20261106-tp4/p3/verify-v1/result.json > /root/dsv41-indexer-port/results/indexer/20261106-tp4/p3/verify-v1/run.log 2>&1

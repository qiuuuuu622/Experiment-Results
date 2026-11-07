#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p3-final-v1
export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
/root/dsv41-indexer-resume-20261107/.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_full --deep-gemm-package /root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1 --output /root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p3/verify-v1/result.json > /root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p3/verify-v1/run.log 2>&1

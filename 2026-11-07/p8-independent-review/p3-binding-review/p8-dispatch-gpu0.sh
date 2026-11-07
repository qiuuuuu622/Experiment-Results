#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p8-dispatch-final-v2
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_dispatch --manifest results/indexer/20261107-resume/p8/admission-single.json --environment-digest 8528280c7188f033097340d7931fe440c0d58fbf0cd8e2ff228d25876290776a --output verify-8528280c7188.json > verify-8528280c7188.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_dispatch --manifest results/indexer/20261107-resume/p8/admission-single.json --environment-digest f0d078d72c43717f86c735eb54fcb066bc887b56fc8a57a9b485fc8fd1f8c7c1 --output verify-f0d078d72c43.json > verify-f0d078d72c43.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_dispatch --manifest results/indexer/20261107-resume/p8/admission-single.json --environment-digest 9653c78abe440326fb86f873ffbb24c7e64b07924ecd42ac79fd43529b742e41 --output verify-9653c78abe44.json --deep-gemm-package /root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1 > verify-9653c78abe44.log 2>&1
touch GPU0-COMPLETE

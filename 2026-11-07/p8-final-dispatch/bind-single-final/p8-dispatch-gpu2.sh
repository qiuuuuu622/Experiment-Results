#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p8-dispatch-final-v2
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=2
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_dispatch --manifest results/indexer/20261107-resume/p8/admission-single.json --environment-digest c2388e732a9519fa24e48abd4116bbfd73ba3d5cd8687204e7520b4c8b7f9867 --output verify-c2388e732a95.json > verify-c2388e732a95.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_dispatch --manifest results/indexer/20261107-resume/p8/admission-single.json --environment-digest aee8d03e85ce978b43a088b62718f6060fffe2b18db87134074f07c36f9e9d9b --output verify-aee8d03e85ce.json --deep-gemm-package /root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1 > verify-aee8d03e85ce.log 2>&1
touch GPU2-COMPLETE

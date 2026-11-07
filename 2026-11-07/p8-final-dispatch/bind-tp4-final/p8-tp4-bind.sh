#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p8-tp4-bind-final-v1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m benchmarks.kernels.dsv41_indexer.verify_dispatch --scope tp4 --manifest results/indexer/20261107-resume/p8/admission-tp4.json --environment-digest e57a8a58f8fa674625cfb91c924e69f90e9e5a3752dda98ffb58fcc8069aeff2 --output verify-e57a8a58f8fa.json --deep-gemm-package /root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1 > verify-e57a8a58f8fa.log 2>&1
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m benchmarks.kernels.dsv41_indexer.verify_dispatch --scope tp4 --manifest results/indexer/20261107-resume/p8/admission-tp4.json --environment-digest 79adeb1d054c8c5b118d5588471097107c13d72075cce7da56eff679c231ec7a --output verify-79adeb1d054c.json > verify-79adeb1d054c.log 2>&1
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m benchmarks.kernels.dsv41_indexer.verify_dispatch --scope tp4 --manifest results/indexer/20261107-resume/p8/admission-tp4.json --environment-digest 660bbec902c46f39c7ce7bc43ea0f198a60affee2987383af24d77851cd3d5cf --output verify-660bbec902c4.json > verify-660bbec902c4.log 2>&1
touch COMPLETE

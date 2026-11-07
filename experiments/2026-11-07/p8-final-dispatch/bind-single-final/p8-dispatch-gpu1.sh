#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p8-dispatch-final-v2
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_dispatch --manifest results/indexer/20261107-resume/p8/admission-single.json --environment-digest 7414f0932e1ef941ba51f06f07eb96ef8037b81497f3a5ceb13f339a6350a578 --output verify-7414f0932e1e.json > verify-7414f0932e1e.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_dispatch --manifest results/indexer/20261107-resume/p8/admission-single.json --environment-digest 89cf099835780f3929ce13130121ccca7c23bbb1cda35458b2c372eeaaa1c0c6 --output verify-89cf09983578.json > verify-89cf09983578.log 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_dispatch --manifest results/indexer/20261107-resume/p8/admission-single.json --environment-digest f1fc1c214e8cb1f566b31d3137234e464d246ba5111c3aeb89a15f9999e796ff --output verify-f1fc1c214e8c.json --deep-gemm-package /root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1 > verify-f1fc1c214e8c.log 2>&1
touch GPU1-COMPLETE

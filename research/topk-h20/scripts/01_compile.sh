#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-topk-h20-tuning-20261107
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export MAX_JOBS=4
.venv/bin/python tune_topk.py --phase compile --output results/compile.json

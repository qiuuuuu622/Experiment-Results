#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-topk-h20-tuning-20261107
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
gpu="$1"
export CUDA_VISIBLE_DEVICES="$gpu"
case "$gpu" in
  0) rows="1 8";;
  1) rows="32";;
  2) rows="64";;
  3) rows="128";;
  *) exit 2;;
esac
.venv/bin/python tune_topk.py --phase benchmark --rows $rows --widths 65536 131072 262144 --execution eager graph --trace --output "results/gpu${gpu}-benchmark.json"
.venv/bin/python tune_topk.py --phase verify --rows $rows --widths 262144 --output "results/gpu${gpu}-verify.json"

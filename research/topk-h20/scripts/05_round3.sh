#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-topk-h20-tuning-20261107
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
gpu="$1"
export CUDA_VISIBLE_DEVICES="$gpu"
case "$gpu" in
  0) rows="256";;
  1) rows="32";;
  2) rows="64";;
  3) rows="128";;
  *) exit 2;;
esac
variants="native ant_streaming stream_hist2 stream_warpagg"
.venv/bin/python tune_topk.py --phase benchmark --rows $rows --widths 131072 262144 --execution eager graph --variants $variants --output "results/round3/gpu${gpu}-random.json"
.venv/bin/python tune_topk.py --phase benchmark --rows $rows --widths 131072 262144 --distribution indexer --execution eager graph --variants $variants --output "results/round3/gpu${gpu}-indexer.json"
.venv/bin/python tune_topk.py --phase verify --rows $rows --widths 262144 --variants stream_hist2 stream_warpagg --output "results/round3/gpu${gpu}-verify.json"

#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-topk-h20-tuning-20261107
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
gpu="$1"
export CUDA_VISIBLE_DEVICES="$gpu"
case "$gpu" in
  0) rows="8"; variants="native ant_streaming ant_c8_p18_fixed ant_c2_p78_fixed";;
  1) rows="32"; variants="native ant_streaming ant_c2_p78_fixed";;
  2) rows="64"; variants="native ant_streaming ant_c2_p78_fixed";;
  3) rows="128"; variants="native ant_streaming stream_hist2";;
  *) exit 2;;
esac
for round in 1 2 3; do
  .venv/bin/python tune_topk.py --phase benchmark --rows $rows --widths 131072 262144 --distribution indexer --execution eager graph --variants $variants --output "results/final-v1/gpu${gpu}-round${round}.json"
done
.venv/bin/python tune_topk.py --phase verify --rows $rows --widths 262144 --variants $variants --output "results/final-v1/gpu${gpu}-verify.json"

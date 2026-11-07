#!/usr/bin/env bash
# Equivalent to auxiliary compile commands already issued during first compilation.
set -euo pipefail
cd /root/dsv41-topk-h20-tuning-20261107
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda MAX_JOBS=4 .venv/bin/python tune_topk.py --phase compile --variants ant_c4_p36_fixed ant_c4_p18_fixed --output results/compile-c4.json &
CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda MAX_JOBS=4 .venv/bin/python tune_topk.py --phase compile --variants ant_c2_p78_fixed ant_c2_p39_fixed --output results/compile-c2.json &
CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda MAX_JOBS=4 .venv/bin/python tune_topk.py --phase compile --variants ant_c8_p9_fixed --output results/compile-p9.json &
wait

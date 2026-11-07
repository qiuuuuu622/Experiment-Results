#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-vision-20261107
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4
exec .venv/bin/python -u -m torch.distributed.run \
  --nnodes=1 --node_rank=0 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29861 \
  runs/frozen-rmsnorm-v1/benchmarks/kernels/dsv41_vision/bench_vision.py \
  --model /model --images images --source-root runs/frozen-rmsnorm-v1 \
  --mode data --batch-sizes 4 8 \
  --variants eager eager_rope eager_rmsnorm eager_rope_rmsnorm \
  --rmsnorm-backend single --verify-weight-shards \
  --reference-dir runs/dp4-reference-v6 --reference-label dp4_fullweights \
  --correctness-only --output runs/rmsnorm-module-correctness-v1.json

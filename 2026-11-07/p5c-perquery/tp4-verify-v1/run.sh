#!/usr/bin/env bash
set -eu
cd /root/dsv41-indexer-resume-20261107/runs/p5c-tp4-verify-v1
out=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5c-perquery/tp4-verify-v1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
.venv/bin/python capture_identity.py "$out" before
if .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m benchmarks.kernels.dsv41_indexer.verify_perquery_tp4 --output-dir "$out"; then
  rc=0
else
  rc=$?
fi
.venv/bin/python capture_identity.py "$out" after
printf '%s\n' "$rc" > "$out/exit-code.txt"
exit "$rc"

#!/usr/bin/env bash
set -eu
cd /root/dsv41-indexer-resume-20261107/runs/p5-model-numerical-v2
out=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5-model-shapes/numerical-diagnostic
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python capture_identity.py "$out" before
if .venv/bin/python -m benchmarks.kernels.dsv41_indexer.diagnose_prefill_model --output-dir "$out"; then
  rc=0
else
  rc=$?
fi
.venv/bin/python capture_identity.py "$out" after
printf '%s\n' "$rc" > "$out/exit-code.txt"
exit "$rc"

#!/usr/bin/env bash
set -eu
cd /root/dsv41-indexer-resume-20261107/runs/p5-model-verify-v1
out=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5-model-shapes/dynamic-verifier-v1
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python capture_identity.py "$out" before
if .venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_prefill_model --output "$out/verify.json"; then
  rc=0
else
  rc=$?
fi
.venv/bin/python capture_identity.py "$out" after
printf '%s\n' "$rc" > "$out/exit-code.txt"
exit "$rc"

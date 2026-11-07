#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export MAX_JOBS=8
out=results/indexer/20261107-resume/audit
mkdir -p "$out"
cp frozen-audit-source.json "$out/frozen-source.json"
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_grouped --output "$out/grouped.json" > "$out/grouped.log" 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_grouped --paired --output "$out/grouped-paired.json" > "$out/grouped-paired.log" 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_prefill --output "$out/prefill.json" > "$out/prefill.log" 2>&1
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_resume_modules --output "$out/topk.json" > "$out/topk.log" 2>&1
.venv/bin/python -m pytest -c tests/kernels/dsv41_indexer/pytest.ini tests/kernels/dsv41_indexer/test_grouped.py tests/kernels/dsv41_indexer/test_prefill.py -q > "$out/pytest.log" 2>&1

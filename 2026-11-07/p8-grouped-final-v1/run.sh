#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p8-grouped-final-v1
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export MAX_JOBS=8
TASK_PY=/root/dsv41-indexer-resume-20261107/.venv/bin/python
TASK_OUT=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p8-grouped-final-v1
"$TASK_PY" -m benchmarks.kernels.dsv41_indexer.environment --output "$TASK_OUT/environment.json" > "$TASK_OUT/environment.log" 2>&1
"$TASK_PY" - <<'NATIVE' > "$TASK_OUT/native-identity.json"
import hashlib,json
from pathlib import Path
from benchmarks.kernels.dsv41_indexer.common import check_vllm
installed=Path(check_vllm())
files=[]
for name in ['deep_gemm','vllm']:
 for p in sorted((installed/name).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ['.py','.cuh','.h','.so']):
   files.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(dict(native_patch='unmodified installed package; no P3 zero-prefetch override',files=files),indent=2))
NATIVE
"$TASK_PY" -m benchmarks.kernels.dsv41_indexer.verify_grouped --paired --output "$TASK_OUT/verify.json" > "$TASK_OUT/verify.log" 2>&1
"$TASK_PY" -m benchmarks.kernels.dsv41_indexer.benchmark_grouped --paired --output "$TASK_OUT/final.json" > "$TASK_OUT/final.log" 2>&1
"$TASK_PY" -m benchmarks.kernels.dsv41_indexer.benchmark_grouped_planning --output "$TASK_OUT/planning.json" > "$TASK_OUT/planning.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p5-model-20261107-final-v3
out=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5-ant-keys-major/model-aligned/isolated-source
mkdir -p "$out/frozen"
cp /root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5-ant-keys-major/model-aligned/final-v3/frozen/source.tar.gz "$out/frozen/source.tar.gz"
export CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
py=/root/dsv41-indexer-resume-20261107/.venv/bin/python
"$py" capture_identity.py "$out" before
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu --format=csv > "$out/gpu-before.csv"
nvidia-smi -i 1 --query-gpu=timestamp,index,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu --format=csv -l 1 > "$out/telemetry.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
for order in forward reverse; do
  flag=()
  if [ "$order" = reverse ]; then flag=(--reverse); fi
  "$py" -m benchmarks.kernels.dsv41_indexer.benchmark_prefill_model --role source --case steady_256k "${flag[@]}" --identity "$out/runtime-before.json" --output "$out/source-$order.json" > "$out/source-$order.log" 2>&1
done
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu --format=csv > "$out/gpu-after.csv"
"$py" capture_identity.py "$out" after

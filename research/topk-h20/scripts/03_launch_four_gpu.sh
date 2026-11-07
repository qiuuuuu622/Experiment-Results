#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-topk-h20-tuning-20261107
nvidia-smi --query-gpu=timestamp,index,uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu --format=csv -l 1 > results/gpu-monitor.csv 2>&1 &
printf '%s\n' "$!" > results/monitor.pid
for gpu in 0 1 2 3; do
  nohup bash scripts/02_benchmark.sh "$gpu" > "results/gpu${gpu}.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "results/gpu${gpu}.pid"
done

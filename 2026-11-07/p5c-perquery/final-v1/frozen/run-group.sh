#!/bin/bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p5c-final-v1
out=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5c-perquery/final-v1
group=$1
if [ "$group" = a ]; then
    gpu=1
    cases='initial_2k initial_8k initial_32k steady_16k'
elif [ "$group" = b ]; then
    gpu=2
    cases='steady_64k steady_256k tail_1m ragged'
else
    exit 2
fi
for order in forward reverse; do
    flag=''
    if [ "$order" = reverse ]; then flag=--reverse; fi
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_prefill_perquery --cases $cases $flag --identity "$out/runtime-before.json" --output "$out/group-$group-$order.json" > "$out/group-$group-$order.log" 2>&1
done
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python capture_identity.py "$out" "after-$group"

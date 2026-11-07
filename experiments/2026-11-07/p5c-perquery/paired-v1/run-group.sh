#!/bin/bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107/runs/p5c-final-v1
out=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5c-perquery/paired-v1
group=$1
if [ "$group" = a ]; then gpu=1; elif [ "$group" = b ]; then gpu=3; else exit 2; fi
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python capture_identity.py "$out" "before-$group"
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python /root/dsv41-indexer-resume-20261107/runs/p5c-paired-v1/paired_pipeline.py --frozen-root /root/dsv41-indexer-resume-20261107/runs/p5c-final-v1 --identity "$out/runtime-before-$group.json" --group "$group" --output "$out/group-$group.json" > "$out/group-$group.log" 2>&1
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python capture_identity.py "$out" "after-$group"

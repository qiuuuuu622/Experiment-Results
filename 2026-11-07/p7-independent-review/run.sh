#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-indexer-resume-20261107
export CUDA_VISIBLE_DEVICES=3
.venv/bin/python results/indexer/20261107-resume/p7-review-p3/repro.py > results/indexer/20261107-resume/p7-review-p3/result.json

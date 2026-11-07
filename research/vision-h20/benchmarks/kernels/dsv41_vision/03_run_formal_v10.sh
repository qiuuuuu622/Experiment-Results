#!/usr/bin/env bash
set -euo pipefail
cd /root/dsv41-vision-20261107
bash benchmarks/kernels/dsv41_vision/01_formal_dp4_v10.sh > runs/formal-dp4-v10.log 2>&1
bash benchmarks/kernels/dsv41_vision/02_formal_tp4_v10.sh > runs/formal-tp4-v10.log 2>&1

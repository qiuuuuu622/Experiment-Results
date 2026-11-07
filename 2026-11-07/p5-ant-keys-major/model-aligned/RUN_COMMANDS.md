# 可复现实验命令

固定snapshot：`/root/dsv41-indexer-resume-20261107/runs/p5-model-20261107-final-v3`。从`final-v3/frozen/source.tar.gz`解压可恢复源码；使用独立venv。下面三个role分别用GPU1/2/3，在各自GPU顺序执行forward、reverse；同GPU不并发。

```bash
cd /root/dsv41-indexer-resume-20261107/runs/p5-model-20261107-final-v3
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
PY=/root/dsv41-indexer-resume-20261107/.venv/bin/python
OUT=/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5-ant-keys-major/model-aligned/final-v3
CUDA_VISIBLE_DEVICES=1 "$PY" capture_identity.py "$OUT" before
# role=full/gpu=1, role=source/gpu=2, role=consumer/gpu=3
CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m benchmarks.kernels.dsv41_indexer.benchmark_prefill_model \
  --role "$role" --identity "$OUT/runtime-before.json" --output "$OUT/$role-forward.json" > "$OUT/$role-forward.log" 2>&1
CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m benchmarks.kernels.dsv41_indexer.benchmark_prefill_model \
  --role "$role" --reverse --identity "$OUT/runtime-before.json" --output "$OUT/$role-reverse.json" > "$OUT/$role-reverse.log" 2>&1
# 等所有GPU进程完成后
CUDA_VISIBLE_DEVICES=1 "$PY" capture_identity.py "$OUT" after
```

上面的PY/OUT仅为命令缩写。实际命令始终指向`.venv/bin/python`。复跑应使用新输出目录，不能覆盖本轮不可变结果。先创建该目录的`frozen/source.tar.gz`，再采before身份。

独立GPU1正反复核脚本完整保存在 `isolated-source/run.sh`；它含before/after身份、全卡状态、GPU1每秒温度/时钟/功耗监控。脚本只终止自己启动的只读监控进程，没有修改时钟/功率或其他进程。

本地整理命令（从优化repo根目录）:

```bash
.venv/bin/python results/indexer/20261107-resume/p5-ant-keys-major/model-aligned/scripts/make_manifest.py
.venv/bin/python results/indexer/20261107-resume/p5-ant-keys-major/model-aligned/scripts/summarize_model.py
.venv/bin/python results/indexer/20261107-resume/p5-ant-keys-major/model-aligned/scripts/finalize_decision.py
```

模型batch采用f422... driver，专用Ant core957557...，完整SHA见`DECISION.json`。旧synthetic与smoke证据分开保存，不将其数据冒称final-v3。

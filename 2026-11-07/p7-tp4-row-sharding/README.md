# P7 真实多卡复制与行分摊实验

状态：真实四卡实现、正确性和多批性能已完成，最终全零恢复验证1920项通过，P7已完成本阶段验证。详见 REPORT.md；旧 native 静态路径已明确排除，不能用于准入。

实现位于 `vllm_dsv41_indexer/row_shard.py`，入口为
`benchmarks.kernels.dsv41_indexer.verify_tp4` 和 `benchmark_tp4`。

- 复制模式每 rank 完整 32 个 index heads，逻辑输入相同，物理 cache 页独立置换。
- 分摊模式先按 request 完整分组，ragged 使用最大组优先分配。组不足另测显式 split_rows；逆映射恢复原 query 行序。
- source 与四个不同 Q/head weights 的 consumer 保持同一行所有权，source 候选留在所有者本地，只聚集各层 logical top-k。
- replicated-input 每 rank 事先有 Q/weights/metadata，计入本地 pack；root-input 只有 rank 0 的输入有效，单个预分配 byte buffer 打包后 scatter。非 root 验证输入故意填错，禁止依赖其值。cache/table 均不进入通信载荷。
- 空 rank 不运行 indexer kernel，但参加 scatter/all-gather。未分摊完整 index heads。
- 随机投影是合成模块：FP32 hidden×Wq / hidden×Whead；每 head 的 Q maxabs/448 scale，Q E4M3，scale 合入 FP32 head weight。不是完整模型的投影/RoPE，保留 FP8 cache。GEMM 的 M 维变化可能改变 FP32 舍入，验证单列 raw Q、raw weights、FP8 Q、带 scale weights 的误差及 top-k。
- 投影中的 torch abs/amax/where/div/clamp 会产生临时张量。forward 全部计时，记录 CUDA graph pool allocated delta；不宣称所有临时内存都提前申请。
- 性能每 variant 20 warmup、30 samples、每 sample 10 调用，正反序均测。每次调用各 rank GPU 时间取最大，再在 sample 内平均；另外记录主机调用＋完成同步墙钟。变异系数超过 10% 追加三轮并保留全部。
- CUDA graph 包含完整 pack、scatter（若有）、kernel、logical output gather。初始化通信、JIT、计划、图捕获在计时外。基准没有服务或完整模型。

## 已执行

```sh
.venv/bin/python -m pytest -c tests/kernels/dsv41_indexer/pytest.ini --confcutdir=tests/kernels/dsv41_indexer tests/kernels/dsv41_indexer/test_row_shard.py -q
# 11 passed
.venv/bin/python -m ruff check vllm_dsv41_indexer/row_shard.py benchmarks/kernels/dsv41_indexer/{tp4_cases,verify_tp4,benchmark_tp4}.py tests/kernels/dsv41_indexer/test_row_shard.py
# All checks passed
```

2026-11-07 总控确认 h20-4 GPU0–3 全空闲后授予独占，执行：

```sh
cd /root/dsv41-indexer-resume-20261107
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m benchmarks.kernels.dsv41_indexer.verify_tp4 --output results/indexer/20261107-resume/p7/verify-tp4.json > results/indexer/20261107-resume/p7/verify-tp4.log 2>&1
```

## 代表性能矩阵（正确性通过后）

1. compact decode：请求 1/8/32/128；上下文 16K/64K/256K；ratio1，eager+graph；replicated 与 group shard；input replicated/root。
2. compact verify：请求 1/8；每请求 2/4/6/8 行；64K；组不足对比 split_rows。
3. full/source+4consumer：请求 1/8/32；64K 与 256K；固定随机输入；完整组合成本。
4. 投影＋compact：请求 1/8/32/128；64K；固定随机投影权重；与 score/top-k-only 模式独立比较。
5. ratio2 验证 256K 逻辑上下文对应 128K index keys，明确两个长度。

上述矩阵已运行并补齐64请求、4K/128K及ratio2。实际矩阵、代码版本与样本以原始JSON和REPORT.md为准。

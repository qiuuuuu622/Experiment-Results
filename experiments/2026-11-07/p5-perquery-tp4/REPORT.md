# P5c 真正四卡 per-query consumer 正确性补验

2026-11-07，SSH `h20-dsk`，4 张独立 H20 / 4 个 torchrun rank。最终退出码 0，3 个形状、每形状 7 个输入阶段全部通过。**本轮只有模块正确性验证，没有性能计时或 TP4 加速结论；不与旧 P7 通信性能合并。**

本轮 core 保持冻结 SHA `77a7ffbf34ebe4495e560612c9b6943b17214b5fd01a9cdfe613d5887dbe9b1c`。只新增 `benchmarks/kernels/dsv41_indexer/verify_perquery_tp4.py`，没有修改 core、旧 P5b 结果或已封存的 `p5-model-shapes`。

## 结果

- 1008 次“形状 × 阶段 × backend × 执行方式 × rank × ownership 模式”比较全部精确相同，合计 86016 次逻辑 top-k 行比较；这些是比较次数，不是 86016 个独立输入。
- rank0 另做 3584 行 scorer/稳定选择参考检查。direct 与 gathered 的有效 compact scores 逐位相同；相同 logits 的稳定 top-k 精确一致。
- 相对当前 native full-score 的最大分差 `1.9073486328125e-06`，所有已测行 top-k 集合重合率 100%。
- 辅助 FP32 参考的最大分差为 `0.0019866228103637695`，单独保留。它不等于 FP32 strict 门槛全部通过；本项证明的是分摊/收集与复制路径一致，没有把 native 数值贴近度冒称绝对 FP32 精确性。
- 前后源码、native headers/.so、包版本身份一致，四卡 UUID 不同。初始逻辑 Q/weights/requests/lengths/candidates/hash 和重建的逻辑 cache hash 各 rank 一致，物理页表 hash 四者各异；每个动态阶段也记录到四个不同页表 hash。

## 形状与行所有权

所有形状为 ratio1 consumer、H32、D128、top512、逐 query `[Q,2048]` 候选，使用固定随机量化输入。它们是模块 fixture，不是加载模型后的真实激活。

| Q | 每请求 query 数 | index-key 上限 | request-group 分摊的每 rank 行数 | split-row 分摊 | 发布路径 |
|---:|---|---:|---|---|---|
| 64 | [17,9,25,13] | 16384 | [25,17,13,9] | [16,16,16,16] | 外部候选去重 |
| 64 | [7,57] | 65536 | [57,7,0,0] | [16,16,16,16] | source-unique 兼容候选 |
| 128 | [33,31,64] | 65536 | [64,33,31,0] | [32,32,32,32] | source-unique 兼容候选 |

每种输入同时验证 direct 与 gathered_wgmma；每个 backend 都执行：

1. 每 rank 完整复制计算。
2. 优先保持请求组完整的行分摊。
3. 逐行分摊对照。

第 2、3 个形状分别覆盖 2 个、1 个 empty rank。空 rank 在该 ownership 变体中不构造或调用本地 indexer，仍参与逻辑输出 collective；其他变体照常执行。direct 强制最多 5 行一 chunk，gathered staging 限制最多 3 行，因此同一输入还覆盖两条路径不同分块边界与尾部不足一块。

## 动态输入与 Graph 验证顺序

7 个阶段分别为 initial、候选/length 更新、非空长度但空候选、zero-query 同分、request ID 与物理页重排、全零长度+页重排、恢复。

候选默认逐 query 不同；构造时强制包含最后一个可见块，从而覆盖尾部不满 8 个 key 的映射。外部候选分支加入重复值与 -1，source-unique 分支保持有效项唯一。K scale 乘 1.3，包含非 2 的幂的 FP32 scale。

每个阶段先更新输入，然后 poison 所有路径的候选发布状态、score、输出和 gathered staging；row-shard 的本地 receive buffer 也清零，检验 row packing 是否被捕获。随后 **replay 所有 Graph 并保存全部输出**，之后才允许 Eager 或 oracle 刷新任何准备缓冲。各 Graph 保存结果与 Eager、rank0 稳定参考按排序后的逻辑位置精确比较。

输出不足部分为 -1，不重复、不越界。zero-query 场景还直接检查结果为有效候选中逻辑位置最小的 512 个。全零长度、空候选输出全部 -1，最后恢复输出与初始阶段完全一致。

Graph 的 shape/storage、descriptor 形状、chunk 布局和 RowPlan 行所有权固定。request ID 动态更新为双射重命名，保持原组的行成员关系；任意改变分组行成员或行数需要重新规划，不在此固定 Graph 契约内。

## 四卡数据与通信

每 rank 都持有完整的本地 index cache。逻辑 cache 内容相同，物理页由 rank 独立的 permutation 放置。请求表只在本地查找；cache 和 physical page ID 从未加入通信字段。

`RowExchange` 在本地 pack 本 rank 所有的 Q/weights/request IDs/lengths/candidates，再计算本地行的 scores/top-k。NCCL 收集的是 int32 逻辑 top-k，按既定行所有权还原原始 query 顺序。rank0 参考的广播也只有逻辑 top-k。初始身份元数据只交换 SHA/设备信息，没有传输物理页表值。

本轮使用 replicated-input 模式；没有把输入分发成本隐藏在性能数字中，因为没有性能数字，也没有测 root-only 输入模式。通信与本地 pack 都包含在捕获的模块调用中。

## 设备与版本

| rank / 物理 GPU | UUID |
|---|---|
| 0 / 0 | GPU-PSEUDONYM-35e2765a41c51b7b128731d8 |
| 1 / 1 | GPU-PSEUDONYM-0cdcd3fce9f25fb1af81d50a |
| 2 / 2 | GPU-PSEUDONYM-d63e7ca0b567d269ab193c54 |
| 3 / 3 | GPU-PSEUDONYM-3a62a7aa1eaf409a737a6709 |

- 父 P5c source archive SHA：`b4c1a3dbda6778ea48d226b44ff77809dce604a50fd63b9b3aac15184e83e41b`。
- 新 verifier SHA：`7b3f1c9ceeaf6b9ad58f792ba921b1f92e3c20adf2d44e46ac9a618e8ca9a5c1`。
- 本轮冻结 archive SHA：`100a0baa88bea1474c76357993e39582f993e2df7c0f9ffacafb7d1af4bb7641`。
- 实际 run：`/root/dsv41-indexer-resume-20261107/runs/p5c-tp4-verify-v1`。
- 实际初始输出：`/root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5c-perquery/tp4-verify-v1`。
- 完成后原样复制归档：`results/indexer/20261107-resume/p5-perquery-tp4/final-v1`；归档没有改变原记录的运行路径。

`final-v1/rank0.json` 至 `rank3.json` 保存逐 rank 完整结果、计划与源码前后 hash；每 shape 的增量文件、`verify.log`、`runtime-before.json`/`runtime-after.json`、`frozen/source.tar.gz`、`run.sh` 和退出码均已保存。native binary build commit 仍按实际身份记录为未知，未用 wheel 名称冒充当前 HEAD 构建。

## 复现

在原冻结目录执行 `bash` 调用 `final-v1/run.sh`。核心命令为：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 .venv/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m benchmarks.kernels.dsv41_indexer.verify_perquery_tp4 \
  --output-dir <新的输出目录>
```

最终四卡均回到 0 MiB；本轮租约已释放。

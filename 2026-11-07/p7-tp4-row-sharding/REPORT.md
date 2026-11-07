# P7：H20 真四卡复制与按 query 行分摊

本轮在 `h20-4` 的 GPU0–3 运行了真实四进程 NCCL 实验，每个 rank 都保留完整 32 个 index heads、完整逻辑 index cache，以及独立置换的物理页表。没有运行完整模型或服务，没有改变 FP8 index cache。实现及实验代码在 `vllm_dsv41_indexer/row_shard.py` 与 `benchmarks/kernels/dsv41_indexer/*tp4*.py`。

**结论：行分摊有收益，但阈值不能只看 query 行数。** 小请求/短上下文、输入从 root 分发、verify 请求组不足时，打包和通信可能超过计算节省。保留复制模式，默认不自动启用未经同形状证据验证的分摊。

## 最终实现

- `RowPlan` 最大请求组优先分配，保持同请求 verify 组完整；显式 `split_rows` 用于组数不足的对照。原行号的逆映射恢复最终顺序。支持非整除、非连续相同请求、空 rank。
- 每个 rank 只计算归属 query，但始终计算全部 32 个 index heads。cache 和物理页号不参与通信。
- 复制输入模式：Q/weights/请求映射/长度/候选事先在各 rank，测量包含每步本地 pack。
- root 输入模式：只有 rank 0 的动态行输入有效，打包到一个 byte buffer 后 scatter。非 root 输入在正确性测试中故意填错，结果不能依赖它。host 端行计划属于预先规划的结构元数据，不在计时中重建。
- 输出以 int32 logical index-key positions all-gather，并恢复原始行序。source 与四个 consumer 共用行所有权，候选仅在所有者本地发布一次，不跨卡发送物理页号。
- 预分配通信区；空 rank 不调用 indexer kernel，仍参加 collectives。
- `native_stable`：每次 forward 都重新执行 NativeSourceIndexer.prepare，重建 rank-local table gather 和 DeepGEMM metadata。所有这些工作包含在 eager 或 graph 的完整成本中。full 使用 DeepGEMM score + Ant 稳定 token top-k；chain 还输出 source 候选供四个 consumer。原生非确定性 cooperative/persistent selector 不作为稳定模式。
- `native_stable` 显式加载 P3 隔离的 mixed-zero prefetch 修复包，路径 `/root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1`。它不是未修改的上游二进制基线。原始 JSON 记录所有相关 header/so 哈希。

P7 实现和本阶段验证已完成；P8 的最终调度选择及 Decision.bind 验证另行进行。

当前 `row_shard.py` SHA256：`4c33cd83aaeafc21cd443b73149d63e24fee695af311c24b12ebd89848720ec1`。

## 正确性与边界

- CPU 行计划行为测试 11 项通过，ruff 通过。
- 真四卡各自独立物理页布局，输出与每卡完整复制参考的逻辑 top-k 集合一致。没有用单卡多 rank 冒充四卡。
- 覆盖单请求空 rank、非整除行、1/2/4/6/8 行 verify 组、ragged 及非连续请求重排、ratio1/2、source+4consumer，以及 eager/graph。
- 动态阶段覆盖混合零长度、517 的部分候选尾块、重复/无效候选、完全同分、请求映射变化、原地物理 cache/table 重排；终版补充“全部零长度＋再置换物理页→恢复原始有效输入”。恢复必须回到最初输出 digest。
- 完全同分必须保留最小逻辑位置，并额外重复两次 graph replay；128 行 full 分摊为 4×32 行，明确跨越原生 cooperative/persistent 分支边界。
- chain 的 source candidate 集合也与完整复制参考逐项比较。所有输出检查唯一、合法、无未来位置泄漏。
- **输出不是承诺按分数排序的 top-k 列表**，比较的是每行集合；无效尾部为 -1。

终版动态结果 `verify-native-allzero-restore.json` 已通过：8case × 6step × eager/graph × 5variants = 480项/rank，共1920项；runner退出码0。中间版本均保留。

## 性能方法

每个 variant 20 次预热、30 个样本，每样本 10 次调用，正序和反序都运行；CV 超过 10% 追加三轮，全部保留。每次调用先取各 rank 的 GPU 时间最大值，再计算样本平均，另记录主机调用及完成同步的墙钟。表中 P50 是正反两序及所有追加轮的样本中位数，不挑最好一轮。

计时包含行整理、必要 scatter、kernel、logical top-k gather 和逆映射。初始化通信组、JIT、规划和 graph capture 在计时外。逐调用 CUDA event 对本身属于测量开销；分项测量不能相加代替完整 forward。显存记录是整个进程的峰值（含同时持有的对照工作区），不是单个 kernel 的独占显存。

### 最终稳定 full/source 路径，256K index keys，CUDA Graph

|模块|请求|四卡各自复制 P50 μs|按行分摊，输入已复制 μs|按行分摊，root输入 μs|
|---|---:|---:|---:|---:|
|full：准备＋DeepGEMM＋稳定top-k|1|60.17|77.98|103.32|
|full|8|142.36|101.00|135.14|
|full|32|395.64|180.75|250.66|
|source＋发布一次＋四consumer|1|123.78|159.67|225.91|
|source＋四consumer|8|294.37|202.56|299.60|
|source＋四consumer|32|754.54|381.24|594.14|

32 请求的 full 为 2.19×、完整 source/consumer 链为 1.98×。上述包含每步 metadata/table 准备；不能拿早期静态准备的结果替代。

来源：`native-stable-full-chain.json`，144 个完整模块正反记录。

### Compact consumer 的形状差异

较早 v1/v2 完整矩阵得到：256K、128 请求约 265.18→127.98 μs；但 1/8 请求变慢。4K、64 请求也变慢（70.52→96.31 μs），说明不能以“行数≥某值”概括所有上下文。每请求 6 行但只有一个请求时，拆行仍慢于复制（45.08→58.79 μs）；8 请求×6 行则约 115.66→102.31 μs。

这些历史实验的实现快照均保留，最终 compact 准入应使用 `compact-final-v3.json` 的当前代码复测，而不是把旧哈希自动称作最终源码。正式调度由 P8 仅对精确形状、输入模式、组布局与投影模式做审查。

最终 v3、64K index keys、graph 的当前代码复测：

|请求×每请求行|复制 μs|行分摊、输入已复制 μs|行分摊、root输入 μs|
|---|---:|---:|---:|
|8×1|50.30|60.82|99.64|
|8×6|115.74|103.24|196.16|
|32×1|101.32|85.30|157.70|
|32×6|362.02|141.01|184.89|

### 通信分项

`phase-breakdown.json` 的 v2 compact、64K、128 请求，在 graph 下：本地 pack 约 12.67 μs，local compute 约 102.46 μs，logical gather/恢复约 18.49 μs，完整 forward 约 126.69 μs；root pack+scatter 约 45.41 μs，完整 forward 约 168.18 μs。分项有独立启动成本，不能机械相加。

最终 `native-stable-phase.json` 补测了 native source/consumer 的准备、计算、收集和完整成本。

## 随机投影实验的数值说明

固定随机 FP32 线性权重，hidden dimension=256；按每个 head 的 `max(abs(Q))/448` 得到 scale，Q 量化为 E4M3，scale 合入 FP32 head weight；cache scale 及 FP8 cache 格式不变。这不是完整模型的投影/RoPE。

改变 GEMM 的 M 维会改变舍入。128 请求时 raw Q 最大绝对差约 3.58e-6，少数值跨越 FP8 量化边界，量化 Q 最大差 2.0；32 请求对应 2.15e-6 和 0.5。scale 后权重误差约 1e-10 量级。当前测试最终 top-k 集合一致，**不能宣称投影路径逐位一致**。

完整误差计数和 nRMSE 见 `projection-errors-final.json`。早期 projection+compact 128 请求 graph 310.50→165.96 μs；最终准入使用当前哈希复测。最终 v3 复测：8请求 77.95→81.84 μs，root输入为115.83 μs；32请求 139.74→109.55 μs，root输入为169.88 μs。torch abs/amax/where/div/clamp 创建的临时张量全部包含在 forward 成本中，记录 graph pool allocated delta。

## 必须保留的负结果与旧版本限制

1. `native-full-chain.json` 是早期 **静态 prepare＋原生 full selector** 的实验。它不能处理动态 metadata/page 生命周期，原生 selector 对同分的选集不稳定。**明确禁止用于最终准入和最终稳定性能结论。** 对应 v2 快照仅供审计。
2. 初版 `verify-tp4.json` 的 kernel 检查成功，但活跃 CUDA graph 引用使 NCCL 退出等待，随后终止了本实验 launcher；不能把那次 runner 标成完整成功。v2 通过显式 graph.reset 修复，后续运行退出码 0。
3. 小 batch、root 输入、较短候选行常无收益，保留复制。全矩阵有真实噪声，追加轮的高波动样本不删除；P90/CV/所有样本存于 JSON。
4. native_stable 依赖明确标记的 P3 隔离修复包；portable 独立保留。结构性的请求分组变化需要重新规划与捕获，不能假装仅改 tensor 内容就完成重规划。
5. 本轮是模块级实验，不能外推模型准确率、TTFT、TPOT 或完整服务吞吐。

## 复现命令

统一前缀（四个真实设备）：

```sh
cd /root/dsv41-indexer-resume-20261107
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

最终动态验证：

```sh
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m benchmarks.kernels.dsv41_indexer.verify_tp4 --full-backend native_stable --patched-deep-gemm /root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1 --output results/indexer/20261107-resume/p7/verify-native-allzero-restore.json
```

稳定 native 完整链：

```sh
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m benchmarks.kernels.dsv41_indexer.benchmark_tp4 --requests 1 8 32 --contexts 65536 262144 --verify 1 --ratios 1 --workloads full chain --full-backend native_stable --patched-deep-gemm /root/dsv41-indexer-resume-20261107/runs/deep-gemm-p3-zero-prefetch-v1 --execution eager graph --output results/indexer/20261107-resume/p7/native-stable-full-chain.json
```

compact 最终复测：

```sh
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m benchmarks.kernels.dsv41_indexer.benchmark_tp4 --requests 8 32 --contexts 65536 --verify 1 6 --ratios 1 --workloads compact --execution eager graph --output results/indexer/20261107-resume/p7/compact-final-v3.json
```

所有 JSON 保存具体 `command_args`（独立 verifier/分项入口见命令和源码）、环境信息、源码/vendor 哈希、每次调用各 rank 的最大值、每轮 P50/P90/CV、墙钟、显存和通信字节。完整原始文件清单见 `inventory.json`，代码快照见 `source-*.tar.gz`。

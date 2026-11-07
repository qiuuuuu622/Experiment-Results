# P3：full/source decode 研究结论与验收证据

状态：P3 的实现、边界验证、分项消融和代表形状正反实验已完成。自动选择策略、完整跨维度回归及四卡比较由 P7/P8 接续。本报告不把代表矩阵称为所有维度的笛卡尔积，不包含模型或服务测试。

## 保留实现

- `NativeFullIndexer`：保留 DeepGEMM FP8 paged scoring，作为 full 性能基线。新 Triton/Gluon full scorer 在已测形状普遍较慢，不提升为默认路径。
- `NativeSourceIndexer`：DeepGEMM score + 确定性 token top-512 + block top-2048，保留 source 的两个输出。它不是仅选块而省掉 token top-k。
- `FullIndexer`/`ChunkedIndexer`：本地页表直接读取 FP8、有效长度限制 tile 工作、fused block max 与独立 scan 两种实现、共享 score arena。作为完整实现/消融/低内存分块路径保留；性能结果不能因其慢而隐藏。
- `full_score.py` 的物理 page ID 在乘 8448-byte page stride 前转为 int64，避免大于 2 GiB cache 的 byte-offset 溢出。128 请求×131072 index keys 的 2,214,592,512-byte cache 已通过联合正确性与正反测量。
- source 只发布一次，四个 consumer 使用不同 Q/head weights。长上下文保留 compact scoring；短上下文采用 Hybrid source 后继续 full-score/stable-top-k consumer 更好。新增组合在 `benchmark_full.py` 中明确实现、验证和测量，尚未偷偷写入自动 dispatch。

## 输入输出契约与限制

输入为 FP8 Q `[rows,32,128]`、包含 Q scale 的 FP32 head weights `[rows,32]`、vLLM FP8 paged cache、每请求本地物理页表、query→request 映射及每 query 可见 **index-key 数**。一页为 64 个 key：8192 字节 FP8 payload 后接 256 字节 FP32 scales。每个 rank 仍保留全部 32 index heads；无 PCP/DCP 分片。

ratio=2 时输入 key cache 已是对应的 index-key 空间，可见长度必须由调用方提供。本模块不把原始 token 长度自动当成 key 数。输出和 source 候选都在 index-key 逻辑坐标：候选块含 8 个 index positions；ratio 只用于明确原 token span 的对应关系。测试中的 ratio2 logical 256K 对应实际 128K index keys。

输出合法 token 位置唯一，token top-k 为 512 项，不足填 -1。source 选择最多 2048 个块，并沿用上游强制保留最新块的规则。候选尾块按真实长度裁剪；候选只在一次 source+consumer 模块组合内复用。

初始化、JIT、分组规划及通信初始化不在性能循环中；原生 `prepare()` 单独测量。动态请求顺序/长度更新必须重新准备 metadata，Graph 连续回放测试将这一步显式纳入捕获。表中组合成本包括 source、候选发布（compact 路径）和四个 consumer；不只计 GEMM。缓存条件为 warm repeated calls，无 L2 flush。

## 已解决的原生边界问题

安装版本的 DeepGEMM scheduler 会跨过零长度 request，而 producer 曾固定预取 `q_idx+1`。当长请求后夹一个/两个空请求时，后续 query 用错 Q/weights：独立复现最大误差分别 0.05560249 / 4.43944454。

补丁让 producer 找到下一非空调度项再预取。它只应用在隔离目录 `runs/deep-gemm-p3-zero-prefetch-v1`，全局 site-packages 和旧实验均未改。修复后同一复现所有有效分数误差 ≤9.54e-7；随后十组 ratio/生命周期测试通过。

- 补丁：`vllm_dsv41_indexer/patches/deepgemm-sm90-empty-query-prefetch.patch`。
- 显式启用：实验入口 `--deep-gemm-package <隔离目录>`；内部在首次 score 前调用 `_lazy_init(); _import_deep_gemm()._C.init(path)`。
- patched include SHA256：`c3d345d8780cdc6f53d68c84d533a8765321c6bdb4464a861beb7757a3affb96`。
- `_C_extension.abi3.so` SHA256：`0042598bc0770e70bc4ea4b9c3d27d8efc8e395d6f766d41e939c746c2bd3046`。
- h20-4 与 h20-dsk 的上述隔离文件哈希一致。完整 native include 及每文件哈希均随批次保存。
- 未修原生与修复原生分别测量。1×64K score-only 修复增加约 0.26–0.31 μs；完整 source+4 compact consumer 增加约 0.8–1.3%。8×256K 未观察到稳定超过 5% 的补丁开销。

默认模块不会自动改外部 DeepGEMM。P8 若选择 NativeSource/NativeFull 且可能出现空行，应显式启用已验证修复，不能把原始 mixed-zero 路径视为正确。

## 正确性证据

`full-final-v1/verify.json` 在 h20-dsk GPU3 新主机实测通过：ratio1/2 × index-key 长度 1、7、517、4099、65536，共十例；每例有混合 0/1/7/511/513/长行、source token/block 选择、四个不同 consumer、三次 Graph 回放、请求重排和物理页重排。相对修复原生打分最大绝对误差 ≤9.54e-7，使用既定 `rtol=1e-3, atol=1e-4`，未降低到 FP16。固定分数的稳定 top-k/source 选择精确一致，检查唯一性、合法范围和未来位置。

真实 1 GiB score arena：1025 行、规划 max_keys=262144，分为 1024+1，所有子模块共享同一 arena。先用 NaN 填 score 存储，zero-Q 解析参考校验 token/source 结果；Graph 再改为前1024行长度0、末行65，输出仍正确。明确：初始前1024行只有513有效 key，末行有256K；这是实际分配/分块/生命周期验证，不是1025行均满256K的性能测试。

`full-final-v1` 六份性能 JSON 内还各自运行 source→publish→4 consumer 的正确性，对 Triton/Gluon/Hybrid 三种 source 分别检查；所有已保留形状均通过。`composition-v2` 额外检查 full consumer 固定分数稳定选择。FP32 与原生 FP8 的固有差异不冒充本次额外误差，也未以逐元素 FP32 误差为理由改变 FP8 算法。

本地 CPU 合约测试：14 passed、1 GPU test skipped；GPU 行为由上述远端验证覆盖。相关 Python Ruff 检查通过。

## 性能结论

所有正式结果均为20次预热、30个样本、每样本10次调用，正反顺序各一轮；CV>10%的测量额外三轮。总计123个追加轮次，全部保留。`timings.csv` 汇总691条（含追加轮次）分项测量；原始样本、P90、host wall time、显存峰值在 JSON。以下是 Graph P50 μs，按正向/反向列出。

| 请求×实际index keys | 原生 source+4consumer | Hybrid source+compact consumers | 判定 |
|---|---:|---:|---|
| 1×65536 | 151.89 / 151.84 | 71.66 / 71.42 | compact 获益约2.12× |
| 8×262144 | 699.33 / 698.62 | 276.80 / 277.65 | compact 获益约2.52× |
| 128×131072 | 4149.73 / 4202.67 | 1684.98 / 1697.74 | compact 获益约2.47× |
| ratio2，8×131072 | 468.50 / 468.28 | 208.40 / 209.02 | compact 获益约2.24× |

source 本身在 1×64K 从72.76降至25.31μs；8×256K从186.64降至149.14μs；128×128K从864.48降至778.44μs。完整 token top-512 和 source block top-2048 均仍计算。

短上下文需组合选择。独立 `composition-v2` 同一批次结果：

| 请求×index keys | 原生完整链路 | Hybrid+compact | Hybrid+full consumer | 选择 |
|---|---:|---:|---:|---|
| 32×4096 | 96.92 / 97.01 | 147.92 / 147.99 | 67.75 / 67.64 | full consumer |
| 64×16384 | 336.88 / 337.14 | 570.64 / 573.45 | 284.22 / 284.26 | full consumer |
| 1×65536 | 151.93 / 151.85 | 71.49 / 71.42 | 108.47 / 108.31 | compact |
| 8×262144 | 696.09 / 695.86 | 276.95 / 276.69 | 706.97 / 706.49 | compact |

full consumer 组合仍使用确定性 top-k，并包含候选mask；不是仅计裸 score。上述形状支持把短窗口与长窗口分开选择，但不能据此臆测所有请求数都共享同一个精确阈值。Eager 的 Python launch/准备成本明显，不能直接套用 Graph 倍数。

Gluon/Triton full scorer 在本轮形状不胜 DeepGEMM。8×256K Gluon source fused max 为258.71μs，separate max为244.10μs，Hybrid source为149.14μs；不能把融合写成必然更快。完整分项还包括原生/新 score、token top-k、block top-k、native prepare、完整组合。

## 复现与数据归属

- `zero-prefetch-v1/`、`verify-v1/`：旧 h20-4 GPU2 完成并恢复到本地，见 `RECOVERY.md`。
- `full-final-v1/`、`composition-v2/`：新 h20-dsk GPU3，正式完成；`hostname.txt`/`gpus.csv` 记录实际主机及设备。两台主机结果不混成同一个统计样本。
- `composition-v1/`：保留失败。额外组合的报告变量覆盖了二维 mask，触发 IndexError；尚未输出完成性能，修正变量后新建 v2，不覆盖失败记录。
- 每批 `run.sh` 保存完整命令，`source.tgz`/`source.json` 为开跑前快照，`native-include.tgz` 保存实际原生 headers。composition 明确在上一批结束后复用解压路径以复用 JIT cache；两份不可变 archive 分开保存。
- 可执行命令：在 h20-dsk 新工作目录中使用 `.venv/bin/python`，先解压对应 `source.tgz` 到 `run.sh` 指定的 run directory，再以 GPU3 执行该脚本。不要在旧结果路径覆盖重跑；重跑须换 output 目录并重新冻结。
- 原生 Python wrappers 核对当前HEAD；二进制身份单独记录，不声称旧 wheel 由该HEAD构建。未加载完整模型、未运行服务、未测试模型准确率/TTFT/TPOT。

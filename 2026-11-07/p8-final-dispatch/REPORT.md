# P8 既有 FP8 indexer 路径最终回归与调度记录

日期：2026-11-07。范围为函数/模块实验，不加载模型，不含 serve、模型准确率或端到端吞吐结论。

**当前结论：原 P2–P7 已验证路径的最终调度、单卡与 TP4 实际绑定验证已完成；用户新要求的 P5b「直接迁移蚂蚁专用 SM90 keys-major」仍在独立实施，未纳入本报告准入，也不能依据旧 grouped-Gluon 实验排除它。整个目标不在此标记完成。**

## 实现与准入

- `vllm_dsv41_indexer/dispatch.py` 是显式 CPU 规划器：输入精确 query/key/request/ratio/group、硬件、Eager/Graph、候选布局、native 空请求补丁状态；TP4 还区分实际 world size、输入是否仅在 root、投影是否分摊、完整请求组及本地 scorer。选择后通过 `Decision.bind()` 绑定已构造的模块 callable；计时与 capture 内无 host 候选资格探测、JIT 或工作区创建。
- `benchmark_final.py` 核验最终模块与 vendor 依赖、被计时 driver/common/inputs 的快照、实际性能文件和正确性证据。driver 差异保留原始 SHA、当前 SHA 及逐 diff 审阅说明，不伪称所有脚本完全相同。native 身份是具有路径和 SHA 的独立 artifact，patched/unpatched 分环境。
- `admission.json` 已完成聚合：**400 条候选比较准入，204 条拒绝，0 个 bundle 失败**。数量是形状×执行模式×候选比较，不是 400 个不同 kernel。仅精确匹配已测点，不外推交叉阈值，不采用蚂蚁 2M 阈值。
- 原始 FP32 score 的稳定同分语义优先。要求确定性时绝不退回非确定性的 native selector；未确认空请求安全时也不退回已知错误的原始 DeepGEMM paged scorer。
- 准入要求两个 A/B 顺序、20 次 warmup、30×10 个调用样本、完整准备/发布/通信成本、墙钟与显存记录。由 raw 重算 P50/CV；波动项保留三轮追加。收益不足 5% 或不超过实测噪声拒绝。选择时使用所有保留轮次的最坏 P50，避免只挑首轮。
- 共享 consumer 只接受调用者已保证的 `shared_prefix`；TP4 只接受每次调用的最慢 rank 统计和包含输入传输/结果汇聚的记录。旧 P7 非稳定 native 或未逐步 prepare 的静态结果不准入。

## 最终验证

|层级|完成证据|
|---|---|
|CPU 行为|`test_dispatch.py` 17 项通过；覆盖 source/env 过期、单序/伪造 CV、wall 缺样、<5% 与噪声拒绝、追加轮次、严格同分、共享布局、空请求以及 TP4 输入/投影/请求组隔离|
|P2 最终 GPU 正确性|24 个 lifecycle case × 4 个独立 consumer；ratio 1/2、重排、长度/候选变化、空候选、无效尾部与 graph 回放|
|P6 最终 GPU 正确性|360 个 full/source selector case + 6 个 ragged case；固定 logits 与稳定参考精确相同|
|单卡真实 dispatch|`bind-single-final/`：8 个 native 环境独立进程，共 **28** 个策略/角色代表项通过 Eager 与连续 3 次 CUDA Graph；source 比较 token top-k 与 forced-latest 候选，chain 实际执行 source + 4 consumer|
|TP4 真实 dispatch|`bind-tp4-final/`：3 个 native/投影环境，共 **7** 个策略/角色代表项、实际四个 rank；Eager 与连续 3 次 graph 的逻辑 top-k 集合等于复制模式|

绑定验证是对最终选择机制和实际 callable 的代表验证；完整形状的分数/选择回归由所绑定的各阶段最终原始报告提供。没有把代表绑定检查称为所有形状的独立再测。

每个绑定进程核验目标 native package/header/二进制 hash。补充的 `loaded-native.json` 记录两台机器实际 importer 的 Python module、`_C.py` facade 与 `torch.ops.loaded_libraries` 中真正 `_C_extension*.so` 的路径/hash，避免仅核对未加载的 vendored 文件。P3 补丁只切换 JIT include 根；未覆盖全局安装。

## 代表性能与限制

下面是本轮最终原始 Graph P50 的正序示例（μs）；双序、P90、全部 raw、追加轮次、墙钟及峰值保留在 JSON。具体准入依据双序及全部保留轮次，不能只凭下表一个数。

|模块/形状|基线|候选|说明|
|---|---:|---:|---|
|单 consumer，8 请求 × 256K|127.57|35.20|compact WGMMA，含候选发布与融合映射；约 3.62×|
|prefill full，64 Q × 64K|628.61|150.87|现有 Triton keys-major；约 4.17×，**不是新 P5b Ant 专用实现**|
|prefill source，64 Q × 64K|707.60|166.57|包含准备、打分、token 与候选选择|
|prefill consumer，64 Q × 64K|639.00|62.92|row-tiled，request-shared 候选，约 10.16×|
|TP4 full，32 请求 × 256K|398.25|180.95|复制 vs 按请求组分摊；每步 prepare + DeepGEMM + stable top-k + 结果汇聚|
|TP4 chain，32 请求 × 256K|754.51|381.30|输入已经复制，source + 4 consumer 与完整汇聚；约 1.98×|
|TP4 chain，同形状 root 输入|754.51|593.69|还包含输入传输；不能把这个模式与复制输入模式混用|

source 和 consumer 独立选路。短 4K/16K 上 source 可保留 DeepGEMM + 稳定 selector，而后续 consumer 使用 full scoring + candidate mask + 稳定 selector；长上下文改用 compact。`composition-v2` 两个实际分支都进入比较，未简单把整条链一起退回旧实现。

TP4 最终 raw 中只有 13 条比较满足当前保守准入门槛；其余 59 条保留并拒绝自动切换。单行、小 batch、root 输入通信较重、Eager host 噪声等不能用单卡 kernel 收益代替整模块结论。投影实验是固定随机权重；FP8 Q 少数舍入边界可能不同，不声称投影逐比特一致，最终逻辑选择按 P7 原始记录报告。

所有实验是 warm-cache 重复调用，无 L2 flush。峰值通常包含同进程并存的 variant 工作区，不能当成某一个 backend 的独占峰值。单卡 Eager 与主机墙钟会受同主机其他实验 CPU 负载影响；所有样本保留。FP8 Q/K 与 scale 语义未变；与 native 的 FP32 score 差异沿用既定回归门槛，未为严苛 FP32 oracle 强行切换 FP16 MMA。

## 矩阵与证据

`matrix.json` 已由最终 raw 对齐：101 个已完成的形状条目，10 个原配对未进入最终分阶段矩阵，1 个 monolithic-driver 超出 1 GiB score 预算的条目明确排除；后者由 P3 专用分块验证覆盖。没有把未测的笛卡尔积组合标成完成。P5b 另行待办。

- `final-p2-p6/`：GPU0 独立冻结源码、native includes、run.sh、P2 25 形状正反与 P6 29 形状正反，以及本轮正确性结果。
- `bundles/`：每阶段实际 raw、正确性、源文件、native 身份与 driver 审阅引用。
- `admission-single.json` / `admission-tp4.json` / `admission.json`：单卡、TP4、合并准入。
- `bind-single-final/` / `bind-tp4-final/`：真实绑定执行报告、命令、日志及对应运行源码包。
- `FINAL-SHA256.json`：当前工具源码与全部交付 artifact 的 hash 清单。
- 各阶段性能 raw 位于同级 `p3/`、`p7/`、`p8-grouped-final-v1/`、`p8-prefill-final-v1/`，原文件未重写。

本轮 GPU 已释放。无服务启动，无模型加载，无 source push 或 PR。下一步只接收 P5b 的独立正确性与包含准备/selector 的公平 A/B 结果，再判断其精确形状是否进入新一轮准入。

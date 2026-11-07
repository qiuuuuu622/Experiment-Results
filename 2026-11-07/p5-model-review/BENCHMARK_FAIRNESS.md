# P5b 模型结构对齐三方实验：独立 CPU 审阅

最终 CPU 复核对象：`benchmarks/kernels/dsv41_indexer/benchmark_prefill_model.py`，SHA256 **`f422bdd3a5593e031c16e9270f0e93fb3dd343402c4eacd13cfe087987c7ef49`**，与总控确认的 `h20-dsk:/root/dsv41-indexer-resume-20261107/runs/p5-model-20261107-final-v3` 冻结版本一致。完整依赖 SHA256 见 [CPU_CONTRACT_AUDIT.json](CPU_CONTRACT_AUDIT.json)。此审阅没有启动 GPU，也没有生成性能结果。P8 既有路径冻结及准入文件保持不变；Ant 专用 prefill 路径尚未准入。

## 结论

最终计划的 23 个 shape 的**计算维度、ratio、因果可见长度和层角色与所提供模型配置一致**。三方比较使用相同 FP8 Q/K、FP32 scale/head weights、相同逐 query 候选和同一稳定 selector；整条 pipeline 的计时包含 K 准备、Ant 动态 prefix、所有 score chunks、consumer mask、token top-k，以及 source 的 block top-k。它可以作为模型结构对齐的模块实验入口。声明的 8 组 shape × 3 个角色共 24 种组合中，`initial_32k/full` 被 main 明确排除：新增 32K 初始长行是为了研究 ratio-1 source/consumer 候选截断，不是因显存不足排除 ratio-2 full；不能将未测项标为完成。

但这不是完整模型或服务实验。消费者路径是 **全长打分后逐 query mask**，不是只计算候选 key 的 compact kernel；DeepGEMM 对照也使用共同的稳定 selector，不能将结果描述为相对 vLLM 原生完整 indexer 的加速。初始化输入仍是固定随机 FP8 值，没有加载模型权重。

独立审阅提出的 FP32 抽样、全 source 行基本不变量、source 候选差异边界解释、初始 prefill 跨 2048 块截断和配置断言均已进入上述最终版本。P7 独立动态 Graph **v2** 的 5 个 fixture、186 项检查、2976 行稳定 token 选择及 1056 行 source block 选择已复核通过。最终 23 shape × 正反顺序的 6 份报告、1055 轮样本、79 个冻结源码文件及运行身份已全部复核；已有 23 个旧 synthetic shape 仍仅作为 kernel 筛选证据。

**最终研究结论：本轮没有新增 Ant 自动准入。**23 点主测中唯一超过两个顺序 5%+噪声门槛的 source/Q4096/256K，在独立隔离复测中只比最佳 current keys-major 快 `0.684%/0.710%`，稳定收益未复现，因此撤销候选。保留 Ant 显式实验构造器，不新增 opt-in factory，不更改旧 P8 400 条准入；不能把这个结果外推为所有 shape 都慢，也不归因于未经证明的具体温度或 GPU 差异。

**初始 FP32 严格门槛仍没有全部通过，已完成原因分析。**例如 full 7 shape 每个顺序 DeepGEMM/Ant 各有 944 个被抽样元素失败，current keys-major 为 943；最大 FP32 绝对误差约 `0.002155`，Ant 与 DeepGEMM 最大差仅 `2.861e-6`。独立 FP64/FP32/D128/D32 分解表明，记录输入上的主要共有误差来自 D128 FP8 tensor-core dot；用该 dot 重构得分后，Ant 残差≤`9.54e-7`、native≤`1.43e-6`。D32 减小误差但仍非精确，top-k 也不保证单调改善。没有改变生产 kernel 或放宽容差。此处是有据可查的实验研究结论，不是“FP32 严格全通过”或模型准确率结论。

证据分别见 [完整原始记录复核](PERFORMANCE_RAW_REVIEW.json)、[隔离复测复核](ISOLATED_RAW_REVIEW.json)、[数值分解复核](NUMERICAL_RAW_REVIEW.json)。forward 仅统计 Ant-vs-DeepGEMM 一次，token top-k 集合变化 **6 行**（full 1/source 2/consumer 3），source 候选集合变化 **5 行**；所有变化已由实际分数或同分边界解释，没有把其他 backend 或反序结果重复计入。

## 已核验的语义和公平性

| 项目 | 源码检查与结论 |
|---|---|
| 真实维度 | 配置 `index_n_heads=32, index_head_dim=128, index_topk=512, candidate_topk_blocks=2048, candidate_block_size=8`。indexer heads 为复制形式，未误用主注意力 TP 后的头数。CPU 审计逐项断言配置值。 |
| 层角色 | ratio-2 full 对应层 2/8/14；ratio-1 source 为层 20；ratio-1 consumer 为 24/28/32/36。CPU 审计核对 `compress_ratios` 和 `index_source_layer_ids`。 |
| 因果长度 | `model_inputs()` 使用 `(final_context - query_len + j + 1) // ratio`。与 `mla_indexer.py` 的 `global_ctx // COMPRESS_RATIO` 相符；ratio 2 初始第一行可见长度为 0，不能 ceil，也不能使用最终上下文长度给所有行。 |
| 多请求 | ragged case 为 Q=[512,128]、final context=[65535,16384]，request IDs 显式重写为各自 query 数，非同长 verify 分组。每请求独立物理页表；score 按请求切片，位置输出为请求内 index-key 坐标。 |
| 候选布局 | source 的输出为 `[sum(Q),2048]`；三个 consumer scorer 使用同一份 source 输出，按 chunk 的 query 区间传入 `apply_candidate_mask()`，没有将候选错误广播为每请求一个集合。source 使用独立固定随机 Q/head weights，避免与 consumer 假设同一层投影；逐行有效块数/候选位置数及相邻行集合变化记录在 raw。 |
| source 双输出 | `source=True` 保留 token top-512，并计算 block max、强制最近块后再选 block top-2048。没有用块选择替代 token 选择。最新块规则与 `candidate_blocks.py` 一致。 |
| 共同 selector | 三条路径固定 `selector='ant'`、`native_gather=False`；token 和 source block selector 共用同一实现，三方速度差不会混入原生 selector 不同开销。 |
| 准备成本 | `ModelPath.forward()` 每次 `prepare()`，只为每请求准备一次 K；Ant 额外每次 `metadata()`，之后遍历所有 chunk 的 score/select。consumer 的全长逐 query mask 在各自 pipeline 内。 |
| 候选生成成本 | consumer case 的 source 候选生成发生在计时外，适合比较单个 consumer；source role 单独包含自己的候选生成。不能把二者相加后称为实测“一次 source+四层 consumer”组合成本。 |
| score-only | 明确排除 K 准备、prefix、mask、selector。它只说明预备好的 scorer 核心速度，完整优化结论应使用 pipeline。 |
| 规划 | descriptor 构造不在普通 pipeline 中；Ant Eager 另计 descriptor-only 和 descriptor+pipeline。所有三方构造记录独立 wall time、额外峰值显存。固定内存、请求分组及 chunk 布局是复用计划的前提。动态 prefix 已进入普通 pipeline。 |
| 分块 | `score_budget=1<<30`；每次 forward 执行所有 chunk，每请求 K 不随 chunk 重复聚集。CPU 审计所有 23 个计划 shape，每个 score arena 均不超过 1 GiB。1 GiB 是单个 score arena 上限，不是三条实现及辅助缓冲总显存上限。 |
| 采样 | 共同 `measure()` 为 20 次预热、30 个样本、每样本 10 次；Eager/Graph 均有，`--reverse` 反转同组路径次序；CV>10% 时 `sampled()` 保留另外三轮。实际双向 raw 仍需 owner 验收。 |
| 缓存与峰值 | 明确 warm repeated/no flush，显存峰值为三条实现同时驻留；不代表单独服务运行的峰值。无效尾部写入差异保留各实现真实行为，不人为让 DeepGEMM 清理本来不读取的尾部。 |

## 必须保留的边界说明

1. **短上下文 bypass 是 batch 条件。**模型源码为 `max_seq_len // ratio <= 512 and not is_current_stream_capturing()`。不是每个 query 的 visible≤512 就跳过。当前 initial2K/initial8K 的 batch 最大长度超过阈值，因此其中早期短行参与打分符合模型执行语义。当前 driver 通过 assert 排除了整个短 batch，未测其性能；报告应明确排除，并保留 ratio1 512/513、ratio2 1024/1025/1026 的边界审计。capture 时该模型提前返回分支关闭，不能将 Eager bypass 推广成 Graph bypass。
2. **候选容量只是上限。**候选有效数为 `min(ceil(visible/8),2048)`；有尾块时有效 key 数不是固定 16384。initial2K、initial8K 都没有触及 2048 块截断；steady16K 也没有超过容量。initial32K 后半部和 steady64K/256K/1M 才有真正过滤。逐 query 布局不等于各 query 的候选集合一定不同。
3. **初始大 Q 与候选截断已纳入最终计划。**CPU 审计结果 `initial_prefill_crosses_candidate_cap=true`。新增 initial32K/source 和 initial32K/consumer 的 32768 行中，恰有 16384 行超出 2048 块容量；每次 forward 执行四个 1 GiB score chunks，正确性检查每个 chunk 的首/中/尾，不能只计最后一个 chunk 或把源码覆盖称为已跑完性能。
4. **满路径对照与原生 baseline 要区分。**这里是原生 DeepGEMM score + 共同 K prepare + 共同稳定 selector。它适合确定哪一个 scorer 更快。若要宣称整体相对 HEAD 原生路径收益，还需要独立原生 pipeline 对照，且不得把原生不稳定 selector 的差异混入三方 scorer 选择。
5. **分块不等于预算内峰值。**steady256K 的 source/consumer 矩形 logits 原始大小为 4 GiB，分成四个 1 GiB chunk；ratio2 full 为 2 GiB，分两块。验证时参考和被测 logits、boolean masks 等会额外占显存，需分别记录 correctness 峰值和 benchmark 峰值。
6. **ragged 模块包装与 HEAD 原生 packed 包装不同。**本实验把每请求 K 准备为 `[requests,max_keys_padded,128]`，各请求单独 score，行起点均为 0，第二个短请求仍使用统一最大 width。HEAD `sparse_attn_indexer.py` 按 chunk 的 `local_cu_seq_lens` 聚集 packed K，向 DeepGEMM 传 `cu_seqlen_ks/ke`，并可通过 `skip_kv_gather` 复用 continuation chunk。两种包装的逻辑可见 key 和结果坐标等价，但 gather 工作量、padding、调用数、工作区组织可能不同。此三方使用相同实验包装，因此 scorer 对比公平；ragged 数值不能直接当 HEAD 完整服务性能。

## 最终代码复核与 GPU 验收边界

| 编号 | 最终 CPU 复核 | GPU/记录验收条件 |
|---|---|---|
| R1 | **已关闭。**独立验证器 v2 先将所有准备 K/scales、prefix、输出 poison，先保存所有 Graph 输出，再执行 Eager/oracle，避免 Eager 预热掩盖 Graph 漏准备。 | [DYNAMIC_RAW_REVIEW.json](DYNAMIC_RAW_REVIEW.json) 独立重算 186 项/2976 行 token/1056 行 source 检查；动态候选、长度、页表、等长请求组重排、全零恢复、零 Q 同分及 16384 候选边界通过。v1 保留为历史，不能作为 prepare 捕获完整性证据。 |
| R2 | 已增加每 chunk 首/中/尾行 FP32 oracle；≤3072 个均匀 key 加双方 top-512 位置，FP8 K 转 FP32，单独比较两者；TF32 关闭。 | **已分析但非严格全过。**11 条完整行数值诊断记录（含补充边界）核对 FP64、FP32、D128 和 D32；主要共有 FP8 dot 误差解释如上。保留所有 strict failures，没有重新标成通过。 |
| R3 | 已增加所有 source 行数量/唯一性/范围/latest 断言；所有相对 native 的变化行都记录候选重合率与实际 block-max 边界误差。固定 logits oracle 仍是首/中/尾抽样，字段明确。 | 逐份 raw 验证 source 双输出通过、所有变化能由实际分数误差解释。精确同分由共同稳定 selector 规定较小位置优先，不要求不同算子 FP32 累加逐位相同。 |
| R4 | initial32K 的 source/consumer 进入最终计划，CPU 审计 4 chunks、16384 条截断行。 | 两个角色均有正反 Eager/Graph 和对应正确性 raw；full 角色仍显式范围排除。 |
| R5 | main 已断言 H/D/topk/候选容量/块大小、source layer、完整 index source layer list 和 ratio。 | raw 的 config hash 必须匹配冻结配置；本结论不自动推广到其他模型配置。 |
| R6 | main 要求 `--identity`，绑定 identity 文件路径及 hash；冻结运行目录由 owner 管理。 | **已关闭。**最终 6 份 raw 绑定同一 final-v3 driver/config/identity；79 个源码成员逐文件匹配 archive；native/source/version before-after 相同，仅阶段说明不同。隔离复测也核对相同身份和原始样本。 |

P5b 的上述模型结构对齐研究项证据已收齐：输入语义、公平性、动态正确性、数值偏差来源和收益不稳定区间都有实现及原始记录。没有自动准入是本轮实测结论；原方案 P5 的逐 query compact consumer（P5c）是另一条独立实现，仍单独推进，不能把 P5b 研究完成等同整体 P5 或总目标完成。

动态图验证使用 `benchmark_prefill_model.py=f422bdd3…`、验证器 `fd406778241d6d9e53f58266d1e4cab9ffc2e4812f09ebe6378f8b8c246ad0a4`、Ant core `957557aa…`，均与冻结 archive 成员 hash 对应。最大分数差为 `1.430511474609375e-6`。before/after 的 source/native/version 相同；identity 文件只在明确的 before/after 阶段描述字段不同。消费者动态 fixture 有 16/16 行、cutoff fixture 有 12/16 行的候选集合实际变化，避免“更新”未触发候选变化。此结果是 H20 单卡模块正确性，不能称为 TP4 或模型准确率测试。

汇总脚本也已复核修正：从 raw 重算 median/CV；按两个 A/B 顺序分别比较全部轮次中最小 baseline 与最大 Ant，并加入噪声保护，两个顺序均达到 5% 才标记稳定收益。表格中“各轮最大 P50 的比值”仍仅是延迟摘要，不单独构成稳定收益或 dispatch 准入依据。

## CPU 审计复现

```bash
.venv/bin/python results/indexer/20261107-resume/p5-model-review/audit_shapes.py
```

输出为 23 个计划 shape 的配置/长度/分块审计通过，另列出 `initial_32k/full` 范围排除；GPU 执行明确标为未执行。原始配置、最终 driver、prefill/Ant 实现及采样工具的 SHA256 都包含在 JSON 内。CPU 审计脚本和最终 driver 均通过 `ruff check`。如后续修改 driver，需重新绑定其版本，不得把本次审阅自动用于新源码。

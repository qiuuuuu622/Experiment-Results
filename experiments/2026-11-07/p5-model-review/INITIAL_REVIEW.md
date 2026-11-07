# 模型结构对齐 driver 初审记录（历史，不是最终验收）

这是独立 CPU 审阅期间、owner 仍在编辑 driver 时发现问题的记录。初版 CASES 只有七组、三个角色共 21 个计划点。该初稿未冻结完整源码，本文不为它补造源码 SHA 或 GPU Result。最终可复现审计、最终源码 hash、23 点计划及已修复结论见 [BENCHMARK_FAIRNESS.md](BENCHMARK_FAIRNESS.md) 和 [CPU_CONTRACT_AUDIT.json](CPU_CONTRACT_AUDIT.json)。

初审确认 ratio-2 的 `(prefix+j+1)//2`、32 个 index heads、D128、source token512+block2048、逐 query `[Q,2048]` candidate mask 和三方共同稳定 selector 正确。全 pipeline 计入每请求 K 准备、Ant prefix、全部分块打分及选择；consumer 对照为 full score+mask，不能称为 compact score。

当时提出的问题及 owner 随后的处理：

| 初审观察 | 最终版本处理 |
|---|---|
| 只有 DeepGEMM 参照，没有模型形状 FP32 诊断。 | 每 chunk 首/中/尾行抽取均匀位置加双方 top-512 位置，以 FP32 score 诊断两者；记录严格门槛失败数。 |
| source 仅首/中/尾行 oracle，未解释其余候选变化。 | 所有行检查有效数、唯一性、范围、latest；所有变化行报告与原生的重合率和 block-max 边界误差。 |
| initial 最大 8K，不会跨过 2048 个候选块容量。 | 增加 initial32K/source 和 consumer，计划共 23 点；full 角色明确范围排除。 |
| 配置只有 hash，未断言关键字段。 | 增加 H/D/token topk/候选容量/块大小/source layer/ratio 等断言。 |
| source 与 consumer 共用 Q，只滚动 weights。 | source 改用独立固定随机 Q/head weights；仍共享同一个 FP8 cache，符合本轮纯函数范围。 |
| 规划和常驻显存解释不足。 | 三方分别记录构造 wall time/额外峰值；Ant 描述符规划和动态描述符整链单列。 |
| 新 ModelPath 动态候选 Graph 未验证。 | 总控交由 P7 独立验证，最终 CPU 审阅不冒充已跑 GPU。 |
| identity 只绑定少数文件风险。 | driver 增加显式 identity 参数；最终冻结包和实际导入二进制 before/after 由 P5 owner 归档。 |

短上下文的核对也在初审时完成：模型 bypass 使用 **batch 最大上下文**并且要求当前不在 Graph capture；不是逐行 visible≤512 就跳过。初始长 prefill 中早期短行继续打分符合模型语义。

初审未启动任何 GPU，没有修改 P5 owner 代码；问题通过总控和 owner 消息协调。本文仅保留审阅过程，最终范围和状态应以正式报告为准。

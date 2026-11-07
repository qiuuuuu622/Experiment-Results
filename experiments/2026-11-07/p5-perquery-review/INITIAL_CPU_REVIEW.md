# P5c 逐 query compact prefill：最终计时入口 CPU 审阅

结论：下列冻结版本没有发现阻止最终正确性和性能实验的 CPU 层问题，已向总控放行。这里仅审阅输入、调用链与成本口径；最终 GPU 原始记录尚待复核，不是 dispatch 准入。

| 文件 | 最终 SHA256 |
|---|---|
| `benchmark_prefill_perquery.py` | `40f50488bfa7213ae3354340c08786e954f7403f2c8d81ebfcd4a2fa8feb96d4` |
| `verify_prefill_perquery.py` | `d8e9efb85e3f9c3890baf69bfacbe07b6cb90099f5733a49fa259579174858ce` |
| `perquery_prefill.py` | `77a7ffbf34ebe4495e560612c9b6943b17214b5fd01a9cdfe613d5887dbe9b1c` |

## 核验结果

- 直接读取和先 gather 的两条 compact 路径都接受真实 `[Q,2048]` 候选、逐行 request ID 和 visible length，没有用每请求共享候选代替。source 采用独立固定随机 Q/weights，候选由真实 source block 选择生成。
- 最终四路径为 DeepGEMM full+perquery mask、current keys-major full+perquery mask、compact direct、compact gathered WGMMA。所有路径使用同一稳定 selector 和相同候选。已有原生 DeepGEMM prefill 被保留为主要数值/性能对照，收益必须相对同次实验最佳 full baseline，不能只选择较慢的 current baseline。
- gathered 后端明确是 **自定义 batched WGMMA**，没有冒称 DeepGEMM 或原版 Ant keys-major。P5c 新文件与 P5b Ant 文件独立，不变更旧 P8 runtime whitelist。
- 完整 compact pipeline 每次执行 publication/规范化，随后执行全部 chunk 的 gather（若有）、score、融合逻辑位置映射/top-k。两条 full pipeline 每次包含完整 K prepare、score、逐 query mask、top-k。
- compact score arena≤1 GiB；gather staging 另有 256 MiB 上限；gathered 的 chunk 行数同时受两者限制。大 Q 不会为所有 query 同时常驻 `[Q,16384,128]` 的候选 K。source/candidate state/descriptor 等是额外空间，不能把两个 arena 上限称为总峰值上限。
- source 构造和候选生成在单 consumer 计时外，source 自身可以另计；这是单 consumer 实验，不是 source+四层 consumer 的实测组合，也不代表跨 consumer 自动复用 publication。
- 分项 score/top-k 对每个 chunk 先准备对应数据；不会把最后一个 chunk 的 staging 用于前面 query。各 chunk 分项之和明示为推导值，完整连续 pipeline 才是最终延迟。
- 正反顺序、Eager/Graph、20 次预热、30×10 个计时调用及 CV>10% 的三轮追加全部沿用共同采样入口；warm cache/no L2 flush，峰值包含全部对照和独立 source 常驻。
- `validate()` 用有界的 32-query full logits 参考核对所有有效候选分数；compact direct/gathered 的全部分数和稳定逻辑选择逐行验证，两个 full baseline 也核对自身真实 mask 后 logits 的稳定选择。所有模型形状沿用 P5b 的 ratio-1 consumer 语义，默认八个计划点包括 initial32K 候选截断。
- `--validate-only` 清空计时 phase，不跳过 source、两条 full baseline 或 compact 的实际验证，可在同冻结入口先验证再正式计时。
- 独立验证器的六组 fixture 覆盖 source 唯一候选与外部去重、短尾、空集合、请求/页表重排、长度变化、全零与恢复。scale 初始乘 1.3，动态再乘 1.0003 后恢复，FP8 payload 不变；避免只有 2 的幂 scale 时隐藏乘法问题。Graph 验证先 poison 准备区，再全部 replay/clone 输出，最后才 oracle/Eager。

## 结果使用限制

FP32 严格失败数必须保留并分析，不能因为两条 compact 分数逐位相同或与 native 容差一致，就宣称数学 FP32 全通过。P5b 已发现共享 FP8 dot 的数值误差，P5c 仍需按自己的最终 raw 核对。

初始短上下文与长上下文都必须报告，不预设 compact 总是更快。调度收益需同一次、同 shape、同 execution 的正反完整 pipeline 对照，检查噪声和各轮；四张 GPU 同时跑独立实验不是 TP4 并行延迟。

CPU 审阅已放行最终 GPU 实验；正式结果和自动选择仍待 source-bound raw、原生身份和 before/after 源码一致性验收。

# P5b 三方 benchmark 公平性 CPU 审阅

审阅日期：2026-11-07。固定被审源码 SHA 见 `BENCHMARK_REVIEW_SOURCE.json`。这是计时契约与代码审阅，不是新的 GPU 实验结果；未修改 P5 owner 文件。

## 已确认的公平性

1. 三方复用同一个 `IndexerInputs` 与候选输入，FP8 Q/K、FP32 head weights、K scale 和逐 query 长度相同。`ratio` 记录实际 index-key 数，不把原 token 数直接当 key 数。
2. 三方都使用 `selector="ant"` 和 `native_gather=False`。DeepGEMM 对照是共同稳定 selector 的完整 FP8 模块，不混入旧 native selector 的额外延迟来夸大新的 scorer 收益。
3. 所有 `*_pipeline` 调用都含本次 key gather/候选准备；Ant adapter 的 `forward()` 还显式包含动态 visible-prefix metadata，再逐 chunk 打分与选 token/source 候选。请求的 key 每次调用只准备一次，多个 query chunk 复用。
4. `*_score_only` 只执行 `score_chunk()`；Ant prefix 在计时前初始化，因此它是“准备好 key 与 prefix 后的打分”。不能将其与完整模块成本混用。DeepGEMM score 返回缓冲的内部 allocation 保留，另外两方使用显式工作区，这一点记录在结果中。
5. TensorDescriptor 与 prefix buffer 构造在固定 storage 的规划期进行；`ant_dynamic_descriptors_pipeline` 与 `ant_descriptor_plan_only` 另在 Eager 测量。Graph 对照只涵盖固定 storage/固定分块，不声称每步重建 descriptor 也能保持相同成本。
6. 每项使用共同 `measure()`：20 warmup、30 样本×10 调用，正反顺序、Eager/Graph，CV>10% 追加三轮；GPU 与 host wall、全部样本和峰值都保留。结果明确 warm repeated calls、无 L2 flush，峰值含同进程所有 variant 驻留。
7. 小/大/混合三套矩阵含相同名义 score 数而 Q/K 比例不同的形状，以及 source、consumer、多请求、ratio2、short-prefix 与 reversed-visibility，适合重新测交叉点，不能直接沿用蚂蚁的 2M 阈值。

## 已反馈 owner 的边界

- 独立 verifier 的空行 overlap 公式会把合法空输出记成 0%；应空目标且输出空时记为 1.0。合法项计数断言已经能验证空输出，问题是报告指标。
- verifier 最大 keys=4099，source 块数尚小于 2048，不覆盖真正的 source 候选 cutoff。建议在性能 source case（64Q×64K）计时前，对每个 backend 的实际 score 调用 `source_reference`，精确检查稳定候选与 forced-latest 规则。
- 动态 Graph 长度翻转步骤目前重点检查 score 与不越界；建议追加该步的稳定 token top-k 和 source 候选精确检查，避免只靠初始静态输出背书。
- Ant 专用 kernel 会给整个不可见 score 区写 `-inf`；DeepGEMM `clean_logits=False` 只保证有效区域。二者外部 top-k 合同相同，但 Ant 多写尾部是其实际实现成本，尤其影响 short-prefix。应披露而不是人为给 DeepGEMM 加清零；将来删除无用 fill 是单独实现/验证项。
- short-prefix 的 1024Q×65536 名义尺寸有约 67M score 槽位，真正 `sum(lengths)=524800`，约 0.5M。应同时记录有效 score 总数/可见 tile 数与输出字节数，避免将存储大小误读成有效计算量。
- driver 自带 3-file hash 只是一部分身份；正式冻结还必须覆盖 common/inputs/verify_prefill、native/vendor 依赖与实际加载的 binary/header 身份，并核对运行前后不变。已有 P8 的单卡/TP4 路径身份不能自动充当新 Ant scorer 的身份。

这些意见已发送总控与 P5 owner。核心三方计时闭包没有发现把不同 selector 或遗漏 key preparation 混在一起的偏置；上述正确性边界和冻结证据需由最终运行记录确认，新的 Ant 路径在确认之前不进入 dispatch。

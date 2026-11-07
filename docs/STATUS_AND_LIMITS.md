# 完成状态和验收边界

以当前方案书和最终报告为准，P0–P8 的模块研究已完成，包括后来补充的 P5b 专用 Ant 迁移和 P5c 逐query compact。完成表示每项有实现结论、相应验证与记录，不表示每条新路径更快，也不表示所有笛卡尔积形状已测试。

|阶段|已交付|必须保留的限制|
|---|---|---|
|P0/P1|环境、来源、统一输入／位置契约、FP32 参考、工作区|预装二进制 build commit 未知，不能把 wheel 名当基线源码构建证明|
|P2|候选发布一次、compact FP8 打分、融合映射、动态正确性|短窗口可能慢，候选不跨 step 隐式复用|
|P3|full/source、双输出、1 GiB arena、源到四consumer链、空行补丁|新 full scorer 没有普遍获益；native_stable 显式依赖隔离补丁|
|P4|grouped full/source/consumer，规则和 ragged verify|共享候选前提、host 资格规划成本不能省略|
|P5b|专用 Ant 完整调度的 FP8 适配、23 模型形状直接 A/B|当前 D128 没有稳定新增胜区；D32 只做数值诊断|
|P5c|逐query `[Q,2048]` direct/gathered、八点四方对照、paired 补证|单consumer计时不含source；新增四卡证据只验证正确性|
|P6|固定 logits 稳定 top-k、长行/cluster/tie/ragged|确定性选集不保证列表有序；大 batch cluster 可能更慢|
|P7|真实 TP4 复制/行分摊、输入传输与输出汇聚、随机投影|不是 PCP/DCP；随机投影不是模型投影；小 batch 可亏损|
|P8|精确形状、依赖绑定、收益与噪声准入、实际单卡/TP4执行|旧 400 比较准入＋204 拒绝；P5c 独立 16 合同，不能外推未测形状|

## 正确性分三个层次

1. **离散结构**：候选规范化、逻辑位置映射、固定 logits 的稳定选择要求精确；输出唯一、范围合法、无未来泄漏、无效尾部 -1。动态 Graph 测试包括先 poison 再回放，避免先跑 Eager 刷新状态掩盖错误。
2. **优化相对 native**：保留 FP8 Q/K、K scale 和 head weights，检查 score 误差、nRMSE、top-k 集合与边界。P5c 正序模型矩阵已测 top-k 100% 重合；P5b 仍有少数边界差异，报告不隐藏。
3. **相对 FP32/FP64 oracle**：初始 `rtol=1e-3, atol=1e-4` 并未全部通过。P5b Ant/native 各有 3,327 个抽样元素失败，最大约 0.002155；P5c 正序模型矩阵优化 4,120/native 4,121 个失败元素，最大约 0.001664。native 也失败不构成自动放宽门槛；数值诊断、误差及选集影响独立保留。这不是模型精度验收。

## 四卡证据不要混用

P7 有真实四进程性能实验，按每次迭代最慢 rank 汇总，并计入行整理、必要输入传输和结果汇聚。P5c 后补四卡验证有 1,008 项比较、86,016 次逻辑行比较和独立页表，但没有新的 TP4 性能测量。四张卡各跑一个独立形状属于并行实验调度，不是 TP4 延迟。

## 历史更正与失败记录

- 早期 `backend="gluon"` 调用 grouped scorer，不是专用 Ant prefill；该负结果不能排除专用实现。
- 早期 request-shared consumer 不代表模型逐query候选；P5c 独立补齐。
- P7 静态 prepare／非稳定 selector 早期结果不用于最终稳定准入。
- 部分旧 P5c 正反候选 raw hash 不同，原始无序 tensor 未保存，不能事后重建证明。paired 补充使用同对象和 canonical 核验，独立保存。
- 旧方案书、失败 runner、调试日志原样归档，其中“待完成”等状态对应当时，不覆盖当前完成结论。

这次分支发布只做导出、整理、CPU/静态/完整性检查。没有重跑 GPU，没有修改主线程优化代码，没有推送优化源码仓库，没有部署服务。

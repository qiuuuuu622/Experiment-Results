# P8 独立CPU审阅阶段记录

本记录是独立审阅，不是P8最终验收；没有增加GPU实验。

## 已证实并复核修复

1. 候选选择初版只比较初始P50，忽略已通过准入的追加轮最坏P50。构造A初始75/追加90、B稳定80，旧版选A；修复后选B。仅合成CPU输入，不冒充GPU性能数据。
2. 准入初版信任CV字段，未从raw samples重算；构造1/99各15个样本、median50而伪报CV0，旧版接受2倍加速。修复后拒绝“CV is inconsistent with raw samples”。

初始证据`dispatch-review.json`、修复后独立复核`dispatch-review-fixed.json`保留各自dispatch源码哈希。

## 源码依赖核验

`source-dependency-review.json`用P4/P5真正的运行前逐文件manifest与当前模块比较，不替换旧source.tgz总哈希。当前参与P4/P5的module/vendor执行依赖无差异；排除dispatch.py/row_shard.py是因为这两条独立driver未导入它们。旧冻结整体差异包括后续P3/P7 driver、调度/测试工具、测试缓存；它们仍在真实原始快照中保留。

P4/P5自身driver的变化仅GPU元数据修正及排除Gluon性能候选，`benchmark-only.patch`已显式保存。不能将这一结果泛化为所有benchmark都是“非执行报告工具”：P3的source+consumer组合closure就在benchmark_full.py里，被计时的组合算法需要保留其测量时driver来源与实现哈希。已向负责agent建议按family保留driver/common/inputs/helper依赖和已审阅差异，不用一个新总hash掩盖旧执行代码。

## 已转交负责agent的适配检查

- P3 composition-v2只测三种完整组合，新增精确名称是`hybrid_source_full_consumers`。短4K/16K形状主要收益来自这一组合；旧normalize(full)固定读取全量timing名称会KeyError或漏掉该策略，需要显式适配实际存在的测量项。
- native_identity应是可核验的文件内容SHA/完整manifest，不是仅非空路径字符串。P3隔离patched include和P4/P5未修安装包必须保留为不同原生身份；不能伪造统一环境摘要。
- CPU全集在审阅时为16 passed、1 failed：TP4 rank-max测试加aggregation后仍选fallback `tp4:replicated`，未选候选`tp4:shard_request_groups_replicated_input`。已交P8负责人修正；本记录不宣称该问题已解决。修复后全套结果以其最终日志为准。

已完成GPU证据P4/P5的REPORT、raw JSON、原始样本、source/native identities及运行后无变化检查均已提供给P8，不因CPU报告工具变化无目的重跑GPU。

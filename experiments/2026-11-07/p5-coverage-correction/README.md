# P5 迁移覆盖更正与新增直接对照

日期：2026-11-07。最新状态：覆盖结论已更正；专用内核的旧 23 形状及 grid 合成筛选已保存，模型角色对齐的 23 点主矩阵尚未完成。下文保留最初新增任务的说明；最新执行清单见[根方案书](../../H20优化方案书.md)，已完成合成证据见[检查点](../p5-ant-checkpoint/CHECKPOINT.md)。合成筛选不能当作模型主路径结论。

此前`backend="gluon"`调用`grouped_score.grouped_logits`，属于grouped decode/verify内核用于prefill的实验。它不是蚂蚁固定SHA `6e6d5d9e13915adb01db0856b15d6908eaaa2c48`的`sm90_fp8_prefill_keys_major.py`。现有keys_major有K复用，但没有完整覆盖专用调度机制。

因此，历史报告中“Gluon无额外收益”仅限定被测grouped实现与其代表形状；不能用于排除蚂蚁专用实现，也不能代表这条迁移路线已完成直接对照。P5原有数据、日志、源码快照保持历史证据，数值结果不撤销。

待完成P5b：保留FP8 cache、K scale、FP32 head weights与打分语义，适配32行全heads共享驻留、Q/K TMA、三级K缓冲、双head异步WGMMA和visible-tile persistent调度；不复用BF16舍入点或FP4精确累加假设。同输入、同可见长度、同selector比较DeepGEMM/现有keys_major/专用适配版；裸score与准备+metadata+score+topk全模块分开计时，既定正确性和正反采样规范，不照搬2M阈值。无预设收益，验证前不勾选。

此目录的[优化方案书](H20优化方案书.md)保留最初覆盖更正时的历史快照；最新范围、模型角色、逐 query 候选和未完成清单以[根方案书](../../H20优化方案书.md)为准。

[蚂蚁参考源码](https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/python/sglang/kernels/ops/attention/dsv4/sm90_fp8_prefill_keys_major.py)

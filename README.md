# H20 DSK41 FP8 Indexer 实验结果

本仓库保存已完成实验的原始数值、日志、源码快照、许可证与审计证据。范围是 **H20 上 FP8 index cache 的函数／模块级研究**，不是完整模型推理或服务性能报告。

截至本次发布：**P0–P6 已完成各自独立研究阶段；P7、P8 尚未完成。** 后续正在进行的实验不混入本次已完成结果。

|目录|内容与状态|
|---|---|
|[2026-11-06 checkpoint](2026-11-06/h20-fp8-indexer-checkpoint/)|冻结于20261106-153700的历史checkpoint公开副本，包含原始成功及失败记录。P3子目录是未完成阶段的历史证据，不代表P3验收通过。|
|[2026-11-07 audit](2026-11-07/h20-fp8-indexer-audit/REPORT.md)|新主机143项正确性检查及4项pytest通过；核验旧P4/P5/P6快照及采样记录，明确历史证据和适用范围限制。|
|[P3 full/source 完成报告](2026-11-07/p3-full-source/REPORT.md)|直接页表打分、source/四consumer组合、真实1GiB arena与零长度原生预取修复；保留失败尝试和最终正反数据。|
|[P4 最终源码绑定](2026-11-07/p8-grouped-final-v1/REPORT.md)|新跑7形状正反、6形状host重规划成本，31项正确性通过，运行后源码及native哈希未变。|
|[P5 最终源码绑定](2026-11-07/p8-prefill-final-v1/REPORT.md)|新跑32条shape/mode/order记录，27项正确性通过，运行后源码及native哈希未变。|
|[许可证与来源](licenses/)|Apache-2.0许可证、原始来源说明、Ant依赖源码哈希。源码包保留各文件原有许可证声明。|

## 已完成阶段

- **P0/P1**：固定基线与环境，建立输入输出契约、FP32诊断参考及原生对照。
- **P2**：[候选发布与compact consumer](2026-11-06/h20-fp8-indexer-checkpoint/P2报告.md)，含实际四卡复制正确性；不能将复制模式等同于四卡行分摊收益。
- **P3**：[full/source decode](2026-11-07/p3-full-source/REPORT.md)，原生DeepGEMM打分与确定性选择组合、source发布和四consumer，以及隔离的mixed-zero预取修复。
- **P4**：[grouped verify](2026-11-06/h20-fp8-indexer-checkpoint/p4/REPORT.md)，实现同请求key tile共享，报告无收益形状以及候选共享资格重规划成本。
- **P5**：[FP8 prefill](2026-11-06/h20-fp8-indexer-checkpoint/p5/REPORT.md)，32条主实验记录及[独立Gluon补充](2026-11-06/h20-fp8-indexer-checkpoint/p5/GLUON_REPORT.md)8条记录；无收益实现不进入自动路径。
- **P6**：[确定性top-k](2026-11-06/h20-fp8-indexer-checkpoint/p6/REPORT.md)，360项固定logit检查及6项ragged检查、29个性能形状的正反顺序记录。

### 必须保留的解释

基线Python wrapper对应vLLM `29be941d8d5746cd98135bdd5e710eeebd8d7014`，参考Ant `6e6d5d9e13915adb01db0856b15d6908eaaa2c48`。预装vLLM二进制构建commit未知，不声称二进制来自当前HEAD重新编译。

P3/P4/P5/P6性能主要为各自独占GPU的单卡函数实验。P3修复使用单独DeepGEMM include，不修改全局安装；P4/P5本次明确使用原始安装包，原生身份分别记录，不能拼接为一个伪造的统一环境。**P7真正四卡行分摊及通信成本、P8最终dispatch与统一回归尚未完成**。原始结果包含Eager、CUDA Graph、P50/P90、样本、追加轮次及适用条件；不要将Graph数字当作Eager，也不要外推为TTFT、TPOT、吞吐或模型准确率。未启动完整模型或`vllm serve`。

数值回归比较优化和原生路径；FP32参考另行诊断。不能声称全部分数逐位一致。固定相同logits的稳定top-k、候选规范化和逻辑位置映射要求精确；原生不保证稳定同分的回退不能冒充确定性。

P4动态共享资格host重规划成本、P5不同query候选集合/显存峰值的范围、P6长行无收益区间均详见各报告。快照内方案书和报告是历史记录，其当时的勾选状态保留原样；最新阶段状态以本README和11月7日审计为准。

## 公开副本、哈希与复现

原始私有证据未修改。公开导出仅做以下处理，详见[PUBLIC_EXPORT.json](PUBLIC_EXPORT.json)：

1. 四份`environment.json`删去无关的完整机器清单、平台与解释器路径，保留软件版本、二进制哈希、GPU能力/拓扑、CUDA编译信息。
2. `p2-source.tgz`移除22个`__pycache__`/`.pyc`缓存条目；源码字节与所有性能数值保持不变。因此此公开tgz的哈希不同于私有原包。
3. 历史私有清单保留为`ORIGINAL_PRIVATE_*`，仅用于追溯原始checkpoint，不作为处理后副本的校验清单。P3公开副本省略无关hostname/GPU UUID清单，执行主机和设备索引仍在报告中明确。

**公开文件以根目录[SHA256SUMS.json](SHA256SUMS.json)为准**。所有旧报告中引用的私有原包哈希依旧保留，导出差异均有记录。旧P4保存时快照不是每次早期调优的完整源码封存，其他历史版本绑定限制见11月7日审计；不补造运行证明。

```sh
# 在任意已有Python虚拟环境中执行；只校验，不跑GPU实验
/path/to/.venv/bin/python verify_repository.py
```

各阶段报告、归档脚本及审计`run-regression.sh`提供原始命令。它们保留原实验主机别名和目录，复现时需要根据自己的机器修改；本仓库不含认证信息或模型权重。实验源码快照作为证据发布；优化代码仓库未推送。

# H20 DSK41 FP8 Indexer 实验结果

本仓库保存H20上 **FP8 index cache 函数／模块级实验**的原始样本、日志、源码快照、许可证及独立审阅。没有启动完整模型或 `vllm serve`，不能外推为模型准确率、TTFT、TPOT或服务吞吐。

**最新状态：P0–P4、P6、P7以及P8既有路径已归档；P5b专用Ant prefill的模型对齐研究已验收，不新增自动调度。P5整体仍未完成，P5c真实逐query compact consumer正在独立实施，未包含在本次发布中。** 详见[最新H20优化方案书](H20优化方案书.md)。

## 本次P5b结论

[完整报告](2026-11-07/p5-ant-keys-major/REPORT.md)覆盖真实DSV4.1 Flash结构约束下的23个形状、正反测试和独立复测：32个index heads、dim128、top512；ratio2 FULL层2/8/14、ratio1 SOURCE层20、ratio1 CONSUMER层24/28/32/36。逐query可见key数为 `floor((prefix+j+1)/ratio)`，候选为 `[query,2048]`，每块8个index-key位置。

三方采用同一FP8输入、准备和稳定selector，比较DeepGEMM scorer、已有keys-major和Ant专用prefill适配。模型主矩阵包括初始prefill、大块prefill、256K上下文、1M尾chunk和ragged请求；1GiB score arena分块执行全部工作。consumer本次采用 **full score + 逐query候选mask**，是单consumer打分路径对照，不是compact优化或source+四consumer链路。

唯一约6%的主矩阵候选收益在空闲单卡相同合同的独立正反复测中降至约0.7%，因此撤销准入，不改dispatch。原因没有通过控制实验单独定位，不能归咎于某个时钟、温度、设备或分配因素。**这个性能结论仅适用于本轮D128原生累加语义适配版。4×D32只做了数值诊断，没有模型主性能数据；不能据此否定所有Ant累加变体。**

优化对native数值回归通过；初始FP32逐元素strict门槛没有通过。Ant与DeepGEMM在模型主矩阵正序抽样各有3327个失败元素，最大差约0.002155；独立逐head/FP64诊断在代表性行定位主要偏差来自D128 FP8 WGMMA点积累加。固定相同logits的稳定选择要求精确，不把原生比较代替高精度oracle，也不承诺全部score或top-k逐位一致。

|证据|内容|
|---|---|
|[P5b最终报告](2026-11-07/p5-ant-keys-major/REPORT.md)|模型主表、孤立复测、精度解释、适用范围及未完成P5c说明。|
|[23点模型矩阵](2026-11-07/p5-ant-keys-major/model-aligned/MATRIX.json)|角色、Q、原token prefix/context、实际index K、有效score、allocated score与分块预算。|
|[模型原始结果与冻结源码](2026-11-07/p5-ant-keys-major/model-aligned/final-v3/)|46条shape/order记录，P50/P90、CV追加轮、host wall、score-only、规划与峰值；前后79份源码及589份native依赖身份一致。|
|[详细性能表](2026-11-07/p5-ant-keys-major/model-aligned/final-v3/DETAILS.csv)|逐角色/形状/执行模式/实现的完整数值，不能只取最有利一轮。|
|[独立GPU1复测](2026-11-07/p5-ant-keys-major/model-aligned/isolated-source/)|相同source大形状正反、完整原样本及温度/时钟/功耗；支持撤销唯一候选。|
|[动态Graph独立验证v2](2026-11-07/p5-model-shapes/dynamic-verifier-v2/REPORT.md)|186项检查、2976行token、1056行source候选精确参考，先验证Graph再刷新Eager，覆盖候选/长度/物理页/请求重排、全零和恢复。|
|[逐head数值诊断](2026-11-07/p5-model-shapes/numerical-diagnostic/REPORT.md)|FP8 dot、FP32/FP64、scale/head归约、PTX、最差输入与边界补充；保留失败尝试，不将代表行解释泛化为全部输入。|
|[最终独立审阅](2026-11-07/p5-model-review/BENCHMARK_FAIRNESS.md)|计时合同、模型语义、动态验证、原始样本/身份/准入和数值解释审阅。|
|[最终决定](2026-11-07/p5-ant-keys-major/model-aligned/DECISION.json)|当前D128版无新增admitted shape，P5b研究完成，P5c范围单列。|

## 已归档的其他阶段与历史证据

|目录|内容与范围|
|---|---|
|[2026-11-06 checkpoint](2026-11-06/h20-fp8-indexer-checkpoint/)|冻结于20261106-153700，保留成功/失败记录；当时P3子目录未完成，不代表后来的最终结论。|
|[2026-11-07 audit](2026-11-07/h20-fp8-indexer-audit/REPORT.md)|143项正确性及4项pytest，核验旧P4/P5/P6快照与适用范围。|
|[P3 full/source](2026-11-07/p3-full-source/REPORT.md)|直接页表打分、source/四consumer、1GiB arena和独立DeepGEMM零长度预取修复。|
|[P4最终源码绑定](2026-11-07/p8-grouped-final-v1/REPORT.md)|grouped verify、资格重规划、正反计时与正确性。|
|[既有P5实验](2026-11-07/p8-prefill-final-v1/REPORT.md)|历史32条记录；request-shared或grouped scorer范围不能替代新的模型主矩阵。|
|[P6确定性top-k](2026-11-06/h20-fp8-indexer-checkpoint/p6/REPORT.md)|360项固定logit、6项ragged、29形状正反及无收益区间。|
|[P7真实TP4](2026-11-07/p7-tp4-row-sharding/REPORT.md)|四卡复制/按行分摊与完整通信成本，480项/rank动态验证；不是把四个独立单卡当TP4。|
|[P8既有路径](2026-11-07/p8-final-dispatch/REPORT.md)|400条比较准入、204条拒绝、11环境35个实际代表绑定；不是400个不同kernel或400形状全部重绑，P5b未新增准入。|
|[P7/P8独立审阅](2026-11-07/p8-independent-review/p3-binding-review/REPORT.md)|保留失败复现、修复验收、最终artifact引用和实际native身份核验。|
|[P5覆盖更正](2026-11-07/p5-coverage-correction/README.md)|原`gluon`实际调用grouped scorer，不是Ant专用prefill内核；历史负结果不能替代本次直接A/B。|
|[P5b合成筛选检查点](2026-11-07/p5-ant-checkpoint/CHECKPOINT.md)|旧23个synthetic形状和当时状态原样保留；不是当前完成状态。|
|[P5b grid消融](2026-11-07/p5-ant-grid/REPORT.md)|39/78/156/312，四张卡各跑一个独立形状，不是TP4延迟，也不能替代模型主矩阵。|
|[Ant机制审阅](2026-11-07/p5-ant-review/REVIEW.md)|固定上游文件、许可证、FP8数值反例与CPU机制审阅。|
|[许可证及来源](licenses/)|源码包保留原许可证和来源哈希。|

历史快照中的勾选状态按当时记录保留。旧ratio2 consumer是通用API压力形状，不是真实模型consumer角色；旧request-shared候选不能证明真实逐query compact覆盖。P5b完成不能替代正在实施的P5c。

## 来源、公开副本与复现

基线Python wrappers对应vLLM `29be941d8d5746cd98135bdd5e710eeebd8d7014`；参考Ant `6e6d5d9e13915adb01db0856b15d6908eaaa2c48`。预装vLLM二进制build commit未知，没有把wheel名当成当前HEAD重编译证明。各阶段native环境分别有身份记录，不能拼成一个伪造统一环境。

本次发布保留所有原始数值和源码计算逻辑，私有原证据不修改。公开脱敏与历史处理详见[PUBLIC_EXPORT.json](PUBLIC_EXPORT.json)：个人Mac工作目录替换为`/workspace/...`，必要时去除无关主机/GPU UUID或个人tar owner元数据；旧包移除解释器缓存的既有记录继续保留。许可证文本不改。最新根方案书替换前的副本保存在[合成发布时方案书](2026-11-07/p5-ant-checkpoint/PLAN_AT_SYNTHETIC_EXPORT.md)，历史映射继续可校验。

阶段内部SHA表示原始私有运行溯源；脱敏后不能直接用它校验公开字节。原始私有校验清单以`ORIGINAL_PRIVATE_*`保留。**公开文件以根[SHA256SUMS.json](SHA256SUMS.json)为准**，[PUBLIC_DEPENDENCIES.json](PUBLIC_DEPENDENCIES.json)保留公开对象、私有原SHA和公开SHA的映射，包括P5b报告的外部动态/数值/grid证据。P5c活动代码和结果没有被混入已完成证据。

```sh
/path/to/.venv/bin/python verify_repository.py
```

该命令只核验公开文件和依赖映射，不运行GPU实验。GPU原始命令、固定源码和环境见[P5b复现命令](2026-11-07/p5-ant-keys-major/model-aligned/RUN_COMMANDS.md)及各阶段报告；路径/主机别名需按复现环境调整。历史`audit.py`保留其私有源仓库相对路径，不冒充可直接从公开仓库运行的独立复现入口。不要覆盖已冻结结果。

优化代码仓库未推送；本仓库的源码快照作为实验依据发布，不包含认证信息或模型权重。

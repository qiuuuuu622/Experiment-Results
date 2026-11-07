# H20 DSK41 FP8 Indexer 实验结果

本仓库保存已完成实验的原始数值、日志、源码快照、许可证与审计证据。范围是 **H20 上 FP8 index cache 的函数／模块级研究**，不是完整模型推理或服务性能报告。

截至本次发布：**P0–P4、P6、P7及P8既有FP8路径研究已完成并归档；P5b 专用 Ant prefill 的旧 23 形状与 grid 合成筛选已保存，但模型角色对齐的 23 点主矩阵尚未完成。整个优化目标仍未完成。合成筛选不能当作模型主路径结论。**

最新执行范围见根目录[H20优化方案书](H20优化方案书.md)。本次只发布已冻结的合成证据，排除正在变化的 `model-aligned/` 与模型主矩阵审阅文件。

**P5覆盖更正：**此前`backend="gluon"`调用grouped decode/verify内核，不是蚂蚁专用`sm90_fp8_prefill_keys_major.py`。其无收益结论不能排除蚂蚁专用路线。见[覆盖更正及未完成清单](2026-11-07/p5-coverage-correction/README.md)。历史快照和原始数据不覆盖。

|目录|内容与状态|
|---|---|
|[2026-11-06 checkpoint](2026-11-06/h20-fp8-indexer-checkpoint/)|冻结于20261106-153700的历史checkpoint公开副本，包含原始成功及失败记录。P3子目录是未完成阶段的历史证据，不代表P3验收通过。|
|[2026-11-07 audit](2026-11-07/h20-fp8-indexer-audit/REPORT.md)|新主机143项正确性检查及4项pytest通过；核验旧P4/P5/P6快照及采样记录，明确历史证据和适用范围限制。|
|[P3 full/source 完成报告](2026-11-07/p3-full-source/REPORT.md)|直接页表打分、source/四consumer组合、真实1GiB arena与零长度原生预取修复；保留失败尝试和最终正反数据。|
|[P4 最终源码绑定](2026-11-07/p8-grouped-final-v1/REPORT.md)|新跑7形状正反、6形状host重规划成本，31项正确性通过，运行后源码及native哈希未变。|
|[P5 最终源码绑定](2026-11-07/p8-prefill-final-v1/REPORT.md)|新跑32条shape/mode/order记录，27项正确性通过，运行后源码及native哈希未变。|
|[P7 四卡行分摊](2026-11-07/p7-tp4-row-sharding/REPORT.md)|真实TP4、每步metadata准备与稳定top-k，四卡480项/rank动态验证；保留复制/分摊、root输入通信、投影和无收益区间。|
|[P8 最终调度与回归](2026-11-07/p8-final-dispatch/REPORT.md)|既有路径400条比较准入、204条拒绝、0 bundle失败；11环境的35个代表真实绑定通过。P5b未纳入。|
|[P7/P8 独立审阅](2026-11-07/p8-independent-review/p3-binding-review/REPORT.md)|保留审阅发现、失败复现及修复后的验收；核验41个最终artifact引用和实际native身份。|
|[P5b 合成筛选检查点](2026-11-07/p5-ant-checkpoint/CHECKPOINT.md)|59 项旧证据、23 形状正反与 4 个独立单卡 grid 形状；2,639 项 CPU 核对通过。主模型矩阵未完成。|
|[P5b grid 消融](2026-11-07/p5-ant-grid/REPORT.md)|39/78/156/312 的完整原始样本与源码；四卡分别跑一个单卡形状，不是 TP4 延迟。|
|[P5b 独立机制审阅](2026-11-07/p5-ant-review/REVIEW.md)|Ant 原始来源、许可证、FP8 数值反例和计时公平性审阅；CPU 审阅不是 GPU 验证。|
|[许可证与来源](licenses/)|Apache-2.0许可证、原始来源说明、Ant依赖源码哈希。源码包保留各文件原有许可证声明。|

## 已完成阶段

- **P0/P1**：固定基线与环境，建立输入输出契约、FP32诊断参考及原生对照。
- **P2**：[候选发布与compact consumer](2026-11-06/h20-fp8-indexer-checkpoint/P2报告.md)，含实际四卡复制正确性；不能将复制模式等同于四卡行分摊收益。
- **P3**：[full/source decode](2026-11-07/p3-full-source/REPORT.md)，原生DeepGEMM打分与确定性选择组合、source发布和四consumer，以及隔离的mixed-zero预取修复。
- **P4**：[grouped verify](2026-11-06/h20-fp8-indexer-checkpoint/p4/REPORT.md)，实现同请求key tile共享，报告无收益形状以及候选共享资格重规划成本。
- **P5已有实现（迁移覆盖未完成）**：[FP8 prefill](2026-11-06/h20-fp8-indexer-checkpoint/p5/REPORT.md)，32条主实验记录及[独立Gluon补充](2026-11-06/h20-fp8-indexer-checkpoint/p5/GLUON_REPORT.md)8条记录；负结果仅限所测grouped实现，不能替代蚂蚁专用内核的直接A/B。
- **P6**：[确定性top-k](2026-11-06/h20-fp8-indexer-checkpoint/p6/REPORT.md)，360项固定logit检查及6项ragged检查、29个性能形状的正反顺序记录。
- **P7**：[真实四卡复制与行分摊](2026-11-07/p7-tp4-row-sharding/REPORT.md)，保持32个index heads及每rank本地页表，测量包含行整理、必要输入传输和结果汇聚；最终文件由[inventory](2026-11-07/p7-tp4-row-sharding/inventory.json)指定，旧静态native和中止运行由[excluded-runs](2026-11-07/p7-tp4-row-sharding/excluded-runs.json)明确排除。
- **P8既有路径**：[最终准入与实际绑定](2026-11-07/p8-final-dispatch/REPORT.md)，400条是形状×执行模式×候选比较，不是400个不同kernel；真实绑定为单卡28个、TP4 7个代表项，不能写成400个形状全部重新绑定。

### P5b 本次合成证据的范围

旧 consumer 候选是随机 `[requests,2048]` 的 `request_shared` 布局，不是模型 source 层产生的逐 query 候选。**ratio2 consumer 仅为通用 API 压力形状，不是真实模型层角色**。实际主矩阵应分别覆盖 ratio2 full 层 2/8/14、ratio1 source 层 20、ratio1 consumer 层 24/28/32/36，并按真实因果长度与候选布局计时。

旧 23 形状和 grid 结果不足以证明 Ant 专用实现对真实模型主路径有收益或无收益。`runtime-identity.json` 是 post-run 采集；历史 verifier 的逐次源码绑定限制保留在检查点。没有因此把 P5b 勾选完成，也没有把新内核自动纳入 P8。

### 必须保留的解释

基线Python wrapper对应vLLM `29be941d8d5746cd98135bdd5e710eeebd8d7014`，参考Ant `6e6d5d9e13915adb01db0856b15d6908eaaa2c48`。预装vLLM二进制构建commit未知，不声称二进制来自当前HEAD重新编译。

P3/P4/P5/P6性能主要为各自独占GPU的单卡函数实验。P3修复使用单独DeepGEMM include，不修改全局安装；P4/P5本次明确使用原始安装包，原生身份分别记录，不能拼接为一个伪造的统一环境。**P7已完成真正四卡行分摊及完整通信成本测量；P8已完成既有路径的最终dispatch与回归，P5b专用Ant prefill仍在独立实施**。原始结果包含Eager、CUDA Graph、P50/P90、样本、追加轮次及适用条件；不要将Graph数字当作Eager，也不要外推为TTFT、TPOT、吞吐或模型准确率。未启动完整模型或`vllm serve`。

数值回归比较优化和原生路径；FP32参考另行诊断。不能声称全部分数逐位一致。固定相同logits的稳定top-k、候选规范化和逻辑位置映射要求精确；原生不保证稳定同分的回退不能冒充确定性。

P4动态共享资格host重规划成本、P5不同query候选集合/显存峰值的范围、P6长行无收益区间均详见各报告。快照内方案书和报告是历史记录，其当时的勾选状态保留原样；最新覆盖状态以本README及[P5覆盖更正](2026-11-07/p5-coverage-correction/README.md)为准。

## 公开副本、哈希与复现

原始私有证据未修改。公开导出仅做以下处理，详见[PUBLIC_EXPORT.json](PUBLIC_EXPORT.json)：

1. 四份`environment.json`删去无关的完整机器清单、平台与解释器路径，保留软件版本、二进制哈希、GPU能力/拓扑、CUDA编译信息。
2. `p2-source.tgz`移除22个`__pycache__`/`.pyc`缓存条目；源码字节与所有性能数值保持不变。因此此公开tgz的哈希不同于私有原包。
3. 历史私有清单保留为`ORIGINAL_PRIVATE_*`，仅用于追溯原始checkpoint，不作为处理后副本的校验清单。P3公开副本省略无关hostname/GPU UUID清单，执行主机和设备索引仍在报告中明确。
4. 本次P7/P8增量导出182个文件。公开副本把P8 manifest/bundle内的个人工作目录替换为`/workspace/vllm_dsv41_flash_opt`；两份绑定源码包同步处理内嵌路径及个人tar owner metadata。数值、布尔值、状态、源码计算逻辑和许可证文本未改；每个变化记录私有原SHA与公开SHA。

[公开依赖映射](PUBLIC_DEPENDENCIES.json)将 P8 的 41 个引用及本次 P5b 的 63 个导出对象对应到仓库内公开文件。P5b 新增 63 个文件含 59 个核验对象、检查点文件和最新根方案书；全部保留原字节。唯一需要恢复的是旧 `capture_identity.py`：撤销后来为模型主矩阵加入的 before/after 输出及新增依赖记录后，其字节数和 SHA 与旧检查点完全一致；owner 已把新 helper 保存到未导出的模型目录后，按明确授权将旧私有路径恢复为原字节；59 项原始证据重新核验一致，过程记录在 PUBLIC_EXPORT。阶段内部的artifact/source SHA仍表示原始私有运行的溯源身份；个人路径替换或tar元数据处理后，不能用它们校验公开副本。请使用根目录公开清单，验证器也核验此依赖映射。所有私有原文件保持不变。

**公开文件以根目录[SHA256SUMS.json](SHA256SUMS.json)为准**。所有旧报告中引用的私有原包哈希依旧保留，导出差异均有记录。旧P4保存时快照不是每次早期调优的完整源码封存，其他历史版本绑定限制见11月7日审计；不补造运行证明。

```sh
# 在任意已有Python虚拟环境中执行；只校验，不跑GPU实验
/path/to/.venv/bin/python verify_repository.py
```

检查点 `audit.py` 保留私有源仓库的原始相对路径，并引用未完成且未导出的模型配置；公开副本的完整性请运行根 `verify_repository.py`，不要把该历史审计脚本误作独立 GPU 复现实验。

各阶段报告、归档脚本及审计`run-regression.sh`提供原始命令。它们保留原实验主机别名和目录，复现时需要根据自己的机器修改；本仓库不含认证信息或模型权重。实验源码快照作为证据发布；优化代码仓库未推送。

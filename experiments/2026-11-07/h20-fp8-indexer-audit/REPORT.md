# P4/P5/P6 证据审计与新主机正确性回归

日期：2026-11-07。主机：h20-4。隔离目录：`/root/dsv41-indexer-resume-20261107`。仅使用物理GPU3，未运行模型、服务或性能扩展矩阵，未触碰其他进程。历史目录`/root/dsv41-indexer-port`保持只读。

结论：P4/P5/P6均有实现、独立正确性与原始性能证据，可勾选**对应独立研究阶段**，但须保留以下适用边界；这不等于四卡P7、最终dispatch/P8或全Goal已完成。当前实现无需修复，GPU3已释放供总控继续实验。

## 版本及证据核验

- 冻结57个模块、kernel头文件、benchmark及test文件，本地与新主机逐文件SHA256一致，回归期间复核无变化。`frozen-source.json`、`source-identity.json`保留核验结果。
- 新回归源码：`regression-source.tgz`，SHA256 `bd32ac0fdd11903168c32e138c2e0c71b39dd6e3421d41e9480446296c8eb069`。
- 历史5份tgz均能读取并计算所有内部文件哈希。P4 final、P5主实验、P5 Gluon的3个已记录归档SHA全部吻合。P6源码manifest与snapshot内全部8个文件哈希吻合；`06_topk.sh`在manifest中写benchmark路径、包内写scripts路径，显式映射后内容一致。
- P4最终核心实现与当前一致；两个benchmark/verify文件只有变量命名/换行格式变更。P5原32条实验快照与后来的Gluon版本明确分开，当前与Gluon快照一致。P6快照代码与当前一致。
- 历史全部JSON可解析、已记录性能运行均completed。递归审计2,896个测量节点（包括初轮及追加轮次，不是独立shape数量），每个均20预热、30样本、10次内部调用，未发现采样数量违例。原始高波动节点仍保留，不删除、不把重试样本当成独立shape。
- `historical-audit.json`记录每个JSON及tgz哈希、状态、shape数量、测量节点、采样检查、CV记录、归档内部文件及当前代码差异。复核工具为`benchmarks/kernels/dsv41_indexer/audit_resume.py`。

证据强度限制：旧P4 final快照本来就是保存时checkpoint，不能声称它封存了每一次早期调优的精确源码；旧P5主快照只打包专属源码，共享依赖为单列哈希；旧性能JSON中的source主要是基线身份，未全部嵌入每个新kernel的运行前完整哈希。P6提供内部manifest而没有更早独立锚定的tgz总哈希。本次不补造历史运行绑定关系，以新回归的冻结快照和未来P8性能重新绑定最终版本。

## 新主机回归结果

|阶段|命令模块|通过数|重点|
|---|---|---:|---|
|P4|verify_grouped|31|group1/2/4/6/8、ratio1/2、517/4099等非整齐长度、ragged、源候选、共享consumer、不同集合fallback、物理页置换、graph长度/候选变化|
|P4|verify_grouped --paired|31|相同覆盖，双query异步WGMMA|
|P5|verify_prefill|27|FP8 DeepGEMM/row-tiled/keys-major/indirect/Gluon，multi-request、ratio1/2、强制小预算分块、非整齐可见长度、固定score稳定top-k、graph变化|
|P6|verify_resume_modules|48+6 ragged|512/2048，streaming/cluster，随机/同分/tie-bin overflow，1×4K、8×64K、32×256K、513×16K，NaN无效尾部、graph、非对齐窗口|
|P4/P5|pytest test_grouped.py test_prefill.py|4|候选共享资格边界、alias请求不同候选、NaN尾部、逻辑坐标和graph更新|

合计143项脚本检查，另4项pytest通过（5.31秒）。没有重新运行完整360项top-k历史矩阵；此为明确选取的迁移回归，历史完整矩阵仍保留。当前原生wrapper身份检查通过；实际预装二进制构建commit仍未知，不声称当前HEAD重编译。

首次top-k full/cluster编译占用约数分钟，不能当作模块调用延迟。没有把本次正确性回归的运行时间当作新性能结果。

## 独立研究阶段勾选依据及边界

|阶段|证据结论|仍应明确的边界|
|---|---|---|
|P4|可以勾选独立研究阶段：旧默认/paired各31项、7形状正反性能/调优及6形状重规划成本，当前两路径62项再次通过|共享consumer仅在预规划共同候选前缀契约下启用；动态通用host资格重规划成本0.4–15ms已记录，不能隐去。不同集合保留逐行。小请求full无收益则保留原生；N6/8原生对照是展开单行，非原生融合。|
|P5|可以勾选独立研究阶段：32主记录+8独立Gluon记录，当前27项与专项test通过|共享候选consumer性能不能外推每query完全不同集合；不同集合可用request alias表达但未做完整性能矩阵。未测4096×256K组合。显存峰值含同进程其他对照缓冲。Gluon无额外收益不进入dispatch；<5%形状保留基线。|
|P6|可以勾选独立研究阶段：旧360+6严格选择检查、29形状×正反性能，当前48+6通过|cluster16未启用；ragged未独立做完整prefill链路性能。128×256K token top-k稳定selector无收益；要求严格同分语义时不能回退不保证确定性的原生。selector独立收益不是完整indexer收益。|

P4/P5/P6都未验证模型准确率、TTFT、TPOT或服务吞吐；四卡复制/分摊与通信属于P7，本轮没有抢做。最终调度及固定条件性能回归由P8负责。无需重复调优已验证无收益路径。

## 复现与文件

新主机运行的完整命令保存在`run-regression.sh`，使用独立`.venv/bin/python`及`CUDA_VISIBLE_DEVICES=3`。脚本会覆盖指定输出，复跑应选择新目录或先归档。

```sh
ssh h20-4
cd /root/dsv41-indexer-resume-20261107
bash run-audit-regression.sh
```

本地历史只读审计：

```sh
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.audit_resume \
  --output results/indexer/20261107-resume/audit/historical-audit.json
```

交付：REPORT.md、historical-audit.json、frozen-source.json、source-identity.json、regression-source.tgz、run-regression.sh、grouped/grouped-paired/prefill/topk的JSON和同名日志、pytest.log。新代码仅audit_resume.py和verify_resume_modules.py；未改动prefill/grouped/native/topk实现。

## CPU独立接口审阅建议（P3/P7，非新增GPU验证）

审阅对象：当前`module.py`、`full_score.py`、`contract.py`、`row_shard.py`及`verify_full.py`。没有修改其拥有文件，下面是接下来的验证建议，不能记为已通过。

1. **行分摊的空rank及非均衡组。** `RowPlan.make`优先保留请求组，单请求会导致其余rank为空；`RowExchange`设计为这些rank不构造kernel但参与collective。P7应验证1请求、3请求、非整齐verify1/2/6/8混合组与split_rows对照，检查填充行不进入结果和所有rank均进入collective。恢复顺序应按`plan.inverse`比较原始query顺序，不只比较整体集合。
2. **root输入传输的证据。** root模式打包scatter融合字节buffer，含FP8 Q/FP32 weights或hidden及int32元数据；应故意毒化非root输入，确认结果只由rank0输入决定。各rank使用不同物理页映射且改变页布局后复放，不在通信fields中夹带物理页ID。记录分组规划/输入打包/传输/结果收集，不能只比较本地compute。
3. **投影语义与临时内存。** `FixedProjection`是固定随机权重的合成线性投影，不是模型完整RoPE/投影。分摊改变GEMM的M维，浮点结果可能随kernel调度产生差异；应分别比较raw Q、量化FP8 Q、包含Qscale的head weights以及最终top-k。`forward`中abs/amax/where/div/clamp会创建临时tensor，因此docstring中的“所有临时存储在计时前分配”表述需修正，或进一步实现显式workspace；当前若完整forward计时并报告Graph内存池即可诚实包含这些成本。
4. **分组与请求生命周期。** Full/Grouped构造时按请求分组；graph期间请求ID可更新，但同组必须仍指同一请求。改变组成员需要重新plan/capture。P3当前生命周期验证用单行source准备更新并正确包含`prepare()`；不能由此推断任意group_rows>1的重排都可直接复放。P7计划所有权与source/四个consumer须一致；请求重新分组需要明确重规划边界。
5. **预算与尾部。** P3的真实1GiB arena测试涵盖跨chunk和长短混合；应保留其原始输出，注明1GiB仅限FP32 score arena，不包含各chunk的source maxima、输出、KV cache。所有selector必须按实际length/block_count消费，无法从NaN尾部推断未写尾部是合法score。
6. **source组合正确性。** source既须输出token512，又须发布唯一候选块；四consumer应各用不同Q/head weights。先比较source固定score选择，再比较consumer逻辑位置和重复/越界/未来位置，避免仅比候选集合而遗漏token top-k。当前`verify_full.py`已实现这些步骤及三次graph生命周期回放；总控需待该阶段本次实跑完成再勾选。

这些审阅建议不改变P4/P5/P6已验证结论；P3/P7最终结果以其负责agent原始日志和总控验收为准。

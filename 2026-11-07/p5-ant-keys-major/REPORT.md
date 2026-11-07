# P5b：Ant 专用 prefill keys-major 的 FP8 适配与实验

**结论：本研究项实现和验证完成；本轮D128原生累加语义适配版未找到经过独立复核、相对当前最快基线稳定超过5%的胜区，不修改自动调度。**

这不是对Ant所有累加变体的否定。4×D32是不同的数值累加变体，本轮仅作数值诊断，没有它的模型主矩阵性能结果；偏离native不等于数学错误，也不能据此断言其无收益。 本报告不能代表P5全部完成：真实逐query compact consumer的gather/GEMM与直接读取对照属于后续P5c。本轮consumer主矩阵明确是full score + 逐query候选mask。

## 实际实现

适配固定Ant commit `6e6d5d9e13915adb01db0856b15d6908eaaa2c48` 的 `sm90_fp8_prefill_keys_major.py`，原文件、Apache-2.0许可证与SHA保存在 `upstream/`。新模块 `vllm_dsv41_indexer/prefill_keys_major_ant.py` 保留其专用prefill结构：32query的全部32 heads驻shared Q；Q/K TMA；K三级缓冲；head双缓冲异步WGMMA；按可见tile prefix的persistent调度。没有用旧grouped decode/verify scorer替代这个实验。

FP8缓存、独立FP32 K scale、已包含Q scale的FP32 head weights保持不变；计算 `sum_h relu(dot(Q_h,K)*K_scale)*weight_h`。去除原FP4-valued路径的BF16中间舍入/BF16 weights及最终BF16转换。每head用D128 FP8 WGMMA对齐native累加路径，双head等待改为wait_group 1，最后wait_group 0；没有降成FP16 MMA。编译PTX见 `edges-chunked.ptx`，共享内存184360字节，确认E4M3 WGMMA和等待组。

早期保留4个D32独立partial dots的版本见 `numerical-v1/`：它对FP32更接近，但对native严格回归失败（maxabs0.001192、180个失败）；native自身对FP32也有184个失败。因此没有将“更近FP32”自动作为native合同通过，也没有笼统称为公式错误。最终D128版保持原生数值兼容；不继承FP4网格下的精确累加承诺。

## 主实验：按真实模型结构选择23个形状

详见 `model-aligned/ARCHITECTURE.md` 与 `MATRIX.json`。真实配置SHA为 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`；32 index heads、dim128、top512；ratio2 FULL对应层2/8/14，ratio1 SOURCE为20层，ratio1 CONSUMER为24/28/32/36层。可见长度逐query取 `floor((prefix+j+1)/ratio)`，候选是 `[query,2048]`、每块8个index-key。

初始prefill覆盖Q2048/8192，source/consumer另加Q32768跨候选截断；steady覆盖Q512/16K、Q2048/64K、Q4096/256K；还有Q256/1M尾块和ragged多请求。最大完整score矩形4GiB，每路径arena限制1GiB并执行全部chunk。单样本进程allocated峰值4.50GiB；nvidia-smi包含allocator保留显存，可更高。

这些是架构约束下的固定随机模块输入，不是部署默认batch或完整模型。TP4不会把index heads改成8；本轮每张GPU独立测试一个角色，不声称是TP4延迟。

三方为DeepGEMM FP8 scorer、现有keys-major、Ant专用适配；共用相同FP8输入、准备和稳定selector。**DeepGEMM列不是原生完整vLLM端到端路径**，而是原生scorer加共同准备/selector。source同时输出token和候选；consumer输入候选由独立seed的source Q/weights生成，同一K cache，计时仅单consumer，不包含source生成或四consumer组合。

完整成本包括K准备、Ant prefix、所有score chunks、consumer mask分配/flags/写入、共同selector；source还含候选生成。另测score-only、共同准备、prefix、mask及每次重新创建descriptor的Eager完整成本。初始化与JIT不进入热采样；逐路径构造墙钟/峰值另报，未将首次所有JIT编译时间分解成独立指标。

每项20预热、30样本×10调用、正反顺序各一次，CV>10%追加3轮；原样本、P90、host wall和峰值均保留。`final-v3/SUMMARY.json`逐项列出两序及追加轮的保守P50/P90/墙钟、CV、score-only、内存、规划成本；`final-v3/TABLE.md`是23点主表。表中max-P50比值仅作描述，准入另要求每个顺序相对当序最快baseline超过5%且超过噪声。

## 模型主表

单位µs；P50采用两序及全部追加轮的最大值。表内比值为描述值，不代表通过最终独立准入。P90/噪声/墙钟/裸score/有效score和分块详见 `model-aligned/final-v3/DETAILS.csv` 与 `SUMMARY.json`。

| Role | Shape | Query rows | Original context | Actual max K | DG P50 µs | Current P50 µs | Ant P50 µs | Ant/best speedup |
|---|---|---|---|---:|---:|---:|---:|---:|
|full|initial_2k|[2048]|[2048]|1024|70.94|66.97|82.22|0.815|
|full|initial_8k|[8192]|[8192]|4096|711.45|729.37|833.48|0.854|
|full|steady_16k|[512]|[16384]|8192|172.26|155.06|170.67|0.909|
|full|steady_64k|[2048]|[65536]|32768|2238.15|2218.17|2259.71|0.982|
|full|steady_256k|[4096]|[262144]|131072|17742.19|17574.06|17818.17|0.986|
|full|tail_1m|[256]|[1048576]|524288|5156.68|4578.96|4556.9|1.005|
|full|ragged|[512, 128]|[65535, 16384]|32767|743.17|629.12|652.7|0.964|
|source|initial_2k|[2048]|[2048]|2048|129.44|125.26|141.16|0.887|
|source|initial_8k|[8192]|[8192]|8192|1356.83|1415.13|1525.95|0.889|
|source|initial_32k|[32768]|[32768]|32768|19604.27|21741.44|20923.87|0.937|
|source|steady_16k|[512]|[16384]|16384|346.5|308.82|316.87|0.975|
|source|steady_64k|[2048]|[65536]|65536|4655.87|4902.54|4679.22|0.995|
|source|steady_256k|[4096]|[262144]|262144|41701.21|40401.54|38128.12|1.06|
|source|tail_1m|[256]|[1048576]|1048576|10578.7|10272.94|9872.35|1.041|
|source|ragged|[512, 128]|[65535, 16384]|65535|1522.73|1345.45|1304.81|1.031|
|consumer|initial_2k|[2048]|[2048]|2048|133.9|129.66|141.47|0.916|
|consumer|initial_8k|[8192]|[8192]|8192|1405.55|1458.85|1571.35|0.894|
|consumer|initial_32k|[32768]|[32768]|32768|20120.82|21149.96|21468.76|0.937|
|consumer|steady_16k|[512]|[16384]|16384|345.29|308.18|316.08|0.975|
|consumer|steady_64k|[2048]|[65536]|65536|4675.86|4618.52|4694.21|0.984|
|consumer|steady_256k|[4096]|[262144]|262144|42681.4|38293.88|38121.7|1.005|
|consumer|tail_1m|[256]|[1048576]|1048576|10919.42|9792.9|9736.25|1.006|
|consumer|ragged|[512, 128]|[65535, 16384]|65535|1546.61|1307.15|1343.82|0.973|

## 唯一候选收益没有独立复现

主矩阵source Q4096、原context262144、ratio1的正反测试曾满足门槛：Ant保守P50 38.128ms，最快baseline40.402ms，约1.060x。两序噪声余量后的比值分别1.0576和1.0562。

随后在空闲GPU1以完全相同core/driver、输入、selector和工作区合同做一次独立正反复测，并保存每秒温度/SM及显存时钟/功耗：

| 顺序 | DeepGEMM ms | 现有 keys-major ms | Ant ms | Ant相对最佳baseline |
|---|---:|---:|---:|---:|
| 正序 | 41.848 | 37.497 | 37.242 | 1.00684x |
| 反序 | 41.837 | 37.494 | 37.230 | 1.00710x |

收益仅约0.7%，因此撤销候选。未控制隔离卡间差异、分配布局、时钟和温度的单独影响，**不复现原因未知**；不能把它归因到某一个因素。没有继续挑选有利轮次或调参凑5%。本矩阵不存在可据此采用的Ant自动选择交叉点，更不能套用蚂蚁2M阈值。最终决定见 `model-aligned/DECISION.json`；isolated全部原样本、时钟及身份见 `model-aligned/isolated-source/`。

## 正确性与数值限制

最终23点全部可见score通过相对native的rtol=1e-3、atol=1e-4；Ant最大绝对差2.861e-6、最大nRMSE9.67e-8。按正序一次计，6行token集合变化（最低511/512），5行source候选集合变化（最低2047/2048）。全部变化由实际边界分差≤2.384e-7或native精确同分解释。没有承诺与native逐位一致。

所有行检查数量、唯一性、越界、source latest block；每个arena chunk首/中/尾行对当前相同logits稳定参考精确一致。独立动态验证 `../p5-model-shapes/dynamic-verifier-v2/REPORT.md` 使用相同最终driver/core：186项检查、2976行token与1056行source候选全行精确参考通过，覆盖候选原位更新、长度变化、页表/请求重排、全零及恢复、NaN poison和Graph先验证后Eager刷新。任意ragged分组改变仍需重新规划，不冒称旧Graph支持任意新布局。

**初始FP32逐元素strict门槛没有通过。** FP32抽样覆盖各chunk首/中/尾、每行<=3072均匀key加双方top512边界。正序23点合计，Ant与DeepGEMM各3327个抽样元素失败，最大绝对差均约0.0021548；current keys-major为3326个。这与“优化对native回归通过”是不同结论。共同偏差不能静默当作FP32通过，也不自动放宽门槛。独立逐head诊断进一步在ratio2 FULL的steady256K row4095（实际131072个keys）观察到：Ant对FP64最大误差0.0021547168；用暴露的D128 FP8 per-head dot加相同scale与顺序FMA重构，残差仅7.15e-7（9行最大9.54e-7），因此主要误差来自D128 FP8 dot，而非候选处理或scale广播。4×D32在该行对FP64最大误差0.000343578，更接近高精度；这是不同累加行为，不能叫作D32错误。独立原始证据见 `../p5-model-shapes/numerical-diagnostic/diagnostic.json` 及其PTX、最差输入和前后身份。

## 附实验、证据和边界

- 旧23个`small/large/mixed`形状是synthetic筛选，见根 `SUMMARY.json/TABLE.md`，不替代模型主矩阵。其中ratio2 consumer、request-shared候选不代表模型真实consumer。
- 独立grid39/78/156/312消融见 `../p5-ant-grid/REPORT.md`：四个筛选形状默认78均为最低完整P50，其他grid未显示稳定收益。只作附表，不泛化为全部模型形状的证明。
- `verify-v2/v3`保留了空输出overlap统计为0的旧指标问题；后续v4修复为双方空时1，并增加动态回放后精确选择检查。最终主要动态证据采用独立model verifier v2。早期legacy correctness JSON没有独立source绑定、native identity仅post-run采集，限制明确保留。
- 模型主批与isolated批都在启动前和完成后采集完整身份：79份冻结源码及589份实际native依赖逐字节一致；wrapper核验当前HEAD，二进制build commit仍未知，未把wheel名当作当前HEAD构建证明。
- 主core SHA `957557aa4ac9944073ae4e2c2a370775df446f93437fa82de821b3051d3184c9`；driver SHA `f422bdd3a5593e031c16e9270f0e93fb3dd343402c4eacd13cfe087987c7ef49`；主source archive SHA `99c0ffdae2fd8497c0f610477c8cc846a9428ed494d3f69ba50423b6b0d22445`。
- 未启动模型/serve，未测模型准确率、TTFT、TPOT；未改变FP8缓存和scale语义。此研究无稳定新增dispatch收益并不意味着P5c逐query compact优化没有空间。

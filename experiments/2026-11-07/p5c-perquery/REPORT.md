# P5c：真实逐 query FP8 prefill compact consumer

**实现、8点完整实验及同对象paired补充均已完成；本阶段未直接修改自动 dispatch，新增准入另由P8独立binding确认。** 已补齐原 P5 的真实 `[Q,2048]` 候选紧凑打分：直接间接读取与候选先 gather 后自定义 batched FP8 WGMMA GEMM 两条路径，都与 DeepGEMM/现有 keys-major 的 fullscore+perquery mask 进行同输入、同稳定 selector 的四方比较。完整覆盖模型 consumer 的8个形状，正反顺序、Eager/Graph以及准备/score/top-k/完整调用的原始样本全部保存。

最终主结论以同对象paired复核为准：64K/256K/1M尾chunk/ragged中direct的Graph保守净加速分别为 **1.473×、6.117×、22.044×、1.308×**；初始2K/8K/32K及16K上下文继续保留full路径。gathered WGMMA在8点均慢于direct，不进入自动选择。新路径若进入P8仍需对应精确形状/可见长度模式的独立实际绑定，不能从本报告直接推成服务加速。

## 原始候选哈希与补充验收

原8点的正反在独立进程生成source输出。initial32K和4个长上下文点的`candidate_sha256`不同；该字段哈希的是无序source输出的原始排列，不能作为候选集合的语义fingerprint。每次实验内四条路径使用同一候选tensor，compact会规范化排序；但历史raw未保存原始tensor或canonical hash，**不能回填、伪造历史canonical一致的字节证明**。

`paired-v1`独立补充覆盖8点source三次重放的canonical集合审核，并对4个拟准入长形状只构造一次inputs/candidates/四路径对象，连续测正反完整pipeline。它保存Q/W/cache/页表/request/length/sourceQW及candidate raw/canonical前后hash。准入应以该同对象补充通过及P8独立binding为准；原8点分项和负收益记录保留，不覆盖原raw。补充已经通过：8点canonical集合三次精确重放；4长点所有输入与候选raw/canonical哈希前后一致，四路径同一对象完成正反。新结果见下节与`paired-v1/AUDIT.json`，没有回填旧raw。

## 版本、环境与范围

- 冻结目录：`h20-4:/root/dsv41-indexer-resume-20261107/runs/p5c-final-v1`。
- 源包：`final-v1/frozen/source.tar.gz`，SHA256 `b4c1a3dbda6778ea48d226b44ff77809dce604a50fd63b9b3aac15184e83e41b`。
- 新模块 `perquery_prefill.py`：`77a7ffbf34ebe4495e560612c9b6943b17214b5fd01a9cdfe613d5887dbe9b1c`；最终 driver：`40f50488bfa7213ae3354340c08786e954f7403f2c8d81ebfcd4a2fa8feb96d4`。
- GPU1测initial2K/8K/32K及steady16K；GPU2测steady64K/256K、tail1M和ragged。**每一形状正反均在同一卡**，两卡只是并行执行独立单卡实验，不是TP2/TP4延迟或吞吐。
- `runtime-before.json` 与两个 `runtime-after-{a,b}.json` 的全部源文件、589项原生文件哈希、版本及源包身份精确相同。安装wheel build commit仍未知，不声称由当前HEAD重新编译。
- 固定模型配置副本与hash在`final-v1/frozen/`。本轮仅ratio1 consumer层24/28/32/36的模块形状：32 index heads、D128、top512、候选2048块×8。未加载模型权重；source和consumer的Q/head weights是各自固定随机输入，维度/因果结构对齐模型，不能称真实模型数值准确率。
- Q表示一个prefill chunk的query token行数，不是请求并发。尤其1M点是**单请求Q256尾chunk**；ragged是两个请求Q=[512,128]、终止上下文=[65535,16384]。

## 实现合同

A `direct` 复用P2已经验证的逐query FP8 WGMMA scorer，直接以每行候选逻辑块和真实request ID查本rank页表。TensorDescriptor在构造期规划，不在计时区重新创建。

B `gathered_wgmma` 将当前query chunk各自的候选FP8 payload与独立FP32 K scale聚集到有限staging，再执行自定义batched WGMMA GEMM。**它不是DeepGEMM，也没有让不同query相互计算对方的K。** 使用相同FP32 scale→ReLU→head weight及P2 head reduction；没有重新量化K或改成FP16。

两路线都接受显式`[Q,2048]`候选，一次模块调用只publish/normalize一次，随后所有chunks顺序共用arena，使用相同Ant稳定top-k并融合compact→逻辑位置映射。不制造一query一个request alias，不依赖请求级候选共享；输入request、length、页表和候选内容可在固定storage内动态改变。source唯一输出走快发布；任意外部列表走去重。输出是无序逻辑top-k集合，合法项唯一，不足填-1；“稳定同分”不意味着输出列表必须排序。

score arena明确不超过1GiB；gather staging独立限256MiB。宽16384时staging容量124行，占268,173,312字节，score仅8,126,464字节。initial32K时direct真实使用1GiB score arena、2个chunk；gather执行265个chunk，所有chunk成本已计入pipeline。候选state、Q/cache、多个对照及验证临时量另占显存；1GiB不是总进程显存承诺。正式采样的最大torch allocated峰值为6,486,343,168字节，含同进程全部对照和source驻留。

## 正确性与已知数值边界

- 最终冻结下42组动态阶段：6形状×7阶段，覆盖两backend、source_unique快发布及任意候选去重、正长度空候选、尾部partial block、16,396 keys跨候选cap、零query、全零长度、request/page重排、非2幂scale及恢复。
- 每阶段先poison候选state/count/score/output和gather K/scale，再Graph replay并clone所有输出，之后才运行oracle/Eager。Graph输出与独立固定score oracle及Eager的逻辑集合精确一致，避免Eager先prepare掩盖Graph漏准备。
- 8个性能形状双向累计101,120 query行逐行验证；所有有效候选score通过相对DeepGEMM的`rtol=1e-3, atol=1e-4`门槛，最大绝对差1.90735e-6。direct与gathered有效score逐位相同；固定logits的逻辑top-k集合精确；本次全部形状最终集合与native参考100%重合。
- 两条full baseline也检查了各自实际masked score对应的稳定逻辑选择及对native的score门槛。DeepGEMM与keys-major都保留，不能只挑较慢对照宣称收益。
- FP32独立诊断**不是全部通过最初严格逐元素门槛**。42阶段opt/native分别6487/6481个抽样失败，最大误差约0.0018990/0.0018992；原生自身也存在该误差。正式矩阵各形状的opt/native失败数几乎相同（initial32K为468/469，其余相同），详见raw。这沿用此前FP8原生误差校准，不改FP8累加，也不隐去诊断或把native一致写成FP32逐位一致。当前新路径对native的误差比这一共同FP32偏差小约三个数量级。
- PTX证据两路线均存在FP8 WGMMA，无FP16 MMA，代表编译0 spill；direct112 registers、gather56，shared均12,544字节。PTX与SHA在最终结果目录。

## 同对象 paired 最终复核（准入的性能依据）

新runner `paired-v1/paired_pipeline.py` SHA256 `5b8ffddff4f4f51b145a862439b57575b9f7c8d221d1612643f6563052da83f3`，明确导入并断言原冻结core/driver hash，没有改原实现。GPU1测64K/256K，GPU3测1M/ragged；每个shape两顺序都在同一进程、同一卡、同一inputs/candidates和四个path对象上完成。

8点各重复source三次，canonical排序集合精确一致；initial32K和4个长形状raw排列确实改变，其余3点raw也相同。此处是新的独立证明，不是对旧raw候选tensor的追溯重建。4个长形状保存10类输入before/after字节哈希（Q、weights、cache、table、requests、lengths、source Q/weights、candidate raw/canonical），全部不变；raw候选在paired两顺序之间也没有重新生成。

64条完整pipeline记录，20预热、30×10，各64轮、1920样本；全部首轮CV≤10%，无追加轮触发。两个分组source/native before/after也完全相同。每order标准P50/P90、host wall及CV见`paired-v1/AUDIT.json`。下表仍用最快full单轮P50 / 最慢direct单轮P50作保守比较：

|形状|最快full Graph μs|direct最慢 Graph μs|Graph保守加速|Eager保守加速|
|---|---:|---:|---:|---:|
|steady_64k|4584.78|3111.53|1.473×|1.476×|
|steady_256k|38000.39|6211.88|6.117×|6.118×|
|tail_1m|9737.42|441.72|22.044×|21.766×|
|ragged|1297.21|991.82|1.308×|1.317×|

因此direct的4点收益在同对象反转顺序下仍成立；gathered WGMMA仍比direct慢，保持显式实验实现，不推荐准入。源候选生成仍在consumer计时外，paired不重测原1090条大Q分项；原8点分项记录和短形状负收益证据全部保留。准入最终还应结合P8非provisional实际binding，不把历史原始候选SHA作为运行时语义键。

## 原8点完整覆盖与历史性能（保留原raw）

下表单位μs。为避免挑最好一次，full列取全部正反/追加轮中**最快单轮P50**，compact列取**最慢单轮P50**，最后一列用最快full除direct最慢。标准每order聚合P50/P90、host wall、CV、显存和所有原始样本见`final-v1/SUMMARY.json`与四份raw JSON。

|形状|DeepGEMM full|keys-major full|direct|gathered WGMMA|direct保守加速|
|---|---:|---:|---:|---:|---:|
|initial_2k|131.47|128.05|317.27|410.40|0.40×|
|initial_8k|1397.72|1451.32|3451.81|5182.83|0.40×|
|initial_32k|20013.50|20989.98|36762.83|52269.14|0.54×|
|steady_16k|343.60|306.32|781.56|911.70|0.39×|
|steady_64k|4638.81|4584.90|3117.51|3811.61|1.47×|
|steady_256k|42041.19|37732.71|6162.90|8300.72|6.12×|
|tail_1m|10722.35|9642.80|440.84|545.40|21.87×|
|ragged|1520.21|1286.80|989.99|1169.45|1.30×|

这组原始独立进程数据得到Graph保守1.47×、6.12×、21.87×、1.30×；Eager保守分别1.47×、6.11×、21.70×、1.31×。它们保留为覆盖与分项证据，**四个新准入点的最终性能依据改用前述paired同对象结果**。这些比值包含publish、全部chunks、gather（B）、score和稳定映射/top-k。它们只代表此处架构合法、固定随机输入的模块实验。

初始2K/8K以及16K上下文候选基本覆盖全长；initial32K虽跨cap，理论有效位置节省仍不足以弥补逐query scorer/发布成本。对应direct保守比值约0.40×、0.40×、0.39×、0.54×，明确保留full路径。B在长上下文也快于full，但比A慢，不以“相对某个baseline有收益”绕过更快A。

## 计时与分项解释

共3,360条variant/phase/order/execution记录；3,561轮、106,830个样本，其中201轮为初次CV>10%之后追加的三轮，全部保留。每轮20预热、30样本×10调用；有Eager/Graph、GPU时间、host wall和峰值。正式源码不改后重跑正反，source/native before/after一致。

full路径为对应scorer + 共同K prepare + perquery mask + 共同Ant selector，并非HEAD原生完整indexer wrapper。当前keys-major与DeepGEMM都参与最快baseline选择。未计模型投影、MoE、通信或完整服务。

source layer20的候选来自独立Q/head weights，生成在consumer计时之外，另作为`source_generation`实测；它采用现有keys-major source作为fixture，并非对source最佳实现的新结论。本次是单consumer，不声称source+四consumer组合成本；该组合另有P2/P3既有范围证据。

准备阶段包含candidate publication/normalize及各chunk gather。B的staging会被下一chunk覆盖，因此score-only/top-k按每个chunk先正确准备后分别测量；其分项和只用于归因，不能冒充真实连续调用延迟。完整pipeline是真实一次forward遍历所有chunks的测量。所有输入保持warm repeated、无L2 flush；初始化、descriptor规划、JIT均在计时外。

## 原始失败与复现

`verify-v1`的TMA offset编译错误已修复为int32 descriptor offset，保留源码/日志；v2是较窄原型通过。v3/v4因verifier错误地逐位置比较无序top-k而失败，修正为集合精确比较；v5因扩展fixture tuple漏一项而中止。它们全部保留STATUS，均不作为最终验收。v6及`final-v1/verify.json`是补齐全部边界和poison-first协议后的通过证据。CPU preflight、独立审阅及无计时大形状pilot也保留。

在冻结目录中按`final-v1/frozen/run-group.sh`运行两组，保持每组正反在同GPU；正式参数与输出由`RUN_MANIFEST.json`记录。正确性入口：

```sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_prefill_perquery --output RESULT_DIR/verify.json
```

结果归纳/采样核验在本地仓库执行：

```sh
.venv/bin/python results/indexer/20261107-resume/p5c-perquery/summarize_final.py
```

本实现可以关闭“真实perquery compact consumer的两条路线实现与函数级研究”缺口；整个优化Goal及P8新准入由总控依据独立binding/最终归档决定。本报告不自行勾选方案书，不推源码仓库。

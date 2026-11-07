# P5 H20 FP8 prefill 模块实验 checkpoint

- 日期：2026-11-06
- 运行结果：completed
- 范围：P5 独立函数/模块实验；不代表整个 P0–P8 Goal 完成。

## 结论与核心结果

已完成 request-local FP8 key 一次准备、DeepGEMM原生对照、row-tiled、真正 keys-major（一个K tile在CTA内跨多组Q复用）、candidate先gather再GEMM及直接间接读取，并执行正反序32条shape/mode/order实验。各条均含Eager和CUDA Graph，保留全部原始样本。keys-major不是仅交换grid维度；本checkpoint用Triton实现，未冒充尚未测试的Gluon扩展。

下表为正反序Graph P50平均值，单位μs；数字都包含本次调用的key准备、打分、token top-k，source额外含候选选择，consumer额外含候选规范化和逻辑映射。最优仅表示本次测得最小值，差异<5%不自动进入dispatch。

|模式|query行|请求数|实际index keys|ratio|原生完整链路μs|最快实测路径|最快μs|加速|
|---|---:|---:|---:|---:|---:|---|---:|---:|
|full|64|1|65536|1|631.12|keys_major|151.29|4.17×|
|source|64|1|65536|1|707.58|keys_major|166.42|4.25×|
|consumer|64|1|65536|1|643.41|row_tiled|63.11|10.20×|
|full|1|1|4096|1|48.17|indirect|7.31|6.59×|
|full|16|1|262144|1|2487.91|indirect|194.67|12.78×|
|full|64|1|16384|1|165.44|row_tiled|46.05|3.59×|
|full|64|1|32768|2|320.41|keys_major|81.47|3.93×|
|full|64|1|131072|2|1257.31|keys_major|301.05|4.18×|
|full|256|1|16384|1|177.18|keys_major|152.73|1.16×|
|full|1024|1|16384|1|661.19|keys_major|565.66|1.17×|
|full|4096|1|4096|1|375.23|deepgemm|364.03|1.03×|
|source|64|2|65536|1|1396.14|keys_major|196.60|7.10×|
|source|256|1|16384|2|282.29|keys_major|161.72|1.75×|
|consumer|16|1|262144|1|2489.16|row_tiled|37.34|66.67×|
|consumer|256|1|65536|1|741.25|keys_major|170.41|4.35×|
|consumer|1024|1|16384|2|704.54|keys_major|585.54|1.20×|

关键观察：

- q64、64K keys，full约631.12→151.29μs（4.17×），source约707.58→166.42μs（4.25×）。consumer约643.41→63.11μs（10.20×），请求内共享候选，最多2048块×8位置。
- q4096、4K keys时，keys-major与DeepGEMM+Ant selector接近，row-tiled更慢；不足5%的差异应保留原路径。没有照搬“2M scores起用keys-major”：q64×32K虽达2M，keys-major相对row-tiled仅约1–2%；q1024×16K才有约7%的稳定优势。调度必须同时看query/key形状和模式。
- 32条记录中各优化版相对原生分数最大nRMSE为7.73e-8，最大绝对差1.44e-6，top-k集合重合均100%。24组独立正确性通过；alias候选、NaN尾部、graph输入变化专项1项通过。FP32误差单独报告，未用过严FP32门槛迫使默认切到FP16累加。

QPS、输出tok/s、TTFT、TPOT、E2E、Prefix hit rate均N/A。本实验不加载模型、不测模型准确率。

## 复现

主机h20-dsk，物理GPU1（CUDA_VISIBLE_DEVICES=1，进程内device0），工作目录/root/dsv41-indexer-port。无服务端口。

```sh
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_prefill --output results/indexer/20261106-tp4/p5/verify-final.json
.venv/bin/python -m pytest -c tests/kernels/dsv41_indexer/pytest.ini tests/kernels/dsv41_indexer/test_prefill.py -q
bash scripts/05_prefill.sh
```

05_prefill.sh包含smoke/matrix各自正反序全部命令。20次预热、30样本、每样本10调用；52个测量项CV>10%自动追加3轮，全部保留在extra_rounds。没有静默删掉噪声结果。原始JSON包含P50/P90、墙钟、峰值、原始样本。多GPU独立实验同期可能有CPU争用，最终总控P8仍须在固定负载下回归。

原生基线使用当前HEAD所对应的cp_gather_indexer_k_quant_cache、fp8_fp4_mqa_logits（FP8 Q、None scale及FP8 K）、top_k_per_row_prefill；source调用原生select_candidate_blocks；consumer全上下文打分后apply_candidate_mask。基线和优化都按同样的request/chunk计划执行。这是request-chunked模块适配，不是运行vLLM服务调度器；多请求不声称复现所有上游packing策略。

## 运行配置

H20 SM90；PyTorch2.13.0+cu130、Triton3.7.1、vLLM0.32.0、TVM-FFI0.1.11；完整环境参见父目录environment.json。HEAD29be941d相关wrapper逐文件核验；预装二进制构建commit未知，不能称为HEAD重编译。
FP8 E4M3 Q/K，Qscale已在FP32 head weights里，cache页64、32 heads×128、top-k512。基准seed20261101，候选seed20261106。ratio1/2统一使用index-key坐标，逻辑上下文=实际key数×ratio；输出不乘ratio冒充原token位置。

source.tgz SHA256：`0703b5893a2a8efbffd1042da3c61a78405376308c07906a9f5d7d62c8319f15`。
其中实测prefill.py SHA256：`6fe4a6c8556ddba16d2aaa99d40862a1f28b0f2c8539b3adf8d816d2f2968a4b`。
source-sha256.txt保留归档时的共享依赖哈希。当前工作区可能已有单独的未测Gluon扩展；上述快照才是这批数据对应的P5源码，不能把后改代码冒充原始版本。

## 限制与证据

- 这是分阶段筛选矩阵，不是所有维度的笛卡尔积。已覆盖query1/16/64/256/1024/4096和实际keys4K/16K/32K/64K/128K/256K及ratio1/2；超大4096×256K组合未做性能测量，不能称为内存失败或已测。实现按1GiB score预算分块；通过小预算强制分块测试验证请求key仅准备一次。
- consumer每request共享候选；每query不同集合可以用相同物理页表的request aliases表达，专项测试通过。性能表中consumer使用request-shared候选，不能外推到所有query候选都不同的成本。
- Q、长度、候选内容可以在稳定request/chunk计划下变化并graph replay；改变query所属request或分组需要重新构造模块/计划并重新capture，无跨步隐式缓存。
- keys-major/row-tiled/indirect均保持FP8打分，原生与优化分数非逐位一致。固定logits的稳定top-k与规范化/映射要求精确；不会声称模型精度结论。
- peak_allocated_bytes包含同一benchmark进程内其他variant预分配的驻留缓冲，不能直接当作单一模块独占峰值。每个check另有preparation_bytes及score_workspace_upper_bound_bytes；如要对比独占显存，需要各variant独立进程再测。
- Eager受Python launch开销影响，详见原始JSON，不能用Graph数字冒充Eager。新backend尚未自动接入dispatch；明确无收益/小于5%的形状保留基线。

证据：smoke-{forward,reverse}.json/.log（各3例），matrix-{forward,reverse}.json/.log（各13例），verify-final.json/.log（24组），pytest.log（1项），source.tgz，source-sha256.txt。smoke-initial-forward是初轮原始记录，同样保留但不重复计入32条最终矩阵。

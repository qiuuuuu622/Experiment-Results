# P4 H20 grouped verify 实验记录

- 日期：2026-11-06
- 本阶段已完成独立正确性、调优与性能实验；整个P0–P8 Goal尚未完成。
- GPU：h20-dsk物理GPU3（CUDA_VISIBLE_DEVICES=3，进程内device0），已释放。

## 结论

已实现真正的同请求 key tile 共享：每CTA读一次FP8 payload/FP32 key scale，多个query复用shared-memory K；保持32 index heads与原FP8 cache ABI。双query pair异步WGMMA与单query顺序版本均实测；6行组按真实6行规划，避免padding8多算空pair。PTX确认使用E4M3×E4M3 FP32 WGMMA，不是FP16替代。

收益有范围：32请求×6行×64K时，完整full从约583μs降到约508μs；较小请求数时full通常仍不如DeepGEMM，不能全局替换。source在部分形状有收益，但1请求×6行×256K只约3%，不足5%应保留原路径。consumer只有候选布局共享时启用，按原生compact逐行做对照，含一次发布；任意不同候选集合保留逐行。

重要限制：当前通用候选共享资格检查在host读取counts/blocks；若每步重新验证，成本可达0.4–15ms，远大于kernel收益。下面快速consumer数字仅适用于预规划且每次回放维持共同候选前缀/最长行顺序的明确契约；不能把任意逐步变化的候选自动放入快路径。每步replan结果单独保留，不隐去。

## 正确性

- `verify-v2.json`、`verify-paired-v2.json`各31组通过。覆盖group1/2/4/6/8×keys517/4099/65536×ratio1/2（真实logical长度除ratio计算可见key）；另混合1+2+4+6+8组。
- full/source固定logits稳定top-k和source候选与稳定参考精确一致；未引入重复、越界或未来位置。默认及pair路径与原生分数按rtol1e-3、atol1e-4回归，FP32容差校准沿用总方案说明。
- 同query行逐行与grouped分数在性能矩阵中逐位一致；两种路径的top-k集合相同。
- 包含NaN无效score尾部、graph改变可见长度、零Q精确同分、共享候选变化、不同候选fallback、请求本地物理页置换。
- CPU共享资格边界测试3项通过：最长合法前缀、不同集合fallback、非法count/空组拒绝。
- 原生N1/2/4为native grouped；N6/8基线明确标注expanded_single_row，不冒充原生融合。

## 性能

以下为CUDA Graph正反顺序P50的平均值（μs），仅便于阅读；原始两个顺序及P90、全部样本、CV追加轮次、主机墙钟、显存峰值都在JSON。候选上限2048块×8，score输出FP32，热cache重复调用，无L2清空。不同GPU同时进行独立实验；Eager包含host启动且易受共享CPU负载影响，最终P8需固定条件复测。

|请求|每请求行|index keys|CTA目标|原生full|grouped full|原生source|grouped source|逐行compact consumer|共享consumer含发布|
|---|---|---|---|---|---|---|---|---|---|
|4|1|65536|512|21.03|42.69|83.28|50.45|24.38|24.39|
|4|2|65536|512|36.40|47.16|99.85|56.23|34.99|23.97|
|4|4|65536|512|55.25|61.09|120.50|73.83|55.76|28.83|
|4|6|65536|512|71.29|82.38|142.84|97.89|66.83|34.72|
|4|8|65536|512|94.81|99.79|164.49|118.57|78.27|40.46|
|32|6|65536|2048|579.20|509.76|678.21|587.76|328.98|176.16|
|1|6|262144|512|65.44|106.84|134.18|130.05|29.65|21.79|

每项20预热、30样本×10调用，forward/reverse各一轮。CTA256/512/1024/2048/4096均正反调优；CV>10%的项追加3轮，保留全部数据。单query顺序版本全7case见benchmark.json；初版pair见benchmark-paired-v1.json（保留无益/低效实验）；最终pair见benchmark-paired.json。

### 共享资格重规划成本

下面是Eager主机墙钟P50（μs，forward示例；reverse同样在planning.json）：

|请求|每请求行|逐行发布+执行|预规划共享发布+执行|仅host资格规划|每次发布+重规划+执行|
|---|---|---|---|---|---|
|4|2|82.63|44.44|492.76|546.53|
|4|4|82.44|48.80|951.54|1079.96|
|4|6|82.45|47.09|1400.86|1532.38|
|4|8|88.35|48.63|2212.66|2332.76|
|32|6|334.91|170.43|14786.42|15139.40|
|1|6|79.13|45.11|382.62|457.21|

资格规划含host校验和设备metadata更新，不能CUDA Graph捕获。这个通用检查用来验证实验输入，不应在任意在线decode每步调用；若调用必须纳入完整成本，收益将消失。没有静默排除这项成本。

## 复现命令与证据

远端工作目录 `/root/dsv41-indexer-port`，Python统一`.venv/bin/python`，无模型/服务/端口。

```sh
CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH .venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_grouped --output results/indexer/20261106-tp4/p4/verify-v2.json
CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH .venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_grouped --paired --output results/indexer/20261106-tp4/p4/verify-paired-v2.json
CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_grouped --output results/indexer/20261106-tp4/p4/benchmark.json
CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_grouped --paired --output results/indexer/20261106-tp4/p4/benchmark-paired.json
CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_grouped_planning --output results/indexer/20261106-tp4/p4/planning.json
.venv/bin/python -m pytest -c tests/kernels/dsv41_indexer/pytest.ini tests/kernels/dsv41_indexer/test_grouped.py -q
```

原始log与JSON同名；`fp8-instructions.txt`为已编译PTX指令证据。`source-v1.tgz`保存早期相关代码；`source-final.tgz`为本次保存时的相关源码checkpoint（共享依赖也在包内），最终P8运行还须在启动前固定整个最终实现版本，不将本checkpoint冒称所有历史运行的完整封存源码。

当前源码checkpoint SHA256：`5a4073f10b37241d6ff4b3088707d879226740408510f3e975a741bfe878e10d`。

未测试模型准确率、TTFT、TPOT或serve集成；不外推本模块收益为模型吞吐。四卡真正分摊与通信属于P7，不属于本GPU3独立实验结果。

# H20 Top-k 大行数调度实验

- 日期：2026-11-07（Asia/Shanghai）
- 运行结果：completed
- 主机：h20-dsk（用户简称 dsk），四张 H20，各自进行独立同卡 A/B；不是 TP4。

## 结论与核心结果

**32 行×128K/256K、64 行×256K 的确定性 top-512，cluster-2 配置有稳定收益；128 行尚无新配置超过原 streaming。** 不统一切换 cluster，不修改已发布的默认调度。所有性能结论仅针对 selector；FP8 Indexer 打分用于构造输入，不计入 selector 时间。

最终表采用固定随机 FP8 Q/K、K scale 和 head weights，经实际 DeepGEMM Indexer 打分得到的 FP32 logits。它比独立正态 logits 更接近模块输入，但不是完整模型采样。每形状三次独立进程运行，均包含正反 A/B；表为全部保留 Graph GPU samples 的 P50，单位 μs。

|行×规划宽度|vLLM 原生|原有最佳稳定策略|新策略／耗时|建议|
|---|---:|---:|---:|---|
|8×128K|20.184|cluster-8：11.418|cluster-2：16.106|保留 cluster-8|
|8×256K|26.739|cluster-8：15.990|cluster-2：23.731|保留 cluster-8|
|32×128K|38.923|streaming：21.446|cluster-2：16.502|约 1.30×，延迟下降23.1%|
|32×256K|68.186|streaming：37.902|cluster-2：25.336|约 1.50×，延迟下降33.2%|
|64×128K|50.106|streaming：21.504|cluster-2：24.826|保留 streaming|
|64×256K|82.781|streaming：61.958|cluster-2：47.859|约 1.29×，延迟下降22.8%|
|128×128K|54.826|streaming：44.477|hist2：44.779|保留 streaming|
|128×256K|79.963|streaming：84.456|hist2：84.256|差异不足5%，保留 streaming|

收益均相对该形状原有最佳**稳定**策略。原生 128×256K 较快，但不能把它的同分语义称作严格确定性。32×256K 的最终合并值为25.336 μs；早期约24 μs仅属筛选批次，不能用更漂亮的早期值替代最终三轮。

### 关键观察

- cluster-8 的大 batch 慢路径可复现：128×256K 正态 logits 约207 μs，streaming约84 μs。Trace 中 persistent cluster 主内核约205.6 μs，plan约1.3 μs。PDL相关后处理可能重叠，不能把各kernel duration相加当总延迟。
- 减小 cluster 并允许每行直接调度有益于32/64行；128行直接 cluster-2约94 μs，仍不胜streaming。改变cluster宽度及路由是两个因素，不能把全部收益只归因于其中之一。
- 直方图2副本无稳定>5%收益；warp合并原子操作正确但显著变慢；4副本超过静态共享内存上限而编译失败。这些负结果与失败均保留。

## 复现

实际工作目录 `/root/dsv41-topk-h20-tuning-20261107`，独立 `.venv`，系统CUDA/PyTorch依赖。所有实际Python执行通过 `.venv/bin/python`。

### 实验顺序

1. 第一轮：`scripts/01_compile.sh`，辅助并行编译命令保存在 `scripts/01b_compile_parallel.sh`；初次包装代码错误及修复后输出分别保留。`scripts/03_launch_four_gpu.sh` 分配GPU0=1/8行、GPU1=32行、GPU2=64行、GPU3=128行。每卡比较原生、streaming、cluster-8/4/2及不同pool。
2. 第二轮：`scripts/04_round2.sh GPU`，GPU0改为256行压力点。加入direct调度，即通过512的路由常量强制每行一个cluster；**512不是声称同时驻留512个cluster**。比较正态logits和FP8 Indexer生成logits。
3. 第三轮：`scripts/05_round3.sh GPU`，测试streaming直方图复制、warp聚合。hist4编译失败，未进入性能或正确性结果；其失败JSON/log独立保留。
4. 最终：`scripts/06_final.sh GPU`，GPU0=8、GPU1=32、GPU2=64、GPU3=128。只复测候选与相关基线，每卡三次独立进程、每次正反Eager/Graph，再做最终动态验证。

每阶段的 `snapshots/*source.tar.gz` 保存当时完整driver/header/scripts。**要复现历史轮次，应先解压对应快照到新目录，调整脚本绝对路径和输出目录；不要用最终runner的默认全variant列表重跑旧轮次。** 最终runner仍保留显式可选但已知编译失败的hist4研究配置，正式脚本不选择它。禁止覆盖冻结结果。

```sh
# 恢复匹配的环境和最终源码快照后，各卡可独立执行：
bash scripts/06_final.sh 0
bash scripts/06_final.sh 1
bash scripts/06_final.sh 2
bash scripts/06_final.sh 3

# 在归档副本中重算最终汇总与审计计数，不需要 GPU：
/path/to/.venv/bin/python collect_results.py
```

### 计时与正确性

- 每策略20次预热、30样本×10调用，正反序；CV>10%自动追加三轮。全部原始GPU和host wall样本保留，不能只选择最低P50。
- CUDA Graph捕获、JIT、分配、输入生成在计时外；实际需要的plan包含在forward里。热缓存重复调用，无L2 flush。峰值包含同进程多个比较工作区，不是独占kernel显存。
- 同一case所有variant共享相同scores；前后SHA256一致。最终每形状三轮的score hash也一致。Indexer fixture采用32heads、D128、FP8 Q/K，实际逐行可见长度记录于JSON，通常从width-rows+1递增至width。
- 固定FP32 logits以稳定argsort为参考，对比逻辑位置集合。k512/2048，random、zero、equal_nonzero、cutoff_ties、tie_bin_overflow；每项验证Eager full、Graph mixed、Graph all-zero、Graph restore。无效尾部写NaN、输出先写-999，再直接回放Graph。
- 正确性共690个配置/分布检查，覆盖2760个执行阶段；另有各benchmark的同输入稳定选集检查。原生仅按自身选择语义核验，不能宣称原生同分稳定。

## 运行配置

|项目|实际值|
|---|---|
|设备|h20-dsk GPU0–3，NVIDIA H20，每卡78SM；UUID与拓扑见原始环境|
|Python/PyTorch|Python3.12，PyTorch2.13.0+cu130|
|CUDA/Triton/TVM-FFI|CUDA13.0，Triton3.7.1，TVM-FFI0.1.11|
|vLLM|0.32.0；Python wrappers按29be941d的7文件清单核验，实际二进制build commit未知|
|来源|已发布93a4e82的核心模块快照；Ant 6e6d5d9e；本轮只修改隔离副本|
|新CUDA实现|cluster-size可配置；streaming直方图2/4副本和warp聚合可配置；确定性开关一直启用|
|编译参数|C++20、O3、sm90a、SGL_TOPK_DET_TIES=1；每variant完整宏在tune_topk.py|
|模型/服务/端口|未加载模型、未启动服务；QPS/输出tok/s/TTFT/TPOT/E2E均N/A|
|性能总量|32个完成的benchmark文件，1620条测量记录，1653轮，49590个GPU样本及同数量wall样本|

各GPU的绝对时间不直接混比；每个形状的策略比较在同卡完成。同机并行实验可能影响主机墙钟，原始wall和追加轮均保留。实验结束后四卡显存占用0 MiB、利用率0%，只停止了本任务的监控进程。

## 限制与证据

### 可确认与尚未确认

- 可确认：在已测形状、分布及固定score语义下，cluster-2改善32/64行；128行没有新的稳定胜区。当前只形成实验候选，不自动修改已发布模块的生产/服务调度。
- 尚未确认：完整Indexer包含打分/发布/通信后的净收益、真实模型分数分布及模型准确率、真正TP4组合下的收益。本轮四卡是独立实验，不是四卡collective。
- Nsight Compute/Systems不可用；已有PyTorch trace可区分kernel耗时，未采集SM stall、DSMEM或DRAM硬件计数器。不能把“原子冲突/带宽饱和”写成已定量证实的唯一根因。
- native二进制和DeepGEMM身份在最终阶段/运行后补充采集，不冒称所有早期轮次都有native before/after证明。每个完成runner的实验源码before/after一致。
- 保留两次失败：初次生成包装代码多出字符导致编译失败；hist4静态shared memory约0x10930字节超过0xc000限制。失败不算正确性或性能成功。

### 证据索引

|内容|路径|
|---|---|
|最终P50/P90、三轮范围、wall、CV|`results/FINAL_SUMMARY.json`|
|原始样本与最终动态验证|`results/final-v1/`|
|全轮次计数及审计|`results/AUDIT.json`，生成脚本`collect_results.py`|
|初始形状/cluster宽度/pool实验|`results/gpu*-benchmark.json`、`results/gpu*-verify.json`|
|direct调度与Indexer生成分数|`results/round2/`|
|直方图消融及失败|`results/round3/`|
|PyTorch CUDA trace|`results/trace-*.json`|
|硬件温度/频率/功耗|`results/gpu-monitor.csv`|
|原生版本/拓扑/二进制来源|`results/NATIVE_IDENTITY.json`、`results/NATIVE_MODULES.json`、`results/DEEPGEMM_IDENTITY.json`|
|各阶段完整源码|`snapshots/`；当前研究源码及完整改动见`experimental-kernels.patch`|
|完整性|`MANIFEST.json`；不包含manifest自身和解释器缓存|

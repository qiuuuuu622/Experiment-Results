# DSV4.1 Flash Indexer opt

H20 上 DeepSeek V4.1 Flash **FP8 index cache 函数／模块级优化**的代码与实验交付。分支：`DSV4.1-Flash-Indexer-opt`。本分支重新组织本次优化成果，`main` 保持原有内容。

基线：vLLM [`29be941d`](https://github.com/qiuuuuu622/vllm_dsv41_flash_opt/tree/29be941d8d5746cd98135bdd5e710eeebd8d7014)；参考：Ant SGLang [`6e6d5d9e`](https://github.com/antgroup/sglang/tree/6e6d5d9e13915adb01db0856b15d6908eaaa2c48)。根目录源码是优化工作区的逐文件快照，**不是声称这些修改已包含在基线 commit 中**。历史实验使用各自冻结快照和实际 native 依赖，不能用单一版本替代全部实验环境。

## 从哪里看

|内容|入口|
|---|---|
|新增：四卡视觉 RoPE / RMSNorm 融合|[报告与核心代码](research/vision-h20/README.md)、[原始性能数据](research/vision-h20/results/)|
|最新：四卡 Top-k 大行数调优|[完整报告与数据](research/topk-h20/REPORT.md)、[研究代码](research/topk-h20/tune_topk.py)、[CUDA 改动](research/topk-h20/experimental-kernels.patch)|
|优化前后机制、source＋4 consumer 链路|[机制详解](docs/MECHANISMS.md)|
|性能对照、正反序、收益和负结果|[性能汇总](docs/PERFORMANCE.md)|
|核心源码与对应实验|[代码地图](docs/CODE_MAP.md)、[完整模块](vllm_dsv41_indexer/)|
|相对基线的修改|[71 个新增／修改文件的 patch](patches/optimization-vs-29be941d.patch)|
|P0–P8 执行清单|[H20 优化方案书](H20优化方案书.md)|
|原始数据、日志、冻结源码、独立审阅|[完整实验索引](experiments/README.md)|
|复现、环境、验证范围|[复现说明](docs/REPRODUCE.md)、[限制与验收](docs/STATUS_AND_LIMITS.md)|
|逐文件版本与完整性|[源码来源](SOURCE_PROVENANCE.json)、[发布校验清单](SHA256SUMS.json)|

## 新增视觉组件结果（2026-11-07）

真实四卡H20、只加载视觉权重。DP4四图：原生159.52–160.57ms，RoPE+RMSNorm为135.77–136.02ms，总延迟降低14.89–15.29%；RMSNorm独立降低6.99–7.67%，同模式输出逐位一致。

RMSNorm整轮已按用户要求中止：四图正反8组完成，八图正式0组。**优化后的多图packed和视觉CUDA Graph对照测试仍待补；当前视觉Graph仅支持FLASH_ATTN，不支持FLASHINFER。** 详见[视觉报告](research/vision-h20/REPORT.md)，不要将已有原生smoke视为融合路径验收。

## 最新 Top-k 调优（2026-11-07）

h20-dsk 四卡分别执行独立同卡 A/B，完成 cluster-8/4/2、persistent/direct 调度和直方图消融。下面是最终三轮正反序的 **selector-only Graph P50**，输入由随机 FP8 Q/K 经实际 Indexer 打分生成；不是完整模型采样，也不是 TP4 延迟。

|形状，top-512|原有最佳稳定路径 μs|新结果 μs|结论|
|---|---:|---:|---|
|32×128K|21.446|16.502|cluster-2，约1.30×|
|32×256K|37.902|25.336|cluster-2，约1.50×|
|64×256K|61.958|47.859|cluster-2，约1.29×|
|128×256K|84.456|hist2为84.256|无稳定>5%收益，保留streaming|

8行仍优先cluster-8；64×128K仍保留streaming。690项配置检查、2760个执行阶段通过，49590个GPU原始样本及同数量wall样本已保存。新代码在 [`research/topk-h20/`](research/topk-h20/)，包含可配置cluster宽度/工作池及负结果实现；根目录已验证模块和默认dispatch保持原行为，尚未验证新selector在完整Indexer链中的净收益。详细统计、失败记录与来源边界见[报告](research/topk-h20/REPORT.md)。

## 已有模块结果

以下为不同实验的代表性 **CUDA Graph 模块耗时**，单位 μs；统计口径和完整证据见性能页。不同 workload 的倍数不能相乘。

|优化|形状与范围|优化前 → 后|结论|
|---|---|---:|---|
|候选 compact consumer|8 请求，256K index keys，含发布／top-k 映射|127.57 → 35.20|约 3.62×；单 consumer|
|source＋4 consumer|8 请求，256K，完整选择链|699.33 / 698.62 → 276.80 / 277.65|约 2.52×；正／反序|
|grouped verify|32 请求×6 行，64K，consumer|327.82 → 177.07|约 1.85×；要求可共享候选布局|
|四卡行分摊|32 请求，256K，source＋4 consumer|754.54 → 381.24|约 1.98×；输入已复制，含通信|
|逐 query prefill consumer|Q4096，256K 上下文|38000.39 / 38009.61 → 6210.16 / 6211.88|保守 6.117×；单卡、单 consumer|
|Ant 专用 prefill 调度适配|23 个模型形状＋独立复测|无稳定 >5% 新胜区|完成直接对照，当前 D128 版不加入自动选路|

**P0–P8 的函数／模块研究和验证已完成，包含补充 P5b/P5c。** 保留了未获收益的实现、失败尝试、旧版快照和范围更正。没有启动完整模型或 `vllm serve`，没有模型准确率、TTFT、TPOT、服务吞吐结论。初始 FP32 strict 门槛并非全部通过；优化相对 native 的误差与高精度 oracle 误差分别记录。

当前 prefill 模型形状采用 32 index heads、D128、top-512、最多 2048 个候选块、每块 8 个 index-key 位置。Q 表示 query 行数，**不能写成并发请求数**。TP4 保留每卡全部 index heads，按 query 行分摊是独立策略。

## 校验与许可证

```sh
/path/to/.venv/bin/python scripts/verify_release.py
```

该命令校验全部发布文件、98份原模块源码、最新Top-k研究的公开映射与研究源码，以及历史归档，不执行GPU实验。[发布验证记录](RELEASE_VALIDATION.json)说明本次实际检查。原有748个公开归档文件位于`experiments/`，内容字节保持不变；新Top-k实验单独位于`research/topk-h20/`，避免把不同版本的证据混用。历史报告保留当时状态，最新结论以本页和当前方案书为准。

新增浏览目录保留 SPDX 和上游归属，见 [LICENSE](LICENSE)、[原模块 NOTICE](vllm_dsv41_indexer/NOTICE.md)、[补充来源说明](docs/ATTRIBUTION.md)及[实验许可证](experiments/licenses/)。不包含模型权重、认证材料或本轮未验证的服务接入草稿。

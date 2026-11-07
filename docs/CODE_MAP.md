# 核心代码与证据地图

**最新Top-k实验代码在独立目录 [`research/topk-h20/`](../research/topk-h20/)**：

- [tune_topk.py](../research/topk-h20/tune_topk.py)：cluster宽度/pool、direct调度、直方图消融的JIT包装、同输入A/B、动态验证。
- [CUDA修改diff](../research/topk-h20/experimental-kernels.patch)：相对本分支93a4e82原Ant头文件的两文件变更。
- [修改后的topk_v2.cuh](../research/topk-h20/vllm_dsv41_indexer/third_party/ant/csrc/deepseek_v4/topk_v2.cuh)、[topk_impl.cuh](../research/topk-h20/vllm_dsv41_indexer/third_party/ant/include/sgl_kernel/deepseek_v4/topk_impl.cuh)：包含正负结果实现。
- [最终执行脚本](../research/topk-h20/scripts/06_final.sh)、[CPU汇总/审计](../research/topk-h20/collect_results.py)、[冻结源码](../research/topk-h20/snapshots/)。

根目录模块继续对应已有P0–P8验收版本；本次发布不把selector实验伪装成已接入默认dispatch的完整模块优化。

根目录导出 98 个模块、benchmark、测试及依赖文件，字节未改写。相对 `29be941d`，其中 64 个新增、7 个修改、27 个未修改依赖；[来源清单](../SOURCE_PROVENANCE.json)逐项列出。可直接阅读源码，不必先解压历史 tar 包。

|机制|源码入口|实验／验证入口|
|---|---|---|
|统一输入、full/source、分块|[module.py](../vllm_dsv41_indexer/module.py)、[contract.py](../vllm_dsv41_indexer/contract.py)、[reference.py](../vllm_dsv41_indexer/reference.py)|[verify_module.py](../benchmarks/kernels/dsv41_indexer/verify_module.py)、[verify_full.py](../benchmarks/kernels/dsv41_indexer/verify_full.py)|
|候选发布、去重、工作区|[prepare.py](../vllm_dsv41_indexer/prepare.py)、[candidate_publish.cuh](../vllm_dsv41_indexer/csrc/candidate_publish.cuh)、[workspace.py](../vllm_dsv41_indexer/workspace.py)|[verify_compact.py](../benchmarks/kernels/dsv41_indexer/verify_compact.py)、[benchmark.py](../benchmarks/kernels/dsv41_indexer/benchmark.py)|
|FP8 候选间接打分|[score.py](../vllm_dsv41_indexer/score.py)|[P2/P6 最终证据](../experiments/2026-11-07/p8-final-dispatch/final-p2-p6/)|
|full score、融合 block max|[full_score.py](../vllm_dsv41_indexer/full_score.py)|[benchmark_full.py](../benchmarks/kernels/dsv41_indexer/benchmark_full.py)、[P3 报告](../experiments/2026-11-07/p3-full-source/REPORT.md)|
|DeepGEMM 空行预取补丁|[补丁及说明](../vllm_dsv41_indexer/patches/README.md)|[最小复现](../benchmarks/kernels/dsv41_indexer/repro_zero_prefetch.py)|
|grouped verify 共享 K|[grouped.py](../vllm_dsv41_indexer/grouped.py)、[grouped_score.py](../vllm_dsv41_indexer/grouped_score.py)|[benchmark_grouped.py](../benchmarks/kernels/dsv41_indexer/benchmark_grouped.py)、[verify_grouped.py](../benchmarks/kernels/dsv41_indexer/verify_grouped.py)|
|既有 prefill 路径|[prefill.py](../vllm_dsv41_indexer/prefill.py)|[benchmark_prefill.py](../benchmarks/kernels/dsv41_indexer/benchmark_prefill.py)|
|专用 Ant TMA/WGMMA prefill|[prefill_keys_major_ant.py](../vllm_dsv41_indexer/prefill_keys_major_ant.py)|[benchmark_prefill_model.py](../benchmarks/kernels/dsv41_indexer/benchmark_prefill_model.py)、[verify_prefill_model.py](../benchmarks/kernels/dsv41_indexer/verify_prefill_model.py)|
|逐query compact／gathered|[perquery_prefill.py](../vllm_dsv41_indexer/perquery_prefill.py)|[benchmark_prefill_perquery.py](../benchmarks/kernels/dsv41_indexer/benchmark_prefill_perquery.py)、[verify_prefill_perquery.py](../benchmarks/kernels/dsv41_indexer/verify_prefill_perquery.py)|
|稳定 top-k 与编译模式|[topk.py](../vllm_dsv41_indexer/topk.py)、[native.py](../vllm_dsv41_indexer/native.py)、[vendored topk_v2](../vllm_dsv41_indexer/third_party/ant/csrc/deepseek_v4/topk_v2.cuh)|[benchmark_topk.py](../benchmarks/kernels/dsv41_indexer/benchmark_topk.py)、[verify_topk.py](../benchmarks/kernels/dsv41_indexer/verify_topk.py)|
|TP4 行所有权与通信|[row_shard.py](../vllm_dsv41_indexer/row_shard.py)|[benchmark_tp4.py](../benchmarks/kernels/dsv41_indexer/benchmark_tp4.py)、[verify_tp4.py](../benchmarks/kernels/dsv41_indexer/verify_tp4.py)|
|最终调度与准入|[dispatch.py](../vllm_dsv41_indexer/dispatch.py)、[perquery_dispatch.py](../vllm_dsv41_indexer/perquery_dispatch.py)|[test_dispatch.py](../tests/kernels/dsv41_indexer/test_dispatch.py)、[test_perquery_dispatch.py](../tests/kernels/dsv41_indexer/test_perquery_dispatch.py)、[最终 P5c 绑定](../experiments/2026-11-07/p5-perquery-review/bind-v2/)|

## 如何看 diff

[optimization-vs-29be941d.patch](../patches/optimization-vs-29be941d.patch)只包含上述三个目录相对基线的 71 个新增／修改文件，不包含完整 vLLM 源码、报告或环境安装。未改的模块／vendor 依赖已在根目录导出，但不会重复出现在 patch 中。

对照顺序建议：`module.py` 了解输入与调用关系 → `prepare.py/score.py/topk.py` 看候选路径 → `grouped_score.py` 看 verify 共享 → `prefill_keys_major_ant.py/perquery_prefill.py` 看两种不同 prefill 优化 → `row_shard.py` 看四卡通信 → `dispatch.py/perquery_dispatch.py` 看准入。

需要复核某个历史数据时，应打开该批次冻结包，不能默认最新浏览源码就是当时全部源码。`SOURCE_PROVENANCE.json` 与实验内部 source identity 的职责不同：前者证明本次发布字节，后者证明当时运行版本。

# 复现与发布校验

最新Top-k研究使用独立[研究目录](../research/topk-h20/REPORT.md)及各轮源码快照；最终命令见[scripts/06_final.sh](../research/topk-h20/scripts/06_final.sh)。复跑历史轮次需恢复匹配快照，不使用最终runner默认全variant列表；其中保留了显式可选、已知编译失败的hist4研究配置。正式最终脚本不选择hist4。公开数据中的GPU UUID使用一致伪名，归档tar的owner已标准化，映射见[PUBLIC_EXPORT.json](../research/topk-h20/PUBLIC_EXPORT.json)，数值样本未改变。

## 先检查公开文件

在此分支根目录运行：

```sh
/path/to/.venv/bin/python scripts/verify_release.py
```

校验三层来源：本分支所有公开文件、`SOURCE_PROVENANCE.json` 的浏览源码、原有 `experiments/` 归档的公开 hash 与 private/public 依赖映射。内部历史 SHA 可能对应脱敏前字节，应使用公开映射，不能把私有源 hash 直接套到公开 tar。

## 本地 CPU 调度测试

使用已有含 pytest 的独立环境，不需要 GPU：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /path/to/.venv/bin/python -m pytest \
  -c tests/kernels/dsv41_indexer/pytest.ini -p no:cacheprovider \
  tests/kernels/dsv41_indexer/test_dispatch.py \
  tests/kernels/dsv41_indexer/test_perquery_dispatch.py -q
```

这两个文件共 38 项，验证精确合同、证据过期、噪声/单序拒绝、绑定等 CPU 行为，不替代 GPU 正确性。其他测试和 GPU verifier 保留在目录中。

## GPU 环境应逐阶段恢复

本仓库是实验模块和证据仓库，不是完整可安装的 vLLM fork，不提供“装最新包即可复现”的承诺。需要 H20/SM90、CUDA、对应 PyTorch/Triton/TVM-FFI、vLLM wrappers、DeepGEMM headers 与实际二进制。已有阶段环境记录例如 [P4 最终环境](../experiments/2026-11-07/p8-grouped-final-v1/environment.json)；P5c 使用[运行前身份](../experiments/2026-11-07/p5c-perquery/final-v1/runtime-before.json)及运行后核验。不同阶段不能拼成虚假的统一环境。

`benchmarks/kernels/dsv41_indexer/baseline.json` 保存 7 个当前基线 wrapper 的 hash；`common.check_vllm()` 核对实际安装文件。wrapper 匹配仍不证明 native 二进制来自同一 commit。`environment.py` 单独记录 binary/header/package 身份。

步骤：

1. 选定阶段和结果文件，读取该批次报告、`run.sh` 或 `RUN_COMMANDS.md`。
2. 在新的实验目录解压该阶段冻结 source 包，恢复相应 native 依赖及独立 `.venv`。需要源码 overlay 时也可在干净 `29be941d` 上使用本分支 patch；历史结果仍以冻结包为准。
3. 执行 wrapper 和 native 身份检查。若使用 P3/P7 native_stable 混合零长度路径，显式启用[隔离 DeepGEMM 补丁](../vllm_dsv41_indexer/patches/README.md)，不改全局包。
4. 先跑对应 verifier，再按记录运行正反序 benchmark，输出到新目录；JIT、工作区与通信初始化在预热前完成。
5. 保存执行命令、输出、所有 raw samples、身份前后 hash。重新测量的结果单列，不覆盖本仓库冻结结果。

|阶段|可执行命令与冻结环境|
|---|---|
|P3 full/source/chain|[P3 报告](../experiments/2026-11-07/p3-full-source/REPORT.md)，其 `full-final-v1/` 和 `composition-v2/` 的 run.sh/source 包|
|P4 grouped|[最终目录](../experiments/2026-11-07/p8-grouped-final-v1/)，run.sh、planning.json、verify.json|
|P5b 专用 Ant|[模型矩阵命令](../experiments/2026-11-07/p5-ant-keys-major/model-aligned/RUN_COMMANDS.md)，23 点矩阵和 isolated-source|
|P5c 逐query|[正式目录](../experiments/2026-11-07/p5c-perquery/final-v1/)，[paired runner](../experiments/2026-11-07/p5c-perquery/paired-v1/run-group.sh)|
|P6/P8 最终 selector|[最终 P2/P6 目录](../experiments/2026-11-07/p8-final-dispatch/final-p2-p6/)|
|P7 TP4|[P7 报告的命令](../experiments/2026-11-07/p7-tp4-row-sharding/REPORT.md)，四个真实 rank，独立物理页|
|P5c 最终调度|[默认绑定 v2](../experiments/2026-11-07/p5-perquery-review/bind-v2/)、[审阅](../experiments/2026-11-07/p5-perquery-review/BENCHMARK_FAIRNESS.md)|

历史脚本中的 `/root/...` 是当时环境路径，公开脱敏的 `/workspace/...` 是占位路径。必须按新环境明确映射；本次整理不改原始脚本，不伪装成任意目录下一键可运行。调度准入表里的 artifact 引用也需要恢复原布局或基于新实验重新生成，不能直接把私有路径 manifest 当可移植运行配置。

## 相对基线应用 patch

在单独的、干净的优化源码 checkout 中：

```sh
git checkout 29be941d8d5746cd98135bdd5e710eeebd8d7014
git apply --check /path/to/Experiment-Results/patches/optimization-vs-29be941d.patch
git apply /path/to/Experiment-Results/patches/optimization-vs-29be941d.patch
```

patch 只恢复所导出的优化模块、benchmark 和测试变化；不会安装依赖或写入服务 hook。新的 GPU 测试和部署是后续操作，不属于此次分支归档。

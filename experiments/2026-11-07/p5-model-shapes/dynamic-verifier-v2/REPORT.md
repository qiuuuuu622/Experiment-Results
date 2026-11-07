# ModelPath 逐 query 候选与动态 CUDA Graph 验证

2026-11-07，`h20-4` 物理 GPU0，独立单卡正确性实验，无性能或 TP4 声明。退出码 0，5 个 fixture 全部通过。

- 共 186 个 phase × backend × execution 检查，2976 行 token top-k 与当前实际 logits 的稳定参考精确一致。
- 其中 1056 行 source 候选与逐行稳定 block-topk 参考精确一致；每个非空行的最新块都保留。
- 各 scorer 与 DeepGEMM 的最大绝对差 `1.430511474609375e-06`，逐项严格 `rtol=1e-3, atol=1e-4` 通过。独立 FP32 诊断由模型形状主基准负责；这里没有将 DeepGEMM 当成 FP32 oracle。
- 独立固定相同 logits（17 个重复 FP32 分数桶、无效尾部 NaN）的三种 selector 全行精确一致。
- source 生成的真实逐 query 候选原位更新后，动态 consumer 16/16 行集合发生变化，跨容量边界 fixture 12/16 行变化；没有把 request-shared fixture 当作逐 query 测试。

| fixture | 角色 | Q/请求 | 最终原始 token | ratio | 阶段数 |
|---|---|---|---|---:|---:|
| ratio2_dynamic | FULL | [8,8] | [1026,1025] | 2 | 9 |
| source_dynamic | SOURCE | [8,8] | [16400,16409] | 1 | 9 |
| consumer_dynamic | CONSUMER | [8,8] | [16400,16409] | 1 | 9 |
| candidate_cutoff | SOURCE | [16] | [16396]，prefix=16380 | 1 | 2 |
| candidate_cutoff | CONSUMER | [16] | [16396]，prefix=16380 | 1 | 2 |

每个阶段覆盖 DeepGEMM、现有 keys-major、Ant 专用 keys-major，以及 Eager 和已捕获 Graph。动态阶段为：base、candidate update、length update、物理页重排、整请求重排、全零+再次页重排、恢复、零 query 同分、再次恢复。请求重排交换完整等长组并重新归一 request ID；descriptor 地址/形状、分组布局和 chunk 计划固定。任意 ragged 分组改变需要重新规划，不在这个固定 Graph 契约内。

**动态准备的验证顺序**：每次更新输入后，先把所有路径已准备的 K 字节设为 FP8 NaN、scale 设 NaN、Ant prefix 清零、score 设 NaN、输出设 -2。随后对所有路径 replay 并保存输出和 source 候选。只有全部 Graph 结果已保存，才执行任何 Eager 或 oracle prepare/metadata。保存的结果再与当前输入重新计算的稳定参考比较。因此 Eager/oracle 不会提前刷新缓冲区，从而掩盖 Graph 漏捕获准备的错误。

ratio2 可见范围的 CPU 检查覆盖偶/奇 prefix 和零可见行；原模型 batch max=1024/1025/1026 时 Eager bypass 分别 true/true/false，capture 时均 false。这里仅核对原模型分支逻辑，不在本身无 host shortcut 的 ModelPath 中伪造 bypass。

版本与证据：

- model driver `f422bdd3a5593e031c16e9270f0e93fb3dd343402c4eacd13cfe087987c7ef49`
- Ant core `957557aa4ac9944073ae4e2c2a370775df446f93437fa82de821b3051d3184c9`
- verifier `fd406778241d6d9e53f58266d1e4cab9ffc2e4812f09ebe6378f8b8c246ad0a4`
- 冻结 archive `9a836a2490655d97498ec1aeb341d43c54c757bef350b026c3667472b179ff46`
- 原始逐项记录：`verify.json`；汇总：`summary.json`；运行日志：`verify.log`。
- `runtime-before.json`/`runtime-after.json` 对源码、实际 DeepGEMM headers/.so、vLLM wrapper/extension、包版本的前后身份核验全部一致。二进制 build commit 未知，没有将 wheel 名字冒称当前 HEAD 构建。
- 复现命令：`bash run.sh`，它切入冻结的 `runs/p5-model-verify-v2`，使用该目录 `.venv/bin/python` 并固定 `CUDA_VISIBLE_DEVICES=0`。完整源码在 `frozen/source.tar.gz`。

上一版 `dynamic-verifier-v1` 保留。它虽数值检查通过，但先 Eager/prepare 再验证 Graph，不能证明动态准备被 Graph 完整捕获；本项验收只采用 v2。GPU0 在退出与身份归档完成后已释放。

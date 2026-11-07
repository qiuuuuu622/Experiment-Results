# FP8 prefill 数值偏差的分层诊断

2026-11-07，`h20-4` GPU0。未修改生产/实验 core、FP8 Q/K 字节、scale/head-weight 语义或默认 dispatch；没有重跑性能。最终诊断及边界补充均退出 0，源码/native headers/.so/包版本前后身份核验一致，GPU 已释放。

## 已定位的主要原因

在本次代表性输入中，Ant 和 DeepGEMM 相对 FP32/FP64 的约 `2.15e-3` 分数偏差主要来自 **D128 FP8 WGMMA 的点积累加路径**。把该路径的每个 head 点积读出，再用相同 FP32 scale 和 head weights 重构 score，Ant 剩余差异最大仅 `9.536743e-7`，DeepGEMM 剩余差异最大 `1.430511e-6`。这排除了这些样本中候选映射、scale 广播或额外 BF16 舍入作为主因。

这里的 FP32/FP64 参考使用的是**同一组已经量化的 FP8 Q/K**。所以测到的是 FP8 tensor-core 计算路径与参考累加的差异，不是拿未量化模型输入比较后产生的量化误差。

范围为 initial Q2048/T2048、steady Q4096/T262144，均 ratio2 FULL。复现原主矩阵选定的首/中/末行，加上唯一 Ant/native top-k 不同的 row1401，共 9 个非空 query、911549 个可见 score、29169568 个 per-head dot。原主矩阵这两形状的抽样 strict failures 分别 **36、247**，本次重新复现。其余主矩阵形状没有逐 dot 分解，因此不把这 9 行宣称为全部 944 个抽样失败的逐项解释。

## 对照结果

严格检查仍使用 `rtol=1e-3, atol=1e-4`，未放宽。

| 点积/累加方式 | dot 对 FP64 最大差 | 重构 score 对 FP64 最大差 | 全可见 score strict failures |
|---|---:|---:|---:|
| D128 默认 FP8 WGMMA | 0.0294799805 | 0.00215471683 | 12227 |
| D128，max_num_imprecise_acc=32 | 0.00521850586 | 0.000366250279 | 244 |
| 4 个独立 D32 FP8 WGMMA，再用 FP32 相加 | 0.00521850586 | 0.000366250279 | 244 |
| FP32 dot + 同一顺序 FP32/FMA epilogue | 0（本次输入） | 0.00000133250728 | 0 |

生产 Ant score 在这批全可见位置上的 strict failures 为 12224，和重构 D128 的 12227 相差 3 个门槛附近点；差异量级与最后 FP32 epilogue 一致。该数是全行范围，不能与原主矩阵的均匀抽样数量直接相加或比较占比。

所有 K scales 实际为 0.5、1、2，乘 scale 相对 FP64 的舍入差实测为 0；顺序 FMA 与拆开乘加的 score 差最大 `9.536743e-7`。FP32 dot 与 FP64 dot 在本次 29169568 个元素上逐值相同；这只说明当前输入范围的结果，不能推广为任意 FP8/任意动态范围均精确。

一个直接例子：steady row4095、key91731、head1。

| 值 | 实测 |
|---|---:|
| FP32/FP64 dot | 42.4276123046875 |
| D128 默认 dot | 42.40625 |
| 4×D32 dot | 42.42626953125 |
| scale | 2 |
| head1 weight | 0.041839562356472015 |
| Ant / DeepGEMM / D128 重构 score | 5.387727737426758 |
| 4×D32 重构 score | 5.38962459564209 |
| FP64 score | 5.3898824542533745 |

原最大分差约 0.002155 在该位置重现，且生产值与 D128 dot 重构完全相同。最差位置的 FP8 字节、Q、scale、weights 已存为 `*-worst-inputs.pt`，可独立复现。

## PTX 与计算机制

保存的 `d128_default.ptx`、`d128_max_imprecise_32.ptx`、`four_d32_default.ptx` 都包含 `wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3`，没有把点积改成 FP16。

- 默认 D128 生成 4 条 K32 FP8 WGMMA，后 3 条继续使用同一 accumulator；诊断 PTX 中没有用于分段合并的 `add.f32`。
- `max_num_imprecise_acc=32` 生成 4 个独立 accumulator 的 FP8 WGMMA，再通过 48 条标量 `add.f32` 合并每个线程的 16 个输出。
- 手动 D32 kernel 每个 segment 执行一条 K32 FP8 WGMMA，四个 segment 在外部按 `((p0+p1)+p2)+p3` 做 FP32 相加。
- 在边界补充的 row1401/3072 共 8329120 个 dot 上，手动 4×D32 与 `max_num_imprecise_acc=32` 逐位相同。

安装版 Triton `warpgroup_mma` 文档明确说明 `max_num_imprecise_acc` 控制 FP8→FP32 中有限精度累加的跨度，原文保存为 `triton-warpgroup-mma-source.txt`。本次未推定硬件内部尾数位数、单条 K32 WGMMA 的具体舍入顺序。D32 仍有显著非零误差，不能借用 Ant FP4 输入下的精确累加假设。

## top-k 的两个具体边界

整体误差减小并不保证每一行 top-k 都更接近 FP64。

**row1401：Ant 与 native 的 1/512 差异。**

| 位置 | FP64 | native | Ant | 4×D32 |
|---:|---:|---:|---:|---:|
| 16519 | 2.421180113119 | 2.420727014542 | 2.420727014542 | 2.421102762222 |
| 58409 | 2.420074327086 | 2.420727014542 | 2.420727252960 | 2.419986486435 |

FP64 边界间隔是 0.001105786033。D128 误差使 native 的两个 logits **恰好同分**，稳定 tie 选择较小位置 16519；Ant 的最后 FP32 差异让 58409 高 1 ULP，改选该位置。这里不是“原 FP64 本来同分”，也不是原子顺序问题。4×D32 保留了 FP64 次序。

**row3072：D32 的反例。**

| 位置 | FP64 | native | 4×D32 |
|---:|---:|---:|---:|
| 25625 | 3.281124229289 | 3.280748844147 | 3.281137943268 |
| 29488 | 3.281234453338 | 3.280871868134 | 3.281109094620 |

FP64 gap 为 0.000110224049。native/D128 保留位置 29488；D32 翻转次序，重合率为 511/512。于是这 9 行中，native 对 FP64 的 top-k 全相同，Ant 只有 row1401 不同，D32 只有 row3072 不同。该有限样本不能作为模型准确率结论。

## 对 v1/v2 与后续准入的含义

此前 `p5-ant-keys-major/numerical-v1/diagnostic.json` 的 `failed_native_gate` 只代表没有通过原生路径贴近门槛，不证明 D32 算法或打分公式错误。当时 D32 反而更接近其 FP32 参考；本次独立分解支持“分段累加可降低总体误差”，同时明确它不是精确累加，也不保证逐行 top-k 单调改善。

当前 D128 默认实现更贴近 native FP8 数值，是保守的兼容性选择，不能因此宣称它通过了原定所有 FP32 strict 门槛。D32/32-step 累加仍需独立的全形状数值、边界选择、Graph 和含准备性能评估才能决定是否进入自动选择路径；本次没有更改 default，也没有给 D32 性能结论。

## 归档与复现

- 主诊断：`diagnostic.json`、`summary.json`、`initial_2k-partial.json`、`steady_256k-partial.json`、PTX、最差输入、日志。
- 边界补充：`boundary-supplement/diagnostic.json`，包含两位置所有 score 变体及 FP64 gap；该运行独立冻结，只增加诊断记录，没有改 core。
- 主诊断脚本 SHA `0332aef408d675af918b5802e5eeb48deb542b05f38d990dec825ae5ec3589ea`；补充脚本 SHA `c7250768f6df86ec5e9ffcf1cd33c48215060793ad93a15752b7afc9b1937b77`。
- 两次运行均有 `runtime-before.json`/`runtime-after.json`、`frozen/source.tar.gz` 和 `run.sh`。主诊断 archive SHA `b1ca0462866630aca2199258d17d5a50cce1fa37c671abe22cfad068ef9f1d37`。
- `attempt1/` 保留 FP8 masked-load 类型转换编译失败；修正为 uint8 载入后 bitcast 原 FP8 字节。`attempt2/` 保留 `max_num_imprecise_acc=0` 在当前 Gluon 不支持的编译错误（F32 accumulator 要求至少 32）；最终实际比较支持的 32，不把失败调用当成数值结果。

复现 `bash run.sh`，再运行 `bash boundary-supplement/run.sh`；二者均使用冻结目录 `.venv/bin/python`，仅 GPU0。

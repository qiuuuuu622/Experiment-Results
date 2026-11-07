# 图片 CPU 预处理组件实验

实现：`vllm_dsv41_vision/preprocess.py` 中 `ParallelImagePreprocessor`。

该组件持有一个有界线程池，调用当前 vLLM 原生 `image_processor` 和 `image_token_types`，以图片为单位并行。输出仍为 `patches`、`vit_grid`、`llm_grid`、`types`，保持输入顺序。它不缓存图片结果，不改变 resize、归一化或 BF16 舍入；空输入返回空字典，worker 异常传播给调用者，实例需要显式关闭或使用 `with`。组件不会修改全局 PyTorch 线程数。

本实验将 PyTorch intra-op 和 inter-op 设为 1，进程限制在最后 4 个可用逻辑 CPU；容器总 CPU quota 为 32 核，不是可见线程数所暗示的独占资源。它只测 CPU，未初始化 CUDA，不包含 H2D、视觉编码器、跨卡通信或服务调度。

验证包括：原生 processor 的完整张量输出逐位比较、图片正序与反序、解码后的图片与从 PNG 文件开始两种输入；单元测试覆盖线程完成顺序与输出顺序、变化输入、空输入、关闭和异常传播。线程池初始化及首次调用单独记录，热调用包含最终 batch 拼装。

`runs/preprocess-parallel-v1.json` 是早期 B4/B8 探索，已有数据全部保留。该版本在抽取可复用组件后中止，状态为 `aborted`，不能称为完整实验或最终组件收益。

最终组件实验使用 `runs/preprocess-parallel-v2.json`，选择 B8 作为代表形状，原生串行与 4 worker，包含/不包含 PNG 解码，正反顺序分别采样；每项 20 次预热、30 个单调用样本。CV 超过 10% 的项目追加 3 轮，保留所有样本。CPU 运行与 GPU smoke/profile 有重叠，GPU 正式采样等待此任务结束，因此不能把这组 CPU 数字与 GPU 中位数直接相加称为端到端实测。

最终 v2 已完成，8 组原生输出对照全部逐位一致，2 项组件契约测试通过。实测 CPU affinity 为 184–187，CUDA 未初始化。

| B8 CPU 路径 | 串行 P50 全轮区间 | 4 worker P50 全轮区间 | 串行 / 4 worker 轮数 |
|---|---:|---:|---:|
| 已解码图片 → patches/grid/types，含 batch 拼装 | 681.32–748.55 ms | 211.73–236.56 ms | 2 / 2 |
| PNG → patches/grid/types，含 batch 拼装 | 977.10–1034.54 ms | 305.04–328.45 ms | 5 / 5 |

裸预处理正反序主轮收益分别为 3.16×、3.22×；含 PNG 的主轮分别为 3.27×、3.18×。包含所有补测，含 PNG 路径的 P50 区间仍明显分离：取串行最低 P50 除以并行最高 P50 为 2.97×。这是轮次区间比较，不是统计置信区间。含 PNG 的 P90 全轮范围为串行 1029.89–1056.72 ms、4 worker 357.97–381.83 ms。

**异常及波动保留：**串行含 PNG 的正向主轮第 10 个样本（零基索引 9）墙钟为 42.2496 s，但进程 CPU 时间仅 1.0481 s，导致该轮 CV 为 307.4%。没有主动暂停该进程，确切原因未知；可能的外部调度或阻塞只是假设。该样本保留在原始 JSON，未剔除。追加三轮 P50 为 1012.43、977.10、982.95 ms，CV 为 4.0%、4.4%、4.3%。

4 worker 含 PNG 的反向主轮 CV 为 12.9%，追加三轮 P50 为 305.04、319.47、328.45 ms，CV 为 9.0%、9.1%、12.7%；最后一轮仍有明显抖动，不能宣称低抖动稳定延迟，但各轮中位数收益都远大于测量噪声。共保存 14 轮、420 个正式单调用样本，另有每轮 20 次预热。

整个进程峰值 RSS 约 1809.9 MiB，包含导入依赖、预解码图片、正确性参考和线程池工作区，不是优化新增内存。4 worker 的进程 CPU 时间 / 墙钟比约 2.42–2.65 核，收益来自独立图片的 CPU 并行，不是 GPU 加速或图片结果缓存。

复现：

```bash
cd /root/dsv41-vision-20261107
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python benchmarks/kernels/dsv41_vision/bench_preprocess_parallel.py \
  --source-root source --images images \
  --output runs/preprocess-parallel-v2.json --batch-sizes 8 --workers 0 4
```

源码冻结：`runs/preprocess-parallel-source-v2.py`、`runs/preprocess-adapter-source-v2.py`。原生预处理源码、模型配置和 8 张输入图片 SHA256 记录在 JSON 中。权重、完整模型和 tokenizer 均不参与此 CPU 实验。

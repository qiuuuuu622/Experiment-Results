# RMSNorm 融合独立数值审查

状态：源码归约顺序已核对；v2 的19项GPU测试通过，覆盖FP32 mean位模式与BF16最终输出。
真实DP4 B4/B8四路径模块对照也全部逐字节一致；这里不代表未测试输入的数学证明。
不能以 RMSNorm 数学公式等价，代替浮点计算顺序一致。

环境为 `torch 2.13.0+cu130`，`torch.version.git_version`
`cf30153c4c131c8164ee7798e5022d810682e2cb`。本地保存实际安装的
`Reduce.cuh`、`SharedReduceOps.h`、`CUDAMathCompat.h`；哈希见 `provenance.json`。
现有真实 trace 的 mean kernel 模板为 `ReduceOp<float, MeanOps<float,float,float,float>, ..., 4, 4>`。

## 原路径必须保留的舍入点

`DeepseekV4RMSNorm.forward` 的顺序为：

1. BF16 输入转换 FP32；`x.square()` 输出独立 FP32 tensor。
2. FP32 mean 沿最后一维 1024 按下面的 CUDA 树归约。
3. `mean + eps`，eps 为传入值转换至 FP32；当前默认 `1e-6`。
4. 对相加结果调用 CUDA float `::rsqrt`。
5. `normalized = x_float * inv_rms`，先舍入成 FP32。
6. `weighted = weight * normalized`，再舍入成 FP32，最后转 BF16。

不能把第 5、6 步改成 `x * (weight * inv_rms)`，也不能提前把 normalized 变 BF16。
实际视觉 wrapper 将 norm 参数也转换成模型 BF16；weight 加载 BF16→FP32 转换应保持原数值。
应关闭 FMA，并检查编译器没有对累加/乘法重新结合。

## 连续 FP32 `[N,1024]`、N≥16 的 mean 顺序

从安装的 `Reduce.cuh::setReduceConfig` 推导：input vectorization=4，
block width=32、height=16，各 warp 独立处理一行，没有跨 warp/CTA 的行归约。
`square()` 的输出为新分配、按行对齐的 FP32 tensor，因此向量化头部 shift=0。
此句仅对本节的最后一维连续布局成立，不能推广到任意 stride。

对 lane `l∈[0,31]` 和 accumulator `i∈[0,3]`，初始化 `a_i=+0.0f`，
按 `j=0…7` 顺序累加 `a_i = FP32(a_i + square[4*l + 128*j + i])`。
随后严格计算 `s = ((a_0 + a_1) + a_2) + a_3`。
warp 归约使用 shuffle-down offsets **16、8、4、2、1**；lane0 得到整行 sum。
`MeanOps.project` 最后乘 `1/1024`，不是每个输入先除 1024。

`tl.sum` 只有在确认其 lane0 加法树匹配后才适用，不能直接替换为任意 1024 元素归约。
若编译器重结合破坏顺序，可用 `add.rn.f32` / `mul.rn.f32` 固定每个操作。

N<16 时原生 block width 会变化，例如 N=1 为256、N=4为128、N=8为64，
各 lane 数据分配和 shared-memory 归约也随之变化。首版应对这些形状保留原路径；
本轮实际图像最少5980 patch行，均满足N≥16。空行、非1024维、其他dtype及布局应明确处理或回退。

**独立审查发现的接口限制：**转置 dense BF16 `[N,1024]` 可以有 stride `[1,N]`；
其 `.float().square()` 保持列主布局，原生 mean 采用另一条归约路径。
因此应限制 `x.stride(-1)==1`，允许带padding的行stride；其他布局必须拒绝或走完整原生表达式。
现有 `torch_mean` 后端若先产生行连续 square，也改变了上述原始布局，不能作为任意stride的等价fallback。
可选 stats 缓冲应要求 `[N]`、FP32、同CUDA device、contiguous，避免按连续地址写入错误缓冲。
以上两项接口检查已在v2落实，并加入拒绝非法输入与FP32 strided weight的GPU测试。

## rsqrt：不能替换成 `1/sqrt`

已按安装版本 git SHA 核对官方 `UnaryOpsKernel.cu`：实数分支经 wrapper 调用
CUDA `::rsqrt(float)`；复数分支才采用 `1/sqrt`，不适用于本轮。
[精确版本源码](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/UnaryOpsKernel.cu#L76)。

Triton `tl.rsqrt` 与 CUDA libdevice rsqrt 的结果是否相同需要逐位检查；不能仅凭名称断言。
若不同，应测试 `tl.extra.cuda.libdevice.rsqrt`，并保留原生作为基准。
`mean+1e-6` 的正常正输入不落在 subnormal 区域，但极端值、overflow、NaN等仍需明确限定。

## 必须单独检查的中间结果

建议小样本导出原/新：square、FP32 mean、mean+eps、inv_rms、normalized、weighted、最终BF16。
对输入分布覆盖真实视觉激活、随机、零、常量、尺度差异、正负值与靠近最终BF16舍入边界的样本。
先定位第一个分歧阶段，再修实现；不自动放宽阈值。
最终以当前加载的真实视觉权重，在DP4 B4/B8检查 eager、RoPE、RMSNorm、两者组合的逐字节输出。
新的 RMSNorm profile/数据使用独立 `rmsnorm-*` 目录，不覆盖已完成 v10 结果。

验证证据：`../rmsnorm-correctness-v2.log`（19 passed）、`../rmsnorm-v2-source/`，
以及 `../rmsnorm-module-correctness-v1.json` 的三项优化相对eager和组合相对RoPE结果。
本轮复刻的归约顺序针对上面记录的PyTorch版本、FP32 mean1024和H20；其他版本/形状不自动继承精确性保证。

# TP4 与 DP4 原生视觉路径数值诊断

日期：2026-11-07。只分析视觉组件；不加载语言模型。此处不是通过精度验收的声明。

## 权重路径审查

远端实际加载的 `vllm/model_executor/layers/linear.py`、`vllm/model_executor/parameter.py` 与当前本地仓库文件 SHA256 完全一致：

- linear.py：`7a9b90937865fa35d955ffc2017f23d3997bc4fba954e2667870b1b4854a38e0`
- parameter.py：`1a4bbd7400fd1ba79e8e1e8d19666cd22881ec83c35e6e01b3fdc78faee23171`

无量化视觉模块的布局审查结果：

| 参数 | 原始逻辑布局 | TP4 每 rank 应加载 |
|---|---|---|
| Attention wqkv weight/bias | Q/K/V 各1024输出 | 三段各取对应rank的256输出，再按Q/K/V顺序拼接 |
| Attention wo weight | 1024×1024 | 输入维取256列 |
| MLP w1 weight | gate/up 各2816输出 | 两段各取704输出，再拼接 |
| MLP w2 weight | 1024×2816 | 输入维取704列 |
| Aligner w1 weight/bias | 普通GELU前置线性层，5120输出 | 输出维取1280行；不能按gate/up二分 |
| Aligner w2 weight | 5120×5120 | 输入维取1280列 |
| Row-parallel bias、norm、patch embedding、分隔符 | 复制 | 完整复制；row-parallel bias只在rank0加入一次后归约 |

当前驱动调用注册的 `weight_loader`，不手工把整块fused矩阵切四份。v8增加了独立的逻辑分块/切片期望检查，再与实际加载参数按字节比较。DP4及TP4的266个视觉参数均通过。代码审查及此检查未发现漏切、顺序错误或重复加载bias的证据。

该检查排除了本次参数装载的明显布局错误，不证明TP4与DP4的浮点执行顺序相同。原生row-parallel先产生各rank的BF16局部矩阵乘输出，再归约；完整权重路径执行一次完整矩阵乘，计算顺序不同。它是需要验证的误差来源，不能仅凭源码推断解释全部差异。

## 层级中间结果

使用同一第一张图片，原生 eager、真实四rank，各模式捕获rank0中间结果。DP4的B1图片归属rank0，由完整视觉权重计算；TP4由四rank权重分片计算。以下指标使用CPU FP64统计，模型推理仍为原BF16。

| 捕获位置 | 相对 L2 | Cosine | 最大绝对误差 |
|---|---:|---:|---:|
| patch_embed | 0 | 1 | 0 |
| block 0 | 0.00400182 | 0.99999199 | 0.015625 |
| block 1 | 0.00537969 | 0.99998554 | 0.015625 |
| block 4 | 0.00947070 | 0.99995515 | 0.0390625 |
| block 8 | 0.01209437 | 0.99992686 | 0.0625 |
| block 16 | 0.08132021 | 0.99669022 | 160.597656 |
| block 24 | 0.08125743 | 0.99669528 | 160.921875 |
| block 31 | 0.09969904 | 0.99503971 | 1876 |
| final norm | 0.09602769 | 0.99538693 | 0.1271057 |
| aligner | 0.05157331 | 0.99867070 | 0.1174316 |

输入后的patch embedding逐位一致，首次已捕获差异发生在block0；block8至16之间出现明显放大。因此差异不是只由最后的all-gather或span拼装引入。block16误差最大的5个patch占误差平方和约29.1%，block31约42.8%，说明总相对误差受少数位置明显影响。

上述为累计漂移，不是每个模块在完全相同输入上的局部误差。稀疏捕获点不能确定block8至16中哪一层、哪一个子操作造成放大。不能仅凭cosine较高接受，也不能据此声称模型任务准确率降低了某一比例。

当前原门槛未通过，不能将TP4切换DP4描述为已验证数值等价的优化。RoPE融合另行与同模式、同路径的原生实现按位比较，二者的验收不能混淆。

## 后续定位边界

若继续定位跨模式差异，应固定相同输入，分别检查：block0 norm1、Q/K/V分片重组、逐head attention、wo局部输出及归约、MLP gate/up重组和w2归约；同时加密block9–16捕获。需要比较NCCL与custom-all-reduce时应显式标注通信后端，保留原路径结果。未完成这些检查前不把差异归因于某个通信实现或某一舍入点。

## 证据与复现

远端：`/root/dsv41-vision-20261107/runs/layerdump-v8/`。

- `data-b1.pt`、`weights-b1.pt`：实际中间结果。
- `drift-analysis.json`：逐层指标、最大误差位置、输入文件及分析器SHA256。
- `drift-analysis.log`：文本输出。
- `drift-analysis-fp32-metric-invalid.*`：初版CPU FP32 cosine统计出现大于1的非法结果，保留且明确作废；最终采用FP64重新统计，不改变任何模型计算。
- `diag-dp4-v8.json`、`diag-tp4-v8.json`：驱动配置、266参数切片校验。

```bash
cd /root/dsv41-vision-20261107
CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/kernels/dsv41_vision/analyze_layer_drift.py \
  --reference runs/layerdump-v8/data-b1.pt \
  --actual runs/layerdump-v8/weights-b1.pt \
  --output runs/layerdump-v8/drift-analysis.json
```

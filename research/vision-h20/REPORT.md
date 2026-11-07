# 四卡 H20：DeepSeek-V4.1 Flash 视觉组件优化

日期：2026-11-07。仅加载 `/model` 的视觉部分，在 `h20-4` GPU0–3 做真实四进程组件实验；不加载语言模型、不启动服务。

**RoPE/CPU阶段已完成。RMSNorm整轮按用户要求中止，状态为 `aborted`：4图正式正反序8组已完成，8图正式0组。下面单独报告已经完成的4图数据，不将8图短测当正式结果。中止后未继续测试，也未重新运行最终整包回归。**

## 核心代码

- `vllm_dsv41_vision/rotary.py`：将每次Q或K的RoPE操作链分别从9个kernel融合为1个；保留FP32分步运算、关闭FMA、最后舍入BF16。不是Q/K联合单kernel。
- `vllm_dsv41_vision/rmsnorm.py`：复刻当前PyTorch CUDA的FP32 mean1024归约顺序，保留eps、rsqrt及两次乘法的舍入点；每次RMSNorm从8个kernel融合为1个，不融合残差加法。仅支持最后一维stride=1，N<16采用原生mean回退；stats缓冲显式检查。
- `vllm_dsv41_vision/preprocess.py`：有序线程池并行调用原生图片转换，保持patches/grid/types及图片顺序，不缓存图片输出。

基线为vLLM fork `29be941d8d5746cd98135bdd5e710eeebd8d7014`。相关Python源码已逐项核对；CUDA扩展来自记录了hash的vLLM 0.32.0 wheel，不宣称整个vLLM从当前HEAD重新编译。

## 已完成：RoPE 四卡正式结果（v10）

正反序各20次预热、30样本×10调用；表中为两轮P50范围，收益由同顺序配对计算。图片数是每次组件调用的图片数量，不是语言模型请求并发。

| 模式 | 图片数 | 原路径P50 ms | RoPE后P50 ms | 延迟下降 |
|---|---:|---:|---:|---:|
| 图像DP4 | 4 | 159.65–160.73 | 146.94–147.62 | 7.96–8.16% |
| 图像DP4 | 8 | 316.60–316.89 | 291.63–291.69 | 7.88–7.95% |
| 权重TP4 | 4 | 244.32–244.44 | 229.66–229.87 | 5.96–6.00% |
| 权重TP4 | 8 | 461.12–462.70 | 433.21–433.29 | 6.05–6.36% |

原始数据：`results/formal-dp4-v10.json`、`results/formal-tp4-v10.json`。只在固定模式内比较；TP4与DP4原生输出存在跨模式数值差异，模式切换不是本次已验收的等价优化。

## 用户中止前已完成：DP4 四图 RMSNorm 正式对照

独立的新一轮四路径对照，沿用20预热、30样本×10调用。没有与旧v10样本混算。

| 路径 | 正序P50 / P90 ms | 反序P50 / P90 ms |
|---|---:|---:|
| 原路径 | 160.568 / 161.617 | 159.521 / 159.894 |
| 仅RoPE | 147.415 / 148.019 | 147.141 / 148.202 |
| 仅RMSNorm | 148.255 / 148.731 | 148.367 / 148.989 |
| RoPE + RMSNorm | 136.021 / 136.683 | 135.766 / 136.079 |

| 同顺序配对比较 | P50延迟下降 | 加速比 |
|---|---:|---:|
| RMSNorm相对原路径 | 6.99–7.67% | 1.075–1.083× |
| 在RoPE之上增加RMSNorm | 7.729–7.730% | 1.08377× |
| 两者组合相对原路径 | **14.89–15.29%** | **1.175–1.180×** |

这8组GPU样本CV为0.17–1.87%。整轮文件 `results/rmsnorm-formal-dp4-v2.json` 保持 `aborted`；原始中断快照另存，不能将其中历史 `running` 字段解释为实验仍在运行。**8图RMSNorm只有先前smoke/profile，没有正式性能结论。**

## CPU与数值验证

8图CPU完整输入准备主轮约3.16–3.27×：包含PNG解码时，原生串行P50全轮范围977.10–1034.54 ms，4 worker为305.04–328.45 ms。它增加CPU并行度，不能与GPU收益相乘推算端到端加速。异常长尾原样保留，补测末轮CV仍有12.7%；见 `results/preprocess-parallel-v2.json`。

- RMSNorm v2函数/接口GPU测试：**19 passed**；FP32 mean与最终BF16输出按位检查。
- 真实权重模块：DP4 **1/2/3/4/7/8图**，三项优化相对原路径、组合相对RoPE全部逐字节一致。见 `results/rmsnorm-module-correctness-v2.json`。
- CPU原生转换与并行输出精确一致；RoPE正式结果同模式逐位一致。
- 没有测试模型任务准确率、服务吞吐、TTFT或TPOT；没有通过降低图片分辨率或改成FP8取得收益。

## 计时与归档边界

GPU计时包含视觉编码器、Aligner、span整理及通信，输入patches已在各卡；排除CPU预处理、首次H2D、权重加载、JIT及通信组初始化。每调用取四rank GPU最大值，再在样本内平均；主机墙钟另按样本取rank最大值。原始数组及主机统计保留在JSON中。显存峰值字段仅rank0，不是四卡最大峰值或纯工作区大小。

精简profile证据支持实际工作减少：B8每rank RoPE为1152→128 kernels、约22–26→1.2–1.4 ms；RMSNorm为1040→130 kernels、21.11–24.62→2.44–2.80 ms。profile资源时间不能冒充模块延迟；正式收益仅采用上面的独立采样。

本目录发布核心代码、测试/复现入口、冻结源码身份及已有小型结果。**不上传权重、层级激活dump或大体积原始trace。**完整证据仍保留于：

- 本地：`vllm_dsv41_flash_opt/results/vision/20261107-phone-screenshots/`，根目录为 `/Users/a1-6/Documents/ChatGPT/答辩学习/`。
- 远端：`h20-4:/root/dsv41-vision-20261107/`；`runs/frozen-v10`、`runs/frozen-rmsnorm-v2`保存实测源码，`runs/profile-*`保存逐rank trace。

复现应使用对应freeze及其原始命令记录；不要把当前源码与早期原始样本混配。本次发布不恢复已中止的测试。

## 待完成

- 8图RMSNorm四路径正式正反序对照仍未完成。
- **优化后的多图packed路径，以及视觉CUDA Graph的对照测试尚缺。** 已有原生packed/manager的smoke与profile，不等于RoPE/RMSNorm优化路径经过正式验收。
- 本轮H20视觉CUDA Graph可用后端为 **`FLASH_ATTN`（FlashAttention）**；`FLASHINFER`不支持，`TORCH_SDPA`也不支持。这是视觉图路径的后端约束，不表示整个vLLM不支持FlashInfer。
- 用户中止后没有再做最终整包回归。上述待办只记录，不继续执行。

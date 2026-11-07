# DP4 RMSNorm 四路径 profile 审查

32份trace覆盖B4/B8、eager / RoPE / RMSNorm / 两者组合、各四rank。
源代码归因链为 kernel correlation → CUDA runtime → 最内层
`vision.py:74` 或 `vllm_dsv41_vision/rmsnorm.py`，不是按通用kernel名猜测。
以下是单次profile的实际GPU资源时间，不是正式模块性能样本。

| batch | 每rank原RMSNorm kernel数 | 融合后kernel数 | 原RMSNorm GPU ms范围 | 融合后GPU ms范围 |
|---|---:|---:|---:|---:|
| B4 | 520 | 65 | 11.729–12.527 | 1.346–1.427 |
| B8 | 1040 | 130 | 21.110–24.619 | 2.436–2.797 |

每个RMSNorm调用从8个kernel降到1个，残差加法未融合。
B8组合路径的RMSNorm为2.437–2.811 ms，RoPE为1.198–1.357 ms，
二者独立下降；没有把RoPE收益重复计入RMSNorm。
这套实现仍以attention/GEMM为主，RMSNorm剩余约1%的GPU工作。
建议先固定已验证的4warps实现进行正式采样，后续若调优warps应另留独立对照，
不把profile时间当正式收益，不通过修改归约顺序换取速度。

真实DP4 B4/B8中，各优化相对eager、组合相对RoPE均为逐字节一致；
接口补强v2增加最后维stride限制和stats缓冲验证，19项GPU单测通过。
profile使用v1冻结源码，正式性能应以补强后的v2冻结和独立正反序采样为准。

原始证据：

- `profile-rmsnorm-v1/source-attribution.json`：每rank每操作归因与实际GPU duration。
- 同目录32个原始trace、`triage-*.txt` 单rank kernel/overlap/fuse三表。
- 同目录 `analysis-b*-*.json`：GPU时间线union、通信和CPU scope数据。
- `rmsnorm-module-correctness-v1.json`：真实权重完整模块精确性。
- `rmsnorm-correctness-v2.log` / `rmsnorm-v2-source/`：接口与函数测试。
- `rmsnorm-numerical-review/`：PyTorch归约树、舍入点、精确版本源码、BSD许可与hash。

正式审核使用独立 `audit_rmsnorm_results.py`，共16measurement（DP4×2batch×4路径×正反），
不修改旧v10的审核与原始结果；明确CPU预处理和初始H2D排除，内存只报rank0。

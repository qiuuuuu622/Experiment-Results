# H20 四卡视觉组件：RoPE / RMSNorm 融合

[性能、范围与待办](REPORT.md) · [核心实现](vllm_dsv41_vision/) · [原始性能数据](results/) · [测试与复现入口](benchmarks/kernels/dsv41_vision/)

仅视觉权重、真实四进程组件实验。RoPE/CPU阶段完成；RMSNorm整轮按用户要求中止（aborted），4图正反8组完成，8图正式0组。RMSNorm独立降低6.99–7.67%，RoPE+RMSNorm组合降低14.89–15.29%，同模式输出逐位一致。

待完成：优化后的多图packed、视觉CUDA Graph对照测试，以及8图RMSNorm正式对照。当前视觉Graph仅支持FLASH_ATTN（FlashAttention），不支持FLASHINFER。已有原生路径smoke不算融合路径验收。

`frozen/`保存实际测量源码；每份结果保留原命令和来源。编号脚本中的远端目录为原实验布局，复现时应将对应freeze放回runs/。本次不上传模型权重、激活dump或大体积原始trace；完整证据仍保留原机器与本地实验目录。

按用户要求直接提交推送，未在发布阶段重新运行实验或完整发布检查。

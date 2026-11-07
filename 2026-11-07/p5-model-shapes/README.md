# P5 模型形状审阅与动态验证交付

当前验收证据是 **最终 23 点 CPU 形状审阅 + dynamic-verifier-v2**。本目录未执行模型加载或性能主矩阵，不提供端到端模型准确率或 TP4 性能结论。

- [REVIEW.md](REVIEW.md)：H32/D128/top512、层角色、ratio2 可见 key 向下取整、逐 query 候选、短上下文 bypass、内存预算及主基准公平语义审阅。
- [shape-audit.json](shape-audit.json)：主基准 SHA `f422bdd3…`，23 个架构形状 CPU 算术检查通过；initial32K/full 明确排除。旧 21 点初审保存在 `shape-audit.initial21.json`，未覆盖。
- [dynamic-verifier-v2/REPORT.md](dynamic-verifier-v2/REPORT.md)：独立 h20-4 GPU0，5 fixture、186 项、2976 行稳定 token top-k、1056 行稳定 source 候选通过。每阶段 poison 准备缓冲后，先保存全部 Graph 结果，再运行 Eager/oracle；最大 scorer 分差 1.43e-6。
- `dynamic-verifier-v1/`：保留原始数据。该版先 Eager/oracle prepare 再 Graph，会暖好准备缓冲，不能证明动态准备完整捕获。该用途由 v2 替代，不能将 v1 作为此项验收证据。
- `MANIFEST.json`：文件 hash；v2 的 `runtime-before.json`/`runtime-after.json` 记录实际源码、native headers/.so 和包版本身份。前后核验一致，退出码 0，GPU0 已释放。
- [numerical-diagnostic/REPORT.md](numerical-diagnostic/REPORT.md)：另行分解主矩阵的共有 FP32 偏差。9 行、29169568 个 dot 证明代表样本的主要偏差来自连续 D128 FP8 WGMMA；4×D32 总体更接近 FP64，但仍有严格失败与一行 top-k 反例。没有放宽门槛或修改默认路径，不能把 native 贴近度冒称所有 FP32 strict 检查通过。

独立 verifier 源码：`benchmarks/kernels/dsv41_indexer/verify_prefill_model.py`。可复现命令和冻结 archive 见 v2 报告及 `run.sh`。

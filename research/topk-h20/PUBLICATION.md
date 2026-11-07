# 公开归档说明

本目录保留本轮实验的报告、驱动、CUDA 变体、失败记录、逐样本数据及各轮冻结源码。根目录模块的默认调度没有被研究变体替换。

- `REPORT.md` 是实验完成时的报告；`SOURCE_FINAL.json` 标识最终研究源码，历史运行使用各自快照。
- `ORIGINAL_PRIVATE_MANIFEST.json` 保留原私有证据的 hash。公开数据仅对 GPU UUID、容器名及 tar owner/扩展元数据作脱敏，数值样本和源码不变。`PUBLIC_EXPORT.json` 将原 hash 映射到公开文件，`MANIFEST.json` 校验本目录当前公开字节。
- `results/SNAPSHOT_CHECK.json` 是原始检查记录；重新检查公开快照可从仓库根目录运行 `.venv/bin/python scripts/verify_topk_publication.py`。无需 GPU。完整分支校验使用 `scripts/verify_release.py`。
- 最终性能复现命令见 `scripts/06_final.sh`，须使用报告记录的依赖和 H20 环境，并调整脚本中的实验目录与 Python 路径。早期阶段需要对应冻结源码，不能用最终驱动替代全部历史版本。
- `stream_hist4` 编译失败已保留；复现最终结果应使用最终脚本显式列出的变体，不运行驱动的全变体默认列表。
- 四卡是独立并行实验，结果不是 TP4 延迟；本轮仅测 selector，未测新 selector 接入完整 Indexer 后的收益。

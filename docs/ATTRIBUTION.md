# 源码来源与许可证补充

本分支保留源码原有 SPDX、版权和 Apache-2.0 文本，不重写历史 NOTICE。基线 vLLM 为 `29be941d8d5746cd98135bdd5e710eeebd8d7014`；Ant 参考固定为 `6e6d5d9e13915adb01db0856b15d6908eaaa2c48`。

- compact scorer 借鉴 Ant `sm90_fp4_indexer_wgmma.py`，改为 vLLM FP8 cache、物理页表、独立 K scale、FP32 head weights；不是复制 FP4 缓存格式。
- 专用 prefill 适配对应 Ant [`sm90_fp8_prefill_keys_major.py`](https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/python/sglang/kernels/ops/attention/dsv4/sm90_fp8_prefill_keys_major.py)。原文件、机制审阅、许可和原始 hash 随[Ant 审阅目录](../experiments/2026-11-07/p5-ant-review/)归档。适配移植调度机制，保留本项目 FP8 数值合同。
- `third_party/ant/` 的 topk 及传递依赖保留原始文件和[来源 hash](../vllm_dsv41_indexer/third_party/ant/sources.json)，包含已有 vLLM/FlashInfer 归属。
- [旧 NOTICE](../vllm_dsv41_indexer/NOTICE.md)描述初期 compact-only 编译。后续 P6 由 `native.load_kernels(mode)` 扩展 full/streaming/cluster，不能把旧句子理解为最新模块仅支持 compact；vendored 头文件仍未修改。
- `candidate_publish.cuh` 是本项目的 CUB 适配，受 source 发布／consumer 复用机制启发，不冒称未修改的 Ant kernel。

代码快照身份和修改清单见 [SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json)。公开历史包曾有路径、设备 UUID 伪名和 tar owner 元数据脱敏，映射在 [PUBLIC_EXPORT.json](../experiments/PUBLIC_EXPORT.json)及 [PUBLIC_DEPENDENCIES.json](../experiments/PUBLIC_DEPENDENCIES.json)。本次新增浏览源码未作字节变换，归档文件也未重写。

# P5c 逐 query prefill compact：已完成函数级研究

最终结论见 [REPORT.md](REPORT.md)。保留 FP8 cache、FP32 scale/head weights，补齐真实 `[Q,2048]` 候选的 direct 与 gathered batched WGMMA（非 DeepGEMM）两路线，显式限制 score≤1GiB、gather≤256MiB。

**最终性能准入依据是 [paired-v1/AUDIT.json](paired-v1/AUDIT.json)**：4长形状同一inputs/candidates/四路径对象正反测试，所有输入bytehash前后不变。Graph保守净加速为64K 1.473×、256K 6.117×、1M尾chunk 22.044×、ragged 1.308×。gather均慢于direct；初始2K/8K/32K及16K保留full路径。该范围是单卡单consumer模块，source生成另测，不是TP4、服务或真实模型准确率。

- [最终动态验证](final-v1/verify.json)：42个poison→Graph replay→clone→oracle/Eager阶段，含source_unique、空候选、partial/cap、重排、非2幂scale与恢复。
- [原8点完整采样](final-v1/SUMMARY.json)：3,360记录、3,561轮、106,830样本，所有分项及失败原型保留。
- [同对象补充](paired-v1/AUDIT.json)：8点source三次canonical集合稳定；4长形状64条pipeline、1,920样本；Q/W/cache/页表/req/length/sourceQW及candidate raw/canonical哈希前后一致。
- [最终源包](final-v1/frozen/source.tar.gz)：`b4c1a3dbda6778ea48d226b44ff77809dce604a50fd63b9b3aac15184e83e41b`，包含许可证；589项原生文件及源码before/after身份一致。
- [完整文件清单](INVENTORY.json)：所有文件哈希及正式/原型证据入口。

原8点中5个shape的候选raw SHA正反不同，因为source top-k列表不承诺排列顺序。历史raw未存canonical SHA，不能事后重建或回填；新paired是独立同对象证明。原raw保持不变。FP32严格诊断并非全通过，native有相同量级偏差；新路径对native门槛通过，详见报告。

本阶段没有修改自动dispatch；独立P8的正式准入与非provisional实际binding由其报告单独确认，不将本研究完成等同于全Goal结束。

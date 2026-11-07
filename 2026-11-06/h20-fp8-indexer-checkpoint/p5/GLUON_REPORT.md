# P5 Gluon 独立补充实验

日期：2026-11-06。此补充实验在原32条checkpoint之后运行，代码版本独立保存；不改写原source.tgz及原始结果。GPU1已释放。

结论：Gluon直接页表读取、按组复用key tile和source块max融合均通过验证；四个代表形状未超过已有最快路径，因此保留为显式实验backend，不进入自动dispatch。

|模式|query行|实际index keys|原生μs|已有最快路径|已有最快μs|Gluonμs|
|---|---:|---:|---:|---|---:|---:|
|full|64|65536|631.09|keys_major_pipeline|151.43|182.77|
|source|64|65536|712.64|keys_major_pipeline|166.87|213.07|
|full|16|262144|2487.13|indirect_pipeline|194.41|209.31|
|full|4096|4096|374.98|deepgemm_pipeline|363.60|415.63|

表内为正反序CUDA Graph P50平均值，包含准备、打分、token top-k，source还包含候选输出。原生仍为当前wrapper的DeepGEMM完整模块适配；二进制构建commit未知。Gluon不预先gather K，多个query共享直接从页表读出的FP8 key tile；source块max在打分中融合，不再单独扫描score。

正确性：27组全部通过，包括原24组和3组Gluon full/source。benchmark额外4形状×正反序=8条记录。采样仍为20次预热、30样本、每样本10调用；CV>10%的项追加3轮，原始样本和墙钟保留在JSON。

复现：

```sh
ssh h20-dsk
cd /root/dsv41-indexer-port
bash scripts/06_prefill_gluon.sh
```

仅使用物理GPU1。运行前需保证该GPU已获分配。脚本执行verify及gluon suite正反序，不启动模型。

证据：gluon-{forward,reverse}.json/.log、gluon-verify.json/.log、gluon-source.tgz、gluon-source-sha256.txt。source snapshot包含prefill.py、grouped_score.py、benchmark/verify、专项test及复现脚本；共享依赖哈希单列。

`gluon-source.tgz` SHA256：`4f3c2819ac7e5e0ced9dd529aff5585a393077204e46aa5a0f6fca2ede1fd57e`。

限制：此Gluon实现只支持full/source，consumer仍使用已验证的row-tiled、keys-major、indirect路径。性能是单卡prefill函数/模块实验，不能当作TP4整体收益；P7负责四卡。4096×4K的最佳优化与原生差异不足5%，仍建议保留原生。

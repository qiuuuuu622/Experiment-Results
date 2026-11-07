# DSV4.1 Flash 模型对齐的 prefill scorer 实验

本目录只测试函数/模块；使用真实模型配置决定维度、层角色、压缩与因果关系，Q/K/权重为固定随机输入。不是完整模型推理或某个部署的默认 batch，也不是 TP4 端到端性能。

## 配置与调用来源

- 固定源码 HEAD：`29be941d8d5746cd98135bdd5e710eeebd8d7014`。
- 实际保存的模型配置：`architecture/model-config.json`，SHA256 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`。
- `architecture/deepseek_v41_attention.py` 1300–1325 行：32 index heads、维度128、top512，Q/weights 为 ReplicatedLinear；TP4 不把 index heads 改成8。
- 同文件 1258–1268 行与配置：index source 层2/8/14 为 ratio2 full；20层为 ratio1 candidate source；24/28/32/36层为 ratio1 candidate consumer。40个主层不是40次独立indexer打分。
- `architecture/mla_indexer.py` 533–552 行：`start_pos=uncompressed_seq_len-query_len`，逐 query 可见 key 数是 `floor((prefix+j+1)/ratio)`。ratio2 连续两个 query 可以具有相同 key 数；不能每行直接加一个压缩key。
- `architecture/sparse_attn_indexer.py` 691–709 行：候选是 `[query,2048]`，每块8个 index-key 位置。source按query发布，consumer按query取用；不假定同请求的query候选集合相同。
- `architecture/deepseek_v41_attention.py` 1483行：当整个call最大逻辑长度压缩后<=512，且当前不在CUDA Graph capture时，绕过indexer打分。不是每个query分别决定。主矩阵所有call最大key数>512，因此初始prefill的前几百行仍打分，未跳过。

## 23个主形状

每个角色包含初始Q2048/8192（prefix=0）、steady Q512/16K、Q2048/64K、Q4096/256K、尾chunk Q256/1M，以及ragged Q[512,128]/最终逻辑长度[65535,16384]。source与consumer另外增加初始Q32768/prefix0，越过2048候选块截断。FULL ratio2不重复这一额外长初始点。

`MATRIX.json` 逐项列出原token prefix/context、实际 index-key 数、精确因果有效score数、完整矩形score容量、1GiB arena、query分块、额外source block-max空间。最大完整矩形4GiB，实际单路径score arena<=1GiB；全部chunk的执行与共享key准备都计入完整成本。不存在把超预算shape静默略过的问题。

## 三方公平范围

1. 当前安装vLLM的 DeepGEMM FP8 scorer；Python wrapper hash核验当前HEAD，二进制构建commit未知。
2. 现有 `PrefillIndexer(backend="keys_major")`。
3. 本轮专用Ant persistent keys-major FP8适配，32query全head shared Q、Q/K TMA、K三级缓冲、head双缓冲、visible-tile prefix调度；core SHA见运行身份文件。

三方保持FP8 Q/K、独立FP32 K scale与含Q scale的FP32 head weights，使用完全相同的key准备和Ant稳定top-k。consumer三方均为 **full score + 原生逐query候选mask + 相同selector**；这不是compact consumer性能结论。source同时输出token top512及候选blocks。

consumer候选由一次独立固定seed的source Q/weights产生，复用相同index key cache；候选输入生成在consumer计时外。测的是单consumer模块，没有把它写成source+四consumer组合收益。候选可能逐行不同，也可能在候选足够覆盖全部位置时相同；每个结果记录实际有效候选数及相邻集合不同数。

## 成本与正确性

完整pipeline包括每次key准备、Ant prefix metadata、全部分块score、consumer mask的flags初始化/分配/写入，以及共同selector；source包括候选生成。另报裸score、共同准备、prefix、consumer mask。静态TMA descriptor规划排除于热路径，但记录每path初始化墙钟/内存，并另外测每次重建descriptor的Eager完整成本。CUDA Graph采用固定buffer和descriptor，prefix仍每次在GPU重算。

全部可见score按rtol=1e-3/atol=1e-4对native回归；所有token/source候选变化按实际native边界与分数误差解释；所有行检查合法性、唯一性、数量及source latest block。每arena chunk的首/中/尾行对相同logits做稳定选择精确参考。每这些行抽取<=3072均匀key和双方top512边界作FP32诊断；FP32严格门槛失败数与原生自身失败数均保留，不能当作FP32逐元素完全通过。更完整的小规模全行稳定参考、动态Graph与page更新在独立验证报告中。

每个variant预热20次，30个样本，每样本10次；正反顺序各跑一次。CV>10%的variant追加3轮，所有原始轮次保留。测量helper同时记录GPU时间、主机墙钟、P50/P90、显存峰值。峰值范围是同进程三方工作区共存，不将其当作单路径独占峰值。

旧23个 `small/large/mixed` 形状仅为synthetic筛选。尤其旧ratio2 consumer与request-shared候选不代表真实模型consumer主路径，不能替代本目录结果。

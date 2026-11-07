# P5c per-query compact prefill 独立 CPU 审阅

2026-11-07。只读审阅 `perquery_prefill.py` 与 `verify_prefill_perquery.py`，未修改 owner 文件、未运行 GPU。版本 hash 和审阅源码快照见 `SOURCE.json`。这些结论是静态审阅，不等于新 kernel 已获正确性或性能准入。

## 数据与映射

`PerQueryPrefillIndexer:152–159` 接受 `[Q,2048]`，而非 `[requests,2048]`。`:209–212` 每次 forward 根据当前逐行 visible length 发布候选；默认路径去重，`source_unique=True` 只适用于 source 输出有效项唯一的前提。`CandidateState.publish` 先排除无效/未来块再排序，partial block 只有逻辑尾部一块，因而紧凑列的线性映射成立。

`_gather_perquery:35–54` 的行是 chunk 局部行，REQS/BLOCKS/COUNTS 参数已经从对应全局 start 切片。紧凑列 c 对应 `blocks[row,c//8]*8+c%8`；用当前 `requests[row]` 查本地 table，随后读取该逻辑位置的 FP8 payload 和 FP32 K scale。没有跨请求复用候选，也没有把 block ID 当成物理页 ID。

`:230–267` 的 Q descriptor、weights、requests、blocks、counts 都使用同一 chunk 起止；`:270–279` selector 读取该 chunk 的 scores/counts 并用对应的逐行 blocks 映射到原逻辑位置。直接路径复用 P2 `_candidate_wgmma_kernel`，聚集路径复用相同 FP8 D128 WGMMA、K scale 乘法、ReLU 和 `_row_head_sum`。未见 BF16 舍入或 FP4 exact-accumulation 假设引入。

## 分块与容量

`:165–176` 宽度为 `min(16384, align64(max_keys))`。direct 受 FP32 score arena 限制；gathered 还受 `132*width` 每行的 staging 限制（128 B FP8 payload +4 B scale）。在 width=16384 时：

- 1 GiB score budget 可容纳 16384 行；不是把全部 Q×16384 scores 永久保存。
- 256 MiB gather budget 可容纳 **124 行**，不是 128 行；后一个数会漏算 FP32 scale。
- 多个 chunk 顺序复用同一 staging 和 score arena，`forward:282–288` 在下个 chunk 覆盖前完成映射选择，输出写到全局行区间。
- q descriptor 以每个 chunk `[local_rows*32,128]` 创建，与 WGMMA 内 `row*32` 对应。最后不足 capacity 的 chunk 通过独立 descriptor 限定行数。

聚集 kernel 将固定 staging 范围全部写回（包括无效尾部的零 payload/1 scale），因此该开销必须纳入完整 pipeline；不能只报告后续 batched GEMM。CandidateState 的全 Q 候选缓冲与输入、输出，以及 Graph 私有池不包含在这两个 arena budget 中，峰值应另测。

## 动态 Graph 契约

该实现的 chunks 只按行数/预算分割，**不按 request ownership 预分组**。kernel 每次从 GPU 读取 requests/table/length-normalized candidates，因此与旧 PrefillIndexer 不同，它可以在固定 shape/storage 下更新任意合法 request ID，不必保持同请求相邻。前提是 Q/cache/table/candidate tensor 的 storage、shape、stride、device 不变，length 仍不超过预设 max_keys，physical IDs 仍有效。

source_unique 的值属于捕获时固定路径；若输入由唯一 source 候选切换为任意外部含重复列表，必须选择去重版本或重新捕获，不能只修改 tensor 内容后继续假设唯一。

暂未发现静态可证明的映射或分块错误，但新 gathered kernel 的 WGMMA shared-memory/layout 与动态图路径必须以 GPU 结果确认。

## 当前 verifier 的待补证据

以下行号对应 SOURCE.json 所冻结的初审版本，owner 后续可能修订。

1. **唯一 source 快路径未覆盖。** `verify_prefill_perquery.py:130–138` 全部使用默认 `source_unique=False`；应该同时验证 source 真实输出的 unique 快发布和外部重复/无效候选慢发布。两者对同一有效集合的 blocks/counts/top-k 应精确相同。
2. **每步 Eager 与 Graph 对照缺失。** `:147–152` 只在初始化 Eager forward，`:183–186` 动态阶段只 Graph。建议每次新输入先保存所有 Graph 输出，再运行 Eager 对照与 oracle，避免 oracle 预先暖准备。
3. **准备/score 尾部 poison 尚未覆盖。** 当前 oracle 在 `audit_outputs:29–32` 重做 gather/score，但不重做 selector，通常能捕获 stale top-k；仍建议在 Graph 前把 state.blocks/counts、staging keys/scales、scores/output 置 sentinel，Graph 后先保存输出，再做任何 oracle gather。尤其 count=0 与 partial-tail 的 NaN scores 不得被 selector 读取。
4. **空候选但非空上下文。** 当前 changed case 只把第一行 length 清零，没有显式构造 `length>0 && 全部 candidates=-1`；需要一行全无效或全部越界列表，counts=0，输出全部 -1。
5. **明确命中 candidate/block 尾部。** keys=20003 是长上下文随机候选，但最后不满 8 的块不保证被采到。补强制选最后一块、1/7/8/9 可见位置，以及 prefix16380/Q16 跨 2048 块容量的 source→consumer 流程；检查 logical top-k 不重复、不越界且属于逐 query 有效候选。
6. **固定 logits selector 证据需准确命名。** `audit_outputs:71–74` 是“每种 scorer 实际输出 logits 对自己的稳定 selector 参考精确一致”；`:98` 称 fixed_logits_selection 可以被误读为两路径注入同一 logits 的独立 selector 测试。已有 direct/gather 分数逐位检查能提供支持，但报告最好直写实际验证内容，或增加共同 tie-heavy logits 测试。

建议上述补测保持少量行，强制 `max_chunk_rows=3`、gather budget=2 行，以覆盖两路径不同 chunk 划分。无需为正确性复制完整性能矩阵。

## 公平性能对照

真实 perquery consumer 的 baseline 为：相同 Q/W、相同物理 cache/table、同一 source 候选矩阵下，full scorer + perquery mask + 同稳定 selector。新 direct/gathered pipeline 必须含候选发布/去重、gather（适用时）、score、映射 top-k。source 已提供唯一候选时，三条路径均按同一 source_unique 条件测量。

聚集版本是自定义 batched WGMMA GEMM，不是 DeepGEMM；源码模块注释已正确说明。各 query 的候选独立时不能借用 request-shared compact 的 key 准备复用收益。score-only、gather-only、publication-only 可作分项，但最终调度依据完整模块成本；无收益的 gathered 形状保留研究结果即可。

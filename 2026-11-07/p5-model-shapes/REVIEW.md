# DSV4.1 Flash prefill：模型形状与候选语义独立审阅

日期：2026-11-07。范围：CPU 源码与形状核对；没有运行模型或 GPU，没有修改 P5 模块/主基准。基线 HEAD：`29be941d8d5746cd98135bdd5e710eeebd8d7014`。

配置来自 `results/indexer/20261101/model-config.json`，SHA256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`。所有引用文件 hash 见 `shape-audit.json`，审阅的主基准副本见 `benchmark_prefill_model.reviewed.py`。

## 结论

真实模型的 prefill 可见 key 数是 **floor((prefix+j+1)/ratio)**。候选缓冲区是 **[query,2048]**，每个 query 单独选块，随后该 query 的候选跨 consumer 层复用。它不是同一请求内所有 query 共用一组候选。

已有 P5 的 request-shared compact 数据以及此前 23 个合成 shape/grid 调度数据仍是有效的函数实验，但不能直接表述成真实模型 prefill consumer 的表现。新的 `benchmark_prefill_model.py` 已用正确的 floor 长度和逐 query 候选 mask；CPU 审阅未发现这两项核心语义错误。性能和数值结果仍需实际验证。

## 模型参数、层角色和位置单位

`model-config.json:106–127` 给出 H=32、D=128、token top-k=512、candidate block top-k=2048、block size=8。主 attention 的 64 heads 不是 indexer heads。`attention.py:1300–1326` 读取这些参数，Q 和 head-weight 投影都是 ReplicatedLinear；不能把 TP4 当成每卡 8 个 index heads。

| Indexer 层（0-based） | ratio | 行为 | index K 所有者 |
|---|---:|---|---|
| 2、8、14 | 2 | FULL 打分 + token top-512 | 对应层自己 |
| 20 | 1 | SOURCE：全长打分 + token top-512 + 每 query 候选块 | 20 |
| 24、28、32、36 | 1 | CONSUMER：各自 Q/W 打分，在该 query 的 source 候选内 token top-512 | 20 |

证据：`attention.py:303–348` 确定 source 层拓扑；`:431–481` 仅在 index source 创建 indexer，并区分 candidate_write 与 uses_candidates。其他 backbone 层可能复用已生成的 top-k，不应统计成新一次 indexer。位置均以 **index-key 坐标**表示；ratio=2 的 key 不是原始 token 一一映射。块 b 覆盖 index-key `[8b,8b+8)`，尾部由逐 query 可见长度截断。

## 可见范围必须向下取整

记某请求这一 chunk 前已有 P 个原始 token，本次 Q 个原始 query，j 从 0 开始。则：

```
T = P + Q
request_closed_keys = floor(T / r)
visible_keys[j] = floor((P + j + 1) / r)
```

query 行数仍是 Q，不除以 ratio。`vllm/v1/attention/backends/mla/indexer.py:533–552` 明确使用 `start_pos = uncompressed_seq_len - query_len`、`global_ctx = start_pos+1+offset`、`global_ctx // COMPRESS_RATIO`；`:546–565` 把 request-local 长度加到 packed K 的 row_start。PCP/DCP 分片不在此次范围。

例：r=2、P=0、Q=5 时是 `[0,1,1,2,2]`；P=1 时是 `[1,1,2,2,3]`。ceil 会把尚未闭合的压缩 key 提前暴露。偶数/奇数 prefix 都应测。初始 prefill 的前一行可以合法地没有任何可见 key。

旧 `benchmarks/kernels/dsv41_indexer/inputs.py:65–67` 的长度是 `length-rows_per_request+1+j`，`verify_prefill.py:17–34` 只是附加 ratio 标签。因此旧 ratio2 case 不等于连续原始 token prefill 的真实长度日程；它仍可作为合成 index-key 窗口测试。新主基准 `model_inputs:35` 已覆盖为正确公式。

## 候选是 per-query，块号相对于请求

- `vllm/models/deepseek_v41/nvidia/model.py:710–720` 分配 `[max_num_batched_tokens,2048]`。
- `sparse_attn_indexer.py:691–709` 取 `candidate_blocks[chunk.token_start:chunk.token_end]`，source 逐行写，consumer 逐行 mask。
- `candidate_blocks.py:31–49` 每 logits row 独立求 8-key block max；block ID 相对该行 request 的 packed start，而非物理页号或全 batch packed 列号。最新可见块 `(visible-1)//8` 强制赋 +inf，必须入选；空行没有最新块。
- `candidate_blocks.py:99–101` 候选寻址是 `row*stride_row+col*stride_col`。`ROW_REPEAT` 只复用边界，不代表候选行共享。
- 原 HEAD `candidate_blocks.py:179–180` 用 torch.topk，边界同分不保证按小位置排序。实验的确定性 selector 应对所有 scorer 一致使用；不要把这个 tie-policy 差异算成 FP8 打分误差。

当前 `PrefillIndexer` 的 compact 特例与此不同：`prefill.py:224–231` 接受 `[requests,2048]`，`:432–435` 给同请求所有行 expand 同一候选。别名 workaround 可表达每行候选，但把每个 query 当请求会重复 key 准备/descriptor、破坏 32-query key tile 复用；必须计入这些成本和显存，不得称作无成本的一次/request 准备。

## Ant 专用 kernel 的公平模型对照

原 Ant 专用 kernel 及适配的 `AntKeysMajorScores` 没有候选输入；适配接口见 `prefill_keys_major_ant.py:424–447`。优先采用：所有 scorer 对相同真实可见长度做 full score，再使用同一 `[query,2048]` mask 和同一稳定 token selector。统一计入 key gather、Ant prefix metadata、mask、top-k；score-only 单独报告，并对所有 scorer 一致排除这些步骤。

source 模式还必须计入逐行候选生成。consumer 独立计时时可预先给定 source 候选，但须明确不含 source；完整组合另测 source 一次 + 4 组不同 Q/W 的 consumer，逐 query 候选只发布一次。模型真实层的 Q/W 不同，固定随机 Q/W 是函数实验，不是模型实际激活。

备选的 per-query compact gather/间接读取是不同实现实验；它应使用同样的逐 query 候选，并报告准备/聚集/映射/打分/selector 全成本。不能用 request-shared compact 速度代表这种路径。

## 对新主基准的逐行审阅

审阅副本 SHA256 `d19eda72c3c5e04d912f5a2549720c06bd1a7b197394b3bd990fb38c391ee1c3`。以下行号对应该副本；后续 owner 修订需另记版本。

- `:26–38`：真实 floor 长度、ragged request ID 覆盖正确。`:15–23` 的现有 7 shape × 3 role 共 21 shape CPU 计算通过。
- `:69–78`：候选按 query 行 mask，正确；`:81–87` pipeline 包含 prepare、Ant metadata 和统一 selector。
- `:208–210`：source 只 roll head weights，Q 与 consumer 相同。作为合成 source fixture 可以保留，报告应明确；完整四 consumer 组合另用不同 Q 和 W。
- `:312`：`perquery distinct` 表述过强。候选是 perquery-indexed，未必每两行集合都不同；尤其可见 key ≤16384 时 2048 个块覆盖全部可见位置。建议输出不同候选集合数量/行间变化指标，并在 64K/256K 等真正稀疏 case 检查不被错误 broadcast。
- `:318`：已明确 source 生成在 consumer 计时外，正确；该值是单 consumer，不是 source+4consumer 链路。
- `:305` 的 `allocated_scores=rows*width` 是逻辑未分块矩阵元素数，不是实际常驻 allocation；实际 `:307` score_arena_bytes 正确。报告应将前者叫 logical/full score elements。
- `:31` 当前通过 `make_case` 造初始等长组后覆盖请求。现有 ragged `[512,128]` 的总行数可被 2 整除，因此有效；未来任意 ragged 输入不能默认满足这个 fixture 前提，应避免沿用隐含整数整除条件。

## 主矩阵与最低补测

主矩阵以原始 token 为单位：initial Q=2048/8192；steady Q=512@T16K、2048@T64K、4096@T256K；tail Q=256@T1M；ragged Q=[512,128],T=[65535,16384]。三个 role 使用相应 ratio，不能同一 K 标签混用。另加 initial32K 的 source/consumer，可覆盖初始 prefill 中从全覆盖到真正候选裁剪的过渡。1M Q128 可作尾部形状补充，不必展开大笛卡尔积。

最低正确性补测：

1. ratio2 prefix 偶/奇、可见 0/1、逻辑最终长度 1024/1025/1026。
2. source/consumer `P=16380,Q=16`，逐行可见 16381..16396，跨越 2048 个候选块容量；所有行 source 输出与稳定参考一致，并验证 newest block 强制保留。
3. 各 query 有不同候选集合，原位改变候选后 CUDA Graph replay；不只测整请求 broadcast 候选。
4. 多请求不同 prefix、长度变化、全零→恢复、物理页表置换。固定 graph 的请求分组布局必须稳定；逻辑请求重排可通过交换整个请求的数据/表后重新归一 ID 验证。分组行数或 owner 布局改变需重建 plan，不可修改 `inputs.requests` 后假装旧静态 chunks 自动更新。
5. 相同固定 logits 的 selector 与稳定参考精确一致；不同 scorer 的差异分别检查分数误差与 top-k 边界。因果/无效尾部用 NaN sentinel 检查。

短上下文 bypass 的真实条件是 `attention.py:1482–1504`：**batch max_seq_len // ratio ≤512 且当前不是 CUDA stream capture**。因此 ratio2 的 batch max 1024、1025 走该 host 分支；1026 不走。混合 batch 由最大上下文决定，不是每行独立 bypass。捕获时不走该 host 分支。模块实验可以验证条件和 fill-all 输出，但不能给本身没有该分支的 ModelPath 虚构性能收益。

## 显存预算（单请求；MiB）

FP8 Q 每行 4096 B，FP32 head weights 每行 128 B；每个 K 有 128 B payload +4 B scale；top-k 输出每行 2048 B；候选每行 8192 B。FP32 score 需 4QK B。计划 1 GiB 是 **每个 score arena 上限**，并非整个进程总显存上限。

| 场景 | ratio | K | 未分块 score | score arena | chunks | 每份 cache/预备 K+scale |
|---|---:|---:|---:|---:|---:|---:|
| initial2K | 1 / 2 | 2048 / 1024 | 16 / 8 | 16 / 8 | 1 / 1 | 0.258 / 0.129 |
| initial8K | 1 / 2 | 8192 / 4096 | 256 / 128 | 256 / 128 | 1 / 1 | 1.031 / 0.516 |
| steady512@16K | 1 / 2 | 16384 / 8192 | 32 / 16 | 32 / 16 | 1 / 1 | 2.063 / 1.031 |
| steady2048@64K | 1 / 2 | 65536 / 32768 | 512 / 256 | 512 / 256 | 1 / 1 | 8.25 / 4.125 |
| steady4096@256K | 1 / 2 | 262144 / 131072 | 4096 / 2048 | 1024 / 1024 | 4 / 2 | 33 / 16.5 |
| tail256@1M | 1 / 2 | 1048576 / 524288 | 1024 / 512 | 1024 / 512 | 1 / 1 | 132 / 66 |
| initial32K（补测） | 1 | 32768 | 4096 | 1024 | 4 | 4.125 |

三种 scorer 同时驻留时自定义两个 arena + DeepGEMM 临时 logits 至少可能达到约 3 GiB；source maxima 再占每 arena 的 1/8，候选、mask flags、参考比较临时张量、CUDA Graph 私有池另计。initial32K 的 Q 为 128 MiB、候选为 256 MiB；tail1M 每份预备 K 为 132 MiB。建议每个独立进程至少预留 16 GiB 测试余量，在每个阶段记录 actual allocated/reserved 和 graph-pool 峰值；这只是保守运行预算，不是实测峰值。若资源不足按 score arena 减块并显式记配置，不静默漏测。

复现 CPU 算术/配置核对：

```sh
.venv/bin/python results/indexer/20261107-resume/p5-model-shapes/validate_shapes.py
```

原审阅版本输出 `passed, shapes=21`。`shape-audit.json` 保留每个请求的前后 8 个可见长度、有效 score 数、arena 大小和全部源文件 hash。后续主矩阵加 2 个 initial32K role 时应更新归档，不能复用旧 21shape 状态宣称 GPU 完成。

## 最终版本补记

主基准已冻结为 SHA `f422bdd3a5593e031c16e9270f0e93fb3dd343402c4eacd13cfe087987c7ef49`。CPU 形状核对重新运行，**23 点通过**；新增 initial32K/source、initial32K/consumer，initial32K/full 明确排除。现 `shape-audit.json` 与 `benchmark_prefill_model.reviewed.py` 对应该最终版本；原 21 点证据保留在 `shape-audit.initial21.json`、`benchmark_prefill_model.reviewed-initial21.py`。上文逐行问题引用属于原版本，不应套用到新行号。

独立动态正确性验证已另完成，见 `dynamic-verifier-v2/REPORT.md`：5 fixture、186 项、2976 行稳定 token top-k、1056 行稳定 source 候选通过。Graph 在 poison 准备缓冲后先保存全部结果，随后才允许 Eager/oracle 更新，覆盖真实逐 query 候选更新、全零恢复、物理页与请求重排。最大 scorer 差异为 1.43e-6；这项 GPU 验证只有单卡正确性结论，不替代主性能矩阵或真实模型准确率测试。

# 性能结果与比较口径

所有数字来自本分支保留的实验记录，本次整理没有重跑 GPU。单位默认 **μs，CUDA Graph P50**。正／反表示 A/B 执行顺序；表中统计方法按阶段说明，不能混成一个统一统计样本。所有路径使用 warm repeated calls，没有 L2 flush。

每策略至少 20 次预热、30 个样本、每样本 10 次调用，正反各一次；CV>10% 追加三轮并保留全部样本。完整 JSON 有 P90、CV、墙钟与显存峰值。初始化、JIT、预规划、Graph capture 通常不在 forward 计时内；动态准备、候选发布、全部 chunk、选择及需要的通信按对应模块合同计入。

## Decode：source 与四个 consumer

下表来自 [P3 最终报告与原始文件](../experiments/2026-11-07/p3-full-source/REPORT.md)。chain 是 source 的 token/block 选择、发布及四个不同 Q/weights consumer 的总和，不含模型 attention/MoE。

|请求×实际 index keys|原生 chain，正／反|Hybrid source＋compact consumer，正／反|大致收益|
|---|---:|---:|---:|
|1×65,536|151.89 / 151.84|71.66 / 71.42|2.12×|
|8×262,144|699.33 / 698.62|276.80 / 277.65|2.52×|
|128×131,072|4149.73 / 4202.67|1684.98 / 1697.74|2.47×|
|ratio2，8×131,072|468.50 / 468.28|208.40 / 209.02|2.24×；通用 API 压力形状|

ratio2 consumer 是通用契约测试，不是真实模型 consumer 角色。source 本身的代表正序：1×64K 为 72.76→25.31；8×256K 为 186.64→149.14；128×128K 为 864.48→778.44。source 仍输出 token top-512 和 block top-2048。

短窗口独立组合复测说明不能统一启用 compact：

|形状|原生完整链|Hybrid＋compact|Hybrid＋full consumer|选择|
|---|---:|---:|---:|---|
|32×4K|96.92 / 97.01|147.92 / 147.99|67.75 / 67.64|full|
|64×16K|336.88 / 337.14|570.64 / 573.45|284.22 / 284.26|full|

[P8 最终代码绑定](../experiments/2026-11-07/p8-final-dispatch/REPORT.md)中的**单 consumer**代表正序为 8 请求×256K，127.57→35.20，约 3.62×，包含候选发布和融合映射。它不是完整 chain，也不能与上表 source 时间直接相加得到另一批次的 chain。

## Grouped verify

来自 [最终 grouped 回归](../experiments/2026-11-07/p8-grouped-final-v1/REPORT.md)，为正反 P50 的均值摘要。具体准入检查两序全部原始记录。

|请求×每请求行×index keys|逐行 consumer|共享 consumer|原生 full|grouped full|
|---|---:|---:|---:|---:|
|4×2×64K|35.01|23.97|36.69|47.22|
|4×6×64K|66.79|34.70|71.65|82.29|
|32×6×64K|327.82|177.07|580.60|511.68|
|1×6×256K|29.63|21.77|65.67|106.96|

consumer 共享有效不表示 full 一定获益。共享候选资格由预规划合同保证；通用 host 重规划的 0.38–17.6 ms 成本另有原始测量，不隐藏在 GPU 数字之外。不同候选布局不得直接套用本表。

## 真实 TP4：端到端模块通信成本

来自 [P7 最终报告](../experiments/2026-11-07/p7-tp4-row-sharding/REPORT.md)，正反及追加轮样本合并 P50。每次调用取最慢 rank，包含 metadata/table 准备、行整理、kernel、结果 gather／恢复；root 模式另含输入 scatter。

|模块，256K index keys|请求数|四卡复制|行分摊：输入已复制|行分摊：仅 root 输入|
|---|---:|---:|---:|---:|
|full|1|60.17|77.98|103.32|
|full|8|142.36|101.00|135.14|
|full|32|395.64|180.75|250.66|
|source＋4 consumer|1|123.78|159.67|225.91|
|source＋4 consumer|8|294.37|202.56|299.60|
|source＋4 consumer|32|754.54|381.24|594.14|

32 请求 full 约 2.19×、chain 约 1.98×，但输入仅在 root 时收益变小。单请求分摊更慢。

64K compact consumer 的最终 v3：8 请求×1 行为 50.30→60.82（root 99.64）；32 请求×6 行为 362.02→141.01（root 184.89）。只有请求数而没有 query 分组、长度和输入分布，无法判断盈亏。

固定随机投影实验 hidden=256，并非真实模型的完整投影/RoPE：32 请求最终为 139.74→109.55（root 169.88）。改变 GEMM 的 M 维会改变舍入，不能声称量化 Q 逐位相同。

## Prefill：专用 Ant 与真实逐 query compact 分开看

P5b 的 [23 个模型形状直接对照](../experiments/2026-11-07/p5-ant-keys-major/REPORT.md)采用 H32/D128/top512、正确角色和逐行可见长度。三方是 DeepGEMM、现有 keys-major、专用 Ant D128 适配，使用共同准备和 selector。consumer 是 full-score＋逐query mask。

source Q4096@256K 主矩阵曾出现约 6% 候选收益；[独占 GPU 复核](../experiments/2026-11-07/p5-ant-keys-major/model-aligned/isolated-source/)为 Ant 37.242 / 37.230 ms、现有 keys-major 37.497 / 37.494 ms、DeepGEMM 41.848 / 41.837 ms。相对已有最快路径只约 0.7%，撤销准入。结论仅限当前 D128 适配，D32 没有完整性能矩阵。

P5c 的 [逐query compact 完整报告](../experiments/2026-11-07/p5c-perquery/REPORT.md)比较四条 consumer 路径：DeepGEMM full＋mask、keys-major full＋mask、direct 候选读取、gathered WGMMA。下表从[同对象 paired 审计](../experiments/2026-11-07/p5c-perquery/paired-v1/AUDIT.json)提取：

|形状|最快 full，正／反|direct，正／反|Graph 保守收益|
|---|---:|---:|---:|
|64K 上下文，Q2048|4587.08 / 4584.78|3111.53 / 3111.44|1.473×|
|256K 上下文，Q4096|38000.39 / 38009.61|6210.16 / 6211.88|6.117×|
|1M 上下文，Q256 尾 chunk|9780.08 / 9737.42|437.74 / 441.72|22.044×|
|ragged 双请求|1297.21 / 1297.82|982.50 / 991.82|1.308×|

表格数值为两序 P50 展示；保守倍数采用审计中的基线下界／候选上界，不是挑最快一轮。**Q 是 prefill query 行数，不是请求并发数。** 这里包含发布、准备、所有分块、score 和映射/top-k；source 候选生成单独计时，不计入此单 consumer pipeline，也没有 TP4 通信。

短形状在原始八点矩阵中无收益：initial 2K，最快 full 128.05 vs direct 317.27；initial 8K 为 1397.72 vs 3451.81；initial 32K 为 20013.50 vs 36762.83；steady 16K 为 306.32 vs 781.56。gathered WGMMA 在八点均慢于 direct，因此不选它。

八点四方正反原始数据共 3,360 条记录、3,561 轮、106,830 个 GPU 样本及同数量 host 样本。历史部分 candidate 原始 hash 因无序输出不同，不能补造旧 tensor；另做八点 source canonical 三次检查、四个收益点同对象正反补充，十类输入字节前后不变，共 64 条记录／1,920 样本。补充与原始数据分别保留。

旧 P5 的 10×／66× consumer 数字使用 request-shared 候选，不代表真实逐query模型；旧 `gluon` 对照是 grouped scorer，不能用于否定专用 Ant。这些证据保留，但范围已更正。

## Top-k：收益与代价同时保留

**2026-11-07补充：大行数不能只在原cluster-8和streaming之间选择。** 新增cluster-2在部分形状有收益；下面是[最新四卡独立实验](../research/topk-h20/REPORT.md)，与后面的旧P6数据分开保存。最终输入采用实际FP8 Indexer生成的logits，三次独立进程、每次正反序；表为全部保留Graph样本P50（μs）。

|行×规划宽度，k512|原生|原有最佳稳定selector|新配置|建议|
|---|---:|---:|---:|---|
|8×128K|20.184|cluster-8：11.418|cluster-2：16.106|保留cluster-8|
|8×256K|26.739|cluster-8：15.990|cluster-2：23.731|保留cluster-8|
|32×128K|38.923|streaming：21.446|cluster-2：16.502|约1.30×|
|32×256K|68.186|streaming：37.902|cluster-2：25.336|约1.50×|
|64×128K|50.106|streaming：21.504|cluster-2：24.826|保留streaming|
|64×256K|82.781|streaming：61.958|cluster-2：47.859|约1.29×|
|128×128K|54.826|streaming：44.477|hist2：44.779|保留streaming|
|128×256K|79.963|streaming：84.456|hist2：84.256|差异不足5%，保留streaming|

32×256K筛选期约24μs不能替代最终三轮合并的25.336μs。每个形状三轮输入hash一致；完整[P90/CV/wall和轮次范围](../research/topk-h20/results/FINAL_SUMMARY.json)、[原始样本](../research/topk-h20/results/final-v1/)保留。690项配置检查、2760执行阶段通过；1620条测量记录、1653轮、49590个GPU样本。

新cluster-2通过调整cluster宽度及路由减少协作/排队；这两个因素需要分开理解。128×256K直接cluster-2约94μs，仍不胜streaming。hist2基本持平，warp聚合明显变慢，hist4静态shared memory超限编译失败。详见[代码diff](../research/topk-h20/experimental-kernels.patch)。这些都是selector-only数据，未证明完整Indexer或TP4通信下的净收益，未写入根模块默认dispatch。

### 历史 P6 / P8 对照

[P6 selector 研究](../experiments/2026-11-06/h20-fp8-indexer-checkpoint/p6/REPORT.md)保留 29 形状、360 固定 logits 和 6 ragged 检查；[P8 最终 raw](../experiments/2026-11-07/p8-final-dispatch/final-p2-p6/)重新绑定最终版本。代表最终 Graph 正反数据：

|行×宽度，k512|原生|稳定 cluster，含每次 plan|结论|
|---|---:|---:|---|
|1×16K|6.45 / 6.48|5.61 / 5.64|小收益|
|8×256K|26.71 / 26.73|15.84 / 15.90|约 1.68×|
|128×256K|78.82 / 78.98|207.26 / 207.44|cluster 更慢|

selector 表不包含打分，不能当完整 Indexer 的倍数。严格同分确定性要求下，回退原生非确定 selector 不是合法的语义等价优化。

## 精度、显存与统计限制

P5c 正序 50,560 query 行对 native 最大 score 差 1.90735e-6，已测 top-k 集合 100% 重合；同 logits 稳定选择要求精确。但初始 FP32 strict 抽样仍有优化 4,120／native 4,121 个失败元素，最大约 0.001664。P5b 同样保留严格误差及边界选择变化分析。相对 native 通过不等于 FP32 oracle 全通过。

多数峰值包括同进程的多个对照工作区，不能标为单 kernel 独占内存。P5c score arena≤1 GiB、gather≤256 MiB，不意味着总进程显存也受此上限约束。Eager/host 可能受其他卡实验的 CPU 负载影响；全部墙钟与噪声追加轮均保留。

最终准入使用精确形状和完整合同，不外推统一阈值；也不能把单卡 consumer 提速、grouped 提速、TP4 提速相乘宣称模型整体提速。

# P5c 最终独立审阅与精确形状调度

**完成：P5c 四路径实验、公平性复核和独立模块调度已验收。** 最终表有 16 个精确 shape×execution 条目：8 个选择 compact direct，8 个保留 full baseline。`PerQueryPlanner.bind()` 在 H20 上以默认模式完成 8 个代表项的真实执行验证，覆盖全部三类被选后端及 24 次动态 Graph 回放。原 P8 的 400 条准入没有修改。

**数值范围仍是 native-compatible 模块实验。最初 FP32 严格逐元素门槛并非全部通过。** 8 个性能形状正序各计一次：50,560 query 行中所有候选分数相对 DeepGEMM 的最大绝对差为 1.90735e-6，最大 nRMSE 为 6.41041e-8，最终 top-k 集合全部重合；独立 FP32 抽样中优化路径有 4,120 个 score 位置超门槛，原生有 4,121 个，最大差 0.00166416。这些是抽样 score 位置数，不是失败 query 行数。没有放宽容差或把 native 一致写成 FP32 数学精确，也没有验证真实模型准确率。

## 同输入、公平成本与身份

最终比较包括 DeepGEMM full+perquery mask、current keys-major full+perquery mask、compact direct、compact gathered WGMMA，使用相同 FP8 Q/cache、head weights、逐行可见长度、`[Q,2048]` 候选和稳定逻辑 top-k。gathered 是自定义 batched FP8 WGMMA，不是 DeepGEMM，也不是 Ant keys-major。source Q/weights 与 consumer 独立，模型维度和因果区间来自配置，数值为固定随机输入。

compact 完整计时含 publication/规范化、全部 chunk 的 gather（如有）、score、top-k 及逻辑映射。full 完整计时含全部 K prepare、score、逐 query mask 和相同 selector。source 生成在单 consumer 计时外；本表不能作为 source+四 consumer 组合延迟。初始化、工作区和 descriptor 规划在计时外；warm repeated cache，无 L2 flush。score arena≤1 GiB，gather staging≤256 MiB，其余状态和同时驻留的各对照另占显存。

原 8 形状正反有 3,360 条分项/完整记录，共 3,561 轮，其中 201 轮为 CV>10% 后追加；独立复核了每轮 30 个 GPU/host 样本、20 次预热和每样本 10 调用，重算中位数及 CV，未相信摘要字段。原始 candidate SHA 哈希的是无序列表，initial32K及长形状正反存在差异。历史未保存原 tensor/canonical hash，**不回填历史集合一致的证明**。

补充 `paired-v1` 对全部 8 形状检查 source 三次重放的 canonical 集合，并对 4 个长形状在同一进程、同一 GPU、同一输入/candidate/四个路径对象上完成双序完整 pipeline。10 类输入 before/after 字节哈希全部不变；64 条 pipeline、64 轮、1,920 样本均通过，无 CV>10% 追加触发。四个 compact 胜区的最终准入只采用这批同对象结果。短形状保留原始 full 回退；其中选择 current 的 initial2K/16K 原正反 candidate 原始哈希也一致。

源包 SHA 为 `b4c1a3dbda6778ea48d226b44ff77809dce604a50fd63b9b3aac15184e83e41b`。审阅核验了 87 个测量实现文件、589 个原生依赖文件、源包内容及所有 before/after identity；本次实际导入的 DeepGEMM Python 入口、facade 和 `.so` 也逐一匹配。安装 wheel 的 build commit 仍未知，没有声称二进制来自当前 HEAD。

## 最终选择

表中 Q 是 prefill query 行数，不是并发。context 是请求的最终逻辑长度，每行可见长度严格为 `context−Q+j+1`；所有新增条目均为 ratio1 consumer，32 heads、D128、top512、2048×8候选、H20 SM90、相同预算。

| Query 行数 | 最终 context | Eager 选择 | Graph 选择 |
|---|---:|---|---|
| 2048 | 2048 | current full | DeepGEMM full |
| 8192 | 8192 | DeepGEMM full | DeepGEMM full |
| 32768 | 32768 | DeepGEMM full | DeepGEMM full |
| 512 | 16384 | current full | current full |
| 2048 | 65536 | direct | direct |
| 4096 | 262144 | direct | direct |
| 256 | 1048576 | direct | direct |
| [512,128] | [65535,16384] | direct | direct |

| paired 胜区 | 最快 full Graph μs | 最慢 direct Graph μs | 保守加速 |
|---|---:|---:|---:|
| Q2048 / 64K | 4584.78 | 3111.53 | 1.473× |
| Q4096 / 256K | 38000.39 | 6211.88 | 6.117× |
| Q256 / 1M 尾 chunk | 9737.42 | 441.72 | 22.044× |
| ragged | 1297.21 | 991.82 | 1.308× |

每个 order 都要求最慢候选轮相对最快 full baseline 轮降低至少 5%，且降幅大于参与比较各轮最大 CV；compact 必须同时胜过两种 full baseline。gathered 在本矩阵均慢于 direct，保留显式实验后端但不被选中。没有推导任意 Q×K 阈值，也没有采用 Ant 的 2M 阈值。

## 已落地的模块接口

- `vllm_dsv41_indexer/perquery_dispatch.py` 定义不可变 `PerQueryContract` 与 `PerQueryPlanner`。完整 query 分组/context 日程、执行模式、候选语义、dtype/cache ABI、heads、维度、预算、单卡模式、硬件及已核验依赖 identity 都进入键。
- `choose()` 仅精确命中；不插值。未测但 ABI 合法的合同返回稳定 DeepGEMM full baseline，并标记 `evidence=False`；不支持的 ratio/TP4/预算/候选布局显式拒绝。
- `bind()` 在 Graph capture 之前核对实际 tensor 形状、request/length 全序列、唯一且有效的候选/latest block 规则、硬件和已导入原生依赖，构建真实选中模块。返回模块的原 `forward()`，执行区没有新增 host 检查/同步。
- 改变长度、请求分组、模式或预算须重新规划；底层 kernel 对动态输入正确不等于旧性能证据仍适用。Q/权重与候选内容可在原合同内更新。source20 来源是调用方合同，bind 检查其唯一/count/latest 结构，不把任意外部列表自动视作 source。
- 这是独立的函数/模块实验调度。full 对照复用测量入口的 `ModelPath`，并非完整 HEAD serving wrapper；没有接入 vLLM serve，没有新的 TP4 自动性能调度。

## 最终验证与保存

CPU：原 dispatch 17 项 + 新 perquery dispatch 21 项，共 **38 passed**。新项覆盖正反/两个 baseline/全部保留轮、噪声和 5% 门槛、实际全日程键、预算与 ABI 拒绝、artifact 变更拒绝、pending manifest 默认拒绝。ruff 通过。

GPU：`bind-v2/verify.json` 为默认非 provisional 模式，8 个代表绑定（initial2K、16K、64K、ragged × Eager/Graph）全部通过，涵盖 current full、DeepGEMM full、direct 三类真实选中实现。每项真实运行完整分数/稳定选择验证，并在 poison 准备区后回放 baseline、Q/candidate 改变、恢复三个阶段；Graph 与 Eager 逻辑集合完全一致。source 连续三次 canonical 一致、长度变更拒绝及未测日程回退也实测。源码/证据执行前后再次核验。旧 `bind-v1` 保留为 provisional 历史，不代替最终验证。

新模块另有独立 TP4 正确性证据：4 个真实 rank、3 形状、7 动态阶段、两 backend×复制/按请求分组/按行分配×Eager/Graph，每 rank 不同物理页表但逻辑输入相同，输出集合一致。结果位于 `../p5-perquery-tp4/final-v1/`。**这不是 TP4 性能结果**，没有借用旧 P7 的通信收益。

审计入口 `audit_final.py` 从原始 JSON 重建并对比最终表、检查真实绑定及四卡逻辑身份。`FINAL_RAW_REVIEW.json` 是机器可读结论，`admission.json` 是最终 16 项选择与原始样本/依赖/证据哈希，`frozen/planner-source.tar.gz` 保存本轮独立调度源码。初审保存在 `INITIAL_CPU_REVIEW.md`。

## 复现

从仓库根目录运行 CPU 收集/审计：

```sh
.venv/bin/python -m benchmarks.kernels.dsv41_indexer.collect_perquery_dispatch --directory results/indexer/20261107-resume/p5c-perquery/final-v1 --paired-directory results/indexer/20261107-resume/p5c-perquery/paired-v1 --output results/indexer/20261107-resume/p5-perquery-review/admission.json
.venv/bin/python -m pytest tests/kernels/dsv41_indexer/test_dispatch.py tests/kernels/dsv41_indexer/test_perquery_dispatch.py -q
.venv/bin/python -m results.indexer.20261107-resume.p5-perquery-review.audit_final
```

远端最终绑定冻结目录为 `h20-4:/root/dsv41-indexer-resume-20261107/runs/p5c-dispatch-bind-v2`。其源代码是测量源包加 `frozen/planner-source.tar.gz`，`.venv` 和 results 指向同一已核验环境；在该目录按 GPU 租约运行：

```sh
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_perquery_dispatch --manifest /root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5-perquery-review/admission.json --output /root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5-perquery-review/bind-v2/verify.json
```

此审阅可以关闭 P5c 的模块实验与独立 P8 形状选择缺口。整个 Goal、方案书与知识库的最终状态由总控依据全部阶段证据更新。

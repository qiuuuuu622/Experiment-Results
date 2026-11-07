# P6 确定性 top-k：H20 GPU2 函数级实验

- 日期：2026-11-06
- 运行结果：completed
- 范围：selector 本身，不包括模型、score kernel、serve、TTFT/TPOT；最终整模块 dispatch 与 P8 回归由总控完成。

## 实现与正确性

新增 `StableTopK`：初始化、prepare、execute 分开；FP32同分固定较小逻辑位置，输出无序唯一索引、不足填-1。`RaggedTopK`支持非对齐prefill窗口与逻辑offset；会消费/修改窗口前最多3个无效score，不能供后续消费者读取这些前缀。
`native.load_kernels()` 保持既有compact无参ABI；新增full/streaming构建、plan/ragged/cluster_floor导出。H20实际78SM，cluster-8 pool采用18，避免照搬H200默认30。cluster-16未启用：缺少非portable设备探测，无需为本研究强启。
不修改任何vendored Ant内核，许可证和sources哈希不变。

360项full/source固定logit选择 + 6项ragged检查通过，与stable argsort的逻辑集合精确一致。覆盖k512/2048，4K/16K/64K/128K/256K，1/8/32/64/128行及513行fallback；随机、全零、非零全同分、超过2048 tie-bin容量、cutoff大同分；NaN无效尾部；连续graph回放和graph内长度变化重plan。

分支覆盖记录：{'register': 120, 'streaming': 160, 'small_cluster': 50, 'persistent_cluster': 30}。原始verify.json每项记录形状、路径和plan。

## 性能结果

Graph P50，正反序范围（μs）。k512原生对照是vLLM cooperative/persistent selector；k2048对照是source原生torch.topk，包含其返回tensor分配。输入全有效随机FP32，同一热缓存连续调用；没有清L2。表中cluster列包括每次调用prepare；小batch不需要GPU plan。

|行|score宽度|k|原生 μs|streaming μs|cluster含plan μs|两种新路径较快者|保守加速|
|---:|---:|---:|---:|---:|---:|---|---:|
|1|4096|512|3.70–3.70|3.21–3.51|3.21–3.37|cluster_replan_each_call|1.10×|
|1|16384|512|6.52–6.52|5.68–5.70|5.68–5.68|cluster_replan_each_call|1.15×|
|1|65536|512|9.77–9.80|12.67–12.72|8.33–8.45|cluster_replan_each_call|1.16×|
|1|131072|512|11.71–11.71|21.38–21.39|9.71–9.87|cluster_replan_each_call|1.19×|
|1|262144|512|14.69–14.70|36.88–36.90|11.60–11.61|cluster_replan_each_call|1.27×|
|8|16384|512|6.78–6.94|5.91–6.04|5.90–6.04|cluster_replan_each_call|1.15×|
|8|65536|512|16.78–16.80|13.06–13.06|10.29–10.31|cluster_replan_each_call|1.63×|
|8|262144|512|26.59–26.61|37.55–37.57|15.95–15.97|cluster_replan_each_call|1.67×|
|32|16384|512|7.69–7.70|6.07–6.09|6.07–6.09|cluster_replan_each_call|1.27×|
|32|65536|512|28.61–28.65|13.16–13.17|13.16–13.19|streaming|2.17×|
|32|262144|512|66.98–67.05|37.74–37.79|42.49–42.53|streaming|1.77×|
|64|16384|512|7.92–7.94|6.26–6.28|6.26–6.28|streaming|1.26×|
|64|65536|512|30.18–30.20|13.25–13.31|13.26–13.29|cluster_replan_each_call|2.27×|
|64|262144|512|83.85–83.87|62.15–62.24|98.81–99.35|streaming|1.35×|
|128|16384|512|13.04–13.09|9.32–9.37|9.33–9.36|cluster_replan_each_call|1.40×|
|128|65536|512|52.38–52.42|22.27–22.29|22.27–22.29|streaming|2.35×|
|128|262144|512|78.99–79.10|84.06–84.09|206.95–207.02|streaming|0.94×|
|1|4096|2048|32.86–32.92|3.07–3.12|3.07–3.11|cluster_replan_each_call|10.57×|
|1|16384|2048|88.84–88.87|6.51–6.53|6.50–6.51|cluster_replan_each_call|13.66×|
|1|32768|2048|58.20–58.22|10.29–10.32|10.30–10.31|cluster_replan_each_call|5.65×|
|8|4096|2048|38.40–38.45|3.12–3.16|3.13–3.16|cluster_replan_each_call|12.19×|
|8|16384|2048|100.74–100.77|6.61–6.67|6.63–6.66|cluster_replan_each_call|15.14×|
|8|32768|2048|61.99–62.00|10.82–10.83|10.80–10.84|streaming|5.72×|
|32|4096|2048|40.04–40.10|3.16–3.20|3.17–3.21|streaming|12.51×|
|32|16384|2048|70.24–70.37|7.14–7.15|7.13–7.14|cluster_replan_each_call|9.84×|
|32|32768|2048|83.77–84.08|10.93–11.06|10.90–10.93|cluster_replan_each_call|7.67×|
|128|4096|2048|60.48–60.53|4.34–4.39|4.35–4.39|streaming|13.78×|
|128|16384|2048|101.78–101.93|11.54–11.59|11.60–11.60|streaming|8.80×|
|128|32768|2048|141.52–141.84|18.76–18.81|18.77–18.80|cluster_replan_each_call|7.54×|

关键结论：
- 小batch长行（实测1/8行）cluster-8有效；32/64行长行streaming更好。不能对所有形状启用persistent cluster。
- 128×256K token top-k无收益：原生约79μs，streaming约84μs，cluster约207–212μs。允许原生同分语义的调用应保留原生；要求严格确定性时必须保留稳定selector，不能把原生回退伪称确定性。
- k2048 source选择在所有代表形状收益显著。这里是block-score selector成本，不是source整个模块收益。
- 32/64/128长行plan本身约1.74–1.85μs；预计划execute和每次重plan总成本分别实测，二者不是简单数字相加（launch并发/PDL会影响总耗时）。

完整Eager、Graph、P90、原始样本、墙钟、总峰值分配与追加轮次见forward.json/reverse.json。
采样每策略20次预热、30样本×10调用；CV>10%追加3轮，全部原样保存。最终调度只应采纳正反序稳定且>5%的收益；未测形状不从此表自动外推。

## 复现

```sh
cd /root/dsv41-indexer-port
CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH .venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify_topk --output results/indexer/20261106-tp4/p6/verify.json
bash scripts/06_topk.sh
```

GPU2映射为进程内cuda:0。原生Python基线通过check_vllm的HEAD文件哈希检查；实际预装二进制构建commit未知，未宣称由当前HEAD构建。
实验版本/拓扑位于forward.json/reverse.json，完整环境沿用P0 environment.json。新kernel只经TVM-FFI编译自身头文件，不重编译vLLM。

## 限制与交付

- 只测固定FP32 logits selector，分数精度归P1/P3/P5；无模型准确率结论。
- CPU墙钟和Eager可能受同主机其他卡实验CPU活动影响，raw samples已保留；P8统一固定条件重跑。
- Ragged正确性6项通过，但本阶段未单独测ragged相对prefill selector的完整链路性能；P5负责prefill准备+打分+选择全链路。
- 本阶段未实现最终auto dispatch；提供显式cluster/streaming选择及实测证据。原生回退的同分语义限制必须保留。
- GPU2已释放，无本任务遗留GPU进程。
- 原始结果：verify.json、forward.json、reverse.json与同名日志；源码：source-snapshot.tgz和source-sha256.json。

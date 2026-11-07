# P5b 合成筛选证据检查点

2026-11-07；独立 CPU 审核。**已有 23 个合成形状与 4 个 persistent-grid 单卡形状的证据已保存；这不是模型架构对齐主矩阵完成，也不是 P5b 整体验收。** P5 owner 正在补模型角色、真实 query 级候选与主矩阵。本检查点不改原始结果，不运行 GPU，不写知识库，不发布或推送。

## 本次核验结果

- `INVENTORY.json` 保存 59 个已有证据文件的 SHA256/字节数及 2,639 项核对，0 项失败。排除正在产生的 `model-aligned/` 实验结果。
- 23 形状来自 small 8、large 7、mixed 8；每套 forward/reverse JSON 均 completed，形状匹配，三项执行源码 hash 与冻结包匹配。
- grid 4 个 JSON 共 192 条 variant/phase/顺序记录，全部 completed。其 driver、模块/依赖源码和引用的冻结包 hash 匹配；每卡独立运行一个形状，不能汇总成 TP4 延迟。
- 两组共核验 1,335 个计时轮次，每轮均 20 warmup、30 samples × 10 calls，同时保存 30 个 host wall samples。首次 CV >10% 的记录都保存了额外三轮，共 315 个追加轮次；追加轮次不稳定仍应如实保留，不能因追加即宣称稳定。
- grid MANIFEST、CPU review MANIFEST、六文件 benchmark review 清单、upstream SOURCE.json 均与对应文件或冻结源码一致。两个 edges PTX SHA 匹配各自 JSON。
- 源包 `frozen/source.tar.gz` 与 grid 的 `frozen-p5-source.tar.gz` 字节相同，SHA256 `31863c46ad5e956f06b9cbea72c9498fbeb17f13f1964fa44d7a37daea8c4280`。Ant adapter SHA256 `957557aa4ac9944073ae4e2c2a370775df446f93437fa82de821b3051d3184c9`。
- 上游固定于 Ant `6e6d5d9e13915adb01db0856b15d6908eaaa2c48`，专用 kernel SHA256 `0bdde39dd1ddb4805492badbd838aa40a51a7dc79f29c070d27b661fc3f66b3f`，Apache-2.0 LICENSE 保存完整。

## 已测角色与必须收窄的解释

旧 23 形状 driver 使用 `make_case(rows, requests, keys, ratio)`；它是通用函数合成输入，不绑定真实模型层。`full` 包含 ratio1/ratio2；`source` 只有 ratio1；两个 `consumer` 分别是 ratio1 的 256Q×64K 与 ratio2 的 1024Q×16K。short-prefix 与 reversed-visibility 也是边界合成模式。

旧 consumer 的候选通过 seeded randperm 随机生成，形状为 `[requests, 2048]`，同请求所有 query 共用候选，即 **request_shared**。它们不是 source layer 实际输出的逐 query 候选，没有测量真实 source→consumer 的候选生成关联。最多 2048×8 个紧凑位置的 API 成本，不能直接代表模型中每 query 不同候选的成本。

用 owner 已保存的模型配置只作范围核对（路径与 hash 见 inventory），实际 indexer 角色是：

|层 ID（0-based）|ratio|角色|
|---|---|---|
|2 / 8 / 14|2|full|
|20|1|source|
|24 / 28 / 32 / 36|1|consumer|

因此 **ratio2 consumer 不是这个模型的真实层角色**；其结果只能保留为通用 API 压力测试。旧 ratio1 full 也不应自动对应上述 consumer 层。后续主矩阵需按实际角色和候选布局单独命名、绑定新的冻结版本与结果。

## 当前性能证据能说明什么

原 `TABLE.md` 的 23 形状汇总采用正反顺序及追加轮次中最大的 P50；grid 报告采用合并全部样本的中位数，两者聚合方法不同，不能逐格混合比较。两批 raw 均保存，核验不重写其数值。

已有三方计时共享 FP8 Q/K、FP32 scale/weights 与稳定 selector；pipeline 含 key preparation，Ant 还含动态 visible-prefix。score-only 已准备好 key/prefix，不能代替完整成本。当前 raw 显示这组合成形状没有足够证据把 Ant adapter 自动替代已有最快路径；grid 的 39/78/156/312 消融也未支持更换默认 78。结论仅限已测合成形状；不替代正在补做的模型架构主矩阵，不证明其他形状永远无收益。

## 身份与历史证据的限制

`runtime-identity.json` 保留 588 个原生文件 hash，实际包为 vendored DeepGEMM，版本 torch 2.13.0+cu130、Triton 3.7.1、vLLM 0.32.0、TVM-FFI 0.1.11。该 identity 明确为 **post-run** 采集，wrapper 在各进程启动时独立核验。本次 CPU 检查核实冻结源码字节与记录一致，没有重新在远端导入或重算全部二进制/header hash，不能追加“独立重验运行时身份”的声明。wheel 的 build commit 仍未知，不能称该二进制由当前 HEAD 编译。

`verify-v2/v3/v4` 各保存 12 个 passed checks，两个 edges 文件各 6 case、PTX hash 可验证；但这些 JSON 本身没有独立每次运行的源码 hash。它们保留为历史正确性证据，不凭当前冻结包反向制造其逐次精确源码绑定。`numerical-v1` 为旧数值诊断，有独立 adapter 文件；不能替代 final adapter 验收。CPU review 的数值反例与公平性审阅不是 GPU 测试。

本轮没有发现文件缺失或所声明 hash 不匹配。上述运行时采集时点、历史 verifier 的逐次源码绑定、真实模型候选布局和主矩阵仍是明确限制。

## 可复验命令

在仓库根目录执行：

```sh
.venv/bin/python results/indexer/20261107-resume/p5-ant-checkpoint/audit.py
```

该命令只读三组原始证据，重建本检查点 `INVENTORY.json`；不访问 GPU、服务或模型，也不修改原证据。`INVENTORY.json` 内对 model-config 的引用只用于解释角色，不意味着新矩阵已完成。

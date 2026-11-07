# P8 最终独立审计（P3 agent）

本次仅做 CPU/只读证据复核，复用已有 `audit_admission_artifacts.py`，没有另跑 GPU 性能，也没有使用 P5b 的 GPU3。结论针对当前冻结的 P0–P8 FP8 模块实验；用户新增 P5b 蚂蚁专用 prefill 迁移仍是独立未完成工作，不能据此结束整个 Goal。

## 证据完整性

- 合并 manifest 为 completed：400 admitted（单卡387、TP4 13），11个环境，0 bundle failure。
- 复用 P5 审阅脚本核验41个唯一源/正确性/性能/native身份 artifact 的原始字节 SHA256；0问题。结果为 `admission-all-bindings-review.json`。
- 原始未修安装包与 P3 隔离修复包属于不同 environment digest，没有统一伪造为同一个native环境。
- dispatch现有CPU测试17项全部通过。核对了严格20warm/30×10、CV重算、追加轮、双顺序及最坏追加轮P50的选择逻辑；未增加重复测试体系。

## 真实绑定复核

- 单卡：h20-dsk冻结 `runs/p8-dispatch-final-v2`，8环境全部passed、0 skipped，共28个代表策略/角色绑定。
- TP4：h20-4冻结 `runs/p8-tp4-bind-final-v1`，3环境全部passed、0 skipped，共7个代表绑定，启动脚本显式torchrun4rank且COMPLETE存在。
- 合计35个代表绑定检查，不应写成“400个形状逐一做dispatch绑定”。400条记录来自完整原始性能/正确性artifact，绑定检查按环境/phase/role/选中strategy/输入模式/投影分组去重。
- 独立在CPU重新构造Dispatcher，按每份报告的实际key/environment重算策略：所有策略名称一致且具有有效证据；报告runtime source digest与合并manifest一致。分别见 `binding-audit.json` 和 `tp4/binding-audit.json`。
- 单卡source绑定验证token输出以及强制最新块语义下的source候选；chain绑定实际source+四个不同consumer，校验五份top-k及source块，不再以旧P3报告代替当前绑定执行。Graph均有三次回放。
- TP4绑定包含RowExchange实际输入整理/通信与逻辑输出收集，比较复制参考的top-k集合；其详细动态边界由P7最终480/rank测试覆盖。

## 原生身份

每个绑定进程在执行前核对实际目标package文件哈希，单卡报告分别核验4040或879个文件。两台主机另有明确标注为运行后导入器审计的 `loaded-native.json`，记录实际导入模块、加载extension路径和SHA；extension均为：

`0042598bc0770e70bc4ea4b9c3d27d8efc8e395d6f766d41e939c746c2bd3046`

只读检查两台venv的外部 `deep_gemm` import spec均为None，不存在外部包抢先导入而绕过vendored身份检查的实际情况。隔离补丁include哈希与P3已验证版本一致。运行后导入器审计没有冒充为之前每个进程的启动时日志。

单卡绑定源码archive SHA256：

`e6739f606f19c6f33c2e3af73f3624e30169bbe6b949e71a9e117738b7b09ef2`

## P7 最后覆盖缺口关闭

已独立读回 `verify-native-allzero-restore.json`：passed、真实4rank、480检查/rank、6个step。最终verifier源码哈希精确匹配。新增step4将全局所有长度置零并重新置换物理页表，断言所有输出为-1；step5恢复原输入并要求输出digest回到初始值。四rank每项逻辑输出hash均一致。

因此前次review里“全局全零长度+同时页表变换”未单测的覆盖说明已关闭。native每步prepare、稳定top-k及token/block双输出此前三个问题也均已修复并重测。

## 审计边界

本结论不扩展到服务、模型精度、TTFT/TPOT或PCP/DCP。单卡的static metadata准入不应被宣传成已为任意动态模式测得性能；TP4 native已按每步prepare计时。新增P5b代码/性能结果必须独立保存原始证据并在最终报告中明确纳入或排除，不能把本次审计当作P5b已验收。

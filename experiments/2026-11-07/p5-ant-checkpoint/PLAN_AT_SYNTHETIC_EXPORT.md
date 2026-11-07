# H20 四卡 DSK41 FP8 Indexer 优化方案与执行清单

## 目标与边界

以 `29be941d8d5746cd98135bdd5e710eeebd8d7014` 为基线，参考蚂蚁
`6e6d5d9e13915adb01db0856b15d6908eaaa2c48`，在 `h20-dsk` GPU 0–3
完成 FP8 index cache 的函数／模块优化、正确性验证及可复现性能实验。
覆盖 prefill、decode、verify、source/consumer、确定性 top-k、四卡复制和按行分摊。

- 不加载模型，不接入 vLLM serve，不测模型准确率、TTFT、TPOT。
- 保留 FP8 Q/K 格式、scale 语义及打分公式；默认每 rank 全部 32 个 index heads。
- PCP/DCP 分片缓存不在范围。按行分摊单独实验。不承诺所有实现更快。
- 优化源码仓库不推送、不发布 PR；已有结果按用户后续明确授权发布到 Experiment-Results。只操作授权任务资源。
- 不设 token 预算。每项必须有实现结论、相应验证和记录才勾选；失败、中止或未验证不能勾选。

本地：`vllm_dsv41_flash_opt/H20优化方案书.md`。
远端：`/root/dsv41-indexer-port/H20优化方案书.md`。
代码：现有 `vllm_dsv41_indexer`；实验：`benchmarks/kernels/dsv41_indexer`；
测试：`tests/kernels/dsv41_indexer`；本轮原始证据：`results/indexer/20261106-tp4/`。

## 当前恢复状态（2026-11-07）

当前目标仍为P0–P8。P0–P4、P6及P7独立研究阶段已验收；P8既有路径的最终调度/绑定验证已完成。P5已有路径实验保留，但蚂蚁专用prefill迁移覆盖不足，新增P5b未完成；P5/P8整体待P5b的模型形状直接对照、调度结论和归档收尾。
本轮以最新本地源码为准，历史目录保持只读，执行副本为两主机`/root/dsv41-indexer-resume-20261107`。
原始证据新增`results/indexer/20261107-resume/`，不覆盖20261106记录。

|任务|子agent|主机/资源|当前工作|
|---|---|---|---|
|P5b专用Ant迁移/直接对照|prefill_p5|h20-dsk GPU1–3|模型角色、因果长度及逐query候选对齐；三方正反序实验|
|架构/调度独立审核|tp4_p7_resume|CPU；grid消融已完成|核验模型配置、层角色、真实可见key数、短上下文分支|
|P8/基准公平性审核|dispatch_p8_resume|CPU|既有P8已验收；审查新模型形状driver及P5b准入依据|
|证据保存/发布审核|full_source_p3|CPU|P7/P8已发布；清点P5b已有合成筛选证据|

总控约60秒汇报，统一GPU租约。用户后续已授权两台主机全部八卡；此前容器外占用已解除。P7四卡实验和P5b grid消融均完成释放，空闲卡按有效实验需求分配，不启动占卡负载。
P4/P5既有路径/P6恢复依据见`results/indexer/20261107-resume/audit/REPORT.md`，最终重测另见P8各冻结结果；这些证据不能替代P5b专用内核的直接对照。
知识库既有路径批次`20261107-resume`已按实际范围归档完成，校验通过20 Results；新P5b批次`20261107-prefill-ant-keys-major`单独保留运行记录，未完成前不关闭。结果仓库最新已发布P7/P8提交`97b6ef19f677210d95a1db6ea53262c30fd65612`。
以下20261106进度按历史保留，不作为当前状态。

## 执行清单

- [x] **P0 固定环境及基线**：固定两个 SHA、快照、许可证，保留历史单卡实验；记录四卡型号、拓扑、驱动、CUDA、PyTorch、Triton、DeepGEMM、TVM-FFI、编译参数。独立 `.venv`，所有 Python 经 `.venv/bin/python`。修复旧 provenance 和硬编码 wheel 版本依赖，分别核验当前 HEAD Python 调用路径及实际二进制。复现单卡及真正四进程 compact 正确性。移除本任务未接入的 serving 草稿。
  - 完成证据：环境与基线清单、命令及原始输出。
- [x] **P1 统一接口和参考**：输入 FP8 Q、含 Q scale 的 head weights、本地页表/cache、query→request、逐行可见长度、prefill 区间/verify 分组、可选候选及 source/consumer 模式。输出唯一逻辑 top-k，不足填 -1。区分候选槽、压缩后 index 位置、原始 token 位置及 ratio 1/2。提供当前 HEAD 原生算法、优化路径、小规模 FP32 参考。显式工作区及初始化/规划/执行；不计 JIT、分配、通信初始化。候选仅同一组合调用复用，无跨步隐式缓存。
  - 完成证据：契约、形状检查、FP32 参考及小规模正确性。
- [x] **P2 compact consumer/发布**：source 唯一候选一次排序计数；外部任意候选去重；一次发布供四个不同 Q/weights consumer 复用；覆盖尾部不足8、短/空/无效候选；比较独立及融合映射。测试新 step、重排、候选/长度变化及连续 graph replay。报告单 consumer 含发布和发布一次+四 consumer。
  - 完成证据：规范化、分数误差、top-k、完整链路 A/B。
- [x] **P3 full/source decode**：从本地页表直接读 FP8 key；按可见范围限制读取及 grid；评估融合候选块最大值；source 同时输出 token top-k。scores 最多 1 GiB，能容纳时保留请求组，否则切分。
  - 完成证据：full top-k/source/consumer 联合正确性，准备/打分/候选/top-k 分项及总耗时。
- [x] **P4 grouped verify**：先 full/source，共享同请求 key tile；每 query 独立可见长度/因果边界；规则及 ragged 组，1/2/4/6/8 行。consumer 只在共享候选布局时共享 key，不同集合保留逐行，不构造昂贵并集。原生多行与展开单行基线明确区分。
  - 完成证据：逐行正确、无未来泄漏、相对逐行净收益。
- [ ] **P5 FP8 prefill（补齐迁移覆盖后验收；已有实现实验保留）**：以 DeepGEMM 为主基线，对比 row-tiled、keys-major（不移植 FP4 解包）。consumer 比较先聚集 K 后 GEMM和候选间接读取。覆盖多请求、chunked、历史前缀、query 区间、ratio1/2。可共享 K 只准备一次，包含准备成本；重测交叉点，不照搬 2M。
  - 完成证据：全链路形状对照、准备成本、显存峰值、交叉点。
- [ ] **P5b 蚂蚁专用 prefill keys-major 的 FP8 适配与直接 A/B**（2026-11-07用户补充）：对照`6e6d5d9e`的`sm90_fp8_prefill_keys_major.py`迁移32 query行全部heads驻共享内存、Q/K TMA、三级K缓冲、跨head双缓冲异步WGMMA、按可见tile分配工作的常驻调度。保留FP8 cache、K scale、FP32 head weights和打分公式；不照搬BF16舍入点，不把FP4精确累加性质套到任意FP8。
  - 三方直接对照：DeepGEMM、现有keys_major、蚂蚁调度FP8适配版；必须同输入、同可见长度、同top-k selector。
  - 分别报告裸打分，以及K准备+调度metadata+打分+top-k完整成本；初始化/JIT另计。使用既定正确性检查及正反20预热/30样本×10调用，CV>10%追加三轮；重新测交叉点，不套2M阈值。
  - 完成证据：原文件/许可/hash、适配源码快照、机制逐项对照、误差与边界、完整原始样本、性能及无收益区间。验证和记录完成前保持未勾选，不预设蚂蚁版本更快。
  - 执行：prefill_p5主责，h20-dsk GPU1–3；tp4_p7_resume独立CPU机制/架构审阅，dispatch_p8_resume审阅公平性。独立模块`prefill_keys_major_ant.py`与新结果目录，先不修改既有P8冻结路径。
  - **模型形状约束（用户2026-11-07补充）**：依据已保存的DSV4.1 Flash配置，Indexer固定32 heads、head dim 128、token top-k 512；主注意力64 heads/512维不是本实验维度。FULL对应ratio2的index层2/8/14；SOURCE对应ratio1的第20层；CONSUMER对应ratio1的24/28/32/36层。prefill候选逐query独立，形状为`[Q,2048]`、块大小8；不得用请求共享候选替代主矩阵。
  - 实际可见index-key数为`floor((prefix_tokens+j+1)/ratio)`，Q表示当前prefill chunk的query token行数，不能直接当并发请求数。现有23形状及grid消融保留为合成筛选，不据此概括模型主路径性能。
  - 待完成主矩阵：各角色测初始prefill `Q=2048/8192,prefix=0`；chunked `Q=512,总长16K`、`Q=2048,总长64K`、`Q=4096,总长256K`；长上下文尾chunk `Q=256,总长1M`；混合请求`Q=[512,128],总长=[65535,16384]`。SOURCE/CONSUMER另加初始`Q=32768,prefix=0`，跨越2048候选块截断，共23点。这些chunk为架构合法的实验工作负载，不宣称是部署默认值。ratio2实际key数约为逻辑上下文的一半。score arena上限1GiB，大矩阵必须分块并计入全部分块成本。
  - 模型短上下文分支按整个batch的`max_seq_len//ratio<=512`且不在CUDA Graph capture时绕过score；主矩阵最大key数超过512，初始prefill中较早的短行仍保留在score测试内。逐query候选可能相同，报告实际候选数量/差异，不能仅据`[Q,2048]`声称每行集合不同。三方consumer对照采用full-score+逐行mask，计时范围为单consumer，不含上游source生成，不冒充source+四consumer组合链路。
- [x] **P6 确定性 top-k**：扩展到 full/source/prefill；相同 FP32 分数优先较小逻辑位置。覆盖长行、cluster、tie-bin 溢出、部分有效行及无效尾部。预规划，分开及合计计划耗时。原生 selector 对照，无收益回退。
  - 完成证据：固定 logits 与稳定参考完全一致，包括边界/长行。
- [x] **P7 四卡复制及行分摊**：复制模式每 rank 独立物理页布局，同逻辑输入。分摊优先请求组，verify 不拆组；组不足再对比拆行。source/consumer 同所有权，不交换物理页 ID。包含行整理、必要输入传输、逻辑 top-k 收集、同步。对比分摊 score/top-k 与包含随机权重 Q/head-weight 投影。小 batch 保留复制，实测阈值。每迭代最慢 rank，额外主机墙钟。
  - 完成证据：与单 rank 参考一致、完整通信及盈亏点。
- [ ] **P8 调度/消融/回归/报告**：仅按明确形状/硬件/模式选择最快稳定路径；消融发布、compact、映射、grouped、prefill、分摊。最终代码重新正反序 A/B，不能复用重构前结果。逐项收益、无收益范围、精度、工作区、限制；纠正原报告 FP8 cache/verify tile 实际共享/卡数/模块范围。
  - 完成证据：最终矩阵、原始样本、源码版本、复现命令。

## 测试矩阵与验收

|维度|取值|
|---|---|
|逻辑上下文|4K、16K、64K、128K、256K|
|Decode 请求|1、8、32、64、128|
|Verify 行/请求|1、2、4、6、8 及 ragged|
|Prefill query 行|1、16、64、256、1024、4096|
|Ratio|1、2，记录实际 index key 数|
|并行|单卡参考、四卡复制、四卡行分摊|
|执行|Eager、CUDA Graph|
|分布|随机、重复 key、零 Q、极小分差、完全同分、长短混合|

分阶段：小规模正确性→代表形状筛选→保留实现全回归；内存超预算提前记录，不能静默漏测。
规范化、逻辑映射、相同 logits 稳定 top-k 必须精确一致。
分数以 FP32 参考 `rtol=1e-3, atol=1e-4` 为初始严格门槛；失败分析，不自动放宽。
报告 nRMSE、max_abs、集合重合、边界分差；选择变化须能用实际误差或同分解释。
检查唯一、合法、无未来，NaN 毒化 score 尾部；独立物理页置换、重排及重复 graph。

每项20预热、30样本、每样本10调用，正反 A/B各一次；报告P50、P90、波动、显存及完整成本。
每条记录候选/有效key/query数和cache条件；代表用例波动>10%追加三轮且保留所有记录。
正确且收益稳定高于噪声才 dispatch，<5%优先基线。四卡GPU时间逐迭代取max，同时报墙钟。

## 当前进度与证据

2026-11-06：Goal 已创建；P0 进行中。分支 `codex/dsv41-indexer-sm90-port`。
本任务遗留未跟踪 `serving.py` 已删除，服务代码不参与实验。
远端 `.venv` 独立建立（继承预装 CUDA 依赖）；四张 NVIDIA H20 空闲。
发现旧脚本依赖 `results/indexer/20261101/provenance.json` 且硬编码 wheel0.32.0，待修复。
知识库 ensure 被旧实验 `sglang/pd-dflash-profile/20260907-tp4-1p1d` 的“进行中”记录阻塞。
已请求用户确认其实际结果；不推断/篡改旧状态，不伪造本轮 Result。
正式性能实验归档前解决阻塞；当前继续源码及正确性工作。

### 实际命令

```sh
git switch -c codex/dsv41-indexer-sm90-port
ssh h20-dsk 'cd /root/dsv41-indexer-port && uv venv --python /usr/local/bin/python --system-site-packages .venv'
ssh h20-dsk 'nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader'
```

后续命令、失败输出、修复和证据逐项追加；当前 P0–P8 无一项冒记完成。

### P0 完成（2026-11-06）

- 14 项契约测试通过；单卡 GPU 和真实四进程 GPU 验证通过，四个 rank 独立页置换及三次 graph replay，输出集合一致。
- 7 个当前 HEAD Python 文件与实际安装 wrappers 的哈希一致；二进制单独采集 SHA256，构建 commit 未知，不视为当前 HEAD 产物。
- 四卡 NV18 互联；PyTorch2.13.0+cu130，Triton3.7.1，vLLM0.32.0，TVM-FFI0.1.11。DeepGEMM 无独立 distribution metadata，后续记录实际导入文件。
- 环境与结果：`results/indexer/20261106-tp4/{environment,verify-single,verify-tp4,baseline}.json`；日志同目录。
- 用户确认旧知识库实验已完成，已只补齐状态；ensure/refresh/validate成功（15 Results）。本轮P0归档到知识库 `results/vllm/dsv41-fp8-indexer/20261106-tp4-p0`。
- 下一步 P1：统一契约和 FP32 参考，逐行/分组 paged FP8 打分及 correctness。

```sh
.venv/bin/python -m pytest -c tests/kernels/dsv41_indexer/pytest.ini tests/kernels/dsv41_indexer/test_contract.py -q
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH .venv/bin/python -m benchmarks.kernels.dsv41_indexer.verify --output results/indexer/20261106-tp4/verify-single.json
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 -m benchmarks.kernels.dsv41_indexer.verify --distributed --output results/indexer/20261106-tp4/verify-tp4.json
```

### 数值门槛校准（用户反馈，2026-11-06）

初始逐元素 `rtol=1e-3, atol=1e-4` 对部分原生 DeepGEMM FP8 分数也失败，不能直接据此认定原生错误或强制改变累加精度。保留该阈值作为诊断，不自动放宽数值或宣称通过。先对相同输入测量原生、旧 WGMMA、显式精确累加三条路径，报告对 FP32 与原生的误差、边界分差及 top-k 重合。更高精度路径仅诊断对照，不因通过初始门槛就进入默认性能路径；compact 默认恢复原 WGMMA 累加。最终验收阈值必须有校准数据支持。

已完成初步：15组full/grouped精确累加通过FP32严格检查；24组compact生命周期×4consumer通过精确累加检查。它们不能替代默认WGMMA的数值验收或四卡性能证据。P1/P2尚未勾选。

校准原始数据：`results/indexer/20261106-tp4/calibration.json`（24条逐行记录，4099/65536 key，signed/positive weights及zero query）。旧WGMMA相对原生最大绝对差9.54e-7，top-k集合100%重合；原生相对FP32最大绝对差1.19e-3，signed weights有839个元素未过初始逐元素门槛。这是合成样本校准，不是模型准确率结论。
确认`max_num_imprecise_acc=0`在当前Triton降低为FP16 MMA（PTX已检查），故仅作诊断，不冒充FP8 Tensor Core优化。另发现QK与KQ矩阵方向影响低精度累加顺序；full/grouped改为与原生一致的KQ方向，继续验证而非直接放宽门槛。

### P1 完成（2026-11-06）

契约 `contract.py` / `module.py`：FP8 Q/K、FP32含Qscale权重、页64的payload+scale ABI、index-key位置输出及ratio1/2原token跨度；分组/1GiB分块规划；初始化与执行分开。原生适配明确N1/2/4与N6/8展开对照；FP32参考在 `reference.py`。
验证：14项CPU契约通过，15组 full/source/grouped 对原生回归与固定score稳定top-k/候选/graph通过；精确累加诊断路径也通过初始FP32门槛。默认compact24组×4consumer生命周期通过。源码方向与原生KQ一致，保留原FP8累加，不默认启用诊断FP16 MMA。
原始证据：`verify-module-native-parity.json`、`verify-compact-wgmma.json`、`calibration.json`（远端同目录，已/将同步本地）。数值回归检查针对原生路径，FP32误差单独报告；其门槛校准原因见前节，不能声称逐位一致或模型精度验证。
P2开始性能测量：`benchmarks/kernels/dsv41_indexer/03_p2.sh`；先GPU0代表形状，最终版本仍需P8重跑。

### P2 完成（2026-11-06）

24组候选生命周期×4consumer通过；融合/独立映射与固定score稳定top-k精确一致。正反A/B代表形状8例完成，Graph/Eager均记录，CV>10%追加三轮。一次发布+4consumer的成本已包含发布；不同Q/weights。
单卡256K、8行，Graph原生单consumer128.35–128.47μs，compact含发布35.12–35.30μs；四consumer512.12–512.21→123.16–123.67μs。仅本轮代表形状，最终版本需P8重测。
证据 `P2报告.md`、`p2-{forward,reverse}.{json,log}`、`p2-source.tgz`；知识库归档 `20261106-tp4-p2`。
下一步P3完整source/full路径分块与融合max消融，随后P4 verify实际共享key tile的性能实验。

### 总控及并行分工（用户要求，2026-11-06）

总控仅负责任务分配、约60秒轮询、证据审查、方案书及知识库归档，不再亲自实现kernel。
第4个独立子agent两次启动被工具拒绝：`agent thread limit reached`；实际3个子agent，4路GPU任务。

|执行agent|阶段|物理GPU|状态|
|---|---|---|---|
|grouped_p4|P3 full/source|0|已确认接管，独立p3/结果|
|prefill_p5|P5 prefill|1|实现与正确性|
|topk_p6|P6 top-k|2|360项selector+6项ragged正确性通过，性能待验收|
|grouped_p4|P4 grouped verify|3|16组正确性通过，pair流水线与调优中|

各GPU实验必须显式CUDA_VISIBLE_DEVICES只指定自己的物理卡；P3/P4同一agent管理两个独立进程/结果目录。
P7四卡通信测试必须等四卡独立实验释放后统一执行；P8最终回归按固定负载条件测量，不将并行实验中的CPU竞争误判为kernel变化。
所有阶段状态仍以最终实现、测试和归档证据为准，子agent口头完成不自动勾选。

### P3完成与八卡执行（2026-11-07）

P3已依据`results/indexer/20261107-resume/p3/REPORT.md`、55文件校验及原始JSON完成研究项验收。ratio1/2十组边界、真实1GiB arena、2.21GB地址均通过；max_abs≤9.54e-7。8×256K完整链路约699→277μs；短32×4K应保留full consumer组合约97→68μs。DeepGEMM零长度预取bug只在显式隔离包修复，patched/unpatched开销分别测量，失败composition-v1保留，v2重新冻结后完成。

用户已明确八卡可用。P7使用h20-4 GPU0–3；P3使用h20-dsk GPU3；P8 compact/topk使用GPU0；P4/P5最终回归分别GPU1/GPU2。P3/P4/P5批次已完成释放后转入审阅与最终调度，空闲卡用于必要补测，不把准备/JIT计入模块性能。

P7已完成真实四rank每rank280项检查，所有rank独立物理页；初轮退出时Graph引用清理问题保留日志，修复后v2/final正常exit0。compact正反144记录、verify128、projection48、ratio2 48、native full/chain144已落盘；最终源码验证/分项和报告尚未完成，P7仍未勾选。

历史checkpoint与新主机audit已发布到用户授权的Experiment-Results，提交`8ecdfea486762af54b6036b47e15d9aedc9e6c6f`，135文件。优化源码repo未推送。新增P3及P4/P5最终批次正在追加归档。

### P5迁移覆盖纠正（2026-11-07）

此前`backend="gluon"`调用`grouped_score.grouped_logits`，是grouped decode/verify内核用于prefill的对照，**不是**蚂蚁专用`sm90_fp8_prefill_keys_major.py`。已有`keys_major`实现K tile复用，但未完整覆盖专用实现的驻shared/TMA/三级缓冲/跨head流水/可见tile常驻调度。因此，“Gluon无额外收益”只适用于当时被测grouped路径，不能排除蚂蚁专用实现；已有P5性能数据和源码快照保留，覆盖结论收窄，P5整体重新保持待完成直到P5b验证收尾。

参考：https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/python/sglang/kernels/ops/attention/dsv4/sm90_fp8_prefill_keys_major.py

P7已完成并独立复核：终版6step真实4rank480项/rank，覆盖全零长度+物理页置换+恢复；动态native稳定full及source/consumer重测144条，旧native静态/非确定性记录禁准入。证据`p7/REPORT.md`及`p7/inventory.json`，P8的实际调度绑定另验。

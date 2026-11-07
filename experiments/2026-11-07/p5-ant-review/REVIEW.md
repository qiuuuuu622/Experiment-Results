# P5b：Ant 专用 prefill keys-major 的独立 CPU 审阅

**结论：值得移植的是数据驻留与异步调度；不能移植其 FP4 数值前提和 BF16 打分语义。** 现有 P5 的 Triton `keys_major` / grouped Gluon 路径没有完整复现 Ant 的专用 kernel，不能用旧路径的结果替代本次 P5b。本文仅审阅固定源码、现有 vLLM 合同及 CPU 数值反例，没有运行 GPU，也没有修改 P5 owner 文件。

## 固定来源

固定参考为 [Ant commit 6e6d5d9e 的原文件](https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/python/sglang/kernels/ops/attention/dsv4/sm90_fp8_prefill_keys_major.py)。下载副本 `sm90_fp8_prefill_keys_major.original.py` 共597行，SHA256 `0bdde39dd1ddb4805492badbd838aa40a51a7dc79f29c070d27b661fc3f66b3f`，已与 P5 实现者 `p5-ant-keys-major/upstream/` 副本核对一致。

同 SHA 的调用者 [sm90_fp8_prefill.py](https://github.com/antgroup/sglang/blob/6e6d5d9e13915adb01db0856b15d6908eaaa2c48/python/sglang/kernels/ops/attention/dsv4/sm90_fp8_prefill.py) SHA256 为 `b0c8984b2b012a22ca21565bb522ef39f5c2baede8dbe25faa6b0120c09bee12`。其235–272、304–340、399–406行补充说明 Q/K 原本来自 FP4 网格，并把 weights 转为 BF16。仓库许可证为 Apache-2.0，副本 `LICENSE` SHA256 `1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747`。本文件行号均指这两份固定原文件。

## 机制对照与必须保留的同步关系

|机制|Ant 原文件位置|移植要求与陷阱|
|---|---|---|
|32 query 行的所有 heads 驻 shared|34–40、179–194、339–346|Q tile是 `[32 heads,32 rows,128]`，不能只改 grid 名称便称为等价移植。固定 tile 会让少量 query 浪费驻留空间，需独立测交叉点。|
|Q/K 都用 TMA|189–194、256–264、535–540|Q global descriptor是 `[rows,H*128]`，每个head取 `[32,128]`；K descriptor是单请求的dense `[keys,128]`。不能直接拿 vLLM `[page,64,1,132]` cache ABI 构建此descriptor。|
|三级 K 流水|39、340–351、360–375、421–461|stage为 `(v-v0)%3`，phase为 `((v-v0)//3)&1`，不是全局 `v%3`。最初最多预取3 tile；消费者完成后才能覆盖stage。|
|head 双缓冲异步 MMA|98–130、216–238|每head发4个D32 MMA，下一head在前一head epilogue期间运行。`wait(4)`依赖这个4-group结构；若合成一次D128 MMA，必须重算wait深度，不能照抄4。最终 `wait(0)` 及每tile CTA barrier必须保留。|
|visible persistent 分配|45–94、353–357、393–419|prefix按每个32-row block的最大可见长度计算；每CTA分一段可见tile序号，不是扫描矩形全部Q×K。prefix重复项对应零可见row block，binary search必须跳过它们。|
|不可见输出|269–305、464–482|可见tile内逐query mask；完全不可见tile另行填-inf，包括输出stride尾部。若适配改成 vLLM 的 untouched-tail 合同，必须明确标注并验证所有下游只按有效长度读取，不得保留“全部尾部已清理”的声明。|

**K scale 索引必须属于当前消费者 tile。** 使用 `key_id=(v-lo)*128+lane`，其中 `lo` 是当前 row block 的可见prefix起点。预取函数的 `p_lo/p_hi` 可能已因 `v+3` 跨到未来 row block，不能用于当前scale取值；也不能直接取 `v*128+lane`。

先用普通全局load读取每tile的128个FP32 scale，再转换为 `key_layout`；它们在M/key轴广播，weights在N/query轴广播。若进一步把scale做三级共享流水，stage、phase和消费/覆盖时机必须与K payload相同。

原weights以head-major `[H,32]` 形式放入shared：全局索引为 `row*stride_wb+head`，shared切片为 `head*32+row`。将dtype改FP32后不能沿用bf16x2的打包解释。Q128KiB＋三级K48KiB＋FP32 weights4KiB，约180KiB shared（未计barrier/对齐）；应记录实际编译shared/register/spill，而非只报理论FLOPS。

## vLLM 数值合同与必须删除的舍入

目标公式（所有位置均为 index-key 坐标）：

```text
D[q,h,k] = dot(FP8_Q[q,h,:], FP8_K[k,:])
V[q,h,k] = max(D[q,h,k] * FP32_K_scale[k], 0)
score[q,k] = sum_h (V[q,h,k] * FP32_head_weight[q,h])
```

head_weight已经含Q scale，不再重复乘Q scale。FP8 K bytes和单key FP32 scale分别保存；把K scale先乘入E4M3 payload再量化会改变输入语义。即使K scale为正，也不要未经实验就把它移到head求和后：实数恒等不代表FP32舍入等价，尤其weights可正可负。

|原版舍入/类型|位置|FP8 vLLM适配|
|---|---|---|
|调用者把weights变BF16|`sm90_fp8_prefill.py:406`|删除。保持FP32原值，不能以“同公式”掩盖额外量化。|
|weights shared为BF16|keys-major:344–346|改FP32并核查shared加载、布局及广播。|
|D128点积结果先转BF16，融合ReLU|134–145|删除 `cvt.rn.relu.bf16x2.f32`；FP32中先乘K scale，再ReLU。|
|head贡献乘权重后再舍入BF16|145–149|删除 `mul.rn.bf16x2` 和位移拆包；保持FP32乘法/累加。|
|所有heads相加后最终转BF16|464|删除 `acc.to(bfloat16).to(float32)`，输出FP32。|

原版每head是4个D32 partial，再按 `((p0+p1)+p2)+p3` 组合，heads按顺序累加。vLLM现有路径使用不同dot累加/heads reduction；**去掉BF16之后仍不能承诺与原生DeepGEMM逐位相同**。FMA contraction也可能改变FP32最后一位；应保留完整数值诊断，不能把mathematical formula相同等同于浮点运算图相同。

## “D32 exact”不能继承：CPU反例

Ant tiled caller的304–308行明确依赖：Q/K各32维来自同一个UE8M0缩放的FP4网格。任意E4M3数值不具备这个共同窄网格。

已经生成 `fp8-d32-counterexample.json`：所有Q/K都是normal、有限、可精确表示的E4M3；第一项 `448×448`，其余31项均 `0.125×0.015625`。

- 精确实数和：200704.060546875。
- FP32逐项先加大项：200704.0。
- 先累计31个小项，再加大项：200704.0625。

精确和本身就不能被FP32精确表示。这不是某个H20 kernel的失败测量，而是直接推翻“任意FP8 D32 partial都精确”的算术前提。保留4×D32可用于忠实移植调度，但应把它作为待比较的累加策略，而非正确性的证明。不要照抄“tile/row split无关、bitwise”的原注释，也不要因为原注释提到 `max_num_imprecise_acc=0` 慢，就把当前FP8的精度要求据此取消。

## TMA、请求边界与生命周期

1. **单个K descriptor只属于一个请求。** 32Q tile不能混不同请求后继续用同一K descriptor。沿用现有 `PrefillIndexer.score_chunk(start,end,request)` 的per-request入口；若另行pack请求，pack/gather成本必须计入。
2. Q/K检查为E4M3、contiguous、16-byte aligned；D=128，当前业务H=32。权重FP32、scale FP32分别检查形状/stride/device，不能依赖原支持函数（它只检查Q/K形状和对齐）。
3. vLLM lengths通常int32，Ant wrapper转int64。若保留int32，不只改参数声明，还需让loop-carried `n_vis` 初始dtype与load一致；验证prefix长度累积范围。
4. chunk的Q切片与K准备缓冲的storage在graph期间固定；descriptor可在规划时复用，但length值改变时prefix必须每次重算。不能使用旧prefix或旧请求的scale地址。
5. 未来tile、最后不满32行/128keys、score stride尾部必须有明确mask/OOB合同。不能凭TMA零填充猜测正确，下面的边界用例必须实际运行。
6. 全零长度仍需产生合法的全无效输出；若保留-inf清理，不能忽略其全矩阵写成本。恢复非零长度后要验证prefix、TMA stage phase和输出均无上一步残留。

## 最低正确性实验，先通过再跑大矩阵

|类别|最小用例与断言|
|---|---|
|移除BF16的解析小例|单有效head、dot=1，weight=1.0001，K scale=1.3；另做两个大而异号weights近乎抵消。对FP32公式精确/严格核对，能直接捕获weights/epilogue的BF16残留。|
|scale/weights广播|不同query使用不同head one-hot权重，不同key设置不同正且非2幂的FP32 scale；明确期望逐 `(q,k)` 值，捕获把key scale广播成query scale或跨request复用。|
|任意FP8与累加|正常随机E4M3、重复K、零Q、正负取消、D32上述反例、大动态范围；先按每head dot→scale/ReLU→weighted sum逐级诊断。与当前DeepGEMM和FP32参考分别比较，报告nRMSE/max_abs/top-k边界分差。|
|tile边界|Q rows=1/31/32/33/63/64/65；keys=1/127/128/129/257；length=0/1/127/128/129/full。至少覆盖最后仅1行、仅1key和output stride按4对齐但非16对齐。|
|visible-prefix重复|多个32-row blocks，max-length形如 `[0,129,0,257,0]`，每block内再混不同长度；与host枚举可见坐标对照，保证空block跳过且每个合法score恰好覆盖。|
|persistent分段|grid_size=1/2/实际SM数，并使一个CTA拿少于3个tile、跨row-block边界、循环超过6个tile；检查K stage复用、barrier phase与scale不串位。|
|因果/污染|无效Q/K位置和score工作区预填NaN；有效输出必须有限且无future key；不可见尾部按声明合同验证。top-k不得读NaN尾部。|
|多请求/chunked/ratio2|至少两个独立物理页布局与不同非2幂scale的请求；不同历史prefix，chunk start非32倍数；ratio2记录真实index-key数，不把原token长度误当可见key长度。|
|graph生命周期|连续三步：混合可见→全零并页表重排→恢复非零且改scale/weight。复用相同storage与graph，恢复应匹配重新计算的参考；prefix/scale不能过期。|
|稳定selector|固定同一FP32 logits，含完全同分和top-512边界极小分差，稳定top-k集合精确；评分路径浮点差异与selector同分语义必须拆开验收。|

建议先做约20个正交小case，而非一开始全笛卡尔积。出现任意无法由实际score误差或精确同分解释的top-k变化，视为失败。沿用当前严格门槛做诊断，失败先定位，不自动放宽容差；原生FP8自身相对FP32的已知舍入作为单独对照，不能混成新kernel的豁免。

## 最低性能与编译核查

- 比较同一批FP8 Q/K/FP32 scale/FP32 weights：DeepGEMM、现有row_tiled/keys_major、Ant专用适配；先只比较打分，再比较 key gather＋prefix＋打分＋稳定top-k完整链。
- `prepare` 的key gather每请求每组合调用一次；Q/K TensorDescriptor构造、prefix工作区规划和初始化分开记录。动态prefix GPU pass、必要的scale加载以及invisible fill必须包含在执行成本中。
- CUDA Graph和eager分别测GPU与host wall；完整调用也做20 warmup、30×10、正反序及高波动追加轮。不能只报WGMMA核心时间。
- 边界比较 `rows×keys` 在约0.5M/1M/2M/4M以及更大形状，兼顾相同score数但不同rows/keys比例；Ant的原阈值是 `1<<21`，是该上游数值路径的经验值，不能直接迁移为vLLM阈值。
- 保存编译资源与PTX/SASS证据：TMA、异步WGMMA及wait实际存在，head双缓冲没有被编译器串行化，未因descriptor/accumulator hoist造成spill。仅代码中调用`is_async=True`不足以证明执行重叠。
- 无收益时保留结果并继续用DeepGEMM，不因“形式更接近Ant”就进入自动dispatch。P8只接受最后源码版本上真实通过的形状证据。

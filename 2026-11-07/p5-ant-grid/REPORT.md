# P5b persistent grid 消融：默认 78 并未掩盖收益

四个代表形状中，Ant grid=78 的完整调用 P50 都是四个 grid 中最低的；156/312 没有产生超过测量噪声的收益，39 明显退化。**保留默认78，不根据本次实验增加grid自动选择。** 这是四张 H20 各运行一个形状的独立单卡实验，不是 TP4 模块延迟。

## 冻结版本和范围

- Ant适配模块 SHA256：`957557aa4ac9944073ae4e2c2a370775df446f93437fa82de821b3051d3184c9`。
- P5不可变源包 SHA256：`31863c46ad5e956f06b9cbea72c9498fbeb17f13f1964fa44d7a37daea8c4280`；本地保存为 `frozen-p5-source.tar.gz`。
- 执行副本：`h20-4:/root/dsv41-indexer-resume-20261107/runs/p5-ant-grid-v1`。主P5目录和owner文件未修改。
- 本次仅新增 `benchmarks/kernels/dsv41_indexer/benchmark_ant_grid.py`；运行前后重新校验module、依赖、vendor及driver哈希，全部一致。
- GPU0/1/2/3分别运行16×64K、64×64K、256×16K、1024×16K，一个请求、ratio1、全部32 index heads。每个形状所有grid及两种基线共享同一组输入。
- keys-major各grid的有效grid均等于39/78/156/312（没有被tile数量截断）。输入为已有FP8 Q/K及FP32 weights/scale；visible length来自同一deterministic fixture，包含chunk的因果边界。

## 正确性

每个grid的FP32 score与grid78**逐位相同**（包括Ant定义的-inf尾部）；稳定top-k集合相同，prefix metadata相同，并通过3次完整graph replay。不是仅用重合率替代grid间的精确检查。

对DeepGEMM分别检验valid score的rtol=1e-3、atol=1e-4；保存nRMSE/max_abs、top-k重合率及变化项距离cutoff。所有输出唯一、合法；只有能由该行实际score误差解释的边界变化才允许通过。具体误差及每个prefix digest见原始JSON。

## 计时方法

每个variant/phase在相同形状上正反顺序各一次；20次预热、30样本×每样本10调用，CV>10%追加3轮，全部保留。eager和CUDA Graph分别测GPU时间与host wall。表格是所有正反及追加轮样本的中位数，单位μs，未选取最好一轮。

score-only之前统一完成key准备及Ant visible-prefix计算，prefix不计入score-only。full包含每次key gather、Ant prefix重建、打分和同一个Ant稳定selector。DeepGEMM自身返回logits的分配保留在其成本中；描述符/工作区预规划不在执行计时中。Ant会清理不可见score尾部，DeepGEMM保持其原有unspecified-tail合同；所有selector都受可见长度限制。全部variant同时驻留，显存峰值是进程总量。

### CUDA Graph完整调用 P50

|Q×K|DeepGEMM|现有keys-major|Ant grid39|Ant grid78|Ant grid156|Ant grid312|
|---|---:|---:|---:|---:|---:|---:|
|16×64K|610.64|56.18|162.35|94.93|107.36|112.35|
|64×64K|607.21|151.32|291.84|163.27|166.32|192.01|
|256×16K|171.68|154.19|297.84|169.56|173.13|199.43|
|1024×16K|644.57|570.16|1092.69|588.28|593.96|605.54|

### CUDA Graph score-only P50

|Q×K|DeepGEMM|现有keys-major|Ant grid39|Ant grid78|Ant grid156|Ant grid312|
|---|---:|---:|---:|---:|---:|---:|
|16×64K|592.44|37.88|142.06|74.22|86.70|91.66|
|64×64K|588.62|131.97|271.00|142.02|144.89|170.58|
|256×16K|151.08|133.63|273.92|146.68|150.03|176.51|
|1024×16K|582.40|507.98|1027.19|523.27|529.30|540.50|

## 判断

- 39个CTA只覆盖约半数SM，这四个形状上并未通过更长的tile重用弥补并行度损失。
- 增加至156/312个CTA引入更多persistent分段。观测到的完整延迟均不优于78；64×64K、256×16K、1024×16K的一些差距不足5%，不应宣称显著差异或新增dispatch。
- 相比现有keys-major，Ant grid78的full延迟在这四个形状上仍更高。对1024×16K约3%的差别按“小于5%保留旧路径”处理；不能以Ant优于此处DeepGEMM就忽略已有更快路径。
- 本次只排查默认grid是否掩盖收益，不证明其他grid/其他形状永远无收益，也不改变P5主矩阵结论。没有重跑其23形状大矩阵，没有修改自动调度。

## 复现

冻结目录中的每张卡分别运行（将`g`设为0、1、2、3）：

```sh
cd /root/dsv41-indexer-resume-20261107/runs/p5-ant-grid-v1
CUDA_VISIBLE_DEVICES=$g .venv/bin/python -m benchmarks.kernels.dsv41_indexer.benchmark_ant_grid --shape-index $g --grids 39 78 156 312 --expected-ant-sha256 957557aa4ac9944073ae4e2c2a370775df446f93437fa82de821b3051d3184c9 --snapshot-sha256 31863c46ad5e956f06b9cbea72c9498fbeb17f13f1964fa44d7a37daea8c4280 --output /root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p5-ant-grid/gpu${g}-shape${g}.json
```

4个JSON共192条variant/phase/顺序记录。每条包含P50/P90、raw samples、CV、wall samples及显存。4个log、源包、driver快照与manifest全部本地和远端保存。

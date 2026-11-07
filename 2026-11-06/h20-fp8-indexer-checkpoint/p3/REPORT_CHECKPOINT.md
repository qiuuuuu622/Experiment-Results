# P3 full/source 阶段记录（尚未完成阶段验收）

- 日期：2026-11-06
- 状态：两轮代表形状性能已完成；P3仍有待补验证，不勾选完成。
- 主机：h20-dsk，物理GPU0；没有模型、服务或端口。

## 已保存的实验

`forward.json`、`reverse.json`各2个代表形状，状态均为completed。每项20预热、30样本、每样本10调用，Eager/Graph各测，正反顺序；CV>10%追加3轮，所有样本及P90/墙钟/显存留在JSON中。
当前形状全部ratio=1；32个index heads×128，FP8 E4M3 Q/cache、FP32 key scale及head weights；本地页64；候选2048块×8，token top512。

每个形状包含12项source→publisher→consumer联合核验（Triton/Gluon/Hybrid三种source×四个不同Q/weights consumer）。在相同已发布候选上对照原生打分：最大绝对差4.77e-7/9.54e-7，固定score的稳定top-k精确匹配。融合/分开block-max路径输出一致。
已有15项contract/GPU测试日志保存，包含小预算共享score arena、ragged group和graph变长；尚不等同于真实1GiB arena验证。

## Graph P50结果（μs）

|顺序|行数|index keys|原生full|原生source|Hybrid source|Gluon fused source|原生source+4consumer|Hybrid source+4consumer|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|forward|1|65536|15.11|72.40|24.88|29.93|152.10|71.16|
|forward|8|262144|116.24|186.99|149.34|259.38|699.76|276.55|
|reverse|1|65536|15.21|72.42|24.87|29.97|152.08|71.23|
|reverse|8|262144|116.34|186.92|149.19|258.12|698.44|277.08|

Hybrid保留DeepGEMM打分，使用Ant确定性token/source selector，再发布候选供compact consumers复用。Source的token top512与候选块均输出。

**结论边界：**代表形状下Hybrid source及完整链路有收益；新Triton/Gluon full scorer本身更慢，不能据此替换原生full路径。8×256K时融合block-max比独立scan更慢，不能全量启用融合。Eager存在CPU调度影响，原始结果保留；这些不是模型吞吐或TTFT/TPOT结论。

## 复现与版本证据

实际脚本 `03_p3_gpu0.sh`，工作目录 `/root/dsv41-indexer-port`；各结果和日志同名。环境见 `environment.json`。
基线29be941d，Ant6e6d5d9e。7个安装Python wrapper已与HEAD哈希核验；预装vLLM二进制的构建commit未知，不称为HEAD编译产物。

**源码快照限制：**P3启动前未保存完整对应source hash/archive。`checkpoint-source.tgz`和`checkpoint-source.json`是现在保存的checkpoint，不能冒称已运行两轮的精确历史源码快照。之后P4为grouped helper添加了PAIR_ROWS；默认未配对路径语义保持，但最终回归仍必须重新绑定源码哈希。

## 待完成（未开始的新实验）

- ratio2真实index-key长度、短/空及尾部不满8的full/source/四consumer联合验证。
- 真实1GiB score arena、跨chunk输出与graph动态长度证据（已有小预算测试）。
- 4K/16K/128K和32/64/128请求的补充代表形状。
- 最终源码绑定的正反回归和完整P3报告；由P8统一决定dispatch。

本次checkpoint只保存已完成记录，未将上述待办标成完成。

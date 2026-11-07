# P8 P4 grouped 最终源码绑定回归

状态：completed。2026-11-07，h20-dsk物理GPU1，独立runs/p8-grouped-final-v1。GPU1已释放。仅函数/模块实验。

paired正确性31项通过；7个shape完成CTA选择及Eager/Graph正反序，另6个shape明确计量host候选共享资格重规划。没有复用旧性能记录。

|请求数|每请求query|index keys|CTA|原生full μs|grouped full μs|原生source μs|grouped source μs|逐行consumer μs|共享consumer μs|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|4|1|65536|512|21.14|42.91|83.44|50.78|24.48|24.50|
|4|2|65536|512|36.69|47.22|100.71|56.41|35.01|23.97|
|4|4|65536|512|55.62|61.32|121.34|74.25|55.86|28.89|
|4|6|65536|512|71.65|82.29|142.87|97.90|66.79|34.70|
|4|8|65536|512|95.13|100.32|165.24|118.98|78.47|40.61|
|32|6|65536|2048|580.60|511.68|677.37|589.37|327.82|177.07|
|1|6|262144|512|65.67|106.96|133.52|130.33|29.63|21.77|

表内为Graph正反P50平均值，仅作摘要，准入必须使用两次原始记录而不是只用平均值。full/source含打分与选择，consumer含本次候选发布。每项20次预热、30样本、每样本10调用；CV>10%自动追加3轮。

结论与限制：

- full仅32请求×6行×64K的当前形状有超过5%收益；其他测量形状保留原生。source多数形状有收益，但单请求×6行×256K约2.4%，不满足5%准入门槛。source收益包含选择器/融合差异，不全部归因于共享K。
- 共享consumer仅在共同候选前缀及最长行优先的预规划契约下有效。每步通用host资格重规划实测约0.38–17.6ms，超过kernel节省，不能隐藏这项成本。不同候选布局或变化破坏契约时应逐行fallback或显式重新规划。
- consumer1行无收益。多个query共享时仍须逐query遵守因果长度；verify包含NaN尾部、长度/候选变化、ratio1/2和ragged。N6/8原生明确为expanded_single_row基线。
- 仅最终paired路径按总控要求运行。旧默认/paired比较保留历史，不重复无益调优；本轮rowwise_gluon与grouped_gluon score/full/source均作为消融。
- 同期GPU0/2/3可能有其他授权实验，Eager墙钟保留共享CPU负载影响，不能冒充独占全机。GPU时间来自本GPU。

源码与复现：

- source.tgz SHA256：`4b373028dd3b469e42f425224c6570e03bd4d39fe98c6f04f5f1605aaa554061`。82文件覆盖模块、vendor、benchmark、tests及许可证。
- benchmark-only.patch仅修正旧硬编码GPU元数据和移除未使用的P5 Gluon性能分支；module/vendor字节不变。
- source-identity.json是运行前全部文件哈希；environment.json及native-identity.json记录runtime/package/header/二进制。原生是未覆写的安装DeepGEMM，未混用P3独立zero-prefetch patch；实际二进制构建commit未知。
- 完整命令run.sh；远端执行`bash /root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p8-grouped-final-v1/run.sh`。
- 原始结果final.json、planning.json、verify.json和同名日志。P8负责执行依赖哈希匹配及最终统一admission，本报告不自动授权新shape外推。

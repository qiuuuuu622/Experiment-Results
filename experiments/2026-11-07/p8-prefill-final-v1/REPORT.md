# P8 P5 prefill 最终源码绑定回归

状态：completed。2026-11-07，h20-dsk物理GPU2，独立runs/p8-prefill-final-v1。GPU2已释放。仅函数/模块实验。

正确性27项通过；smoke/matrix各正反序，共32条shape/mode/order性能记录。所有数据为本次冻结版本新跑；Gluon已无收益，本次性能不重新测试它，verify仍覆盖其正确性。

|模式|query|请求|index keys|ratio|原生μs|最快实测路径|最快μs|加速|
|---|---:|---:|---:|---:|---:|---|---:|---:|
|full|64|1|65536|1|626.53|keys_major_pipeline|151.07|4.15×|
|source|64|1|65536|1|705.47|keys_major_pipeline|166.05|4.25×|
|consumer|64|1|65536|1|638.82|row_tiled_pipeline|62.89|10.16×|
|full|1|1|4096|1|47.83|indirect_pipeline|7.29|6.56×|
|full|16|1|262144|1|2470.96|indirect_pipeline|192.88|12.81×|
|full|64|1|16384|1|164.03|row_tiled_pipeline|45.73|3.59×|
|full|64|1|32768|2|316.66|keys_major_pipeline|80.79|3.92×|
|full|64|1|131072|2|1251.04|keys_major_pipeline|300.23|4.17×|
|full|256|1|16384|1|176.57|keys_major_pipeline|151.95|1.16×|
|full|1024|1|16384|1|658.74|keys_major_pipeline|565.14|1.17×|
|full|4096|1|4096|1|374.15|deepgemm_pipeline|363.11|1.03×|
|source|64|2|65536|1|1387.44|keys_major_pipeline|195.63|7.09×|
|source|256|1|16384|2|280.96|keys_major_pipeline|161.05|1.74×|
|consumer|16|1|262144|1|2475.33|row_tiled_pipeline|37.29|66.37×|
|consumer|256|1|65536|1|736.82|keys_major_pipeline|169.40|4.35×|
|consumer|1024|1|16384|2|704.39|keys_major_pipeline|585.18|1.20×|

32记录中各优化路径对原生最大绝对差1.43051147e-06，最大nRMSE7.7222827e-08，最低top-k集合重合率100.00%。固定logits的稳定选择要求精确；不宣称所有打分逐位相同。

表内为Graph正反P50平均，准入需分别检查两轮原始结果与波动。每次都包含key准备、打分、token top-k；source包含候选选择，consumer包含候选规范化和逻辑映射。20预热、30样本、每样本10调用；CV>10%追加3轮，所有样本保留。

限制和调度条件：

- 原生采用DeepGEMM FP8 prefill、原生gather和selector。consumer原生全长打分后mask；compact路径收益包含减少候选打分范围，不能当作同规模GEMM加速。
- 不直接采用2M scores门槛。query/key长宽比例、source/consumer模式均影响胜负；q4096×4K差异不足5%保留原生。P8自行计算每个已测形状准入，未测形状不外推。
- 性能consumer使用每请求共享候选。每query独立集合可用request alias表达且正确性已测，但没有完整性能矩阵。请求归属/分组变化需要重新plan/capture。
- FP8 E4M3 cache与scale语义不变；ratio2用index-key坐标（对应压缩span），不能把其输出冒称唯一原token。score预算1GiB，未测超大4096×256K组合。峰值含同进程其他对照variant驻留缓冲，不是单模块独占峰值。
- 同期其他GPU有授权实验，CPU墙钟可能受共享host负载影响；原始墙钟及Eager数据保留。

源码与复现：

- source.tgz SHA256：`faf047d61f0c6603bbd4f6d7d4b7253c3a5eada0787290b5e7e58a78fca40dd4`。82文件含完整模块/vendor/benchmark/test/许可证。
- benchmark-only.patch记录元数据GPU修正和已排除Gluon性能分支；module/vendor保持本地冻结字节。source-identity.json运行前哈希，post-run-identity.json运行后全部源码与native身份再次核对。
- environment.json及native-identity.json记录软件版本、native包源码/header/二进制哈希；本次未覆写安装DeepGEMM，不混用P3独立zero-prefetch patch，二进制构建commit未知。
- 复现执行run.sh，远端`bash /root/dsv41-indexer-resume-20261107/results/indexer/20261107-resume/p8-prefill-final-v1/run.sh`；原始数据smoke/matrix-{forward,reverse}.json及同名log、verify.json/log。
- P8负责最后执行依赖一致性和统一admission。本报告不是四卡收益或全模型准确率/TTFT/TPOT报告。

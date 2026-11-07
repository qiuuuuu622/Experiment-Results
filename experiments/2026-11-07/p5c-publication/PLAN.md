# P5c公开发布准备（尚未发布）

本目录只保存独立CPU发布核验；不修改owner代码、raw、方案书或知识库。

- 最终主批：`p5c-perquery/final-v1/`，固定source包`b4c1a3dbda6778ea48d226b44ff77809dce604a50fd63b9b3aac15184e83e41b`、core `77a7ffbf34ebe4495e560612c9b6943b17214b5fd01a9cdfe613d5887dbe9b1c`。
- 必须等4个group/order JSON完成及after-a/after-b身份齐备；核验8形状、20warm/30×10、正反同候选输入、全部重试、GPU与wall样本、1GiB score arena和连续chunk。
- 保留verify-v1–v5失败/中止及各自source，不把历史pass或pilot冒充最终版本；以owner最终报告明确采用的v6/final、P8复核、P7实际四卡验证为验收证据。
- 最终审阅需绑定77a7...；早期review SOURCE中的78aa...作为历史身份保留，不能替换字段冒充重新审阅。
- 公开目录沿用`2026-11-07/p5c-perquery/`、`p5-perquery-review/`及P7最终验证目录原名；先逐项核对owner/审阅manifest，再导出。外部artifact链接全部在PUBLIC_DEPENDENCIES增加原SHA→公开SHA映射。
- 内部私有校验清单保留为ORIGINAL_PRIVATE_*；公开完整性以根SHA256SUMS为准。当前根book被新book替代前保留公开历史副本，旧dependency映射转向历史字节。
- 根book仅在总控确认最新完成状态后复制。私有证据不改；P5b既有公开证据不覆写。
- 递归检查tar/ZIP/Torch文件、许可证、个人Mac路径、token/私钥/密码、无关主机标识及100MiB限制。若GPU UUID需脱敏，采用确定性伪名以保留四卡唯一性，并在PUBLIC_EXPORT记录；数字/布尔/状态/源码计算逻辑保持原样。
- `verify_repository.py`核验全部公开文件与dependency mapping；只有owner+独立审阅完成且根book获准后commit/push结果repo。随后HEAD=origin/main=ls-remote，git clean。不得push优化代码repo。

`audit_final.py`是其中的采样/源码/工作区门槛；缺结果会返回not_ready，不伪造pass，也不能替代P8数值/准入审阅或P7四卡验收。

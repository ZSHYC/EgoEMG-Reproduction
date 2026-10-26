# Classic：最终源码真实 GPU 工程检查

2026-10-26 对匹配公开提交 `658d8c675261daa68836b2b24de0b5e4c57a202c` 的冻结源码，各执行连续两轮、单轮保存、新进程恢复三个真实 GPU 路径。**VEMG2Pose 已通过限定验收；NeuroPose 未通过预测门槛。** [聚合证据](classic-formal-gpu-engineering.json)保留两者实际差异，不把六个训练进程成功当作两个模型都通过。

| 检查 | VEMG2Pose | NeuroPose |
| --- | --- | --- |
| 原批量／每工程epoch训练手样本 | 600／1,200 | 256／512 |
| 每epoch完整验证手样本 | 1,576 | 1,576 |
| 连续／prefix／恢复实际更新 | 4／2／4 | 4／2／4 |
| 所有消费的train／val批次精确核对 | 4／6批通过 | 4／14批通过 |
| checkpoint到实际对象的完整状态恢复 | 精确一致 | 精确一致 |
| 下一epoch首样本更新后最大预测差 | 0.0009765625 rad | 0.0015869140625 rad |
| 固定逐元素门槛的最大比例 | 0.663 | 1.509；8／171,380个元素超限 |
| 最终浮点模型及AdamW moments | 不逐位相同，如实记录 | 不逐位相同，如实记录 |
| 原始三进程证据汇总与新进程复核 | 均成功 | 汇总拒绝，未生成acceptance |

两者的全局、sampler与validation RNG相同，fresh恢复的专用train-loader均精确多一次原生int64 base-seed draw；不能称完整RNG相同。AdamW step/参数组、scheduler、scaler及非浮点counter继续严格比较。原始失败记录保留，NeuroPose的固定门槛未放宽，也未启动其完整训练。

预测比较仅针对独立模型副本的eval/no_grad首个样本，见[事先固定的规则](../docs/classic-canonical-comparison.md)。它不比较整批训练BatchNorm输出，也不证明整条训练轨迹或科研收敛。正式train/val仍需遍历101,714／1,576手，保留max100、patience50、val_loss/min、原0.02目标及同B。

VEMG2Pose已据实际验收另行启动完整训练，运行身份为full cohort，没有engineering训练限制。该训练尚未完成，尚无最终测试成绩。Task6的完整validation报告校准、正常终止后的所选best测试及独立报告仍欠；七项既有legacy成绩保持原样。

匹配源码的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40330846529)两项成功：官方CPU1173 passed／3真实CUDA skip（358.46s），轻量739 passed／390依赖skip（18.91s），pip check通过。本机同78文件组为1173／3（281.44s）及739／390（23.50s）。这些CPU检查不替代上面的实际GPU结果，也不覆盖仍在开发的报告执行流程。

# 进行中的训练快照

生成时间：2026-10-24T07:04:54+00:00。这些是生成时保存的静态快照，**不是实时监控，也不是最终测试成绩**。

| 模型 | 日志收集时间（UTC） | 优化器更新 | 完整验证记录 | 入口 |
| --- | --- | ---: | ---: | --- |
| EMGFormer-S | 2026-10-24T07:00:04+00:00 | 7283 | 57 | [状态](emg-s/status.md) · [曲线](emg-s/training-curves.svg) |
| EMGFormer-M | 2026-10-24T07:00:04+00:00 | 5584 | 43 | [状态](emg-m/status.md) · [曲线](emg-m/training-curves.svg) |

所有状态均为 `benchmark_eligible=false`。输入日志与公开状态的 SHA256 绑定见 [snapshots.json](snapshots.json)；原始日志仅私下保留。

训练目标函数与验证角度 MAE 分开绘制；未使用测试数据、测试指标或检查点权重生成这些快照。结果导出与独立审计要求见[结果指南](../../docs/results.md)。

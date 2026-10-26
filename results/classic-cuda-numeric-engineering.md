# Classic模型：真实CUDA限定数值与完整状态恢复检查

2026-10-26 08:31:56–08:33:18 UTC，两种原模型各完成prefix、fresh-process resume、完整checkpoint verify，六个阶段均退出成功。[聚合证据](classic-cuda-numeric-engineering.json)保留实际源码、运行时、事件和文件身份；无测试集模型成绩。

| 模型 | 尝试 | 实际optimizer执行 | 注入Inf跳步 | 自然overflow跳步 | 原预测shape | resumed峰值reserved |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| NeuroPose | 8 | 7 | 1 | 0 | 1×22×7790 | 247,463,936 bytes |
| vemg2pose | 8 | 7 | 1 | 0 | 1×22×6000 | 1,258,291,200 bytes |

采用原Lightning自动优化、AdamW/Cosine、clip0、有效指尖系数0.02和默认16-mixed GradScaler。首步实际更新；第二次尝试刻意注入Inf后optimizer未执行、参数不变、scale从65536退至32768；随后六次尝试正常更新。新进程恢复模型、optimizer、scheduler、loop/callback、scaler与RNG，prefix结束快照与resume开始完全一致。

每轮仅固定4个train与2个validation手样本，batch1/workers0，共两个工程epoch。完整队列hash/finite扫描另见[CPU队列检查](classic-cohort-cpu-engineering.md)。**这不是600/256批量容量、正式完整训练、验证选模或测试成绩。** 内存数值只属于本次batch1检查，不能线性外推。

先前strict deterministic运行在原插值CUDA反向被实际拒绝，失败产物保持。此次显式采用canonical_backend_nondeterministic_v1：deterministic/cudnn deterministic/benchmark为false、matmul high。完整状态与RNG的恢复一致不构成后续轨迹或逐位确定性保证。

实际执行源为ec2d5b1归档加固定cfbc限定检查文件，列出的7个运行时文件与公开9b5f493逐字节相同；不把较晚公开提交改写为当时checkout。公开仅含聚合和内容hash，不分发检查点、参与者键、机器路径或设备身份。

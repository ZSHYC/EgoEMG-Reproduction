# Classic模型：真实原批量train/validation容量检查

2026-10-26 09:39:05–09:47:40 UTC，六个新进程按固定阶梯完成，均exit0。[聚合证据](classic-cuda-capacity-engineering.json)与[逐字节相同的实际工具](../tools/classic_capacity.py)保留原输入、源码、真实optimizer/scaler和内存身份。每个进程先重验原完整数值检查checkpoint，再独立完成全队列hash/finite扫描。

| 模型 | 实际批量阶梯 | 每child实际optimizer执行 | 每child完整val手样本 | 目标B的train峰reserved | 目标B的val峰reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| vemg2pose | 32→128→600 | 2/2 | 1,576 | 13.488 GiB | 13.488 GiB |
| NeuroPose | 16→64→256 | 2/2 | 1,576 | 16.928 GiB | 16.928 GiB |

全部12次尝试均实际更新，无自然overflow、无注入或调整scaler；train共2,192个手样本、val共9,456个手样本/197个batch。每级train和val使用同一B，保留原最后不足批量及batch-union FK归约。VEMG600的val批量为600/600/376；Neuro256为256×6/40。两阶段峰reserved均低于实际设备总量85%，第二train尝试及val结束剩余均至少4GiB。

reserved包括真实缓存持有量；val开始重置峰计数，但没有empty_cache，不能把该列当作纯val临时激活量或训练吞吐。实际GPU为A80080GB、原16-mixed默认GradScaler、clip0、有效指尖0.02、单rank/workers0及显式canonical非确定性backend。

**目标600/256的这次短程容量门槛已通过，正式完整训练尚未运行。** 每级只有两个train batch，未做val-best/早停竞争、test或长期内存稳定性检查；新的正式训练源码仍需自己的实际GPU验收。它不证明原六rank训练、全局batch/Neuro BatchNorm等价或逐位续训保证。完整队列CPU与原batch1数值/状态恢复证据分别见[CPU报告](classic-cohort-cpu-engineering.md)、[数值报告](classic-cuda-numeric-engineering.md)。

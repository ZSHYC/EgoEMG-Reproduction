# 已完成 MAE 与论文原表的数值参照

截至2026-10-29，已完成11个实验设置的训练与最终测试；论文核心指标尚未全部复现到位。已完成结果的总体MAE数值均高于PDF原表，但数据和协议未完全对齐，因此下表只作数值参照，不能视为同条件性能排名。

论文Table2/3（PDF第8页）是角度MAE，越低越好；Avg为跨测试条件的样本加权均值，条件列的±为跨用户标准差。当前结果使用有效角度加权的pooled MAE，中心和轨迹分列。原表只显示一位小数，差值没有比原表更高的实验精度。

## EMG轨迹：对应Table2

| 模型 | 论文 Avg (°) | 当前 MAE (°) | 数值差 (°) | 协议组 |
|---|---:|---:|---:|---|
| [EMGFormer-S](../completed/emgformer-s/metrics.json) | 14.7 | 16.29 | 约+1.59 | 旧版12,000点/0.01 |
| [EMGFormer-M](../completed/emgformer-m/metrics.json) | 14.2 | 16.49 | 约+2.29 | 旧版12,000点/0.01 |
| [EMGFormer-L](../completed/emgformer-l/metrics.json) | 14.2 | 16.59 | 约+2.39 | 旧版12,000点/0.01 |
| [VEMG2Pose](../original-classic/vemg2pose/metrics.json) | 15.9 | 16.45 | 约+0.55 | 原生7,790点/0.02、旧版队列 |
| [NeuroPose](../original-classic/neuropose/metrics.json) | 16.1 | 16.36 | 约+0.26 | 原生7,790点/0.02、旧版队列 |

## 视觉及融合中心：对应Table3

| 模型 | 论文 Avg (°) | 当前 MAE (°) | 数值差 (°) | 协议组 |
|---|---:|---:|---:|---|
| [ResNet-18](../completed/resnet18/metrics.json) | 5.9 | 7.78 | 约+1.88 | 旧版12,000点/0.01 |
| [ResNet-50](../completed/resnet50/metrics.json) | 5.3 | 6.64 | 约+1.34 | 旧版12,000点/0.01 |
| [ResNet-152](../completed/resnet152/metrics.json) | 5.1 | 5.98 | 约+0.88 | 旧版12,000点/0.01 |
| [ViT-S](../original-vision/vit-small/metrics.json) | 6.0 | 6.46 | 约+0.46 | 原生7,790点/0.02、旧版队列 |
| [ViT-B](../original-vision/vit-base/metrics.json) | 5.8 | 6.76 | 约+0.96 | 原生7,790点/0.02、旧版队列 |
| [RN18 + EMGFormer-S](../completed/fusion-rn18-s/metrics.json) | 5.4 | 6.79 | 约+1.39 | 旧版12,000点/0.01 |

EMGFormer S/M/L的轨迹值来自原训练器最终测试；其独立锁定复算只重新累计中心指标，未独立重推完整轨迹。其他结果的具体复算级别与输入文件SHA见JSON。

同一旧版测试人口内，RN18→RN18+S为7.784887°→6.793533°，误差减少0.991354°（12.73%）；论文为5.9°→5.4°。这支持该旧版设置中融合改善视觉的方向，不支持相对论文的收益优越性。

当前没有证明论文所有模型排序：旧版EMGFormer M/L未得到论文相对S的优势；本次原生队列的ViT-B也未优于ViT-S。不同协议组不做受控交叉排名。

## 核心缺口

- Table3：ViT-L、WiLoR和ViT-S+S融合的最终完整测试成绩仍欠；原生EgoEMG EMGFormer的重新资格/完整训练仍欠。
- Table11/13：独立的EMG2Pose标准基准没有完整250轮训练和科学测试结果；不能把EgoEMG的MAE与该表直接相减。
- Figure13：完整自遮挡分层曲线、Spearmanρ=0.049/p=0.001对应的复算仍欠。
- 数据版本/标签处理/测试人口、窗口和损失、单rank与六rank等差异未消除；尚无PDF数值等价结论。

原始报告与文件字节保持不变。此参照不新增科学实验设置，不把中途验证损失当作最终测试MAE。完整范围见[全文状态](../../docs/full-paper-status.md)。

[机器可读指标与来源SHA](completed-mae-20261029.json) · [CSV](completed-mae-20261029.csv)

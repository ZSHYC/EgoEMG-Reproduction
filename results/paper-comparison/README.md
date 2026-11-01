# 已完成 MAE 与论文原表的数值参照

截至2026-11-01，已完成14个实验设置的训练与最终测试，新增WiLoR完整30轮及ViT-S融合完整150轮、最终测试、独立重放和报告复核。论文核心指标尚未全部复现到位。当前各项总体MAE数值均高于PDF原表；数据和协议未完全对齐，下表只作数值参照。

论文Table2/3（PDF第8页）为角度MAE，越低越好；Avg为跨测试条件的样本加权均值，条件列的±为跨用户标准差。当前结果使用有效角度加权的pooled MAE，中心和轨迹分列。原表只显示一位小数，差值不代表更高的实验精度。

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
| [ViT-L](../original-vision/vit-large/metrics.json) | 5.4 | 6.23 | 约+0.83 | 原生7,790点/0.02、旧版队列 |
| [WiLoR](../original-vision/wilor/metrics.json) | 4.7 | 5.11 | 约+0.41 | 原生7,790点/0.02、旧版队列 |
| [RN18 + EMGFormer-S](../completed/fusion-rn18-s/metrics.json) | 5.4 | 6.79 | 约+1.39 | 旧版12,000点/0.01 |
| [ViT-S + EMGFormer-S](../original-fusion/vit-small-emgformer-small/metrics.json) | 5.5 | 6.34 | 约+0.84 | 原生16槽/7,790点/0.02、旧版队列 |

EMGFormer S/M/L轨迹值来自原训练器最终测试；其独立锁定复算只重新累计中心指标，未独立重推完整轨迹。其他结果的复算级别和输入文件SHA见JSON。

新增ViT-L完成132轮/74,184次更新，原150轮上限及patience30早停保留；从真实108轮检查点续训，六域恢复与完整验证/测试重放均通过。实际验证选优为第102轮。全部中心MAE为6.229674°，相对论文5.4°的数值差约+0.83°；不同数据和协议下，该差值不能单独归因于模型实现。见[报告与来源](../original-vision/vit-large/README.md)。

同一旧版测试人口内，RN18→RN18+S为7.784887°→6.793533°，误差减少0.991354°（12.73%）；论文为5.9°→5.4°。该旧版设置支持融合改善视觉的方向，未建立相对论文的收益优越性。

当前没有证明论文所有模型排序：旧版EMGFormer M/L未得到论文相对S的优势，本次ViT-B也未优于ViT-S。ViT-L这一次MAE低于本次S/B；单seed、旧版队列结果不建立多seed架构结论。不同协议组不做受控交叉排名。

新增WiLoR完成30轮/336,720更新，6轮后显式六域恢复，验证best为第30轮；全部中心5.107042°，原表4.7°，数值差约+0.41°。ViT-S融合完成150轮/296,550更新，无早停，32轮后原生六域恢复，验证best为第108轮；全部中心6.342238°，原表5.5°，数值差约+0.84°。两项各8,612次消费/4,306唯一手样本，完整typed预测新进程重放一致，7/8个评价阶段均退出0。续训不保证与不中断训练逐位同轨迹，原始失败/前缀和原始六文件保留。

## 核心缺口

- Table3的七视觉与两融合均已有可得旧版队列最终报告；这不消除与PDF的数据/训练/评价协议差异，也不证明同条件数值等价。
- 原生官方配方EgoEMG EMGFormer S/M/L仍缺新的GPU资格与完整训练；既有legacy结果按原协议保留。
- Table11/13：EMG2Pose标准基准没有完整250轮训练与科学测试，不能直接拿EgoEMG MAE与该表相减。
- Figure13：完整自遮挡分层曲线及Spearmanρ=0.049/p=0.001对应复算仍欠。
- 数据版本、标签处理、测试人口、窗口/损失、掩码/聚合与单rank/六rank等差异未消除；没有PDF数值等价结论。

原始报告与历史2026-10-29、2026-10-30汇总均保留。此表只纳入通过最终测试与复核的实验；当前训练轮数不计为新科学结果。完整范围见[全文状态](../../docs/full-paper-status.md)。

[机器可读指标与来源SHA](completed-mae-20261101.json) · [CSV](completed-mae-20261101.csv) · [核心运行快照](../core-priority-status-20261101.json)

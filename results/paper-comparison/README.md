# 已完成MAE与论文原表的数值参照

2026-11-02更新：14个模型条目中，ResNet-18主行采用新完成的原生150轮结果；其完整测试、独立typed重放及退出后全部52来源核验已结束。历史legacy RN18结果与2026-11-01完整快照保留，没有覆盖旧实验。模型家族数量不因同一架构补跑原生配方而增加。

下表为数值参照，不能作为同条件架构排名。论文Avg按样本跨条件加权，当前为有效角度pooled；条件mean±SD与Avg分开。数据版本、训练来源、单rank、测试人口、mask和聚合等尚未全部对齐。原表只显示一位小数，差值写成约数。

## EMG轨迹：Table2

| 模型 | 论文Avg (°) | 当前MAE (°) | 数值差 (°) | 协议 |
| --- | ---: | ---: | ---: | --- |
| [EMGFormer-S](../completed/emgformer-s/metrics.json) | 14.7 | 16.29 | 约+1.59 | legacy12,000点／0.01 |
| [EMGFormer-M](../completed/emgformer-m/metrics.json) | 14.2 | 16.49 | 约+2.29 | legacy12,000点／0.01 |
| [EMGFormer-L](../completed/emgformer-l/metrics.json) | 14.2 | 16.59 | 约+2.39 | legacy12,000点／0.01 |
| [VEMG2Pose](../original-classic/vemg2pose/metrics.json) | 15.9 | 16.45 | 约+0.55 | 原生或明确派生7,790点／0.02，旧版队列 |
| [NeuroPose](../original-classic/neuropose/metrics.json) | 16.1 | 16.36 | 约+0.26 | 原生或明确派生7,790点／0.02，旧版队列 |

## 视觉及融合中心：Table3

| 模型 | 论文Avg (°) | 当前MAE (°) | 数值差 (°) | 协议 |
| --- | ---: | ---: | ---: | --- |
| [ResNet-18](../original-resnet/resnet18/metrics.json) | 5.9 | 6.61 | 约+0.71 | 原生或明确派生7,790点／0.02，旧版队列 |
| [ResNet-50](../completed/resnet50/metrics.json) | 5.3 | 6.64 | 约+1.34 | legacy12,000点／0.01 |
| [ResNet-152](../completed/resnet152/metrics.json) | 5.1 | 5.98 | 约+0.88 | legacy12,000点／0.01 |
| [ViT-S](../original-vision/vit-small/metrics.json) | 6.0 | 6.46 | 约+0.46 | 原生或明确派生7,790点／0.02，旧版队列 |
| [ViT-B](../original-vision/vit-base/metrics.json) | 5.8 | 6.76 | 约+0.96 | 原生或明确派生7,790点／0.02，旧版队列 |
| [ViT-L](../original-vision/vit-large/metrics.json) | 5.4 | 6.23 | 约+0.83 | 原生或明确派生7,790点／0.02，旧版队列 |
| [RN18 + EMGFormer-S](../completed/fusion-rn18-s/metrics.json) | 5.4 | 6.79 | 约+1.39 | legacy12,000点／0.01 |
| [WiLoR](../original-vision/wilor/metrics.json) | 4.7 | 5.11 | 约+0.41 | 原生或明确派生7,790点／0.02，旧版队列 |
| [ViT-S + EMGFormer-S](../original-fusion/vit-small-emgformer-small/metrics.json) | 5.5 | 6.34 | 约+0.84 | 原生16槽／7,790点／0.02，旧版队列 |

## 新原生RN18的证据与边界

[完整结果与真实训练曲线](../original-resnet/resnet18/README.md)：150轮／33,750更新，验证best第136轮，8,612消费／4,306唯一手样本。唯一首遍MAE **6.607195°**，有图子集6.469979°；论文RN18 Avg5.9°，数值差约+0.71°。原生batch样本加权与全部消费pooled均另报，完整精度值见原始JSON。

原19阶段/40拥有者及额外独立收集4拥有者均关闭。52个完整输入57.145GB的SHA/文件属性、全部原始预测与检查点链已在退出后重读，未额外运行模型。现仍是41段旧发布、独立train内时间验证和单rank适配；未证明PDF同条件数值等价。

## 历史匹配融合收益保持原来源

原legacy RN18→RN18+S为 **7.784887°→6.793533°**，减少0.991354°（12.73%），来自同一旧人口和原配方。两份原报告及其原始文件SHA保留。该收益没有换用当前原生6.607195°基线；新基线与旧融合的窗口、配方和人口不完全相同，不能混算所谓融合增减。

## 核心剩余项

- 原生S/M/L已完成200轮和中心评价，完整轨迹仍待执行，不能用中心MAE替换Table2轨迹行。
- 原生RN50/152的接续与完整评价、RN18+S本折cotrain/融合训练及兼容来源分析仍欠。
- EMG2Pose三个全源250轮及Table11/13最终测试仍欠；S stride3900与M/L780已按官方配置核对，未缩小源数据。
- ViT-S旧版17,168点密分析已完成，ρ=0.023640393、p=0.002283599；与PDF0.049／0.001不同，RN18路线及真实相机/人口等价仍欠。
- markers2mano、完整多模态与marker输入、腕角/IK重生成和全文其他项目仍需逐项验收，不能由这些MAE宣布整篇完成。

[当前机器可读指标与来源SHA](completed-mae-20261102.json) · [当前CSV](completed-mae-20261102.csv) · [历史2026-11-01快照](completed-mae-20261101.json) · [全文清单](../../docs/full-paper-status.md)

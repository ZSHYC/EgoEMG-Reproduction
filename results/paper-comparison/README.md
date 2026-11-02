# 已完成MAE与论文原表的数值参照

2026-11-02更新：14个模型条目的RN18与RN50主行现采用完整原生／明确派生协议结果。各自完整测试、独立typed重放与退出后52来源核验均结束；历史legacy行和既有快照完整保留。14指模型／表格条目，不是全部不同协议训练次数。

下表为数值参照，不构成同条件架构排名。PDF Avg按样本跨条件加权，当前中心主行为有效角度pooled；条件mean±SD另列在各模型报告。数据版本、训练来源、单rank、人口、mask与聚合尚未全部对齐。论文Avg仅一位小数，差值按约数理解。

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
| [ResNet-50](../original-resnet/resnet50/metrics.json) | 5.3 | 5.88 | 约+0.58 | 原生或明确派生7,790点／0.02，旧版队列 |
| [ResNet-152](../completed/resnet152/metrics.json) | 5.1 | 5.98 | 约+0.88 | legacy12,000点／0.01 |
| [ViT-S](../original-vision/vit-small/metrics.json) | 6.0 | 6.46 | 约+0.46 | 原生或明确派生7,790点／0.02，旧版队列 |
| [ViT-B](../original-vision/vit-base/metrics.json) | 5.8 | 6.76 | 约+0.96 | 原生或明确派生7,790点／0.02，旧版队列 |
| [ViT-L](../original-vision/vit-large/metrics.json) | 5.4 | 6.23 | 约+0.83 | 原生或明确派生7,790点／0.02，旧版队列 |
| [RN18 + EMGFormer-S](../completed/fusion-rn18-s/metrics.json) | 5.4 | 6.79 | 约+1.39 | legacy12,000点／0.01 |
| [WiLoR](../original-vision/wilor/metrics.json) | 4.7 | 5.11 | 约+0.41 | 原生或明确派生7,790点／0.02，旧版队列 |
| [ViT-S + EMGFormer-S](../original-fusion/vit-small-emgformer-small/metrics.json) | 5.5 | 6.34 | 约+0.84 | 原生16槽／7,790点／0.02，旧版队列 |

## 新原生ResNet结果

[RN18完整结果与曲线](../original-resnet/resnet18/README.md)：150轮／33,750更新、验证best136，唯一首遍6.607195°、有图6.469979°；原19阶段／40拥有者及4额外收集者关闭。

[RN50完整结果与曲线](../original-resnet/resnet50/README.md)：150轮／75,900更新、验证best140，唯一首遍**5.880998°**、有图**5.739818°**；论文Avg5.3°，描述性数值差约+0.58°。当前9阶段／20拥有者、20个已复用原资格消费者及额外4收集者均关闭，完整52来源57.145GB、检查点、预测及报告重读通过。

RN50是原RN18配置覆盖backbone和Table8 batch400的明确派生，独立作者YAML未发布。中断后六域恢复及共享指定GPU均披露；最近一次fit调用和接续流程elapsed不是从第0轮累计的论文独占GPU工时。两个模型仍使用旧版41段、独立train内时间验证和单rank，未证明PDF等价。

## 历史匹配融合收益保持原来源

legacy RN18→RN18+S仍为**7.784887°→6.793533°**，减少0.991354°（12.73%）。这保持原配方／人口／报告SHA，没有换用当前原生RN18基线，更没有使用新RN50混算收益。

## 核心剩余项

- 原生S/M/L完整200轮及中心报告已结束；轨迹队列已开始处理S，完整Table2新轨迹结果仍待验收。
- 原生RN152继续训练；RN18融合完整资格结束后已进入独立正式150轮，最终测试仍欠。
- EMG2Pose三项完整来源250轮及Table11/13最终测试仍欠，官方S stride3900／M/L780保持。
- ViT-S旧版17,168点遮挡分析ρ=0.023640393／p=0.002283599保留；RN18路线、物理相机／人口等价仍欠。
- markers2mano、精确作者来源混合、marker／腕角／IK重生成及其余全文项目仍需完成，不能由这些MAE宣布整篇完成。

[最新JSON与来源SHA](completed-mae-20261102-rn50.json) · [最新CSV](completed-mae-20261102-rn50.csv) · [此前2026-11-02快照](completed-mae-20261102.json) · [2026-11-01快照](completed-mae-20261101.json) · [全文清单](../../docs/full-paper-status.md)

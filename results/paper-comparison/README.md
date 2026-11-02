# 已完成MAE与论文原表的数值参照

2026-11-02更新：14个模型／表格条目中，EMGFormer-S主行采用新验收的原生完整轨迹；RN18与RN50主行保留此前完整原生／明确派生结果。旧legacy协议行和全部既有JSON／CSV快照保留，模型家族数量不因补充协议而增加。

下表只作数值参照，不构成同条件架构排名。PDF Avg为跨条件样本加权；当前原生轨迹／中心主行为其各自有效角度pooled，条件跨用户mean±SD另列。数据版本、训练来源、单rank、人口、mask和聚合仍未完全对齐。论文只显示一位小数，差值按约数理解。

## EMG轨迹：Table2

| 模型 | 论文Avg (°) | 当前MAE (°) | 数值差 (°) | 协议 |
| --- | ---: | ---: | ---: | --- |
| [EMGFormer-S](../native-emgformer-trajectory/small/metrics.json) | 14.7 | 16.54 | 约+1.84 | 原生12,000点训练／11,490步轨迹／0.02 |
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

## 新原生完整轨迹S

[S完整结果、原始报告与200轮曲线](../native-emgformer-trajectory/small/README.md)：完整轨迹**16.537696°**，对齐轨迹中心16.474383°；论文Avg14.7°，描述性差值约+1.84°。完整验证1,052／测试2,792请求的原生和独立模型重放、七阶段报告核验及额外CPU原始图重读已完成，16个原拥有者、24个原数据进程及2个额外读取拥有者关闭。16份完整输入16,540,457,757bytes和全部原始包均按真实来源绑定。

S继续使用原生200轮／25,000已保存更新、原中心验证BEST169，没有用轨迹或测试更换权重。12,000点输入经左510／右0上下文得到11,490步；对齐中心映射原6255，既有原生点报告使用6000，两项不能混算。同一个S架构的历史legacy轨迹16.289435°及原生点报告均保留；数据／归约／协议差异不作控制变量改进归因。M／L完整轨迹仍待后续流程结束。

## 原生ResNet结果

[RN18](../original-resnet/resnet18/README.md)：150轮／33,750更新、验证best136，唯一首遍6.607195°、有图6.469979°；完整原始链及额外读取关闭。

[RN50](../original-resnet/resnet50/README.md)：150轮／75,900更新、验证best140，唯一首遍5.880998°、有图5.739818°；论文Avg5.3°。独立RN50 YAML未发布，本结果是RN18配置覆盖backbone和Table8 batch400的明确派生，恢复／共享硬件及工时边界保留。

## 历史匹配融合收益保持原来源

legacy RN18→RN18+S仍为**7.784887°→6.793533°**，减少0.991354°（12.73%）。保留原配方／人口／报告SHA，不换用新原生RN18、RN50或S轨迹混算收益。

## 核心剩余项

- 原生M／L完整轨迹结果仍待验收；三个200轮模型的原中心结果不能替代Table2轨迹。
- 原生RN152继续训练；RN18融合完整资格结束后已进入独立正式150轮，最终测试及新排队的退出后独立验收仍欠。
- EMG2Pose三项完整来源250轮及Table11/13最终测试仍欠，官方S stride3900／M/L780保持。
- ViT-S旧版17,168点遮挡分析ρ=0.023640393／p=0.002283599保留；RN18路线、物理相机和人口等价仍欠。
- markers2mano、精确作者混合、marker／腕角／IK重生成及其余全文项目仍须完成，不能由这些MAE宣布全文完成。

[最新JSON与来源SHA](completed-mae-20261102-s-trajectory.json) · [最新CSV](completed-mae-20261102-s-trajectory.csv) · [此前RN50快照](completed-mae-20261102-rn50.json) · [全文清单](../../docs/full-paper-status.md)

# 已完成MAE与论文原表的数值参照

2026-11-02更新：14个模型／表格条目中，EMGFormer-S／M／L主行采用新验收的原生完整轨迹；RN18、RN50及RN18融合主行采用完整原生／明确派生结果。旧legacy协议行和全部既有JSON／CSV快照保留，模型家族数量不因补充协议而增加。

下表只作数值参照，不构成同条件架构排名。PDF Avg为跨条件样本加权；当前原生轨迹／中心主行为其各自有效角度pooled，条件跨用户mean±SD另列。数据版本、训练来源、单rank、人口、mask和聚合仍未完全对齐。论文只显示一位小数，差值按约数理解。

## EMG轨迹：Table2

| 模型 | 论文Avg (°) | 当前MAE (°) | 数值差 (°) | 协议 |
| --- | ---: | ---: | ---: | --- |
| [EMGFormer-S](../native-emgformer-trajectory/small/metrics.json) | 14.7 | 16.54 | 约+1.84 | 原生12,000点训练／11,490步轨迹／0.02 |
| [EMGFormer-M](../native-emgformer-trajectory/middle/metrics.json) | 14.2 | 16.42 | 约+2.22 | 原生12,000点训练／11,490步轨迹／0.02 |
| [EMGFormer-L](../native-emgformer-trajectory/large/metrics.json) | 14.2 | 16.57 | 约+2.37 | 原生12,000点训练／11,490步轨迹／0.02 |
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
| [RN18 + EMGFormer-S](../original-fusion/rn18-emgformer-small/metrics.json) | 5.4 | 6.19 | 约+0.79 | 原生8通道／12,000点／0.02／repeat1，旧版队列 |
| [WiLoR](../original-vision/wilor/metrics.json) | 4.7 | 5.11 | 约+0.41 | 原生或明确派生7,790点／0.02，旧版队列 |
| [ViT-S + EMGFormer-S](../original-fusion/vit-small-emgformer-small/metrics.json) | 5.5 | 6.34 | 约+0.84 | 原生16槽／7,790点／0.02，旧版队列 |

## 新原生完整轨迹S

[S完整结果、原始报告与200轮曲线](../native-emgformer-trajectory/small/README.md)：完整轨迹**16.537696°**，对齐轨迹中心16.474383°；论文Avg14.7°，描述性差值约+1.84°。完整验证1,052／测试2,792请求的原生和独立模型重放、七阶段报告核验及额外CPU原始图重读已完成，16个原拥有者、24个原数据进程及2个额外读取拥有者关闭。16份完整输入16,540,457,757bytes和全部原始包均按真实来源绑定。

S继续使用原生200轮／25,000已保存更新、原中心验证BEST169，没有用轨迹或测试更换权重。12,000点输入经左510／右0上下文得到11,490步；对齐中心映射原6255，既有原生点报告使用6000，两项不能混算。同一个S架构的历史legacy轨迹16.289435°及原生点报告均保留；数据／归约／协议差异不作控制变量改进归因。M及L亦已完成下述独立验收。

## 新原生完整轨迹M

[M完整结果、原始报告与200轮曲线](../native-emgformer-trajectory/middle/README.md)：完整轨迹**16.419418°**，对齐轨迹中心16.363438°；论文Avg14.2°，描述性差约+2.22°。同一原生200轮／25,000更新、中心验证BEST124、完整1,052验证／2,792测试的四遍原生／独立重放及七阶段报告核验已结束。16原拥有者、24数据进程及2额外CPU读取拥有者关闭；完整16输入／16.54GB、原始包和原选定检查点均经独立重读，0新学习。旧legacy M16.486140°及原生6000中心16.441148°保持，当前轨迹中心对应6255；旧版、单rank、来源和PDF加权差异继续披露。

## 新原生完整轨迹L及三规模关闭

[L完整结果、报告及200轮曲线](../native-emgformer-trajectory/large/README.md)：完整轨迹**16.574095°**，对齐中心16.505456°；论文Avg14.2°，描述性差约+2.37°。原生200轮／25,000更新、中心验证BEST142及完整1,052验证／2,792测试保持。四遍原生／独立重放、七阶段、原16拥有者／24数据进程及额外2CPU读取拥有者成功关闭；额外重读完整16来源／16.54GB和原始包，0新学习。[总关闭凭据](../native-emgformer-trajectory/all-sizes-complete-20261102.json)证实两队列父进程及全部130个不同原拥有者／数据进程身份退出。三规模新轨迹协议完成；历史legacy与原6000点报告保留，全文及PDF等价未完成。

## 原生ResNet结果

[RN18](../original-resnet/resnet18/README.md)：150轮／33,750更新、验证best136，唯一首遍6.607195°、有图6.469979°；完整原始链及额外读取关闭。

[RN50](../original-resnet/resnet50/README.md)：150轮／75,900更新、验证best140，唯一首遍5.880998°、有图5.739818°；论文Avg5.3°。独立RN50 YAML未发布，本结果是RN18配置覆盖backbone和Table8 batch400的明确派生，恢复／共享硬件及工时边界保留。

## 历史匹配融合收益保持原来源

legacy RN18→RN18+S仍为**7.784887°→6.793533°**，减少0.991354°（12.73%）。保留原配方／人口／报告SHA，不换用新原生RN18、RN50或S轨迹混算收益。

## 核心剩余项

- S/M/L原生完整轨迹协议已全部验收；PDF精确人口、作者来源与跨条件样本加权等价仍未证明。
- 原生RN152继续运行；RN18融合150轮／46,950更新、完整验证／测试重放及53来源退出后独立验收现已完成。新同帧4282人口的FP32评价和独立算术仍在运行。
- EMG2Pose三项完整来源250轮及Table11/13最终测试仍欠，官方S stride3900／M/L780保持。
- ViT-S旧版17,168点遮挡分析ρ=0.023640393／p=0.002283599保留；RN18路线、物理相机和人口等价仍欠。
- markers2mano、精确作者混合、marker／腕角／IK重生成及其余全文项目仍须完成，不能由这些MAE宣布全文完成。

[最新JSON与来源SHA](completed-mae-20261102-rn18-fusion.json) · [最新CSV](completed-mae-20261102-rn18-fusion.csv) · [此前SM快照](completed-mae-20261102-sm-trajectory.json) · [此前S快照](completed-mae-20261102-s-trajectory.json) · [此前RN50快照](completed-mae-20261102-rn50.json) · [全文清单](../../docs/full-paper-status.md)

## RN18融合的新增原生协议

[完整原始报告与150轮曲线](../original-fusion/rn18-emgformer-small/README.md)：验证best145；全部中心6.189098°、有图6.079778°，论文Table3 Avg5.4°，描述性差约+0.79°。完整53来源／61.377GB、650个已关闭产物、50原谱系拥有者及4独立读取拥有者均验收；原69轮失败和六域恢复如实保留。

该2792人口不能与4306原生视觉或4282新同帧人口直接计算控制变量收益。上文legacy匹配收益保持原来源；旧6.793533°行移入新快照历史部分，不增加14模型家族数。

[最新数值JSON](completed-mae-20261102-rn18-fusion.json) · [CSV](completed-mae-20261102-rn18-fusion.csv) · [前一轨迹快照](completed-mae-20261102-sml-trajectory.json)

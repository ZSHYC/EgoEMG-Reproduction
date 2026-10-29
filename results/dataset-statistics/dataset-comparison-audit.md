# Table 1：原表数据与原始来源核对

2026-10-28核对所附PDF第4页。六行原值原样保存在[CSV](table1-as-reported.csv)与[JSON](dataset-comparison.json)，没有把引用资料当作本项目训练或采集结果。JSON保留PDF/页文本身份、逐项来源、已核实事实和未解决项；当前没有找到固定官方版本的Table1生成脚本。

| 数据集 | PDF原表 | 原始来源核对 |
| --- | --- | --- |
| Jarque-Bou 2019 | 22人、8通道、18角、26活动 | 原文Data Records明确raw/processed EMG均7列，8是设备通道数；Tables2/3含WRIST_F、WRIST_A两腕角，与原表“无腕角”不符。[原文 Methods、Data Records、Tables2/3](https://www.nature.com/articles/s41597-019-0285-1) |
| NinaPro | 27–78人、10–16通道、22角、53手势 | 必须指定子库与是否含rest。DB1/2/5为27/40/10人和10/12/16通道，22列Glove原始值未校准。[DB1](https://ninapro.hevs.ch/instructions/DB1.html)、[DB2](https://ninapro.hevs.ch/instructions/DB2.html)、[DB5](https://ninapro.hevs.ch/instructions/DB5.html) |
| MyoKi | 36人、16通道、无腕角、74活动 | 原始描述为35人、12个EMG通道、18个运动学通道，并明确校准两腕角，与原表三项不符。[原文 pp.1–3](https://elib.dlr.de/221692/1/s41597-025-05852-6.pdf) |
| PiMForce | 21人、8通道、20角、63手势 | 官方项目论文为21人、8通道、20角、22种交互；附PDF的参考文献作者/标题也与该项目论文不符。[§3.2、B.4.2](https://arxiv.org/html/2410.23629v1) |
| EMG2Pose | 193人、双腕32通道、20角、50手势 | 原论文Table1明确50个被提示的姿态类别，与29个stage分开；§3.1/B.1确认每腕16通道、双腕同时采集，B.4明确不追踪腕关节。[原论文](https://arxiv.org/html/2412.02725v1)、[官方来源](https://github.com/facebookresearch/emg2pose) |
| EgoEMG | 41人、双腕16通道、ego RGB/RGB-D、22角、60手势 | 已有旧版41episode、66,161,725行、每手8通道与22角/60类定义证据；自然人身份、全部模态实物及当前版本标签等价仍待核验。[完整范围](../../docs/full-paper-status.md) |

NinaPro原表脚注“除DB1/2/5外均只有手势标签”也不成立：[DB8官方说明](https://ninapro.hevs.ch/instructions/DB8.html)提供18列校准运动学数据，用于连续回归；其EMG在右臂、手套在左手。DB5的两个Myo在同一侧前臂，不能按设备数量算双腕。NinaPro官方页面自身也有协议段落不一致，JSON保留具体子库口径，不推断全家族统一人数或动作数。

这里完成的是原表数据产物与逐源调查。完整EgoEMG模态盘点尚未完成，外部原始数据没有因此被全部下载，也没有重训这些引用研究。原表值、原始来源值和本项目实测数据在JSON中分别标注。

2026-10-28后续原始来源核对：EMG2Pose原文Table1对gesture的定义是被要求完成的姿态类别，明确提醒不同数据集定义不同；50并非29个stage的另一种计数，也未据此声称独立重建了50类逐帧标签。双腕总32通道与单手HDF5/单手模型16通道分别记录。[Table1、§3.2、B.1/B.4](https://arxiv.org/html/2412.02725v1)

NinaPro的“27–78”仍不能当作每个子库的人数范围：[DB1](https://ninapro.hevs.ch/instructions/DB1.html)、[DB2](https://ninapro.hevs.ch/instructions/DB2.html)、[DB3](https://ninapro.hevs.ch/instructions/DB3.html)分别27、40、11人，和为78只是明确标注的算术解释，不证明作者原表口径或跨库身份去重；DB1的53类可解释为52动作加rest，DB2则49动作加rest。[DB6](https://ninapro.hevs.ch/instructions/DB6.html)有10人、14通道，并列出可穿戴眼动/视野采集，进一步说明全家族不能由单一行概括；它不自动证明存在可用的同步手部RGB姿态基准。所有PDF原值保留，完整模态/逐列验证仍未完成。

2026-10-29补核：Jarque-Bou的[Data Records、Tables2/3](https://www.nature.com/articles/s41597-019-0285-1)将7列数据、8通道设备与两腕角分开；主体视频因隐私不提供，另有两段活动演示。上述是原始文档证据，未下载其MAT逐列实测。

MyoKi的[Methods与Data Records（pp.3、6、9）](https://elib.dlr.de/221692/1/s41597-025-05852-6.pdf)明确18列glove_calibrated以度为单位、含腕传感器17/18；标注webcam视频不公开。PiMForce的[§3.1/B.4.2/D.3.2](https://arxiv.org/html/2410.23629v1)区分磁感手指模块、20维手指目标与另行RGB实验；“20维不含腕关节”是按其完整角定义的推论，不能把RGB演示当作主数据集已发布egoRGB/RGB-D的证明。

本轮7条逐字段判定在[JSON](dataset-comparison.json)的followup_cell_audit_20261029中。NinaPro家族统一行口径、PiMForce完整视觉发布盘点、EgoEMG全部模态实物仍未闭合；Table1不标为全部完成。原六行CSV及全部原表值保持不变。

PiMForce[作者公开目录](https://www.dropbox.com/scl/fo/m7qye1l3ii6hlwgzlajyv/ADhaUZvH2eDJ3ggmGAlN5k0?dl=0&rlkey=eyt5gs66uimhiw3dismnylamw)另核实主ZIP显示7.64GB、独立PressureVision++基准58.96GB，主目录列User1–21；User1有三会话，抽查Session1仅见fsr/join/pps各22个CSV。[目录观察记录](pimforce-public-directory-observation-20261029.json)只证明可见文件清单，未下载压缩包或检查全部63会话，不能由此断言全数据没有图像。

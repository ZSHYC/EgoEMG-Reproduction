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

NinaPro的“27–78”不能当作每个子库的人数范围：[DB1](https://ninapro.hevs.ch/instructions/DB1.html)、[DB2](https://ninapro.hevs.ch/instructions/DB2.html)、[DB3](https://ninapro.hevs.ch/instructions/DB3.html)分别27、40、11人。2026-10-28时仅核实三库相加为78；2026-11-02已从[作者原始研究的Subjects章节](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2016.00009/full)补核其明确报告的78人及67健全／11截肢组成，故不再只作为算术推测。但作者原表范围的具体含义及本项目独立跨库身份去重仍未确立。DB1的53类可解释为52动作加rest，DB2则49动作加rest。[DB6](https://ninapro.hevs.ch/instructions/DB6.html)有10人、14个物理电极，存储EMG为16列、其中2列为空，并列出可穿戴眼动/视野采集。全家族不能由单一行概括，视频采集也不自动证明同步连续手部RGB姿态基准。所有PDF原值保留，完整模态/逐列实物验证仍未完成。

2026-10-29补核：Jarque-Bou的[Data Records、Tables2/3](https://www.nature.com/articles/s41597-019-0285-1)将7列数据、8通道设备与两腕角分开；主体视频因隐私不提供，另有两段活动演示。上述是原始文档证据，未下载其MAT逐列实测。

MyoKi的[Methods与Data Records（pp.3、6、9）](https://elib.dlr.de/221692/1/s41597-025-05852-6.pdf)明确18列glove_calibrated以度为单位、含腕传感器17/18；标注webcam视频不公开。PiMForce的[§3.1/B.4.2/D.3.2](https://arxiv.org/html/2410.23629v1)区分磁感手指模块、20维手指目标与另行RGB实验；“20维不含腕关节”是按其完整角定义的推论，不能把RGB演示当作主数据集已发布egoRGB/RGB-D的证明。

本轮7条逐字段判定在[JSON](dataset-comparison.json)的followup_cell_audit_20261029中。NinaPro家族统一行口径、PiMForce完整视觉发布盘点、EgoEMG全部模态实物仍未闭合；Table1不标为全部完成。原六行CSV及全部原表值保持不变。

2026-10-29历史观察：PiMForce[作者公开目录](https://www.dropbox.com/scl/fo/m7qye1l3ii6hlwgzlajyv/ADhaUZvH2eDJ3ggmGAlN5k0?dl=0&rlkey=eyt5gs66uimhiw3dismnylamw)显示主ZIP为7.64GB、独立PressureVision++基准58.96GB，主目录列User1–21；User1有三会话，抽查Session1仅见fsr/join/pps各22个CSV。[当时的目录观察记录](pimforce-public-directory-observation-20261029.json)未检查完整归档。

2026-11-02新增[完整主ZIP目录及三份原始CSV核验](pimforce-main-archive-audit-20261102.md)：实际ZIP64目录4,242条，21用户键×3会话，全部4,158非目录成员均为fsr/join/pps CSV、各会话三类编号均为0–21。三份完整样本CRC及64,252行数值检查通过，实际join含8路EMG/20手指角；另一进程以独立目录解析器和NumPy读取器完整复核。主ZIP文件清单的图像/视频命名成员为0，但未读取全部CSV内容或单独的PressureVision++包，不能推广为整个发布目录没有视觉数据。另保留原README“21动作”、原论文22交互、EgoEMG表63手势的三方差异；63恰等于会话数，作者是否误写仍无证据。Table1及全文未标完成。

2026-11-02新增[NinaPro十个子库的逐项来源表、CSV及12份来源身份](ninapro-family-source-audit-20261102.md)：补核明确的DB1/2/3共78人；DB9的77人来自DB1/2/5而非新增人口；DB6区分14物理电极与16存储列。DB3/4/7官方文档也列glove字段，DB8/9明确列校准运动学，不能将原脚注概括成所有其余子库只有动作标签。DB10链接的[原始运动研究](https://doi.org/10.1038/s41597-020-0380-3)明确报告45人的第一视角视频并说明MP4发布；该声明不等于本次已核验完整视频实物，更不能与DB9角度拼成同一人口的连续手姿基准。原表六行CSV保持字节不变，完整Table1与全文仍未完成。

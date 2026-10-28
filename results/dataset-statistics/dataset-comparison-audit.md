# Table 1：原表数据与原始来源核对

2026-10-28核对所附PDF第4页。六行原值原样保存在[CSV](table1-as-reported.csv)与[JSON](dataset-comparison.json)，没有把引用资料当作本项目训练或采集结果。JSON保留PDF/页文本身份、逐项来源、已核实事实和未解决项；当前没有找到固定官方版本的Table1生成脚本。

| 数据集 | PDF原表 | 原始来源核对 |
| --- | --- | --- |
| Jarque-Bou 2019 | 22人、8通道、18角、26活动 | 原论文为8通道设备、7个肌肉采集位置；设备能力与实际信号列数须区分。[原文 pp.1–3](https://www.nature.com/articles/s41597-019-0285-1.pdf) |
| NinaPro | 27–78人、10–16通道、22角、53手势 | 必须指定子库与是否含rest。DB1/2/5为27/40/10人和10/12/16通道，22列Glove原始值未校准。[DB1](https://ninapro.hevs.ch/instructions/DB1.html)、[DB2](https://ninapro.hevs.ch/instructions/DB2.html)、[DB5](https://ninapro.hevs.ch/instructions/DB5.html) |
| MyoKi | 36人、16通道、无腕角、74活动 | 原始描述为35人、12个EMG通道、18个运动学通道，并明确校准两腕角，与原表三项不符。[原文 pp.1–3](https://elib.dlr.de/221692/1/s41597-025-05852-6.pdf) |
| PiMForce | 21人、8通道、20角、63手势 | 官方项目论文为21人、8通道、20角、22种交互；附PDF的参考文献作者/标题也与该项目论文不符。[§3.2、B.4.2](https://arxiv.org/html/2410.23629v1) |
| EMG2Pose | 193人、双腕32通道、20角、50手势 | 官方README确认193人、25,253文件、370小时、29个stage；本次未独立核实50手势的定义，不能用29个stage替换。[官方来源](https://github.com/facebookresearch/emg2pose) |
| EgoEMG | 41人、双腕16通道、ego RGB/RGB-D、22角、60手势 | 已有旧版41episode、66,161,725行、每手8通道与22角/60类定义证据；自然人身份、全部模态实物及当前版本标签等价仍待核验。[完整范围](../../docs/full-paper-status.md) |

NinaPro原表脚注“除DB1/2/5外均只有手势标签”也不成立：[DB8官方说明](https://ninapro.hevs.ch/instructions/DB8.html)提供18列校准运动学数据，用于连续回归；其EMG在右臂、手套在左手。DB5的两个Myo在同一侧前臂，不能按设备数量算双腕。NinaPro官方页面自身也有协议段落不一致，JSON保留具体子库口径，不推断全家族统一人数或动作数。

这里完成的是原表数据产物与逐源调查。完整EgoEMG模态盘点尚未完成，外部原始数据没有因此被全部下载，也没有重训这些引用研究。原表值、原始来源值和本项目实测数据在JSON中分别标注。

# Table 1：NinaPro 子库、人数与模态口径

本项核对官方 DB1–DB10 十个页面和两篇原始研究，属于 Table 1 的来源分析。没有下载 NinaPro 原始 MAT／视频或训练其模型；EgoEMG 原表六行、脚注及所有已有实验数值保持原样。

## 78 人的证据范围

此前仅把 27＋40＋11＝78 列为算术解释。现从数据集作者的[原始研究 Materials and Methods／Subjects](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2016.00009/full)核实：该研究明确报告 DB1/2/3 共78人，其中67名健全受试者、11名经桡骨截肢者，三库分别27／40／11人。因此，**78已获得原始文献支持**，不再只是本项目的推算。

这仍不证明 EgoEMG 作者将“27–78”定义成什么范围，也不是本项目独立做了跨库自然人去重。不能把这三个子库的合计当成 DB1–DB10 每个子库人数的上下限。

## 十个子库逐项对照

以下均为原始文献或官方页面的声明，未以实际 MAT 矩阵重新测量。空缺表示本次来源未确立该值，不表示实物一定不存在。各行人数不能相加为整个家族的独立人数。

| 子库及官方来源 | 文档人数／范围 | EMG 电极数／存储列数 | 文档中的运动学字段 | 动作口径 |
| --- | --- | --- | --- | --- |
| [DB1](https://ninapro.hevs.ch/instructions/DB1.html) | 27 | 10／10 | glove 22，未做解剖角校准的传感器信号 | 52＋rest |
| [DB2](https://ninapro.hevs.ch/instructions/DB2.html) | 40 | 12／12 | glove 22，未校准 | 49＋rest |
| [DB3](https://ninapro.hevs.ch/instructions/DB3.html) | 11，截肢者 | 最多12／声明12 | 页面列 glove 22；截肢者实际可用标签需另核实 | 最多49＋rest |
| [DB4](https://ninapro.hevs.ch/instructions/DB4.html) | 10 | 12／12 | glove 22，未校准 | 52＋rest |
| [DB5](https://ninapro.hevs.ch/instructions/DB5.html) | 10 | 16／16 | glove 22，未校准 | 52＋rest |
| [DB6](https://ninapro.hevs.ch/instructions/DB6.html) | 10 | **14／16，其中2列为空** | 页面变量表未列 glove | 7种抓握 |
| [DB7](https://ninapro.hevs.ch/instructions/DB7.html) | 20健全＋2截肢者 | 12／12 | glove 18；页面称未校准，并另述制造商 quick calibration | 通常40＋rest，一位受试者只有38种动作 |
| [DB8](https://ninapro.hevs.ch/instructions/DB8.html) | 10健全＋2截肢者 | 16／16 | **glove 18，页面称已校准** | 回归协议9种动作，另列rest |
| [DB9](https://ninapro.hevs.ch/instructions/DB9.html) | **77，来自DB1/2/5** | 本次变量表未列EMG矩阵 | **声明已校准 angles 22；另注 sensor11 被排除** | 原DB1/2/5的运动学再发布，未设家族统一动作数 |
| [DB10](https://ninapro.hevs.ch/instructions/DB10.html)／MDS1 | 本表对应运动研究45人，15截肢＋30健全 | 通常12／12，原文另有例外 | 原文数据字段表未列连续3D手部关节角 | 10种抓握类型 |

DB5 的两个 Myo 在同一侧前臂，不能按设备数量写成双腕 EMG。DB8 是右侧 EMG、左手手套及镜像双手动作，也不等于双腕同时记录 EMG。[DB5 设置与字段](https://ninapro.hevs.ch/instructions/DB5.html)、[DB8 设置与回归说明](https://ninapro.hevs.ch/instructions/DB8.html)

DB9 官方明确说明77人来自DB1/2/5，不能再计作新增77人。其声明的22列与排除 sensor11 的说明同时保留，未自行改成“21个有效角”，也未把传感器列数自动等同于完整、有效的22解剖自由度。[DB9 Subjects、Dataset variables、Usage notes](https://ninapro.hevs.ch/instructions/DB9.html)

原 EgoEMG 脚注“只有 DB1/2/5 有连续角度，其余只有手势标签”不能作为全部子库的统一结论：DB3/4/7 的页面也列连续 glove 字段，DB8/9 明确声明校准运动学。文档声明与完整实物验证仍分开，尤其未证明每位截肢受试者都有可用的完整姿态标签。

## DB10 的第一视角视频不能与其他子库角度拼成一个基准

DB10 链接的[2020年运动数据原论文](https://doi.org/10.1038/s41597-020-0380-3)明确描述45人、前臂EMG与头戴式第一视角视频；Data Records 和 Usage Notes／Video 说明发布MP4及同步字段。它不是同页另一篇“15截肢＋29对照”的临床研究人口；两篇不能混计。

因此，若 NinaPro 代表整个当前家族，统一“无 Ego RGB”并不准确；若仅代表若干早期子库，就应明确范围。但 DB10 的视频与 DB9 的角度并未被本项目建立成同一受试者、同一时刻的连续手姿基准，不能据此宣称它已提供与 EgoEMG 等价的 EMG＋RGB＋3D手姿任务。

本次匿名访问 MDS1 的 Harvard Dataverse 元数据API返回403，未重试或推断必须登录。因此只验收**作者文献中的视频发布声明**，未验收完整文件目录、视频字节、同步误差或姿态目标。403也不表示数据集不公开。

## 来源、产物与剩余边界

- [逐库 CSV](ninapro-family-source-audit-20261102.csv)：人数范围、物理电极／矩阵列数、运动学、动作及不确定项。
- [结构化记录和12份来源身份](ninapro-family-source-audit-20261102.json)：十页实际HTTP200、两个原始研究的正文／XML字节哈希、章节位置和边界。
- [独立读取核对](ninapro-family-source-readback-20261102.json)：重新读取实际来源与产物，并确认原六行表格字节未改。
- [原表六行 CSV](table1-as-reported.csv)及[综合来源审计](dataset-comparison-audit.md)继续分列原值与来源值。

DB2 官方页面同时出现10次与6次重复；DB8 的回归说明为9种动作，却另有40动作段落。本次原样记录这些差异，没有凭文档猜测修改数据或训练协议。

整张 Table 1 仍缺 EgoEMG 全模态实物等证据，不能标为全部完成。原模型训练、14项既有科学设置和全论文目标保持原范围。

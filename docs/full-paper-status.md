# 全论文复现状态与证据清单

状态快照：2026-10-26（北京时间）。**当前目标是复现论文全文，仍在进行中。** 有歧义时优先采用固定官方源码，并公开说明与 PDF 的差异；六项legacy实验完成不构成全文完成。范围以[全文设计](superpowers/specs/2026-10-26-full-paper-reproduction-design.md)为准，动态进度见[运行状态](run-status.md)，阶段动作见[阶段计划](superpowers/plans/2026-10-26-full-paper-stage1.md)。

原文为 [EgoEMG v1](https://arxiv.org/abs/2605.05712v1)，共 34 页。下文页码使用 PDF 印刷页码。官方源码固定为 [f7cbd048b7092139852270491d3859cd472e0311][official]；源码、resolved 配置、实际发布资产与运行证据分别核对，注释和文件名不能覆盖实际执行行为。本文不把论文数值填入本项目结果。

## 当前实际完成到哪里

- **六项 legacy 实验已完整运行并审计：** EMGFormer-S/M、ResNet-18、RN18+S及独立发布的RN50/RN152。原结果、配方、源码和初始化身份保留在[已完成实验](../results/completed/README.md)及[发布清单](../results/completed/publication-manifest.json)，不因新目标改写。它们与 PDF 原表的数据、标签、输入和训练协议仍有差异。
- **新增损失合成差异已确认：** [11配置实际审计](../results/official-loss-audit.md)证明固定官方EgoEMG配置的两个指尖别名最终相加为0.02，既有legacy训练器/配方为0.01。此前漏核的差异现已明确；旧四实验及当前L/RN50/RN152仍只属于既定legacy协议，不是当前官方合成损失的等价复现。新官方字面实验须另行完成，不能由旧结果补算得到。EMG2Pose直接官方流程仍为0.01。
- **EMGFormer-L 正在完整训练：** 两步真实 GPU 优化、两次各 1,052 样本的完整验证和严格重载已通过，独立 200 epoch 生产训练已启动。它尚无完整训练结束、最终测试、严格结果导出或锁定复算结论；短验收目录与生产目录独立。运行时复现代码为 7b805d4，后续文档提交不能替换该身份。最新进展以[运行状态](run-status.md)为准。
- **RN50与RN152均已完整完成：** [RN50独立结果](../results/completed/resnet50/README.md)完成150轮/23,700更新/150次完整独立验证，第19,118步best的全部中心MAE为6.640058°；最终测试、严格导出和锁定复算均成功。[RN152独立结果](../results/completed/resnet152/README.md)完成150轮/52,650更新/150次完整独立验证，验证最小值选择第52,650步，全部/有图中心MAE为5.982738°/5.869706°；最终测试、严格导出和锁定复算均成功。两者运行来源均为a357423，固定官方版本缺两份独立YAML，属于明确声明的legacy派生对照，保留0.01目标与全部数据限制。
- **DINOv2-S 真实预训练已取得并通过CPU验收：** [下载身份与验收](../results/dinov2-small-pretrained.md)记录匿名HF真实权重、加载后主干tensor摘要、官方位置嵌入插值及合成256px前向/严格离线重载。尚未进行真实cohort优化、GPU容量准入或完整训练；B于17:11完成真实预训练加载、174个主干tensor字节对照及合成前向/严格离线恢复；L仍只完成实际CPU构造/schema。
- **手形统计与聚合图已生成并独立核验：** [实际统计与分布图](../results/dataset-statistics/shape-statistics.md)覆盖 41 个引用 beta 行/82 手，50 项官方 float32/ddof0/range 重算误差为零。PDF Table5 在三位小数有 48/50 项吻合，两处微差保留；真正 P5/P95 单列。聚合箱线图的展示适配、独立参与者计数限制均明示。校准为 3840×3360，包含 rational_polynomial 与 fisheye 模型，与 PDF 720p 的关系尚待核验。
- **60类词表、22角语义、帧与冻结中心覆盖已核验：** [手势定义](../results/dataset-semantics/gesture-vocabulary.md)、[关节语义](../results/dataset-semantics/joint-angle-semantics.md)、[帧覆盖](../results/dataset-semantics/coverage.md)及[独立中心点读](../results/dataset-semantics/center-coverage.md)区分定义与各分母。66,161,725个共享同步帧有全部60类；已另实读33,508个冻结窗口/67,016手中心，1,420,084个有效角度。train/val/test中心分别观测51/32/59类；不能从词表或全帧覆盖推成每个角色60类。模型误差和全帧有效性仍未由此计算。
- **classic官方模型、数据与CPU训练/恢复及完整队列数值准入已核验，未完成正式训练：** [当前范围](classic-emg.md)保留原tuple/2D mask、固定scalar与新7790网格。新增[真实CPU队列证据](../results/classic-cohort-cpu-engineering.md)对两配方各完整准入及重新hash/扫描一致，train/val/test为101,714/1,576/4,306手样本；finite覆盖冻结足迹65,046,370行。原双别名0.02、val_loss/min、clip0与patience50保留；兼容裁决入身份。独立CUDA限定入口已实现/CPU审查，最终cfbc源码真实16-mixed/scaler的batch1限定检查已完成，各8次尝试/7次实际更新/1次注入跳步及完整状态恢复一致；600/256容量、正式full train/val-best/test与Task4仍欠。
- **EMG2Pose metadata、mini与S/M/L真实CPU优化/全窗口重载已核验。** [数据证据](../results/emg2pose-mini-data.md)记录30文件、4,218,756行全字段/原mask位级复核；[早期模型证据](../results/emg2pose-mini-runtime.md)为各单窗口随机初始化检查，[新增真实mini证据](../results/emg2pose-mini-optimization.md)为各2次更新、526窗口×2完整工程评价和严格恢复一致。mini全部原val，优化与工程检查复用它，不能视作独立验证或论文性能；Table11/13仍需完整数据及三个完整训练。
- **公开CPU工程实现已独立审查与整体检查。** `9b5f4931d53db0d536e0ebe64e386e8048872ab8`的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40292505908)为官方CPU **880 passed/2 skipped**（仅真实CUDA正例），轻量609 passed/273可选运行时skip，pip check通过。它覆盖classic cohort、canonical CUDA入口的CPU检查、预取及派生TaskA；不覆盖后续未集成worker/artifact、GPU容量或完整训练。历史源与实际执行身份继续保留。
- **扩展代码与聚合证据已公开，原运行身份保留。** [全论文分支](https://github.com/ZSHYC/EgoEMG-Reproduction/tree/codex/full-paper-reproduction)保留各训练提交，阶段CI和CPU/实际数据证据分别记录在[发布来源](publication-provenance.md)与[运行状态](run-status.md)。完整数据接收和多个必需模型仍未完成；绿测试、mini或中心统计不改变正式实验的完成状态。

EgoEMG 有 14 个主要模型设置（5 EMG、7 视觉、2 融合），EMG2Pose 另有 S/M/L 三项，markers2mano 还需独立训练与质量验证。现有六项只是这份清单中的legacy协议实验子集，不能写成“原表已复现6/17”。全文还包括数据统计、预处理和标注重生成、派生评价及全部定性图。

## 状态、证据与验收含义

| 状态 | 具体含义 |
| --- | --- |
| 未实现 | 本项目尚无满足该项要求的实现或产物；官方文件存在不改变此状态 |
| 工程验证 | 已执行真实构造、有限训练、读取或重载等明确检查；只支持所列范围 |
| 完整运行 | 已达到独立实验预算并完成验证选模、最终测试；仍需对应导出与审计 |
| 分析产物 | 已有由真实数据生成的底表、统计或图，尚需核验其协议、身份和计数 |
| 已核验 | 指定产物及其来源、单位、分母、样本集合、聚合规则已经核验；仅对声明的协议成立 |
| 外部依赖 | 缺少指定资产、访问条件、训练规格或存储；应继续推进不依赖它的工作 |

一项可同时有多个状态。下表“官方入口”只证明所列源码或配置已发布；“尚欠”才是本项目仍须完成的工作。数据统计要读取实际数组，派生分析要固定 checkpoint 和样本键，训练结果要有完整结束、仅用验证选模、冻结测试及可重算误差和/有效分母。公开文档不包含原始参与者标识、机器私有路径、账号文件或受限模型。

| 证据编号 | 实际证据 | 支持范围及限制 |
| --- | --- | --- |
| E1 | [四模型完整结果、曲线与身份](../results/completed/README.md) | S/M 各 200 epochs，RN18/融合各 150 epochs；原训练器轨迹结果与锁定中心复算分开，未额外复算整段轨迹 |
| E2 | [共同人口与环境清单](../results/completed/publication-manifest.json) | 同一冻结中心及 crop-present 人口、初始化来源、配置与运行身份；不是其他模型的证据 |
| E3 | [队列覆盖](../results/cohort-coverage.json)、[crop 就绪核验](../results/crop-readiness.json) | 41 份 crop、2,792 单手测试中心、2,688 有图中心；实际测试 40 名用户，User/Both 各有 1 人与训练交集 |
| E4 | [原始选择性资产清单](../assets_manifest.json)、[下载指南](download.md) | 原 21 文件的 EMG/标签与元数据范围；manifest 声明其他字段不代表那些数组已下载 |
| E5 | [本项目数据适配](../egoemg_repro/data.py)、[指标](../egoemg_repro/metrics.py)、[训练](../egoemg_repro/train.py)、[视觉与融合](../egoemg_repro/visual.py) | 已有 20+2 有效性、整体/逐手/split/user、整体手指/腕角；不能从这些汇总恢复逐关节或逐手势误差 |
| E6 | [动态运行记录](run-status.md)、[L 配方](../configs/emgformer_l.json) | 新 L 的短验收与生产运行、额外资产获取分别登记；配置和运行中状态均不是完整成绩 |
| E7 | [词表](../results/dataset-semantics/gesture-vocabulary.json)、[22角语义](../results/dataset-semantics/joint-angle-semantics.json)、[实际帧覆盖](../results/dataset-semantics/coverage.json) | 60类/30·18·12族群、22维单位/mask/名义限位、157条class/split/activity聚合记录；帧审计实读并重哈希三个完整数组，导出层核对审计与守恒；不推断窗口或模型误差 |
| E8 | [冻结中心覆盖](../results/dataset-semantics/center-coverage.json)与[范围/分母](../results/dataset-semantics/center-coverage.md) | 原12000窗口、train stride1200/eval12000；实际点读12字段共6,232,488bytes，33,508窗口/67,016手中心，178个role/split/class/activity cells；本次未全payload重hash，模型误差仍空 |

## 全部主要模型设置

| 数据/模型 | 官方发布内容与已核验接口 | 本项目状态与真正欠缺 |
| --- | --- | --- |
| EgoEMG EMGFormer-S | 官方 EMGFormer 与 S 配置 | **完整运行、已核验（legacy）**，见 E1/E2；原表精确等价未证明 |
| EgoEMG EMGFormer-M | 6 层、256 维、8 heads、FFN 1,024 | **完整运行、已核验（legacy）**，见 E1/E2；不替代 L |
| EgoEMG EMGFormer-L | [L YAML][large]：8 层、384 维、12 heads、FFN 1,536；实际 16,320,806 参数、22 输出、510/0 上下文 | **工程验证；完整训练进行中**，见 E6；12,000 点输入的轨迹对齐长度 11,490，尚欠完整结束与最终审计 |
| EgoEMG vemg2pose | [官方配置][vemg]及真实 LSTM 模型；实际预测22角、tuple输出、1790/0上下文，5,963,948参数，基类元数据仍写20 | **CPU训练/epoch恢复已核验，未正式完整训练**；目标手8ch/WL7790输出6000；固定scalar/literal mask及filtered适配已实现。真实cohort与batch1 FP16/scaler限定检查已完成，原批量容量和正式选模/test仍欠 |
| EgoEMG NeuroPose | [官方配置][neuropose]及真实 CNN 模型；16槽、tuple输出、0/0上下文、22角、6,354,969参数 | **CPU训练/epoch恢复已核验，未正式完整训练**；WL7790输出7790；先原零填16槽再归一化/全槽旋转。原Lightning clip0与val_loss语义已有真实CPU证明，真实cohort及batch1 CUDA/scaler限定检查已完成，原批量容量和正式完整训练仍欠 |
| EgoEMG ResNet-18 | [官方 RN18 配置][rn18] | **完整运行、已核验（legacy）**，见 E1/E2；非其他视觉模型的替代 |
| EgoEMG ResNet-50 | [ResNetVisionPose][resnet]支持该 backbone；[派生配方](../configs/resnet50.json)，含角度头总参数24,568,406 | **150 epochs完整legacy实验与锁定复算完成**；23,700更新/150次full val，best第19,118步，全部中心6.640058°。独立YAML未发布，损失/队列/单卡配方与官方字面/PDF等价仍未证明，见[结果](../results/completed/resnet50/README.md) |
| EgoEMG ResNet-152 | 同一真实类；[派生配方](../configs/resnet152.json)，总参数59,204,182 | **150 epochs完整legacy实验与锁定复算完成**；52,650更新/150次full val，验证选择第52,650步，全部/有图中心5.982738°/5.869706°。独立作者YAML未发布，保留损失/队列/派生配方限制，见[结果](../results/completed/resnet152/README.md) |
| EgoEMG ViT-S/14 | [官方独立配置][vits]、[DINOv2 实现][vit]；[本项目配方](../configs/vit_small.json)与严格来源记录/安全导出 | **真实预训练与CPU工程验收、未完整运行**；[实际HF资产](../results/dinov2-small-pretrained.md)88,240,510 bytes及主干tensor身份、官方插值、合成256px前向和严格离线恢复均核验；另有随机初始化的实际AdamW工程测试。缺真实cohort优化、GPU容量验收和完整训练 |
| EgoEMG ViT-B/14 | 真实 DINOv2 类；[明确S派生的配方](../configs/vit_base.json)，实际86,182,166总参数 | **真实预训练与合成CPU前向/严格离线恢复已核验、未完整运行**；独立作者YAML未发布，真实cohort优化、GPU容量/反传及完整训练仍未完成 |
| EgoEMG ViT-L/14 | 真实 DINOv2 类；[明确S派生的配方](../configs/vit_large.json)，实际303,833,622总参数 | **独立CPU进程构造/schema已核验、未完整运行**；独立作者YAML未发布，真实权重、GPU/优化器/检查点容量仍待验收 |
| EgoEMG WiLoR | [WiLoRViTPose][wilor]真实 22 角类及 module YAML；无独立 fusion/vision_wilor YAML | **未实现完整接入、外部依赖**；缺固定专用源码/配置/权重/许可与独立角度训练配方。缺权重时官方类可能保留随机主干，构造成功不证明预训练成功；MANO 监督文档不是同一 22 角任务 |
| EgoEMG RN18+S | 真实残差融合类、清洁分支来源及精确零残差适配 | **完整运行、已核验（legacy）**，见 E1/E2；保留原 S635/RN6319 身份 |
| EgoEMG ViT-S+S | [8ch/WL12000 配置][fusion-vits]；另有发布 16 槽/WL7790 配置 | **未完整运行**；须先完成 clean ViT-S，再接同分区 S；现 builder 的 RN18 路径键和 schema 不能直接复用，8/16 槽分支不得混载 |
| EMG2Pose EMGFormer-S | [官方 EMG2Pose S 配置][e2p-s]，16通道、20角、独立数据协议 | **真实mini CPU两次优化、526窗×2与严格完整状态恢复已核验**，见[新增证据](../results/emg2pose-mini-optimization.md)；无独立选模/正式测试，全量仍在接收 |
| EMG2Pose EMGFormer-M | [官方 EMG2Pose M 配置][e2p-m] | **同样完成2次真实更新、526窗×2工程检查**；完整独立训练/验证/测试仍未完成 |
| EMG2Pose EMGFormer-L | [官方 EMG2Pose L 配置][e2p-l] | **同样完成2次真实更新、526窗×2工程检查**；没有完整科研结果，不能用EgoEMG-L或mini数值填表 |

classic 配方的 precision=16-mixed 和 gradient_clip_val=0 还需要真实实现检查：现有复现训练器只为 bfloat16 启用 autocast，裁剪值直接设 0 会清零非零梯度，不能照搬数值宣称忠实训练。两类模型 tuple 已对齐，不能重复裁剪。RN50/152、ViT-B/L/WiLoR 缺独立 YAML 是固定提交内的已核验发布缺口，不是模型类不存在；新增派生配方必须明确其来源和差异。[官方配置说明][config-readme]也将配置定位为研究记录。

## Tables 1–15

| ID / PDF 位置 | 性质与必须产物 | 状态、实际证据 | 尚欠与验收条件 |
| --- | --- | --- | --- |
| Table 1，§3.3 p.4 | 数据集比较；EgoEMG 为作者数据定义，其余行为外部引用 | **部分来源已核对，比较产物未实现**；E4 支持部分 legacy 字段；未找到官方独立生成脚本 | 逐列来源映射、EgoEMG 全模态实物盘点及机器可读比较表。无需把 Jarque-Bou/NinaPro/MyoKi/PiMForce/EMG2Pose 的原研究重新训练 |
| Table 2，§5.1 p.8 | 作者 EMG 主结果；S/M/L、vemg2pose、NeuroPose 的轨迹 Gesture/User/Both 与加权 Avg、参数量 | **S/M 部分完整运行（legacy）**，E1/E2；L 训练中，其余无完整运行 | 五模型各自完整证据；逐用户 mean±SD、样本权重、轨迹 mask/分母；模型同名不能证明与 v1 原表等价 |
| Table 3，§5.2–5.3 p.8 | 作者单帧视觉/融合主结果；7 视觉与 2 融合 | **RN18、RN50、RN18+S有完整legacy运行**，E1/E2/E3及[RN50](../results/completed/resnet50/README.md)；仍不等价于原表 | RN152、ViT-S/B/L、WiLoR、ViT-S+S尚无完整结果；所有模型逐条件用户SD、真实初始化/选择、匹配人口与官方字面配方仍须逐项核验 |
| Table 4，A.1 p.15 | 60类完整词表与描述、30/18/12族群，数据定义 | **定义、实际帧与冻结中心覆盖已核验（legacy）**，E7/E8；完整词表保留raw50；中心有241未分配、66 raw，59类视图33,201窗口，与帧统计分开 | 全帧逐手mask统计仍欠；不同新grid须另报，中心标签统计不能填Table14/15模型误差N。PDF措辞差异单列 |
| Table 5，A.2 p.16 | 作者82手、10维β统计 | **分析产物、已核验（legacy）**；[统计与来源](../results/dataset-statistics/shape-statistics.md)含50项官方重算和实际P5/P95，PDF显示48/50吻合 | 两个微小显示差异保留，原因未定；episode/beta行数不能独立证明人口身份。官方std(ddof0)和range优先，不能把range混称真正P95−P5 |
| Table 6，B.3 p.20 | 22 DoF语义、单位、左右约定和有效性，数据定义 | **语义表与冻结中心逐角mask已核验**，E7/E8；0–19手指rad、20腕FE/21腕RU磁盘deg，模型统一rad；实际中心1,420,084/1,474,352角有效。原literal与legacy20+2分列，腕名义限位未知 | 全帧逐角有效性与标签重生成仍欠；语义相同不证明legacy轴角投影等价PDF landmark IK；不从通用模型末两项臆造腕限位 |
| Table 7，C.1 p.21 | 作者 S/M/L 架构、参数与训练 GPU-hours | **部分构造/完整运行证据**；S/M E1，L E6；官方真实构造有参数计数 | 三模型统一总/可训练参数统计与实际完整训练用时、GPU 数/型号/精度/验证开销；A800 一步 profile 不能填原文 4090 GPU-hours |
| Table 8，C.2 p.22 | 7 个视觉 backbone 的配置与参数，非另加 7 次实验 | **RN18/RN50部分已核验**；RN152已工程验收、DINO-S真实预训练已核验，其余配方/初始化未闭合；上表及[模型源码][resnet] | 每种实际 resolved 配置、256 输入/patch 策略、回归头、冻结状态、权重身份及参数量；PDF 超参和缺失独立 YAML 必须分栏 |
| Table 9，D.1 p.23 | Table 2 五模型的逐手/逐 split 轨迹评价 | **S/M 细分证据已有，完整表未生成**；E1 原训练器轨迹 by_hand_split；额外审计只复算中心 | 固定 best 生成五模型完整左右表与分母；无需另训左右两套模型，中心审计不能代替轨迹 |
| Table 10，D.2 p.23 | S/M/L 中心帧 EMG 参照，为轨迹模型的派生评价 | **S/M 中心结果已核验（legacy）**，E1/E2/E3；L 尚无最终结果 | 固定来源 checkpoint、TDS 中心对齐和同视觉 population、逐用户/样本计数；原文关于 L 最优条件的文字与显示数值不符，不作为验收约束 |
| Table 11，D.3 pp.23–24 | 作者 EMG2Pose S/M/L 实测；† 基线是第三方原论文引用 | **作者三行未完整运行**；公开数据来源见下节，完整 mini 仅用于工程接入 | 完整数据及 User/Stage/User+Stage 协议、20 维、16 通道、三训练及独立 val/test；引用 SensingDynamics/NeuroPose/vemg2pose/CLDM 的 † 数值应明确来源，不能声称本项目重训 |
| Table 12，D.4 p.24 | EgoEMG S/M/L 的 10 个逐指/关节组派生误差 | **未生成**；E5 只有整体 fingers/wrist；[官方分组][constants]存在 | 累积逐角误差和/分母后生成重叠的 proximal/mid/distal、五指、腕 FE/RU 组，按主轨迹协议聚合；现有汇总不能逆推出这 10 组 |
| Table 13，D.4 p.24 | EMG2Pose S/M/L 的 8 组派生误差，无两腕组 | **未实现，依赖 Table 11** | 三个完整模型与标准三 split 的逐角计数、同官方组映射；无需为该表再训练三遍 |
| Table 14，D.5 p.25 | RN18、S、RN18+S 在 59 类的 3 族群同中心比较 | **已有三个 checkpoint 与匹配人口，分析未实现**，E1/E2；[评价][gesture-eval]与[聚合][gesture-paper]源码存在 | 逐样本键、gesture/active/split/hand/crop/mask、三误差及 raw 排除记录；本项目 2,792 中心不能冒称原表 4,278 样本 |
| Table 15，D.5 p.27 | 同 Table 14 底表生成 59 类 ID/name/N/三 MAE/Δ 与排序 | **未实现**；官方词表和表/散点生成入口存在 | 未舍入误差和/分母、按真实 Δ 排序、各族群人数与样本权重；原文 54/59 改善是参考结论，不是当前实测 |

Table 12 的官方组映射：proximal=[0,1,4,5,8,9,12,13,16,17]，mid=[2,6,10,14,18]，distal=[3,7,11,15,19]；五指各占连续四维。各组重叠，不能把组均值相加还原总体。

## Figures 1–13

| ID / PDF 位置 | 性质与必须产物 | 状态、实际证据 | 尚欠与验收条件 |
| --- | --- | --- | --- |
| Figure 1，p.3 | 原采集设备、布置、marker/MANO 与穿戴示意 | **来源说明部分已有，采集未重做**；[论文指南](paper-guide.md)与原创示意只支持说明 | 原系统设备/拓扑/同步/标定记录与可核验图；公开数据 benchmark 不要求重新招募，但不能将原照片说成本项目新采集 |
| Figure 2，p.4 | 3 类同步样例，每行 3.9s 的 GT、双腕 EMG、ego RGB、外部 RGB-D | **未实现完整面板**；E3/E4 有 EMG/crop 中心链 | 固定匿名 episode/time/gesture、原始全帧/外部 RGB-D、GT mesh 与时间偏差/有效性；crop+信号曲线不足以证明完整多模态同步 |
| Figure 3，p.5 | (a)82手β箱线图；(b)参与者×手势split矩阵 | **(a)聚合展示图已生成并核验；(b)图未生成**；[实际图与统计](../results/dataset-statistics/shape-statistics.md) | (a)公开图隐藏单点/离群点，属于展示适配；官方[β脚本][beta]画左右柱图，未声称逐像素重画。(b)按实际release绘制并保留User/Both用户交集 |
| Figure 4，p.6 | EMGFormer remapping/TDS/RoPE/head/增强架构 | **S/M 已构造、L 工程验证；图未重生**，E1/E6 | 从实际配置、形状与对齐生成图；PDF 的 7790ms、8/16 通道与 TDS 简图差异见下节，不能用示意覆盖源码 |
| Figure 5，p.6 | 256 crop→ResNet/ViT→池化/角度头 | **RN18 部分完整运行；图未重生**，E1；[真实实现][vit] | 全部 7 backbone 构造与初始化；具体头激活由实际类裁决，不能以一张通用图代表七项实验 |
| Figure 6，p.7 | 视觉预测+EMG 时间池化残差融合 | **RN18+S 已核验（legacy）；ViT-S+S 未运行，图未重生**，E1/E2 | 两对独立来源、真实形状、零残差等价性与架构图；官方极小正态初始化与本项目精确零适配分别记 |
| Figure 7，p.9 | 自遮挡、深度歧义、运动模糊三类同例 GT/EMG/Vision/Fusion 定性图 | **未实现**；E1 已有三模型，尚无关联误差的渲染图 | 固定样本与选择规则、同一角度/mesh 空间、合法模型与相机/视角、逐例未舍入 MAE/有效分母；论文单例数值不填本项目图 |
| Figure 8，p.17 | markers2mano 图 Transformer 架构 | **未复现；外部依赖**；[官方推理入口][m2m]存在，模型包未随仓库发布 | 实际图谱/PE/参数/权重、forward 与 778 顶点重建；脚本 fallback 宽度/层数不是已核验论文训练配方 |
| Figure 9，p.18 | marker 训练的 8 类增强可视化 | **未实现** | 实际变换概率、参数、组合顺序、固定 seed 前后 marker/mesh；EMG channel rotation 不能替代 marker 增强 |
| Figure 10，p.18 | 11 源原始规模与约 712K 平衡混合物 | **来源调查已有，样本混合未重做** | 各源版本/许可/计数、确定性采样/去重/划分 manifest 与真实饼图；H2O/H2O-3D 身份待作者清单消歧 |
| Figure 11，p.21 | 原 RGB 上 MANO 重投影及 UmeTrack/MANO 重建质量 | **未实现；外部资产依赖**；[IK][ik]与重投影相关官方入口已发布 | 随机抽样清单、原全帧/标定/变换、MANO 与 UmeTrack 资产、实际 overlay 和质量统计；crop 不能代替全帧重投影 |
| Figure 12，p.26 | 59 类 Vision/Fusion 散点，点大小=N、颜色=族群 | **未实现，依赖 Table 15**；[官方生成入口][gesture-paper]存在 | 从同一未舍入底表自动生成散点/标注/表格，保留 raw 排除与样本数；不需新训练，但需真实派生评价 |
| Figure 13，p.26 | 密采样自遮挡与视觉/融合误差、收益趋势和相关 | **未实现；外部几何资产依赖**；[评价][occlusion-eval]、[几何算法][occlusion]、[绘图][occlusion-plot]存在 | 1/4 stride 的实际密网格、GT mesh/相机/分辨率与过滤、逐样本底表、bin/SEM/趋势/逐样本 Spearman；原文 17,168 个手样本不是现有 2,792 中心 |

## 数据、采集、标注与算法清单

这些项目与模型训练并列验收。消费公开标签不等于重生成标签；引用原采集元数据不等于重新采集。

| ID / 论文位置 | 必须覆盖内容 | 当前实际证据、状态与欠缺 |
| --- | --- | --- |
| D01，§3.1/A.3 | 原采集人数、人口、时长、prompt 与双手执行 | **部分来源核对**：E4 的 41 episodes、66,161,725 行只能说明发布记录，按 2kHz 约 9.189h；不能证明 >10h 原录制、性别/年龄或自然人去重。缺匿名人口/原采集记录；无需伪造重新招募 |
| D02，§3.1/A.3 | EMG/IMU/mocap/双视频的采样、时间戳软同步与线性插值、校准诊断 | **部分工程证据**：E3 有中心映射与 crop 覆盖，新校准 JSON 已读取；缺全部模态、视频原生尺寸、stale/delta 分布和同步核验。原文未报告统一同步误差数值，不凭空新增该论文结果 |
| D03，A.3 | release schema、投影框/左手翻转/256 crop、全模态盘点 | **legacy 获取与读取已核验**，E3/E4/E5；缺从 mocap+全帧重生 crop、当前 v3 manifest/checksums 与原 v1 队列对应。选择性清单不是完整 release |
| D04，A.4 | DC、50/100Hz notch、bandpass/rolloff、mV 与边界、原始 EMG | **未重做原始预处理**；现有为整 episode 预过滤+训练行归一化；[官方过滤入口][filter]已发布。缺 raw 和实际频域 mask/边界验证，严格时间隔离需独立分区滤波及边界丢弃 |
| D05，B.1.1 | 21 虚拟 marker、MANO 顶点索引、左右 convention、局部坐标和刚体恢复 | **官方推理规格部分可读，未重运行**，见[官方推理入口][m2m]；缺 raw markers/valid、合法 MANO 模型和实际重建产物 |
| D06，B.1.2 | 图 Transformer、8 维 Laplacian PE、最短路 attention bias、β/θ6D→MANO | **未实现、外部发布缺口**；推理脚本导入未发布的 markers2mano 包；缺真实训练架构/图谱/配置/权重。不得以未训练替身验收 |
| D07，B.1.3 | 11 源参数/mesh、tier 采样、约 712K 混合、去重和 train/val/test | **来源调查，不是采样完成**；下节逐源列访问情况。缺精确样本清单、采样 seed/权重、版本与来源划分；图中比例不是实采数据 |
| D08，B.1.3 | 50% 不增强；8 类为 bone length、global scale、近邻 swap、dropout 插值、neighbor blend、Gaussian noise、局部 drift、spike outlier | **未实现**；缺官方变换训练代码、精确参数/组合与概率、确定性检查和图9产物；不能用姿态模型 EMG 增强替代 |
| D09，B.1.3 Eq.2 | vertex MSE + 5×marker MSE 的 annotator 训练、选模与质量评价 | **未运行**；缺训练入口、schedule、选择规则、权重与完整曲线。姿态基线 L1+tip 损失不等于该训练 |
| D10，§3.2/B.1.3 | marker/vertex 误差、invalid rate、InterHand 子集与重投影质量 | **未重测**；论文 aggregate 只作参考，缺单位、有效帧判据、数据划分与误差分布。3.2mm 的正文 per-marker/附录 per-vertex 名称矛盾待裁决；不是角度真值精度 |
| D11，B.2 | 腕带 marker 构系、FE/RU 几何公式、左右符号、退化帧/有效性 | **仅消费 released 腕角**，E5 单位转换已核对；未从原 marker 重生成，缺原始腕带/手 marker、顺序/正交化/阈值及可确认生成入口 |
| D12，B.3 Eq.3 | MANO→UmeTrack landmark IK、对齐/限位/L-BFGS、收敛/残差 | **官方可读实现存在，未重生成**，见[batch_ik_mesh.py][ik]；其 mapper 训练不是 D06。缺 MANO pose/β/transform 与合法模型；legacy 元数据描述另一算法，不可当同一链路 |
| D13，§3.3/A.2 | 每参与者左右各一β、82手统计 | **实际统计与聚合图已核验（legacy）**，见[产物](../results/dataset-statistics/shape-statistics.md)；metadata引用行去重，每行左右各一次。episode/参与者自然人身份的独立证明仍缺，不能将逐帧重复值当独立手 |
| D14，§3.4/§4.2 Eq.1 | split、MAE、逐用户 SD、加权 Avg、轨迹/中心区别 | **legacy 部分已核验**，E1/E2/E3/E5；与 PDF 队列不等价；需所有新模型逐项复核。± 是跨用户 SD，不是多 seed 误差条 |
| D15，§4.3/C.1 | 左右 remapping、dropout/frequency mask/noise/jitter 等 EMG 增强 | **现有仅声明的官方 rotation 适配**；不能称覆盖 PDF 全部增强。classic 默认/with_aug 与实际归一化由 resolved 配置裁决，缺真实新配方训练 |
| D16，§4.3/C.1 | TDS/SE/RoPE/回归形状、输入与上下文对齐 | **S/M已核验，L工程验收通过**，E1/E6；classic原模型tuple/真实shape已核验，数据和训练链尚待适配。按真实上下文校验分母，非因果/全窗行为不宣称在线延迟 |
| D17，§4.4/C.2 | 7 视觉模型的真实主干、初始化、角度头、损失和训练 | **RN18/RN50/RN152 legacy 完整，其余未完整运行**，见模型表；DINO-S仅真实预训练/CPU工程证据，DINO-B已通过真实预训练CPU检查，L与WiLoR尚欠真实权重闭环；缺失配方仍须声明 |
| D18，§4.5/C.2 | 两对残差融合、时间池化、分支来源、末层初始化 | **RN18+S legacy 已核验，ViT-S+S 未运行**；E1/E2。精确零是明确适配，官方极小正态与 PDF zero 的区别保留 |
| D19，D.6 Eq.4 | camera-space 自遮挡 z-buffer、5×5邻域/5mm、三角面积加权 | **[官方算法][occlusion]存在，本项目未运行**；缺 GT pose/β/faces/world→camera/K/真实尺寸与 validity；不得以 crop 遮挡或 2D 框重叠替代 |
| D20，D.6 | 1/4 stride 密测、bin、SEM、paired gain、趋势与 Spearman | **未运行**，见 Figure13；[官方绘图][occlusion-plot]的实际过滤/样条与 PDF 不同，按下节裁决并保留全部底表 |

## 资产与外部依赖

| 资源 | 已确认事实与当前状态 | 尚欠/继续执行条件 |
| --- | --- | --- |
| legacy 已用数据 | E4 的原 21 文件约 18.39GB，另有视觉对齐字段；E3 的 41 份 crop 约 45.04GB 已核验 | 不包含所有 raw/mocap/MANO/camera/video；字段存在于 schema 不是获取证明 |
| β 与 calibration | [左β][left-beta]、[右β][right-beta]、[校准][calibration]共4,646bytes已取；beta统计/聚合图已核验 | 参与者身份独立证明与相机适用性待核验；统计的两处PDF微差和图形适配详见实际产物 |
| raw 与 markers | 官方 legacy 公共目录已定位；双手 raw 约 4.234GB，双手 marker+valid 约 36.124GB | 本次清单未确认这些大数组已下载；需实际 receipt/hash/读取，不能以 HEAD 成功代替 |
| 遮挡几何 | 必要 pose、world/camera 变换、tracked、marker-valid、β 约 37.778GB，再加校准与尺寸证据 | 未获取完整必要字段；需合法 MANO 模型和 legacy webcam→current head 字段映射，不能静默 β=0 或尺寸 fallback |
| current v3 | [ASSET_SETUP][asset-setup]写当前百度路径与 preview/checksums；[LICENSE][official-license]仍写完整 current release 未发布 | 当前目录实物、完整 manifest/checksums 与 v1 队列未闭合。官方 CLI 路线需百度登录，不将“链接可打开”写成“全量已匿名下载” |
| EMG2Pose | [Meta 官方数据与代码][emg2pose]给匿名 S3；完整包 462,824,048,640 bytes（约431GiB），metadata 与30文件 mini 已完整取得并校验 | 完整数据尚未取得。[直接转换](../egoemg_repro/emg2pose_data.py)已通过mini全字段验收；[分块xz归档](../egoemg_repro/emg2pose_archive.py)已通过独立审阅，正式全量接收已启动且首块后新块提交成功，仍明确incomplete。动态容量不足将停止并保留可恢复进度，不能缩小cohort；首块压缩比不保证全量容量。不能同时保存全tar/H5/Zarr/memmap |
| EMG2Pose 发布权重 | Meta 参考权重对应其 classic 模型；EgoEMG 作者 EMGFormer 权重是另一资产 | 当前代码 val/test 已修复不追溯清洗旧选模史；不得以旧 released checkpoint 或 mini 替代三个新完整实验 |
| MANO | 限定项目范围查找已发现左右 MANO 模型文件；[官方许可][mano]与具体格式兼容性尚待核验；安装 smplx 不等于完成资产验收 | 不能继续称模型完全不存在或必须重新下载；须核验现有文件的格式、版本、许可与 WiLoR/标注/遮挡接口，不再分发模型或公开私有位置 |
| markers2mano | [推理/模型加载入口][m2m]与[质量评价入口][m2m-eval]已发布，.gitignore 排除 markers2mano；默认 checkpoint 为作者私有运行 | 缺模型包、11 源训练预处理/采样清单、训练入口/配置/增强、train/val 来源划分、v51 权重公开来源。MANO→UmeTrack mapper 训练不能补这个缺口 |

下面是 annotator 的 11 个来源，不是“全部需要账号”，也不是“已经全部获取”。本页记录 2026-10-26 调查快照；公开链接与传输、许可检查、完整混合物是不同门槛。

| 来源 | 官方路径/可推进部分 | 实际缺口 |
| --- | --- | --- |
| GigaHands（PDF 写 GigaHand） | [作者仓库](https://github.com/brown-ivl/GigaHands)的 hand_poses HTTPS 包匿名 HEAD 可达，约11.30GB | 未下载；需包内许可与参数/抽样核验，不必为 mesh 训练获取全部多视角视频 |
| HOT3D | [官方 toolkit](https://github.com/facebookresearch/hot3d)；其 [HF clips](https://huggingface.co/datasets/bop-benchmark/hot3d)部分训练 GT 可匿名获取 | 全量 VRS 路线需邮箱领取限时链接；clips 不是作者完整采样证明，仍缺确切混合物与版本 |
| ARCTIC | [官方数据说明](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md)，raw_seqs/meta 可独立于数百 GB 图像获取 | 需本人注册与 research license；相关模型另有许可，未取得训练数据 |
| HOI4D | [官方页](https://hoi4d.github.io/)公开 Hand Pose/Camera/Annotations 链接 | 参数包传输与大小未验证，不笼统写必须申请账号 |
| HanCo | [官方数据页](https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html)提供 shape/calib/meta 分包 | 调查时官方旧下载端点 404，需恢复或核验正式镜像；research-only 条款与采样待闭合 |
| DexYCB | [官方页](https://dex-ycb.github.io/)有全包/按 subject 分包与 MANO 相关标签 | 未发现独立完整 MANO-only 小包；可单 subject 集成，不能当全量 mixture 完成 |
| H2O / H2O-3D | 文字引 [Kwon H2O](https://github.com/taeinkwon/h2odataset)，图写 H2O3D 76K、与 [H2O-3D](https://github.com/shreyashampali/ho3d)相近 | 数据集身份未决；前者需注册、后者有公开链接，不能混同；需作者精确来源清单 |
| InterHand2.6M | [官方页](https://mks0601.github.io/InterHand2.6M/)独立 annotations 含 MANO pose/shape/trans | 未获取/核验；无需仅为参数训练下载全部 RGB |
| FreiHAND | [官方页](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)有训练/eval 资源说明 | 调查时官方 v2 zip 端点 404，需正式端点恢复/镜像和 research-only 条款核验 |
| HO3Dv3 | [作者仓库](https://github.com/shreyashampali/ho3d)公开 v3 与 evaluation GT 链接 | 未验证完整传输；custom academic/noncommercial 及再分发限制，需参数提取/采样清单 |
| HO3Dv2 | 同一作者仓库公开 v2 与 evaluation GT | 同 v3；两版本录制重叠与训练权重须按作者清单核验，不能当独立新增帧直接相加 |

## PDF 与官方实现的裁决记录

采用“官方代码优先”时，应写明实际源码、resolved 配置和影响；无法由已发布内容裁决的项继续标为待核验。不能通过改写旧结果或挑选论文显示值来消除差异。

| 项目 | PDF / 旧数据陈述 | 固定官方源码与本项目处理 |
| --- | --- | --- |
| 输入/时间 | 7,790 samples；图4/6写7790ms；图示8通道而C.1写16 | [L][large]等当前配置为目标手8通道、12,000点；7,790samples在2kHz为3.895s。新实验逐配方声明；官方 unified 配置可能含多来源，不当 legacy 队列已等价 |
| 标签算法 | B.3为 landmark L-BFGS IK | [官方 IK][ik]存在；实际 legacy metadata_summary 明写 axis-angle projection，legacy README 又描述 anatomical Euler 路径。应保留版本/生成算法差异，不能凭同22维宣称同标签 |
| 滤波/归一化/优化 | PDF无幅值归一化；EMG/视觉/融合预算与部分学习率不同 | 以实际配置和运行身份为准；现四模型训练专用标准化、整 episode filtered、RN18与融合配方均已冻结，新运行不得追溯改写 |
| classic 内部矛盾 | vemg 注释称无归一化/原架构，数据却设16槽 | [实际 YAML][vemg]启用 per-dataset 归一化，首卷积仍8通道；NeuroPose需16槽。先记录显式数据/模型裁决，再真实验收；基类 out_channels=20不代表实际预测20角 |
| 视觉独立配方/头 | 表8给7模型配方；图示与文字激活不同 | RN50/152、ViT-B/L/WiLoR独立YAML未发布；ResNet/DINOv2头为ReLU，WiLoR角度头为GELU。不能将脚本引用缺失文件视为完整recipe已具备 |
| 融合初始化 | PDF末层zero | 当前官方为极小正态权重、零bias；本项目已声明精确零适配，旧 clean 分支身份保留。发布16槽配置不与8通道分支静默混载 |
| β统计/图 | Table5末列写P95–P5；Fig3a箱线图、PCA无固定语义 | [beta脚本][beta]实际std(ddof0)、末列ptp=max−min，画2×5左右柱图且附语义名；以range说明官方算法，真正分位差另算，图形/语义差异保留 |
| 手势评价人口/误差 | T14/15称所有split、59类；PDF族群Δ与逐类显示Δ加权不全相符 | [官方eval][gesture-eval]默认gesture split、取EMG token中点，逐样本MAE未按传入mask筛角；[paper脚本][gesture-paper]依赖输入先排raw。本项目masked/aligned中心不能静默当同一指标；须固定实际过滤、底表和差异 |
| Table10文字 | 段落称L最强Gesture/Both，显示表中S更低 | 保留原文内部矛盾，输出真实模型结果与选择规则，不以文字结论约束结果 |
| 遮挡工程前提 | 论文相机/原生分辨率与legacy webcam字段 | [eval入口][occlusion-eval]根路径可能指向scripts/config，硬编码校准、读current head字段，β缺失默认0、尺寸失败默认1920×1080；这些为静态风险，未运行不称实测失败。必须核验字段/路径/尺寸，不能默用fallback出图 |
| 遮挡统计/曲线 | 固定0.05 bin、LOESS/趋势置信带；正文与图内ρ不同 | [实际绘图][occlusion-plot]先滤occlusion非空且vision MAE>0.01，0.05基础bin合并至默认至少100样本，末不足10丢弃；使用加权三次UnivariateSpline，非LOESS；Spearman用过滤后逐样本。只(a)绘SEM带，趋势无对应置信带。按代码生成并明确差异，不抄ρ/p |
| 数据人数/时长 | 41参与者、6heldout、>10h | E3/E4实际41记录、40测试用户、User/Both各5用户且各1与train交集；发布行数约9.189h。标签数不自动证明自然人数，采样人数不等于窗口数 |

## 外部引用、未来工作与未展示项目

- Table 1 中其他数据集和 Figure 10 的原始来源规模属于外部资料；annotator 的实际抽样混合与训练属于作者工作，两者均需来源但不能混作本项目实测。
- Table 11 的 † 结果来自对应原论文。VQ-MyoPose 文中只提不同子集；CLDM/VQ 未在 EgoEMG 报告实验。无需创造论文没有的 EgoEMG 训练行。
- RGB-D、IMU 和时序多帧视觉是论文说明的数据模态及未来方向；应核验数据存在性、同步和 schema，不捏造原论文对应模型成绩。
- PDF 没有独立定量消融表、推理 latency/FPS/throughput/FLOPs 或多 seed 实验。C.1 被正文称有 ablations，实际未展示；B 被称有逐手势标签误差分布，实际只有方法与 aggregate。登记为“论文提及未展示，待官方资产”，不虚构数值。官方额外 joint-range/t-SNE 图也不误编号成 v1 的13幅图。
- consent/IRB/补偿/匿名化、许可、稳定托管/校验及 checklist 应映射到作者实际公开文件；原自述不能替代本项目证据，也不要求伪造新伦理审批。承诺未来发布的资源不当已取得资产。

## 继续顺序与全文完成门槛

1. 保持 L 独立完整训练，待真实结束后完成选定 best 的最终测试、严格导出、锁定复算；观察超时不重复启动。
2. 已生成并核验β统计与聚合图、60类词表/22维语义及中心覆盖；继续补齐各模型逐手/中心子表和统一参数/实际用时统计。每项仍须保留协议、读取身份与实际计数。
3. 从固定 selected-best 做一次足够细的派生评价，保存匿名 sample key、逐角 error/valid、gesture/hand/split/crop，可共同支持 Table12/14/15、Fig7/12；不能从旧总体均值逆推细分结果。
4. 实施经典模型 tuple/context/16槽/精度适配，以及 ResNet/DINOv2 分派、真实初始化和明确派生配方；ViT-S完整结束后再以 clean 分支训练 ViT-S+S。WiLoR先闭合专用资产与22角配方。
5. 完成 EMG2Pose mini 工程验收，独立解决完整数据存储/转换与三训练；并获取遮挡必要几何、MANO/相机证据后先最小集成、再完整密采样分析。子集检查不替代全量。
6. 持续查找 markers2mano 缺失模型包、训练规格、权重和精确11源清单；先完成公开可得来源与标注下游检查，真实发布缺口逐条保留，不以未训练替身宣布完成。

全文仅在本页所有必需作者实验、数据/标注算法、分析与定性产物达到各自验收，引用/未来项正确归类，公开结果与同一证据一致，且无未完成或未验证必需项时完成。阶段计划结束、配置存在、mini、绿测试、短训练或四基线结果均不能替代这个门槛。

[official]: https://github.com/zhenqis123/EgoEMG/tree/f7cbd048b7092139852270491d3859cd472e0311
[large]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/egoemg_emgformer_large.yaml
[vemg]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emg2pose/regression_vemg2pose_egoemg.yaml
[neuropose]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emg2pose/regression_neuropose_egoemg.yaml
[rn18]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/fusion/vision_resnet18.yaml
[resnet]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/resnet_vision.py
[vits]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/fusion/vision_vit_small.yaml
[vit]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/vit_vision.py
[wilor]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/wilor_vit.py
[fusion-vits]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/fusion/fusion_vits_s_center_8ch.yaml
[e2p-s]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/emg2pose_emgformer_small.yaml
[e2p-m]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/emg2pose_emgformer_middle.yaml
[e2p-l]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/emg2pose_emgformer_large.yaml
[config-readme]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/README.md
[gesture-paper]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/paper/generate_per_gesture_appendix.py
[gesture-eval]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/eval/analyze_per_gesture.py
[beta]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/paper/plot_beta_diversity.py
[constants]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/constants.py
[paper-figures]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/paper/plot_paper_figures.py
[m2m]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/mano/infer_mano_for_egoemg.py
[m2m-eval]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/eval/eval_m2m_checkpoints.py
[ik]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/ik/batch_ik_mesh.py
[filter]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/data/generate_egoemg_filtered_paper_memmap.py
[occlusion-eval]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/eval/evaluate_occlusion_fusion.py
[occlusion]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/occlusion.py
[occlusion-plot]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/paper/plot_occlusion_fusion.py
[asset-setup]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/docs/ASSET_SETUP.md
[official-license]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/LICENSE
[left-beta]: https://drive.google.com/file/d/1B8fhGsjNhy7-pkeqaA8Ru1kmMtAb-z2L/view
[right-beta]: https://drive.google.com/file/d/19SMv0e-aL1gNSpN5I5qcv2ROHxIxCJDv/view
[calibration]: https://drive.google.com/file/d/1atBjy4EgNLNzw87SiOvC4CCjNrljxfcc/view
[emg2pose]: https://github.com/facebookresearch/emg2pose
[mano]: https://mano.is.tue.mpg.de/

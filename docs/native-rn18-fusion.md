# RN18＋EMGFormer-S 原生融合

本实验使用固定官方版本 `f7cbd048b7092139852270491d3859cd472e0311` 的 `fusion/fusion_rn18_s_center_8ch`。旧版已完成的 RN18＋S 结果保留原身份；当前新路线尚未完成真实分支权重交接、模型接入和正式训练。

[实际配方与训练足迹分析](../results/native-rn18-fusion-feasibility-20261101.json)、[配置接口](../results/native-rn18-fusion-protocol-20261101.json)、[81项工程检查与独立CPU准入](../results/native-rn18-fusion-reader-and-cohort-cpu-20261101.json)和[真实完整输入两遍准入](../results/native-rn18-fusion-full-cohort-admission-20261101.json)分开记录。

| 项目 | 固定源码实际设置 |
| --- | --- |
| 模型 | MidFusionPoseFormer，ResNet-18＋8通道小型EMG分支 |
| 输入 | 12,000点；训练stride1,200、评价stride12,000；repeat1 |
| 监督 | 原始中心22角及generated mask；缺图保留独立vision mask |
| 批量 | train/val/test均200；另声明的val100未被原数据模块使用 |
| 加载 | 10workers、persistent=true、prefetch2、pin_memory=true |
| 优化 | Adam，lr0.0006、weight_decay0.0001；cosine150、eta_min0.00001 |
| 预算 | 150轮，每轮完整验证，val_mae/min选top3＋last，无EarlyStopping |
| 损失 | 两个指尖别名实际累计为0.02 |
| 视觉分支 | 执行配置不冻结；保留原BatchNorm/训练行为。源注释“冻结”与实际字段冲突，遵循可执行字段 |
| 残差 | 原始末层Normal(std=1e-5)＋零bias；未改成精确零权重 |

实际数据仍是41人旧版EgoEMG，没有伪称包含未获得的ShowEE/Incre。原统计资产的`egoemg_unified`命名查找失败，读取器明确使用同一原资产中的`egoemg`每手、每通道float32向量。原`filtered_paper`请求只在IO层指向实际`filtered`文件；原始文件、标签及统计不改写，也不称为本项目训练集内估计的统计。

两遍真实输入准入得到完全相同身份`1736318c9faf5740417ca676cd520d47ada84d16eac31f865e9cdf24a9f2e38c`：train/val/test为62,470／1,052／2,792个唯一手部样本。每遍完整哈希53份消费payload，共61,377,230,438字节，含40份实际消费crop；元数据仍有41段。12,000窗口与7,790窗口消费的episode集合不同，不能用前者冒充后者的41份crop/重复评价人口。全部可能随机输入足迹与中心标签均扫描，无非有限项；两进程构造RNG未变，0模型／0优化器更新／无CUDA，原6个拥有进程均关闭。

新代码入口为`rn18_fusion_literal`、`rn18_fusion_data`、`rn18_fusion_cohort`。完整准入始终重新读取并哈希输入，现有JSON不能作为免检凭据。源码内容、文件来源、只读typed布局、归一化、图像参数、角色与分区分别绑定；轻量环境40项纯检查通过，41项真实运行时检查明确skip，完整运行时81项实际通过。

初始化还需单独验收：视觉必须使用符合足迹与训练来源要求的完整RN18 LAST，EMG必须使用原指定cotrain-S配方的兼容验证BEST。当前原生EMG-S的12,000分区仅证明足迹兼容；其200轮/0.02配方不能自动冒充150轮cotrain初始化。已完成的ViT兼容初始化与本融合的留出区重叠，不能复用；新7,790视觉分区也有重叠。需要按本融合的固定留出区准备原cotrain-S初始化，再完成原生构造、恢复、GPU资格、正式150轮和独立最终测试。

本页不增加已完成的14项科学设置。完整论文、密网格遮挡与增益相关、标注与预处理等剩余目标继续保留。

RN18兼容cotrain-S初始化的[两次完整轨迹数据准入](../results/rn18-cotrain-full-input-admission-20261101.json)已完成：62,470训练/1,052验证，16份payload/16,540,457,757字节，4次整文件哈希及全部可能轨迹目标检查一致，12原拥有进程关闭。初始化测试集不打开；未来融合留出区身份固定44e9ebc8…b4bd0。新的训练/恢复适配正在工程验证，正式150轮初始化、完整验证BEST和RN18融合150轮仍未完成；数据准入不授予权重交接资格。

# EgoEMG 论文与实现阅读指南

本页解释三个任务如何从数据走到指标，并区分**论文 v1 的描述**、**固定官方源码的行为**和**本项目的适配**。它不是本项目的实验结果；完成条件见[复现覆盖矩阵](reproduction-matrix.md)，实时进度见[运行状态](run-status.md)。

阅读依据为用户提供的 34 页 [arXiv v1 论文](https://arxiv.org/abs/2605.05712v1)与官方提交 [`f7cbd048b7092139852270491d3859cd472e0311`](https://github.com/zhenqis123/EgoEMG/tree/f7cbd048b7092139852270491d3859cd472e0311)。下文页码是 PDF 印刷页码；所有官方源码链接固定到这个提交。本仓库不再分发论文、数据、参与者信息或模型权重。

## 1. 任务与数据：预测的是关节角度

论文的三个任务共享每手 **22 维角度**：EMG 窗口预测姿态轨迹，单张第一视角手部图像预测中心帧姿态，EMG 加图像预测同一中心帧姿态。输出不包含手在世界坐标中的平移、全局朝向或 MANO 手形参数。外部 RGB-D、IMU 和多帧视觉虽属于采集系统，却没有进入论文主基准。来源：[§3–4，pp.3–7](https://arxiv.org/pdf/2605.05712v1#page=3)。

| 项目 | 论文 v1 的采集与发布描述 | 本项目使用与边界 |
| --- | --- | --- |
| 人数与动作 | 41 人、超过 10 小时、60 类动作；30 单手、30 双手；单手动作也由双手镜像执行 | 官方 legacy v2 的 41 episode / 41 subject、66,161,725 行；按 2 kHz 行数折算约 9.19 小时，不等同于原始录制时长口径；人数相同不能证明是同一标签版本 |
| 肌电 | 每腕 8 通道，2 kHz；双腕采集共 16 通道 | 每次模型调用取目标手腕的 8 通道，两手分别构造样本、共享模型 |
| 运动捕捉 | 每手 21 个反光标记，120 Hz；腕带另有标记 | 读取发布的角度与有效性字段，不重建原始 mocap |
| 图像 | 头戴 RGB 1280×720、60 fps；外部 RGB-D 1280×720、30 fps | 读取发布的 256×256 单手 RGB crop 与中心帧索引 |
| 同步 | 主机时间戳软同步，线性插值到共同时间线 | 遵循发布的行/视频帧映射；不宣称独立测得同步误差 |
| 格式 | episode parquet、视频 MP4、逐 episode crop LMDB | legacy memmap v2 加 LMDB，经适配器读取，不冒充当前完整 v3 release |

采集与 schema 来源：[§3.1，p.3；A.3，p.14](https://arxiv.org/pdf/2605.05712v1#page=14)。legacy 检查与下载范围见[可行性审计](feasibility-audit.md)和[下载说明](download.md)。当前官方源码还提供合并 EgoEMG、ShowEE、Incre 的 [unified dataset 配置][up-dataset]；本项目没有因此把这些额外训练数据算入 legacy 队列。

**视觉裁剪有真值条件。** 论文的手框由 mocap 关键点投影产生，不是由待评测的视觉模型检测出来。因此这里测的是“给定已对齐手部 crop 的姿态回归”，尚不包括实际部署中的检测失败、裁剪偏移和同步误差。论文也没有用这组结果证明自由生活或复杂物体交互中的端到端性能。来源：[A.3，p.14](https://arxiv.org/pdf/2605.05712v1#page=14)。

## 2. 标签从哪里来：两条路径合成 22 维

论文的标注链路是：

```text
21 个手部 mocap markers → markers2mano → MANO 姿态/手形 → UmeTrack landmark IK → 20 个手指角
腕带 markers + 手部方向 → 前臂坐标系 → 2 个腕角
                                                        合并为每手 22 维
```

### 2.1 markers2mano 与逆运动学是两个不同步骤

**论文描述。** markers2mano 先把 marker 坐标变换到局部坐标，再用 Graph Transformer 预测 MANO 的姿态 θ∈R48 和手形 β∈R10。21 个 marker token 加入图拉普拉斯位置编码（8 个特征向量），注意力加入骨架最短路径距离偏置。形状头对 token 做均值池化，姿态头展平 token 并预测 16 个关节的 6D 旋转，再转为轴角。MANO 前向得到 778 个网格顶点。训练样本来自 11 个公开手部网格数据来源，并用 marker 位移、交换、丢失与噪声等增强模拟采集误差。来源：[B.1，pp.17–19](https://arxiv.org/pdf/2605.05712v1#page=17)。

之后的 MANO→UmeTrack 转换才是标量角度 IK：固定模型间坐标对齐后，使 UmeTrack 的 20 个 landmark 接近 MANO 对应关节。论文采用带 strong-Wolfe 线搜索的 L-BFGS，学习率 0.1，100 次外层、每次至多 50 次内层迭代，batch 256，并以 sigmoid 重参数化约束关节范围。这个步骤拟合的是 MANO 生成目标，不是独立测量的真实角度。来源：[B.3，pp.19–20](https://arxiv.org/pdf/2605.05712v1#page=19)。

**固定源码与证据缺口。** 官方 [batch_ik_mesh.py][up-ik] 支持 landmark/chamfer 损失、稀疏锚帧求解后插值，以及可优化的全局朝向和平移；存在该脚本不等于能证明下载的每个标签文件都由论文所述参数生成。论文也未给 markers2mano 的完整隐藏宽度、层数、训练日程和可核验的标签生成记录。本项目使用 legacy 的 `generated_joint_angles_*`，不重新训练 annotator、不重新运行 IK，且未证明 legacy 标签与 v1 表格监督逐文件等价。

### 2.2 腕角独立计算

腕带标记建立前臂坐标系，手腕到中指 MCP 的方向相对此坐标系给出屈伸与桡/尺偏角；论文规定伸展和桡偏为正，左右手使用相应法向约定。它不通过 MANO 手指 IK 得到。来源：[B.2，p.19](https://arxiv.org/pdf/2605.05712v1#page=19)。

| 维度 | 手指/关节 | 运动语义 |
| --- | --- | --- |
| 0–3 | 拇指 CMC、CMC、MCP、IP | 屈伸、外展/内收、屈伸、屈伸 |
| 4–7 | 食指 MCP、MCP、PIP、DIP | 外展/内收、屈伸、屈伸、屈伸 |
| 8–11 | 中指，同上 | 同上 |
| 12–15 | 无名指，同上 | 同上 |
| 16–19 | 小指，同上 | 同上 |
| 20 | 手腕 | 屈伸 |
| 21 | 手腕 | 桡偏/尺偏 |

排序来源：[Table 6，p.20](https://arxiv.org/pdf/2605.05712v1#page=20)。论文表格以度报告，但**磁盘、损失和显示单位必须分别检查**：[本项目数据适配器](../egoemg_repro/data.py)读取 legacy 手指角弧度和腕角度数，把腕角转成弧度后合并；[指标累计器](../egoemg_repro/metrics.py)累计弧度绝对误差，在输出 MAE 时统一乘 `180/π`。手指有效标记和手腕有效标记独立，不能因手指无效而忽略仍有效的腕角。

论文报告 EgoEMG marker-to-mesh 误差 4.3 mm、MANO→UmeTrack landmark 拟合误差 2.8 mm。这些是标记/模型几何误差，不能当作 22 维角度真值精度。正文与附录对公共数据上 3.2 mm 的指标描述也不同；3.6% 与 EMG2Pose 12.7% 的无效帧率来自不同管线/数据，不能写成同一输入上的受控消融。来源：[§3.2，p.4；B.1/B.3，pp.19–20](https://arxiv.org/pdf/2605.05712v1#page=19)。

## 3. EMGFormer：压缩时间，再建模长程关系

### 3.1 窗口长度与通道数需要按版本阅读

**7,790 是采样点数。** 2 kHz 下为 `7790 / 2000 = 3.895 s`。A.3 和 C.1 写 samples，Figure 2 写约 3.9 s；Figure 4/6 的 `7790ms` 标注与这些内容冲突，不能照图把输入解释为 7.79 秒。来源：[p.4、pp.6–7、p.14、p.21](https://arxiv.org/pdf/2605.05712v1#page=21)。

**8/16 通道在 PDF 内部存在歧义。** §4.3 每次预测一只手，Figure 4/6 标 8 通道；C.1 却写 16 通道及 `16→256` 首层。双腕采集 16 通道，不足以证明每次前向拼接双腕。固定发布配置明确使用 `target_hand`、首层 `8→256`、12,000 点窗口（6 s）；本项目初始实验遵循这一发布兼容结构。部分官方融合设置把**同一腕的 8 通道**映射到 16 个槽位，也不能称为双腕联合输入。参见 [S 配置][up-small]、[TDS slim][up-tds-config]与[视觉适配器](../egoemg_repro/visual.py)。

### 3.2 TDS、SE、RoPE 与回归头

TDS 前端通过时间卷积降采样，以较少 token 保留局部变化；每个 stage 的 SE 按整窗均值调节特征通道。随后 Transformer 的自注意力在时间 token 之间交换信息，RoPE 在 Q/K 上编码位置，最后逐 token 的 MLP 输出 22 维角度。

论文给出两次初始 Conv1d 和两个 TDS stage。2026-10-28已对固定官方S/M/L原始工厂完成[六次CPU前向与架构重绘](../results/dataset-statistics/emgformer-architecture.md)，下表各长度已由真实hook测量核对。时间卷积均为 `padding=0,dilation=1`，也符合 `floor((T−k)/s)+1`；7,790列仅是诊断输入，实际这三个EgoEMG配置均为12,000点，不能外推到任意前端配置。TDS 内的 `8×32=256` 是潜在特征分组，不是电极数量。来源：[C.1，pp.21–22](https://arxiv.org/pdf/2605.05712v1#page=21)、[TDS 实现][up-tds]。

| 时间算子 | k / stride | 论文 7,790 点 | 发布兼容 12,000 点 |
| --- | --- | ---: | ---: |
| 输入 | — | 7,790 | 12,000 |
| 初始卷积 1 | 11 / 5 | 1,556 | 2,398 |
| 初始卷积 2 | 5 / 2 | 776 | 1,197 |
| Stage 1 输入卷积 | 9 / 5 | 154 | 238 |
| Stage 1 TDS 时间卷积 | 5 / 1 | 150 | 234 |
| Stage 2 输入卷积 | 3 / 1 | 148 | 232 |
| Stage 2 TDS 时间卷积 | 3 / 1 | 146 | 230 |

S/M 前端形状为 `(B,256,T′)`。总时间 stride 为 50，名义步距对应 40 Hz；`146 / 3.895 ≈ 37.5 token/s` 包含边界裁剪，说明论文“约 37 Hz”的数值可以与该长度相容。仅这些卷积的局部感受野是 511 点（255.5 ms）；global SE 和后续非因果注意力再引入全窗依赖。不能把局部卷积感受野直接称为全窗，也不能把整个模型称为在线因果预测器。

| 模型 | Transformer 层数 | 隐藏宽度 | 头数 | FFN 宽度 | 论文参数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 3 | 256 | 4 | 512 | 3.5 M |
| M | 6 | 256 | 8 | 1,024 | 6.6 M |
| L | 8 | 384 | 12 | 1,536 | 16.3 M |

来源：[Table 7，p.21](https://arxiv.org/pdf/2605.05712v1#page=21)。上表末列仍是论文报告值；本项目现已对原始S/M/L统一实测总参数与可训练参数，分别为3,460,134、6,617,382、16,320,806，三者均全部可训练。实际配置、初始化状态摘要、FP32原生对齐和[PNG/SVG](../results/dataset-statistics/emgformer-architecture.md)均可追溯；没有数据读取、优化器更新或新精度结果。固定 [S][up-small]、[M][up-middle]、[L][up-large] 配置的结构数值与表格一致；M 的配置名是 `middle`。M/L 文件头部残留的 “3-layer small” 注释不应覆盖显式配置值。论文未交代 L 的 256→384 衔接，源码的 [Transformer][up-transformer] 则明确有输入 Linear 投影。

**实现差异。** 论文点名 RoPE，但没有完整实现规格；固定源码使用基数 10,000，对各 head 的 Q/K 偶奇成对旋转，选定的 S/M/L 均 `causal=false`。论文图中的回归头标 GELU，发布默认 [head 配置][up-head] 为 ReLU、隐藏 512、dropout 0.1。本文不把这些后续源码细节当作 v1 明文。

**轨迹对齐。** 固定前端报告 `left_context=510, right_context=0`；本项目先裁去目标前 510 行，再按官方线性插值把 token 预测对齐到目标长度。因此 12,000 点输入的轨迹评估长度为 11,490，中心对应裁剪后索引 `6000−510=5490`。`left_context` 是对齐约定，不代表 Transformer 的未来信息被屏蔽。见 [BaseModule][up-base]和[训练对齐实现](../egoemg_repro/train.py)。

## 4. 视觉与融合：一个中心帧，两种信息来源

**视觉分支。** 论文比较 ResNet-18/50/152、ViT-S/B/L 和 WiLoR。ResNet 从 ImageNet 初始化，ViT 从 DINOv2 初始化，WiLoR 使用专用 MANO 手部预训练；这些初始化和训练预算不同，不能据此单独归因“CNN 比 Transformer 更好”。RN18 的发布实现为 crop→ResNet 特征→全局池化→`512→512→22` 回归头，输出 `(B,22,1)`，ReLU 与 dropout 0.1。来源：[§4.4，p.6；C.2，p.22](https://arxiv.org/pdf/2605.05712v1#page=22)、[ResNetVisionPose][up-vision]。

**中心帧残差融合。** EMG 的 TDS+Transformer 输出先做可学习时间注意力池化，汇聚成单个向量，与视觉投影特征拼接后预测修正量：`ŷ = ŷvision + Δŷ`。因此修正量取决于 EMG 与视觉特征，并不是先独立预测两个姿态再平均。固定 [TemporalAttentionPool][up-pool] 为 `Linear(C,C/4)→Tanh→Linear(·,1)`，对时间 softmax 后加权求和；S 分支的 pooled EMG 为 256 维，视觉投影为 256 维，拼接后再投影回 256 维。来源：[§4.5，p.7](https://arxiv.org/pdf/2605.05712v1#page=7)、[中心融合实现][up-fusion]。

论文称残差末层零初始化；固定源码实际用标准差 `1e−5` 的极小正态权重与零偏置。本项目严格加载独立验证选出的两条分支，再把残差末层权重、偏置设为精确零，并全量微调视觉分支。这是显式适配，记录于[融合配置](../configs/fusion_rn18_s.json)和[初始化代码](../egoemg_repro/visual.py)。初始融合采用官方 8 通道 S 结构与 12,000 点输入，没有把 8 通道权重猜测性扩成 16 通道；因此也不是发布 16 通道融合 checkpoint 的原样验证。

发布预裁剪已处理左手图像方向，本项目不再次水平翻转；EMG legacy 适配器直接读取每手发布通道顺序，本项目没有额外加入论文未完整披露的左→右电极排列。EMG 训练仅施加所选官方 rotation transform（通道循环平移 −1、0 或 +1），不能因此声称完全重现论文的镜像和全部增强。缺失 crop key 保留为归一化黑图，并计入全中心结果；另报完全相同的有图子集。缺失整个 episode 数据库会失败，不会悄悄删去整个参与者。详见[视觉指南](vision.md)。

## 5. 预处理、损失与训练配方

**论文预处理。** 逐通道去均值，FFT mask 去除 50/100 Hz 干扰，20–850 Hz 带通，900 Hz 以上柔性衰减；不做幅度归一化，保留 mV。来源：[A.4，p.14](https://arxiv.org/pdf/2605.05712v1#page=14)。**本项目**读取 legacy 整 episode 预过滤信号，在最终保留训练窗口覆盖的唯一行上计算每手每通道均值/标准差，再用于所有角色。它既不同于论文“无幅度归一化”，也不同于使用全数据统计的发布 checkpoint 输入。时间区间无交叠仍不能消除整 episode FFT 带来的跨分区依赖，严格预处理隔离尚需从原始 EMG 分区过滤并丢弃边界。

**损失。** 论文写 L1 角度损失加权重 0.01 的 fingertip-distance；本项目按有效角度 mask 计算**弧度** L1，并调用固定官方 UmeTrack 前向运动学的手指距离项。只把前 20 个手指角传入该 FK，腕角的有效性仍独立参与角度 L1。损失是训练目标，不能把它的数值直接写成角度 MAE。官方距离实现还对时间索引作抽样；这不是每个 2 kHz 时刻都计算几何损失的承诺。来源：[C.1/C.2，pp.21–22](https://arxiv.org/pdf/2605.05712v1#page=21)、[官方 metrics][up-metrics]、[OfficialLoss](../egoemg_repro/train.py)。

**2026-10-26补充核验：** 上述legacy训练器的指尖项实际系数为0.01；固定官方EgoEMG experiment经Hydra合成保留两个指尖别名，Lightning将它们累加为0.02。此前漏核了合成后的这一行为，不能把单份YAML的0.01当成当前官方实际目标。EMG2Pose三项没有该叠加。完整来源、11份合成配置和实际归约见[损失审计](../results/official-loss-audit.md)；作者原checkpoint训练历史仍未据此证明。

| 设置 | 论文 v1 | 本项目已选配方 |
| --- | --- | --- |
| EMG 窗口 | 7,790 点；8/16 通道描述冲突 | 12,000 点、目标手 8 通道 |
| EMG 优化 | AdamW，200 epochs，batch800，lr1e−4→5e−6，WD1e−4，500 steps warmup | S/M：AdamW，200 epochs，batch500，lr5e−5→1e−5，WD0.01；无该 warmup |
| EMG 增强 | channel dropout、频域 mask、Gaussian noise、±40 ms jitter | 所选官方 rotation；固定窗口，不启用 jitter |
| RN18 优化 | 150 epochs，batch900，lr1e−3，AdamW | 150 epochs，batch900，lr1e−4，AdamW |
| 融合优化 | AdamW，lr1e−4，200–300 epochs，batch64–600，具体 pair 设置未完整给出 | Adam，lr6e−4，150 epochs，batch200 |
| 选模 | PDF 未提供足以重建的独立验证划分/选模记录 | 官方训练区间内时间留出，完整独立验证选最优 |
| 精度模式 | bf16，6×RTX4090 | bf16，实际 GPU 与配置写入 run 元数据 |

配方依据：[C.1/C.2，Tables 7–8，pp.21–22](https://arxiv.org/pdf/2605.05712v1#page=21)、[S 配方](../configs/emgformer_s.json)、[M 配方](../configs/emgformer_m.json)、[RN18 配方](../configs/resnet18.json)、[融合配方](../configs/fusion_rn18_s.json)。M 的[独立集成记录](../results/emgformer-m-integration.json)包含两步真实训练与严格重载；此后 S/M 已分别完成 200 轮完整训练、独立验证选模和训练器最终测试，并通过各自的严格结果导出，独立复算待完整裁剪后执行。复现层保留模型架构，但数据队列、归一化、验证和训练配方的这些差异使其属于 **legacy-release reproduction**。

## 6. 划分和评价：名称相同不保证统计口径相同

论文持出 10 类动作和 6 位用户，定义 Gesture、User、Both 三种泛化轴；正文对 User 写“持出用户的所有动作”，图示与实际发布标签的关系需要按 release 核验。不能自行合并 User 与 Both、重复计数，或硬编码 `1/2/3` 的顺序。本项目 [LegacyStore](../egoemg_repro/data.py)从 manifest / metadata 解析 `frame_split_id` 的名称表，要求 `train/user/gesture/both`，由[划分代码](../egoemg_repro/partitions.py)在固定 episode 网格上保留按中心归类的测试窗口。双手共享时间划分。

固定上游 [dataset 配置][up-dataset]把 `user/gesture/both` 用作 `val`，又令 `test=${dataset.val}`。新训练不能把这直接当独立验证。本项目在足够长的官方训练连续区间尾部确定性留出验证，并剔除跨训练/验证/测试输入行足迹的训练窗口。约 12% 是目标比例，短区间与完整窗口约束会改变实际比例。验证仍是**已见训练用户的时间留出**，不等于额外未见用户验证。当前数量与排除数以 sidecar 和[运行状态](run-status.md)为准。

**实际窗口队列与分组交集。** 已下载的 41 个 episode 对应 41 个不同用户标签，但其中 1 个只有 10,109 行，短于 12,000 点输入，实际测试窗口覆盖 40 人。训练、验证、Gesture 请求窗口的用户集合相同，均为 36 人；User 与 Both 的用户集合相同，均为 5 人，其中各有 1 人也出现在训练窗口中。这是按请求窗口、合并双手的集合统计，不代表逐条件具有有效监督的训练用户数。公开数量与来源哈希见[队列覆盖记录](../results/cohort-coverage.json)。

已逐一核对保存的 33,508 个窗口中心与发布的 `frame_split_id`，类别不一致数为 0。固定官方加载器同样[按中心标签取 split](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/datasets/egoemg_memmap_dataset.py#L903-L918)，[预览构建说明](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/data/build_egoemg_preview_memmap.py#L13-L21)也记录单个 recording 可含四类标签。这些源码没有证明该用户交集的成因或科学合理性。因此本项目将 User、Both 解释为 **legacy 发布中心标签分组**，不宣称严格无用户交集，也不由此改写论文 §3.4 的持出用户定义。本次冻结实验保留原标签、窗口和所选检查点，不依据测试结果删改队列。

令有效角度集合为 V，报告的 pooled MAE 为：

```text
pooled_MAE_deg = (Σ_(j,t)∈V |prediction_rad − target_rad| / |V|) × 180/π
```

| 统计项目 | 正确读法 |
| --- | --- |
| EMG 主表 Table 2 | 每个窗口的完整预测轨迹；本项目另保存 center 指标 |
| Vision/Fusion Table 3 | 窗口中心帧；比较时需同中心、同手、同 crop population |
| Gesture/User/Both 的 `mean ± SD` | 论文是按用户计算的均值与用户间标准差，不能当训练种子误差条 |
| Avg | 论文为跨测试 split 的样本加权汇总，不能简单平均三个显示均值 |
| 本项目 pooled 汇总 | 按有效角度数累计误差和与分母；缺失标签时要明确它与等窗口平均的区别 |
| 本项目 user_summary | 用户 MAE 等权均值与总体标准差 `ddof=0`；与 pooled MAE 分开；当前运行时为整个评估集合的用户汇总，不伪装成逐 split 用户 SD |
| 缺图比较 | 同时报告全部请求中心、crop 有效子集及缺失数；不能只保留有图样本再与全量 EMG 比 |

来源：[§4.2，p.5；Tables 2–3，p.8；Table 10，p.23](https://arxiv.org/pdf/2605.05712v1#page=8)、[指标实现](../egoemg_repro/metrics.py)。计划中的独立训练后审计将从已选定 best 复算逐 split/逐手用户均值与总体 SD，并给出 EMG 的匹配 crop-valid 中心结果；它不参与选模或调参。全中心结果须在预先声明的数值容差内与冻结运行时一致，审计产物通过核验前仍属待完成。模型推理与有效标签计数、手指/腕角分量、逐手/逐 split 结果都必须一起追溯。GPU 有限值检查和几步训练只能证明管线可执行，不证明收敛或论文精度。

## 7. 主表参考：仅来自论文，不是本项目结果

以下数值从 v1 第 8 页 Tables 2/3 核对；单位为度，越低越好。它们仅帮助定位基线和理解量级，**不得复制进本项目 measured results 或用作缺失结果的默认值**。`±` 为用户间标准差，Avg 为论文的样本加权值。

| Table 2，轨迹 | 参数 | Gesture | User | Both | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMGFormer-S | 3.5 M | 12.8±1.4 | 15.6±2.5 | 17.4±1.2 | 14.7 |
| EMGFormer-M | 6.6 M | 11.8±1.6 | 15.6±1.4 | 17.4±0.9 | 14.2 |
| EMGFormer-L | 16.3 M | 11.7±1.5 | 15.7±3.0 | 17.7±1.1 | 14.2 |
| vemg2pose | 6.0 M | 15.0±1.4 | 16.3±1.7 | 17.3±1.3 | 15.9 |
| NeuroPose | 6.4 M | 15.8±1.2 | 15.7±1.3 | 16.3±0.7 | 16.1 |

| Table 3，中心帧 | 参数 | Gesture | User | Both | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| V-RN18 | 11.5 M | 5.1±1.3 | 6.8±1.7 | 6.8±0.9 | 5.9 |
| F-RN18+S | 15.5 M | 4.6±1.3 | 6.4±1.4 | 6.4±0.8 | 5.4 |
| V-ViT-S | 21.9 M | 5.0±1.3 | 7.3±2.4 | 7.2±0.5 | 6.0 |
| F-ViT-S+S | 25.9 M | 4.5±1.3 | 6.8±1.2 | 6.8±0.6 | 5.5 |
| V-RN50 | 23.5 M | 4.5±1.3 | 6.2±2.2 | 6.2±0.7 | 5.3 |
| V-RN152 | 58.2 M | 4.3±1.2 | 6.1±2.3 | 6.1±0.7 | 5.1 |
| V-ViT-B | 86.2 M | 4.8±1.3 | 7.0±2.3 | 6.9±0.6 | 5.8 |
| V-ViT-L | 303.8 M | 4.4±1.3 | 6.6±2.2 | 6.6±0.7 | 5.4 |
| V-WiLoR | 631.6 M | 3.9±1.3 | 5.6±2.4 | 5.7±0.7 | 4.7 |

论文数据支持 M/L 的轨迹 Avg 优于所列传统 EMG 基线，但 Both 上 NeuroPose 更低；扩大模型没有在每个 split 稳定改善。两种轻量融合相对各自视觉分支均降低 Avg 0.5°，不能推广为优于所有单视觉模型：RN50、RN152 和 WiLoR 的 Avg 更低。Table 2 与 Table 3 监督粒度不同，不能直接用它们混算融合相对 EMG 的收益。论文的 [Table 10，p.23](https://arxiv.org/pdf/2605.05712v1#page=23)提供中心帧 EMG 参考；其中 S/M/L 的显示 Avg 分别为 14.5/14.9/14.6，而不是轨迹表的 14.7/14.2/14.2。

补充 EMG2Pose 的 [Table 11，p.24](https://arxiv.org/pdf/2605.05712v1#page=24)是另一数据集、20 个手指角和 User/Stage/User+Stage 协议；不能将该表替代 EgoEMG 22 维测试，也不能把在该数据集独立训练称为跨数据集迁移。逐关节、手势、遮挡分析属于额外实验，完成基础三模态链路并不等于已经重做这些分析。

## 8. 何时可以称为“复现完成”

本项目已批准的主目标是固定官方模型在可核验 legacy 队列上的完整训练、独立验证选模、重载与最终测试，再发布可重新计算的结果。它不以论文参考 MAE 填表，也不要求在版本证据不足时声称精确数值等价。

严格 v1 数值复现还需要论文使用的数据/标签/滤波文件、确切划分与窗口计数、完整训练与初始化配置、checkpoint 选择记录的等价证据；这些条件目前未闭合。工程测试通过、模型名字相同或参与者数量相同都不能替代它们。范围、逐项交付物和仍欠缺的证据统一列于[复现覆盖矩阵](reproduction-matrix.md)。

[up-dataset]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/dataset/egoemg_unified_angle_regression.yaml
[up-small]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/egoemg_emgformer_small.yaml
[up-middle]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/egoemg_emgformer_middle.yaml
[up-large]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/egoemg_emgformer_large.yaml
[up-tds-config]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/module/featurizer/tds_slim.yaml
[up-tds]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/featurizers/tds.py
[up-transformer]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/decoders/transformer.py
[up-head]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/module/head/mlp.yaml
[up-base]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/base.py
[up-vision]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/resnet_vision.py
[up-fusion]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/mid_fusion.py
[up-pool]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/_pooling.py
[up-metrics]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/metrics.py
[up-ik]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/ik/batch_ik_mesh.py

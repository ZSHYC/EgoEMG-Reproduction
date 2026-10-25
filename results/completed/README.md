# 已完成实验：legacy 四基线

截至 2026-10-25，**获批 legacy 四基线的完整训练与审计均已完成**。E1（EMGFormer-S）与 E4（EMGFormer-M）各完成 200 epochs、25,400 次更新；E2（ResNet-18）完成 150 epochs、10,650 次更新；E3（RN18+S 融合）完成 150 epochs、47,400 次更新。四者均通过独立完整验证选模、原训练器最终测试、严格导出及锁定中心复算。下表是本项目实际 legacy 成绩，误差越低越好；论文 v1 精确数值等价仍未证明，其他论文扩展未运行。

| 模型 | 全部中心 MAE (°) | 有图中心 MAE (°) | EMG 轨迹 MAE (°) | 验证选中更新 |
| --- | ---: | ---: | ---: | ---: |
| [EMGFormer-S](emgformer-s/metrics.md) | 16.313757 | 16.276525 | 16.289435 | 635 |
| [EMGFormer-M](emgformer-m/metrics.md) | 16.490835 | 16.453937 | 16.486140 | 635 |
| [ResNet-18](resnet18/metrics.md) | 7.784887 | 7.690786 | 不适用（单帧） | 6,319 |
| [ResNet-18 + EMGFormer-S](fusion-rn18-s/metrics.md) | 6.793533 | 6.692349 | 不适用（中心融合） | 36,024 |

全部中心指标来自原训练器最终测试，并由一次锁定中心复算按 **MAE 绝对容差 0.0001°、有效角度计数精确相等**逐组核对；有图中心来自这同一次复算的 `crop_present` 子集。轨迹指标来自原训练器最终测试，锁定产物检查核对了轨迹分母，**没有额外复算整段轨迹指标**。额外锁定核验中的 EMG 前向仍产生时间序列预测，经对齐和插值后取中心预测，只重新累计中心指标。S 成功的原复算产物经身份核验后接纳，未重放 S 推理；M、ResNet-18 与融合后续独立复算完成。ResNet-18 与融合均只输出中心指标，没有轨迹成绩；它们的原训练器有图结果也通过独立审计核对。审计不参与选模或调参。

四个模型共享相同测试请求、划分、归一化和 crop-present 人口，完整身份与共同人口摘要见 [publication-manifest.json](publication-manifest.json)。上表为按有效角度加权的 pooled MAE；逐用户均值及总体标准差另见各 `audit.json`，不能与 pooled MAE 或论文条件统计混算。

两条 EMG 完整曲线均表现为训练损失继续下降、验证误差在早期最低点后上升；最终测试使用验证选出的第 635 次更新，不能用最后一轮替代。这是当前配方存在过拟合的迹象；本次仅运行 seed 42，S/M 的误差差异不足以证明跨随机种子的优劣，也没有据测试结果重新调参。ResNet-18 按完整验证中心 MAE 选出第 6,319 次更新（第 89 个 epoch），最佳验证 MAE 为 7.825145°；融合选择第 36,024 次更新（第 114 个 epoch），最佳验证中心 MAE 为 6.832980°。四模型都只有 seed 42，本次结果不证明跨随机种子的普遍优劣。

| 采样／人口 | 双手中心请求 | 有效角度数 | 无效角度数 |
| --- | ---: | ---: | ---: |
| 全部中心 | 2,792 | 59,572 | 1,852 |
| 有图中心 | 2,688 | 59,132 | 4 |
| EMG 完整轨迹（仅 S/M） | 2,792 个窗口，每窗对齐 11,490 点 | 684,715,272 | 21,046,488 |

全部中心中有 **104** 个请求缺少对应裁剪；这些请求仍保留在全部中心统计中。原 JSON 保留弧度绝对误差和、有效分母、逐手、逐 split、手指／腕角、匿名逐用户记录，支持重新计算度数指标。

## 完整产物

| 模型 | 原严格导出 | 锁定中心复算 | 训练／验证曲线 |
| --- | --- | --- | --- |
| EMGFormer-S | [JSON](emgformer-s/metrics.json) · [Markdown](emgformer-s/metrics.md) | [audit.json](emgformer-s/audit.json) | [PNG](emgformer-s/training-curves.png) · [SVG](emgformer-s/training-curves.svg) |
| EMGFormer-M | [JSON](emgformer-m/metrics.json) · [Markdown](emgformer-m/metrics.md) | [audit.json](emgformer-m/audit.json) | [PNG](emgformer-m/training-curves.png) · [SVG](emgformer-m/training-curves.svg) |
| ResNet-18 | [JSON](resnet18/metrics.json) · [Markdown](resnet18/metrics.md) | [audit.json](resnet18/audit.json) | [PNG](resnet18/training-curves.png) · [SVG](resnet18/training-curves.svg) |
| ResNet-18 + EMGFormer-S | [JSON](fusion-rn18-s/metrics.json) · [Markdown](fusion-rn18-s/metrics.md) | [audit.json](fusion-rn18-s/audit.json) | [PNG](fusion-rn18-s/training-curves.png) · [SVG](fusion-rn18-s/training-curves.svg) |

所有报告与审计 JSON/Markdown 保留原字节。`metrics.md` 保留导出当时的 `--no-plots` 和待审计提示；后续审计结论以本页及对应 `audit.json` 为准。曲线复用由逐字节相同 `metrics.json` 绘制并检查过的产物，图中训练损失与验证 MAE (°) 分列展示，不是同一种量。

### EMGFormer-S

![EMGFormer-S 训练损失与验证 MAE](emgformer-s/training-curves.png)

### EMGFormer-M

![EMGFormer-M 训练损失与验证 MAE](emgformer-m/training-curves.png)

### ResNet-18

![ResNet-18 训练损失与验证中心 MAE](resnet18/training-curves.png)

### ResNet-18 + EMGFormer-S

![融合训练损失与验证中心 MAE](fusion-rn18-s/training-curves.png)

## 实际环境与初始化

四个已完成运行的环境原文件摘要及安全白名单见 [publication-manifest.json](publication-manifest.json)：Python 3.11.12、NumPy 1.26.4、Torch 2.2.2+cu118、CUDA runtime 11.8、cuDNN 8700、A800-SXM4-80GB、bfloat16、matmul `high`。`cuda:0` 是运行进程内的逻辑编号，不表示服务器物理卡号。S/M 随机初始化，建模前 seed 为 42；ResNet-18 使用 ImageNet1K_V1 主干，清单保留其权重枚举、官方来源链接、实际 torchvision 0.17.2+cu118 与主干张量摘要及作用域。S/M 原环境文件没有记录 torchvision 版本，不能由视觉运行的版本反推。

融合加载独立完整验证选出的 S 第 635 步与 RN18 第 6,319 步，其 checkpoint 内容摘要、选择来源及初始协议文件摘要保存在清单中。随后按官方初始化逻辑重置 EMG 最终层并记录精确零残差初始化；目标手 8 通道、共享 12,000 点输入的配置与发布的 16 通道融合检查点不同。原八个初始化协议文件与完成时字节相同，不能把这些有意声明的差异隐藏为论文原配置。

四模型额外锁定核验各一次，S 原成功产物没有重放；审计队列于北京时间 **2026-10-25 21:18:54**记录 `complete`。清单保留实际接纳、完成与退出 receipt 的内容摘要，共同人口 SHA256 为 `4cf005e68d2c3db3af7ac67028779c8535c1075218a04995662ca55b8f0ee371`。

## 科学边界

这是官方 legacy 发布兼容实验。41 份记录中有一份不足完整窗口，实际测试覆盖 **40 名用户**；发布标签中的 User、Both 条件分别与训练窗口有 **1 名用户交集**，不能称为严格未见用户测试。整 episode FFT 预过滤仍有跨时间依赖；本实验采用 12,000 点输入，而 PDF 描述为 7,790 点。数据、标签、滤波、输入和配方与 PDF 原表的精确等价性仍未证明。

全部限制、已完成的获批范围及未运行的论文扩展见[复现覆盖矩阵](../../docs/reproduction-matrix.md)，当前进度见[运行状态](../../docs/run-status.md)。本目录只发布脱敏汇总、匿名用户序号和内容摘要，不发布原始参与者标签或映射、机器路径、原始协议文件、数据及权重。

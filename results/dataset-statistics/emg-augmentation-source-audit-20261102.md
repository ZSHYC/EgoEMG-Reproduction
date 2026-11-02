# EMG数据增强：论文、官方配置与实际训练来源

已用真正的官方Hydra组合核对9条命名配方，并把S/M/L三份已完成200轮／25,000更新的训练身份、构造来源及原生点报告逐项绑定。三项都是`ExtractToTensor → RotationAugmentation`，GPU批量增强均为`null`；val/test只有`ExtractToTensor`。这确认已运行的官方主配方，没有把论文描述当作当前程序已启用的配置。

依据用户提供的EgoEMG.pdf：§4.3物理p.6，Appendix C.1物理p.21；完整PDF SHA在[证据JSON](emg-augmentation-source-audit-20261102.json)。

## 逐项核对

| 项目 | 论文描述 | 实际S/M/L主配方及原代码 | 结论 |
| --- | --- | --- | --- |
| 通道dropout | 每通道25% | 主链没有；可选`emgformer_regression_aug_best`为独立Bernoulli p=0.25，min0／无max上限 | 代码有该预设；不是本次主链 |
| 频率掩码 | 3个、至多128bins | 主链没有；可选best为3次rFFT带遮盖，宽度可取0–128，可能重叠 | 代码有该预设；不是本次主链 |
| Gaussian噪声 | SNR25–35dB，p=0.5 | 主链没有；可选best相同范围／概率，Tensor分支用Torch RNG | 可选参数吻合；未宣称已按它训练主模型 |
| 时间jitter | ±40ms | dataset随机向前移动窗口起点：`randint(0,min(stride,leftover))`，上界排除；train stride1200时最大1199点／599.5ms | 窗口采样jitter，不等同于对称±40ms信号平移 |
| 左手镜像／channel remapping | §4.3称左手统一为右手约定 | target_hand路径保持已存8列顺序，按手归一化；所选Rotation对两手均环移−1／0／+1 | 环移不是手性反转；存储前物理列映射仍未证明 |

主配方见[官方S配置](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/egoemg_emgformer_small.yaml)、[M配置](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/egoemg_emgformer_middle.yaml)、[L配置](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/egoemg_emgformer_large.yaml)及[实际选中transform](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/transforms/rotation_augmentation.yaml)。可选预设见[aug_best](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/transforms/emgformer_regression_aug_best.yaml)，[原算子](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/transforms.py#L575-L805)保留其概率／FFT／噪声语义。

此处核对输入EMG增强。网络decoder自身的dropout另属模型结构，不等同于每个输入通道25%置零。可选best清单虽包含RandomTimeMask，但num_masks=0，实际不做时间遮挡；它也不等同于±40ms窗口jitter。

时间偏移来自[原生dataset取窗](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/datasets/egoemg_memmap_dataset.py#L1735-L1740)。当episode没有剩余帧时不偏移；其余偏移受真实剩余帧限制，并非每个窗口都能取到1199。原适配器仍使用同一范围，只在建分区时排除可能跨越未来holdout的anchor，既有训练未由本次核查修改。

## 实际解析的9条官方配置

| 配方 | 最终train链 | GPU批量增强 | 本次证据范围 |
| --- | --- | --- | --- |
| EMGFormer-large | ExtractToTensor → RotationAugmentation | `null` | 完整200轮训练／构造来源已绑定 |
| EMGFormer-middle | ExtractToTensor → RotationAugmentation | `null` | 完整200轮训练／构造来源已绑定 |
| EMGFormer-small | ExtractToTensor → RotationAugmentation | `null` | 完整200轮训练／构造来源已绑定 |
| NeuroPose-default | ExtractToTensor → RotationAugmentation | `null` | 原配置解析；本次未重新准入其完整fit |
| NeuroPose-with-aug | ExtractField → RotationAugmentation → ToFloatTensor → RandomChannelMask → RandomTimeMask → RandomFrequencyMask → RandomGaussianNoise | `null` | 原配置解析；本次未重新准入其完整fit |
| RN18-fusion | ExtractField → RotationAugmentation → ToFloatTensor | `null` | 当前正式fit的不可变配置已绑定；未声称训练完成 |
| VEMG-default | ExtractToTensor → RotationAugmentation | `null` | 原配置解析；本次未重新准入其完整fit |
| VEMG-with-aug | ExtractField → RotationAugmentation → ToFloatTensor → RandomChannelMask → RandomTimeMask → RandomFrequencyMask → RandomGaussianNoise | `null` | 原配置解析；本次未重新准入其完整fit |
| ViT-S-fusion | ExtractField → RotationAugmentation → ToFloatTensor | `null` | 原配置解析；本次未重新准入其完整fit |

RN18与ViT-S融合虽然在defaults中引用`emgformer_regression_aug_extended`，最终experiment又把train列表替换成三项，batch增强也为null。因此，看到一个预设文件或defaults名称不能推断整套增强已进入训练。RN18这里额外核对了正在运行fit的不可变saved_config，没有读取其活动检查点。VEMG/NeuroPose的`_with_aug`配方确实选择best链；这次解析不代表新增了这些完整增强实验。

## 实际执行与适用边界

在原固定运行环境执行真实CPU组合与三条原始环移分支：显式32×8 float32合成信号，分别观察−1／0／+1，结果与通道环移逐元素一致，均不是反序镜像；val链保持值不变。Python／NumPy／Torch RNG已在探针后恢复。没有真实数据重放、模型／优化器构造、新学习前向、优化更新或CUDA，也没有重跑旧模型／数学／工程套件。

实际进程结束后，另一个不导入Torch的进程独立重读22份官方来源与7份实际fit元数据的SHA／代际，核对5份实际适配／运行帮助模块；审计进程与额外读取器的PID/start均已确认关闭。三份完整训练的来源绑定不是新一次完整训练或原始数据科学验收；原有独立报告继续作为训练／测试完成证据。

按用户的官方代码优先要求，当前结果继续对应已绑定的命名配方。可选best增强的训练效果属于独立实验，本次没有建立它与论文精度差异之间的因果解释。原始filtered列在写入前是否已经实现物理左手镜像，不能由运行时取列及环移证明；Figure9／B.1.3的marker八类增强也没有因此完成。

[结构化证据与全部原始参数](emg-augmentation-source-audit-20261102.json) · [实际审计worker原字节](emg-augmentation-source-audit-20261102.py) · [S完整轨迹](../native-emgformer-trajectory/small/README.md) · [全文状态](../../docs/full-paper-status.md)

归档worker包含本次服务器来源路径，需配合相同环境／路径或据实际环境调整；它是实际执行源码记录，不是新的通用训练入口。整个论文目标仍未完成。

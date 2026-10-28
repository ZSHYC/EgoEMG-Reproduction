# 原生 ViT-S + EMGFormer-S 融合构造

本入口执行固定官方f7cbd048的`MidFusionPoseFormer`与`EmgPredictionModule`，使用发布评价对应的训练配方`fusion_vits_s_center_16ch_wl7790`。当前模块覆盖配置、分支初始化及完整状态重载；没有因此完成融合训练或增加一个模型成绩。配置差异见[实际五配方审计](../results/fusion-recipe-audit-20261028.md)。

## 原配置与输入

保留16槽补零、原8通道位置`[10,12,0,1,2,4,5,6]`、窗口7,790、训练stride400、评价stride7,790、所有角色repeat2与实际batch200。`val_batch_size=100`只是配置声明，原datamodule没有使用它。保留原bf16/high、AdamW、cosine150、clip1、有效指尖系数0.02及原回调。它们是配方描述，尚无本入口的GPU容量或完整训练证明。

`compose_fusion`只给未修改的Hydra配置增加来源绑定；任何其他配置或类型变化都会拒绝。`fusion_protocol`输出原模型/优化器/回调/数据语义及相对源码摘要，并隐去声明的私有初始化根路径；它不改配置，也不创建模型、读取数据或授予科学实验准入。

## 使用真实本地分支文件构造

```python
from pathlib import Path
from egoemg_repro.fusion_literal import compose_fusion
from egoemg_repro.fusion_literal_runtime import build_fusion_runtime

upstream = Path("/path/to/pinned/upstream")
config = compose_fusion(upstream)
prepared = build_fusion_runtime(
    upstream, config,
    vision_checkpoint="/path/to/vision-state.pt",
    vision_sha256="<the actual 64-character file SHA256>",
    emg_checkpoint="/path/to/emg-state.pt",
    emg_sha256="<the actual 64-character file SHA256>",
)
module = prepared.module
prepared.check()
```

输入为本任务拥有、来源明确的本地checkpoint，支持原`state_dict`包或直接tensor state映射，名称允许原`model.`前缀。视觉文件须包含完整原ViT-S backbone及512隐藏宽度/22输出的head；只加载到部分视觉张量会拒绝。文件字节数、SHA256、typed state摘要与传输摘要分别记录。文件摘要只能证明接收的字节，不能代替训练、验证选模与数据来源证明。

EMG初始化仍调用官方`_load_pretrained_backbone`：仅加载允许前缀中名称和shape都相同的张量。8→16首卷积权重按原算法跳过；不扩展、不补造，也不声称整个分支严格恢复。`identity.initialization.emg_transfer`列出所有匹配和跳过项及原因，匹配项另记实际加载摘要。原`load_state_dict`的目标dtype转换规则保留；比较转换后的精确值，原输入dtype与摘要另外保留。

原残差末层继续使用标准差1e-5的正态权重与零bias，初始化后没有强制置零。TemporalAttentionPool仍在原优化器的`_unassigned`组，所有可训练参数应恰好出现一次。文件缺失或SHA错误会在构造前拒绝，避免落入原factory的随机/联网初始化兜底；来源文件在构造前后和后续`check()`时核对稳定性。

## 保存并独立重载完整融合状态

```python
import torch

torch.save({"state_dict": module.state_dict(),
            "initialization": prepared.identity}, "/path/to/fusion-state.pt")
packet = torch.load("/path/to/fusion-state.pt", map_location="cpu", weights_only=False)
restored = build_fusion_runtime(
    upstream, config,
    state=packet["state_dict"],
    expected_initialization=packet["initialization"],
)
```

完整重载不再消费两份分支文件。它验证完整初始化记录、来源、运行时版本、所有keys/shapes/dtypes/有限性，然后以原factory建立同结构模型并严格加载每个tensor；构造期间禁用不必要的预训练下载，之后恢复环境原值。完整state可以是训练后的权重；`check()`允许学习参数变化，但不允许修改配置、源码、初始化记录或仍由fresh实例持有的输入文件。

## 当前验收边界

CPU工程检查使用真实DINO-S预训练来源生成的原视觉格式状态（其姿态head仍是工程初始化）及原factory随机EMG-S状态，明确不冒充训练完成的分支。检查覆盖原forward/loss/backward/优化器、完整视觉与部分EMG传输、离线构造、新进程精确预测重载及损坏输入拒绝。输出始终为`scientific_eligible=false`、`production_ready=false`、`full_paper_complete=false`。

完整融合实验仍须：绑定真实验证best分支和来源；原联合EMG/RGB读取、归一化、augmentation与jitter安全独立划分；完整数据核验；原batch200的GPU构造/训练/恢复验收；150轮完整预算与原选模；冻结测试及独立报告。旧版数据与声明的EgoEMG/ShowEE/Incre合并来源差异继续保留。此处的工程通过不能填写论文精度表。

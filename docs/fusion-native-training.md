# ViT-S + EMGFormer-S 原生融合训练

当前在实现[全文设计](superpowers/specs/2026-10-26-full-paper-reproduction-design.md)内的正式融合实验。[实施计划](superpowers/plans/2026-10-28-native-fusion-training.md)分开登记权重交接、完整训练、冻结评价和实际GPU/完整运行；本文的接口存在不表示完整150轮已完成，也不增加已完成模型数。

## 官方执行配方

固定上游`f7cbd048b7092139852270491d3859cd472e0311`，实际合成实验`fusion/fusion_vits_s_center_16ch_wl7790`。目标手8通道按原位置放入16槽，其余补零、不插值；输入7,790点、训练stride400，三个角色都repeat2、实际batch200。保留有效0.02指尖项、原增强与生成标签mask、bf16/clip1、AdamW1e-4/wd0.01和cosine150/eta_min1e-5；150轮、不早停、完整验证按`val_mae/min`选择最终融合best。

分支初始化遵循原配置：视觉取完成训练的**last**，EMG取完成训练后独立验证选出的**best**。融合模型自己的最终测试仍使用融合验证best。原生8到16通道加载器按精确名称和形状匹配：105项EMG张量加载、5项跳过，不扩展或重塑首卷积。最后残差权重保留原std1e-5正态初始化和零bias。

## 训练来源交接

`fusion_initializer_handoff.admit_fusion_initializers`要求完整注册融合输入，重新打开两个实际full producer的原生检查点、轮次记录、完整验证历史与选模。EMG导出还必须由`read_emg_initializer_export(..., source_fit=...)`核对原生所选best和全部110项张量，工程文件、自报训练完成或只有文件hash不能通过。

已完成ViT-S的训练早于后来为ViT-L增加的激活检查点功能，因此两者的fit读取源码不同。交接支持已经发布的旧读取器，其完整SHA256固定为`e44dc947111bec3aa858fb5fc36493650cda24f2093915bcd133a2d090d070da`；另核对其三个公共依赖的旧/当前实际字节一致，才在独立模块名中加载。原报告、当前视觉模块及运行中的EMG初始化源码不修改。不执行任意调用者提供的Python文件。

两个producer的分区从当前实际metadata、frame split重新构造；共同EMG、标签、图像和元数据payload必须同源。输入足迹按所有可能抖动范围合并，EMG/视觉训练与**新融合整个验证区间**和全部测试输入的交集都必须为零。兼容EMG的receipt、导出和重建分区还必须绑定同一个未来融合partition摘要。

```python
from egoemg_repro.fusion_initializer_handoff import admit_fusion_initializers
from egoemg_repro.fusion_literal_runtime import build_fusion_runtime, fusion_optimizer_identity

branches = admit_fusion_initializers(
    cohort, vision_fit=completed_vision_fit,
    vision_source_root=frozen_vision_source,
    emg_fit=completed_compatible_emg_fit,
    emg_export=compatible_emg_best_export,
)
branches.check(require_full=True)
prepared = build_fusion_runtime(upstream, config, **branches.runtime_kwargs)
configured = prepared.module.configure_optimizers()
optimizer_identity = fusion_optimizer_identity(
    prepared, configured['optimizer'], configured['lr_scheduler']['scheduler'],
)
```

`engineering_fusion_initializers`是独立、明确的工程入口，只绑定可信本地文件及当前cohort；它不声称权重经过训练，`check(require_full=True)`必须拒绝。不能通过改一个scope字段升级为正式交接。

## 优化器与人口

`fusion_optimizer_identity`检查原生八个参数组，特别保留`_unassigned`中的时间注意力参数；所有可训练参数必须各出现一次，没有额外冻结、外来张量或重复。它同时核对各组原始学习率、AdamW设置、cosine预算/下限及scheduler与实际optimizer的归属，不替换原优化器。

实际旧版完整输入为395,276训练、3,152验证、8,612测试次手样本消费；测试仅有4,306个唯一手样本。原缺图行为保留，完整评价需分开all/crop-present、重复消费/唯一人口及有效角分母。当前缺失的合并语料、旧标签、整episode滤波、公开归一化和单rank/workers0适配仍需披露，不能宣称PDF原表数值等价。

正式fit/eval接口和最终实测证据在上述计划后续任务中实现、核验。完整兼容EMG150轮尚在运行，当前没有新的ViT-S融合最终MAE。

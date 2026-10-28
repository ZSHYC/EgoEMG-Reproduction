# WiLoR 22角实验：原文与固定代码配方核对

这是只读配置/调度器核对，尚未构造新的WiLoR模型、消费数据或开始训练。完整身份和来源摘要见[JSON](wilor-angle-recipe-audit-20261028.json)；先前真实预训练CPU模型实证见[已有记录](wilor-pretrained-cpu.md)。

所附PDF第22页C.2/Table8经重新提取和页面渲染核对：WiLoR为AdamW、初始LR `1e-5`、cosine、batch18、30epochs，显示631.6M参数和1576维主干。固定官方版本的[22角类](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/wilor_vit.py)实际池化1280维特征，head为1280→512/GELU/Dropout0.1→22，256×256输入保留原高度裁剪为192×256；不把原文1576维或通用head的ReLU覆盖到该代码上。先前真实测量631,631,657参数/52,499冻结参数仍是已有CPU工程实证。

固定仓库包含`module/wilor_vit.yaml`，搜索到的三份WiLoR实验均是归档的EMG融合配置；没有独立视觉WiLoR训练YAML。它们不能代替Table8单帧22角实验。候选配方因此只能明确标为“公开ViT-S视觉配方的派生”：整体替换module映射，另应用原文的LR/batch/epochs；不能合并后留下DINO专有的`backbone_type`、`pretrained`或`freeze_backbone`参数。真实权重路径也必须另行准入，不能接受原类在文件缺失时静默留下随机主干。

本次在不构造模型的情况下实际合成了这个候选配置，并调用[原始优化器/调度器工厂](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/utils.py)，只对一个CPU标量参数运行30次优化器/调度器步。继承的`CosineAnnealingLR`为T_max30、eta_min1e-5，等于初始LR；实际0–30步的31个LR全部为1e-5。它是常量LR，不能写成“已实现递减cosine”，也不能为得到衰减而静默修改官方floor。这个标量探测不是WiLoR优化或性能结果。

其他继承字段仍要在未来实现计划中前瞻裁决和披露：weight_decay0.01、两个原生backbone/head学习率组、BF16、原梯度裁剪1、val_mae/min/top3+last/早停patience30、seed42，以及7790窗口/训练stride780/评价stride7790/repeat2。原base保留的fingertip别名与实验fingertip_distance在实际Lightning路径相加为0.02，不能只抄单个0.01键。原文BF16句子明确指通用视觉主干；对WiLoR沿用BF16是代码谱系派生，不能宣称原文独立指定。单卡与原六卡、legacy与原声明合并数据的差异也仍须记录。

首次只读脚本因Hydra结构化配置禁止直接新增`pretrained_path`而退出1，原日志保留；改用显式新增字段后合成及标量探测实际退出0。没有调整原源码、断言、LR floor或任何运行中的训练。当前只完成了配方风险核对；WiLoR完整输入、实际模型优化、容量、GPU验收、一次新实现审查和完整30轮/全测试仍欠。

# NeuroPose 原生连续训练

本路径执行官方 NeuroPose 的一次全新连续训练。原 `canonical_fp16_eval_copy_v1` 恢复后训练轨迹检查仍保持失败，相关源码/阈值/产物不修改。[执行裁决](../results/neuropose-native-execution-ruling-20261028.json)记录既有恢复与 fresh/fresh 诊断、推论边界及新路径的精确验收条件；这不是新的 GPU 资格或科学成绩。

新模块 `neuropose_native_train` 保留原厂模型、16槽/7790点、22角完整轨迹、生成2维mask、固定scalar归一化与原增强次序。使用原 Lightning loss、AdamW/Cosine、val_loss/min/patience50 的选模/早停，以及 native top-one/last checkpoint。完整模式为101,714训练/1,576验证/4,306测试手样本，B256/16-mixed/GradScaler、max100epochs、单rank/workers0。单rank的全局批量与BatchNorm统计不等于原六rank，旧版数据也未证明与PDF/缺失合并数据等价。

`fit_neuropose(..., scope='full')`必须先提供新的原始GPU资格文件，以及实际完整cohort、numeric与三阶capacity来源。完整入口仍等待评价/资格模块及真实GPU验收完成；调用者布尔值、旧失败记录、CPU工程结果、改summary或调batch/precision均不能打开完整训练。没有`resume`或覆盖已有输出的接口；失败/中断保留为不完整，不能自动重启或续训。

```python
from egoemg_repro.neuropose_native_train import (
    fit_neuropose, read_neuropose_fit, reopen_neuropose_best,
)

# ordinary small memmap corpus; explicitly CPU engineering only
fit_neuropose(data, upstream=upstream, config=neuropose_recipe,
    partitions=frozen_partitions, output=output, scope='cpu_engineering')
record = read_neuropose_fit(output, require_full=False)
selected = reopen_neuropose_best(output, require_full=False)
```

CPU工程为B2/32-true/两轮，最多前4个训练样本及该小队列全部验证；CUDA工程为原B256/16-mixed/两轮、前512训练及完整1,576验证。两者从身份到reader都与full区分，默认reader拒绝工程结果。CPU RNG快照不读取或初始化可见CUDA设备。

每轮训练和验证均检查完整唯一样本覆盖及原native损失；实际optimizer post-step hook计数与尝试次数分列。epoch journal在完整验证后提交，native checkpoint携带身份、实际完整轮次/初始模型指纹/journal及RNG，保留普通模型/optimizer/scheduler/scaler/loop/callback状态。reader重开实际输入、配置和来源，核对journal、checkpoint与真正val_loss最优文件，检查状态有限性、实际AdamW计数、完整RNG和摘要；不以last替代best。

严格重开最佳权重后，原生完整验证必须重现选中轮次的typed预测/输入摘要和验证损失。后续完整测试还需提前冻结的完整验证native/独立预测配对、新进程完整测试复算、逐手/split/匿名用户/关节/指组的trajectory与center统计，以及真实曲线和公开报告。NeuroPose轨迹长度7790、窗口中心3895，不能套用VEMG的裁后6000点。

当前实施计划为[原生连续NeuroPose](superpowers/plans/2026-10-28-neuropose-native-uninterrupted.md)。完整数据GPU资格、一次新实现最终审查、完整训练和最终科学报告仍是必需后续；整个论文目标保持进行。

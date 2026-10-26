# Classic：限定工程运行的比较规则

2026-10-26 固定此规则，用于新源码的连续两轮、单轮 prefix 和新进程恢复检查。它不改变官方模型、算子、AdamW、损失、归一化、train/val 批量或原生 callback。**CPU 检查与源码审查不构成真实 GPU 验收；两模型均须另行执行新源码三进程检查。** 原 6adc 失败记录保持失败。

## 精确恢复与独立运行分别比较

checkpoint 到实际恢复对象的模型、optimizer、scheduler、scaler、原生可恢复 callback 字段和保存 RNG 继续按 typed 摘要精确比较，在第一次恢复 forward 前检查。原生不恢复的 ModelCheckpoint.current_score 只在比较副本中规范，原 callback 和 checkpoint 不改写。

独立 GPU 运行采用已声明的非确定性后端。最终模型浮点参数、浮点 buffer 和 AdamW moments 要求 key/shape/dtype 相同且全部有限，记录实际差异和是否逐位相同，不承诺接近或收敛等价。AdamW step、参数组、scheduler、scaler 和非浮点 buffer/counter 仍要求精确一致；每条运行的完整验证和 first-minimum 选模分别核验。

## 所有工程输入批次精确比较

每个实际消费的 train 和 validation batch 都记录 EMG、target、generated mask、sample IDs 的 typed SHA，连同 role、epoch、batch index 和实际 batch size。连续 epoch0 对 prefix、连续 epoch1 对 resumed 逐条比较，包含最后一个 partial batch。只记录有界摘要，不保存输入 tensor，也不在完整生产训练中增加这些逐批 hash。

## 首个样本的独立预测检查

每个工程 epoch 的第一个训练 batch，在真实完整 batch 更新前后，使用独立原模型副本的 eval/no_grad 前向检查该 batch 的第一个样本。诊断前后恢复全部 RNG，训练模型本身不会额外执行这个前向。此检查不代表完整训练 batch 的 forward，尤其不代表 NeuroPose 训练 BatchNorm 的输出。

预测单位为 radians，保留实际 float16/float32 dtype；两次输出必须 dtype/shape 相同、全部有限。每个元素以 float64 算术检查固定对称规则：

```text
abs(x - y) <= 2**-10 + 2**-10 * max(abs(x), abs(y))
```

这是事先选定的工程门槛，**不是由 FP16 epsilon 推导出的非线性前向、反向或 AdamW 误差保证**。不使用 mask、均值、rounding 或转成 FP16 来获得通过。记录最大绝对差、最大门槛比例及原始逐位相等性。初始更新前模型和 optimizer 的身份仍精确比较；默认 exact 比较也继续可用。调用方不能传入自定义容差。

## 原生 loader 生命周期

真实 stock-Lightning CPU 追踪发现，fresh Trainer 在 epoch 边界恢复时创建两次训练迭代器，使专用 train-loader generator 比连续运行多消耗一次 int64 base-seed draw。workers0 下，该夹具的实际顺序、随机增强、sampler/validation/global RNG 及小模型/optimizer 均相同；这个 CPU 追踪本身不证明 GPU 等价。

新 GPU 检查仅接纳这个精确的正向一次 draw 关系，不搜索任意 offset，也不回退保存的 generator。全局 Python/NumPy/Torch/CUDA、train sampler 和 validation loader RNG 必须精确一致。例外绑定 workers0、单 rank、epoch 边界、Torch2.2.2+cu118、Lightning2.5.0.post0、三个 Lightning loop 文件及 Torch DataLoader/sampler 的固定 SHA，并绑定完整实际 runtime 摘要。其他偏移、流或源码变化均拒绝。

v2 acceptance 从原始三条运行的 identity/result/checkpoint/witness/输入摘要、完整队列、数值与容量证据重新核验；v1 或调用方通过标记不能代替。通过也只支持限定工程执行/恢复范围，不证明整条训练轨迹、原六 rank 等价或论文性能。Task6 的完整 validation 报告校准及正式测试前 profile 绑定仍须单独完成，不能借用上述单样本门槛。

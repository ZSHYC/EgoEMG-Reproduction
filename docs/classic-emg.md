# vemg2pose 与 NeuroPose：当前实现范围

当前已接入固定官方模型、原始数据变换和真实 Lightning CPU 训练及 epoch 边界恢复。**正式 GPU 训练、完整验证选模后的冻结测试及科研结果导出尚未完成。** 这些工程检查不能填入论文性能表。

| 项目 | vemg2pose | NeuroPose |
| --- | --- | --- |
| 真实参数数 | 5,963,948 | 6,354,969 |
| 实际输入通道 | 目标手 8 通道 | 目标手放入 16 槽 |
| 输入长度 | 7,790 | 7,790 |
| 原模型左上下文 | 1,790 | 0 |
| 原 tuple 的预测/目标长度 | 6,000 | 7,790 |
| 实际角度数 | 22 | 22 |

模型保留官方返回的 `(prediction, target, generated_valid_mask)`，其中 mask 为二维，不再裁剪、插值或替换成旧复现层的逐角 mask。固定官方 scalar 归一化、NeuroPose 先填原始零槽再归一化、原 rotation/with_aug 顺序均经过完整官方 reader/Compose 的实际对照。VEMG 的配置声明 16 槽但卷积要求 8 通道，以及原 reader 漏选 legacy filtered 字段，两项兼容裁决都记录在身份中。

新分区是独立的 `window_length=7790 / train_stride=780 / eval_stride=7790`、验证比例 0.12；不是已有四模型的 12,000 点队列。原始标签非有限值会明确拒绝，不悄悄补零。模型/数据接口和两份基础配方见 [classic_emg.py](../egoemg_repro/classic_emg.py)、[classic_emg_data.py](../egoemg_repro/classic_emg_data.py)、[VEMG 配置](../configs/vemg2pose.json)及 [NeuroPose 配置](../configs/neuropose.json)。

## 已验证的训练行为

[classic_emg_train.py](../egoemg_repro/classic_emg_train.py)直接构造原 `EmgPredictionModule`，使用原自动优化、AdamW、cosine、clip0、LRMonitor、ModelCheckpoint 和 EarlyStopping；保留原 raw loss 双别名及有效指尖系数 0.02。选模量是原 Lightning 按 batch-size 加权的 `val_loss/min`，不是 degree MAE 或全局有效角数量加权。原 cosine T_max100、EarlyStopping patience50 在缩短的工程运行中仍保持原值。CSVLogger 是明确记录的日志适配。

两项真实模型各以小 memmap 完成两轮连续运行，并与暂停一轮后只携带 last、搬迁到新目录的离线恢复对照。每条完整路径有四次实际优化更新和两次完整**夹具**验证；模型、AdamW、scheduler、loop、callback、增强输入以及 Python/NumPy/Torch/loader/sampler RNG 均匹配。新进程实际读取自己的 JSON 身份和 checkpoint 后可严格恢复。检查点保存携带不递归的 selected-best，支持离线重建原 best。

该子阶段的 35 项定向测试实际通过；无 Torch 层为 10 passed / 25 个运行时依赖 skip。独立审查另外执行五项关键恢复/破损状态测试，并实际注入最终状态 rename 后的 fsync 失败及低容量，确认不会遗留成功标志或破坏 checkpoint、源数据和邻目录。曾发现的 JSON tuple/list 身份及失败终态问题已修复，原失败证据保留。

最终 CPU 源码 SHA256 为 `52b1ce655f3c4e107e3d809e403cd507fbcfd655f7ac9ea7bc0b9a592b18b0d8`，测试为 `6ecc55104b24fdbcc8379e50897152f1b94a695da683d6ed80ba52a14c176875`。这些摘要绑定实际被测文件；后续文档提交不能替换真实执行身份。

## 仍需完成

- 实际 CUDA `16-mixed`、GradScaler overflow skip、scaler 恢复、真实 batch600/256 显存准入；CPU32 不能证明这些行为。
- 完整真实队列的非有限值准入、新分区与单卡/原六 rank 批量及 NeuroPose BatchNorm 差异记录。
- 原 max100、patience50 的完整训练/每轮完整验证，以及按有限 val_loss 选定 best 后执行冻结 test。100 是上限，不能为了凑轮数删除早停。
- 严格正式导出、逐关节/手/split/用户统计及独立复算。旧 legacy 的 0.01 目标不能改名为官方合成的 0.02。

当前仅开放 `--engineering --device cpu`；正式入口会明确拒绝，不自动降级为 CPU。已验证的恢复范围为 workers0 的 epoch 边界，没有多 worker 或中途 batch 的位级续序承诺。容量保护、源码/配置/分区/数据/归一化/检查点身份检查继续生效。[实施计划](superpowers/plans/2026-10-26-classic-emg.md)中的整个 Task3 和 Task4 保持未完成。

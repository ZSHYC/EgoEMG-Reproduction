# 固定官方源码的损失合成与选模审计

2026-10-26 新发现并独立复核的协议差异：当前官方配置经 Hydra 合成后，EgoEMG 多个 experiment 同时保留 `landmark/fingertip: 0.01` 和 `fingertip_distance: 0.01`。官方 Lightning 将后一名称映射到前者，逐 key 累加，实际指尖项总系数为 **0.02**。

既有复现层及其 legacy 配方只使用 **0.01**。此前这一合成差异漏核；已完成的四项 legacy 实验、当前 L/RN50/RN152 训练和现有 DINOv2 配方因此都不能称为固定官方代码的损失等价复现。原结果、配方、源码及现有训练身份保留；今后的官方字面协议须保留真实 resolved loss 与回调行为，并与 legacy 结果分开登记。

## 实际核验方法

官方版本固定 `f7cbd048b7092139852270491d3859cd472e0311`，已核验源码/config干净。审计实际使用 Hydra compose，而非只读单份 YAML；取出未改动 `EmgPredictionModule._step` 的 alias 映射、初始化和逐项累加 AST 执行。给定 float32 MAE=2、指尖距离=3，EgoEMG 实测 loss=2.059999942779541，EMG2Pose 实测=2.0299999713897705，分别对应总系数0.02与0.01。此算术检查不涉及模型训练。

来源：[base loss_weights](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/base.yaml)、[S experiment](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/egoemg_emgformer_small.yaml)、[实际Lightning归约](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/lightning.py#L645)。[机器可读审计](official-loss-audit.json)保留11份实际合成结果、callbacks、源文件与归约AST摘要，以及既定legacy配方摘要。

## 固定版本的实际配置

| Experiment | 指尖有效系数 | 选模指标 | 最大epochs | EarlyStopping patience |
| --- | ---: | --- | ---: | ---: |
| EgoEMG-S | 0.02 | `val_mae` | 200 | 200 |
| EgoEMG-M | 0.02 | `val_mae` | 200 | 200 |
| EgoEMG-L | 0.02 | `val_mae` | 200 | 200 |
| RN18 | 0.02 | `val_mae` | 150 | 30 |
| RN18+S | 0.02 | `val_mae` | 150 | 无 |
| DINOv2-S | 0.02 | `val_mae` | 150 | 30 |
| vemg2pose | 0.02 | `val_loss` | 100 | 50 |
| NeuroPose | 0.02 | `val_loss` | 100 | 50 |
| EMG2Pose-small | 0.01 | `val_mae` | 250 | 无 |
| EMG2Pose-middle | 0.01 | `val_mae` | 250 | 无 |
| EMG2Pose-large | 0.01 | `val_mae` | 250 | 无 |

RN50/RN152 与 ViT-B/L 缺独立作者 experiment，本项目旧配方由 RN18/ViT-S 派生；此处只报告实际存在的官方 experiment。最大epochs是上限；回调触发早停时须记录真实结束轮数，不可为了填满预算关闭官方早停。classic 两项使用 `val_loss` 选模，不能套用旧训练器的仅MAE选择。EMG2Pose 三项只有base的0.01，没有该alias叠加；新增EMG2Pose入口直接复用官方Lightning。

该结果证明当前固定源码的合成与执行行为，无法证明作者未公开的原训练历史、发布checkpoint或论文数值曾使用哪一项权重。这里也不据此推断其他预处理、采样、有效分母或精度协议等价。

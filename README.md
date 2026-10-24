# EgoEMG Reproduction

基于固定官方实现的独立复现项目，研究 **EgoEMG: A Multimodal Egocentric Dataset with Bilateral EMG and Vision for Hand Pose Estimation**。本仓库不是官方仓库。

**完整 EMG/标签已下载校验，最终版本的真实数据 GPU 检查已通过；EMGFormer-S 完整训练已启动，尚未完成论文定量复现。**

截至 2026-10-24：41 人 legacy release 的 21 个 EMG/标签文件共 18,392,993,230 bytes，已全部下载校验；3 个视觉对齐字段已核验并固定。训练、验证、测试分别为 63,172、1,052、2,792 个单手窗口样本。约 45.04 GB 的官方视觉裁剪仍在下载。

72 项完整运行时测试全部通过。EMGFormer-S 已用真实数据、batch 500 完成两步训练、严格重载、两次一致验证及使用保存配置迁移目录恢复。ResNet-18 batch 900、融合 batch 200 已用真实裁剪重复样本完成有限值 GPU 步检查，峰值显存分别约 13.968、8.633 GiB。这些属于运行与显存验收，不代表完整 cohort 精度或训练收敛。

- [运行状态与证据边界](docs/run-status.md)及[脱敏集成证据](results/integration-evidence.json)
- [环境、数据、训练和评估命令](docs/training.md)
- [选择性下载与校验](docs/download.md)
- [视觉裁剪、对齐与模型约定](docs/vision.md)
- [已批准设计](docs/superpowers/specs/2026-10-24-egoemg-reproduction-design.md)与[可行性审计](docs/feasibility-audit.md)

## 固定版本与实验范围

| 项目 | 使用版本 / 约定 |
| --- | --- |
| 论文参考 | [arXiv:2605.05712v1](https://arxiv.org/abs/2605.05712v1)，41 人版本 |
| 官方源码 | [zhenqis123/EgoEMG](https://github.com/zhenqis123/EgoEMG)，固定 [`f7cbd048b7092139852270491d3859cd472e0311`](https://github.com/zhenqis123/EgoEMG/tree/f7cbd048b7092139852270491d3859cd472e0311) |
| 初始数据 | 官方公开 legacy `egoemg_v2_memmap`，41 episodes、66,161,725 rows |
| 独立环境 | Python 3.11.12、PyTorch 2.2.2+cu118、torchvision 0.17.2+cu118、NumPy 1.26.4 |
| 初始基线 | EMGFormer-S；ResNet-18；ResNet-18 + EMGFormer-S 残差融合 |
| 主 EMG 配置 | 发布兼容的 8 通道、12,000 点窗口；与论文 7,790 点设置分别记录 |

EMG 单手模型分别使用目标手的 8 个通道；双手共享划分，不把单手输入称为双侧融合。清洁重训练的融合配置使用官方 8 通道 S 分支和 12,000 点窗口，以便严格载入独立训练的 EMG 权重。它不同于发布的 16 通道融合检查点配置；融合残差末层的精确零初始化也是显式记录的复现调整。

## 快速开始

以下命令在仓库根目录、有 CUDA 和 Conda 的运行主机执行。环境和数据保留在本项目目录内；数据完成下载后再执行训练。

```bash
bash scripts/bootstrap_upstream.sh upstream
bash scripts/setup_environment.sh "$PWD/.venv"
PY="$PWD/.venv/bin/python"
"$PY" -m pip install -e '.[test,vision]' --no-deps

"$PY" -m egoemg_repro.download --root data/legacy --metadata-only
"$PY" -m egoemg_repro.download --root data/legacy --workers 4
"$PY" -m egoemg_repro.partitions --data data/legacy --output outputs/partitions.json
```

先执行有界真实数据检查，使用新的输出目录：

```bash
"$PY" -m egoemg_repro.train \
  --data data/legacy --upstream upstream \
  --config configs/emgformer_s.json --output runs/emg_s_check \
  --device cuda:0 --batch-size 2 --num-workers 0 --max-steps 2
```

`--max-steps` 只做有界集成检查，限制验证批数并跳过最终测试。完整训练、恢复、独立评估及视觉数据命令见[训练指南](docs/training.md)。默认 batch 为 EMG 500、ResNet-18 900、融合 200，均保留对应发布配方，并通过 GPU 内存检查。

## 评估约定与已知限制

- 依据 release 的 split 名称解析 Gesture / User / Both；独立验证来自原训练区间的连续时间块，不再把官方测试别名当成验证集。
- 在构造重叠窗口前划分时间，并剔除跨角色的完整输入足迹。双手使用相同时间区间，初始关闭时间抖动。
- 归一化仅统计最终保留训练窗口覆盖的唯一行，按手和通道计算；检查点仅依据独立验证选择。
- EMG 同时报出完整轨迹与中心帧指标；视觉和融合报中心帧。角度 MAE 转为度一次，以有效角度计数重算加权均值，并保留手、split 和用户分组。
- 原始生成标签与论文标签流程尚未证明等价；当前结论只能按 legacy release 复现解释。
- 默认 EMG 是官方整 episode FFT 预过滤信号。时间窗口互不重叠仍不能消除上游滤波依赖，不能据此宣称严格预处理隔离。
- 视觉使用真实视频帧映射。单个缺失 crop 保留为黑图并计数；同时报告全部有标签中心与 crop 有效子集。缺失整个 episode 数据库会报错。

EMGFormer-S 完整训练正在运行，后续依次进行 ResNet-18 与融合；每阶段通过真实数据短训练及重载检查后再进入完整训练。完整独立验证选模与最终测试尚待完成。仓库不将合成测试、少量预览或发布检查点载入当作论文表格复现结果。

## 归属与资产

论文及原实现作者：Ziheng Xi、Jiayi Yu、Yitao Wang、Yanbo Duan、Jianjiang Feng、Jie Zhou。

本仓库不分发论文 PDF、参与者元数据、信号、视频、模型权重或 MANO 资产。外部资产保留原有使用条款。上游基线声明 MIT，继承的 emg2pose 部分使用 CC-BY-NC-SA-4.0；请参阅[固定版本的许可证](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/LICENSE)。本仓库原创部分适用 [LICENSE](LICENSE)。

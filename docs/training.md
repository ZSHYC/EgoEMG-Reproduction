# 环境、数据与训练

这些命令描述仓库当前接口；执行命令不等于已经获得结果。EMG 与标签已下载校验，真实 EMG 短训练、重载和迁移恢复已通过。视觉 crop 仍在下载；视觉/融合已使用真实裁剪完成有限值 GPU 步检查，完整 cohort 训练尚待执行；最新状态见 [run-status.md](run-status.md)。

## 1. 固定源码和独立环境

在仓库根目录执行。环境脚本要求已有 Conda 和可用 CUDA，并保留至少 15 GiB 磁盘空间；可通过 `EGOEMG_CONDA_BIN` 指定 Conda 可执行文件。

```bash
bash scripts/bootstrap_upstream.sh upstream
bash scripts/setup_environment.sh "$PWD/.venv"
PY="$PWD/.venv/bin/python"
"$PY" -m pip install -e '.[test,vision]' --no-deps
EGOEMG_UPSTREAM="$PWD/upstream" "$PY" -m pytest -q
```

固定版本为 Python 3.11.12、PyTorch 2.2.2+cu118、torchvision 0.17.2+cu118、NumPy 1.26.4。完整依赖见 [environment.yml](../environment.yml)。引导脚本固定上游 `f7cbd048b7092139852270491d3859cd472e0311`，遇到未提交的上游改动会停止；训练入口也检查源码身份。不要让其他项目的环境承担本项目依赖。

## 2. 获取与检查 legacy 数据

```bash
"$PY" -m egoemg_repro.download --root data/legacy --metadata-only
"$PY" -m egoemg_repro.download --root data/legacy --workers 4
```

三份小元数据共 33,680 bytes；完整初始 EMG/标签选择清单为 18,392,993,230 bytes。下载器先检查全部剩余文件所需空间，再保留至少 15 GiB 余量，使用正常公开 Drive 确认流程、断点续传、精确字节数和 SHA256 receipt。仅校验成功的文件会从 `.part` 原子改名。元数据 SHA256 固定到已检查版本，其他文件的 SHA256 是实际接收字节的本地记录，不代表发布方签名校验。

同一目标目录只运行一个下载进程；`--workers 4` 是该进程内部的独立文件并行。停止旧进程后可重跑同一命令恢复；成功文件会重新验证。详见 [download.md](download.md)。

元数据和 `frame_split_id.dat` 可用后，先生成并检查独立划分：

```bash
"$PY" -m egoemg_repro.partitions \
  --data data/legacy --output outputs/partitions.json \
  --window-length 12000 --train-stride 1200 --eval-stride 12000 \
  --val-fraction 0.12
```

这会记录覆盖、短区间、丢弃窗口和哈希，不改写源数据。当前真实 split 检查得到 63,172 / 1,052 / 2,792 个训练 / 验证 / 测试单手样本。验证比例是对足够长连续训练区间的近似留出，不保证最终样本数恰好为 12%。训练驱动会依照所选配置再次确定性生成划分并保存在运行目录；外部 `outputs/partitions.json` 用于先行检查，不是训练 CLI 的输入参数。

`val` 在本仓库明确指训练区间内的独立时间验证。最终测试依据 release 的 User / Gesture / Both 标签和固定中心识别，不沿用可能把测试映射为验证的上游别名。所有训练窗口输入足迹与验证、测试足迹分离；双手共享区间；新训练的均值/标准差仅使用保留训练窗口覆盖的唯一行。

## 3. EMGFormer-S

先完成全部必需信号与标签下载，然后执行有界检查。如后续需要融合，先完成第 4 节的视觉对齐字段下载并固定 `data/legacy` 清单，再训练两个分支；在分支训练之间追加文件或 receipt 会改变数据身份。crop 可以在独立目录继续下载。

```bash
"$PY" -m egoemg_repro.train \
  --data data/legacy --upstream upstream \
  --config configs/emgformer_s.json --output runs/emg_s_check \
  --device cuda:0 --batch-size 2 --num-workers 0 --max-steps 2
```

`--max-steps` 限制优化步数，验证至多 4 个 batch，并且跳过最终测试。这种结果只用于验证真实数据、有限值损失、优化和保存流程，不能作为完整 MAE 结果或融合分支初始化。不要将仅执行过有界验证的权重当成完整验证选出的最佳模型。

完成集成核验后，使用新的目录执行完整配置：

```bash
"$PY" -m egoemg_repro.train \
  --data data/legacy --upstream upstream \
  --config configs/emgformer_s.json --output runs/emg_s_full \
  --device cuda:0
```

当前配置为 8 通道、12,000 点窗口、200 epochs，默认 batch 500，保留官方该配方批量。已观察的合成 batch 500 峰值显存约 20.865 GiB，不替代真实数据吞吐与稳定性测量。若通过 `--batch-size` 调整实际 batch，驱动会保存覆盖后的配置；这会改变优化行为，比较结果时必须披露。

训练使用固定官方模块和发布损失，先设置随机种子，再初始化模型。验证选模后才执行最终官方测试；同时输出 EMG 轨迹指标与中心帧指标。与视觉/融合比较时使用同一中心帧定义，不能把轨迹 MAE 当成中心帧 MAE。

## 4. 视觉裁剪和 ResNet-18

视觉训练需要额外的视频帧映射和完整所需 episode 的 crop。下面按顺序执行，避免独立下载进程同时修改 `data/legacy/acquisition_ledger.json`：

```bash
# 先等 EMG 下载进程完成，再将对齐字段写入 legacy 目录。
"$PY" -m egoemg_repro.crops download \
  --root data/crops --data data/legacy --fields-only --workers 2

# 下载公开清单的全部 41 个 episode crop LMDB，约 45.04 GB。
"$PY" -m egoemg_repro.crops download --root data/crops --workers 2
```

第一次少量整合检查可在 crop 下载命令后加 `--episodes episode_000000`；这不是完整 cohort，不能用它替代完整数据训练。需要并行获取对齐字段时使用独立 `--index-root`，完成后再校验合并到 legacy 目录，具体流程见 [vision.md](vision.md)。

视觉样本使用 `image_webcam_frame_index[center]`，兼容新命名 `image_head_frame_index`；不能拿 EMG 的 `frame_index` 当视频帧编号。官方左手裁剪已经翻转，读取时不再额外翻转。图像按 ImageNet 统计归一化，中心标签保留 22 角度的逐角有效掩码。

已实现的 RN18 训练接口如下；真实数据有界运行与完整训练尚待核验：

```bash
"$PY" -m egoemg_repro.train \
  --data data/legacy --upstream upstream --crops data/crops \
  --config configs/resnet18.json --output runs/rn18_check \
  --device cuda:0 --batch-size 2 --num-workers 0 --max-steps 2

"$PY" -m egoemg_repro.train \
  --data data/legacy --upstream upstream --crops data/crops \
  --config configs/resnet18.json --output runs/rn18_full \
  --device cuda:0
```

RN18 默认 ImageNet 预训练 backbone，首次运行可能需要获取对应权重；其来源由实际初始化配置记录。当前完整配置 150 epochs、batch 900；真实裁剪重复样本的 GPU 内存检查约 13.968 GiB，保留官方该配方批量。单个缺失 crop key 会产生归一化黑图并保留样本；缺少整个必需 episode 数据库会停止。选模基于全部有标签中心，另报 crop 有效子集与缺失计数，避免模型之间悄悄改变分母。

## 5. ResNet-18 + EMGFormer-S 融合

融合需要同一数据、划分和归一化协议下，经过**完整独立验证**选出的 EMG 与 RN18 `best.pt`。有界检查产物不能作为这两个初始化分支。当前融合接口如下，真实数据运行证据尚待完成：

```bash
"$PY" -m egoemg_repro.train \
  --data data/legacy --upstream upstream --crops data/crops \
  --config configs/fusion_rn18_s.json --output runs/fusion_check \
  --vision-checkpoint runs/rn18_full/best.pt \
  --emg-checkpoint runs/emg_s_full/best.pt \
  --device cuda:0 --batch-size 2 --num-workers 0 --max-steps 2

"$PY" -m egoemg_repro.train \
  --data data/legacy --upstream upstream --crops data/crops \
  --config configs/fusion_rn18_s.json --output runs/fusion_full \
  --vision-checkpoint runs/rn18_full/best.pt \
  --emg-checkpoint runs/emg_s_full/best.pt \
  --device cuda:0
```

该配置采用官方 8 通道 S 分支、12,000 点 EMG、Adam、学习率 `6e-4`、weight decay `1e-4`、150 epochs；默认 batch 200，保留官方对应批量；真实裁剪重复样本的 GPU 内存检查约 8.633 GiB。分支权重严格匹配，未将 8 通道权重扩展为发布的 16 通道结构。残差末层在载入分支后显式置零，初始融合预测与视觉分支一致；这一调整及分支哈希写入 provenance。它是明确声明的 legacy 独立验证重训练方案，不能冒称原论文或发布融合检查点的逐项复现。

## 6. 恢复、评估与记录

恢复时使用该运行的配置和实际 batch。下面以 EMG 为例：

```bash
"$PY" -m egoemg_repro.train \
  --data data/legacy --upstream upstream \
  --config runs/emg_s_full/config.json --output runs/emg_s_full \
  --resume runs/emg_s_full/last.pt --device cuda:0

"$PY" -m egoemg_repro.train \
  --data data/legacy --upstream upstream \
  --config runs/emg_s_full/config.json --output runs/emg_s_full \
  --evaluate runs/emg_s_full/best.pt --device cuda:0
```

视觉/融合恢复与评估仍提供 `--crops`。检查点包含已初始化模型与初始化身份，恢复无需重新依赖原始分支文件。恢复会检查源码、配置、数据、划分和归一化身份；使用未改动的运行代码。新的非恢复运行要求新的空输出目录。若需要迁移运行目录，使用保存的完整状态并保留所有协议文件。

每个运行保留解析后的 `config.json`、`partitions.json`、`normalization.json`、`provenance.json`、`environment.json`、官方模块配置、`progress.jsonl` 和检查点。`last.pt` 是继续训练状态；`best.pt` 是验证选出的状态，检查点 schema 还保存自包含最佳状态。完整结束后 `summary.json` 记录最终测试；独立评估写 `evaluation.json`。

指标保存弧度误差和与有效角度计数，并在序列化时只转换一次为度。报告包括手、split、用户、手指/手腕分项，池化均值从误差总和与有效计数重算，用户均值及标准差单独报告。缺失标签、图像和样本的数量必须与结果一并解释。

## 解释结果前的边界

默认使用的公开生成标签未证明与论文标签完全一致；官方整 episode FFT 预过滤也带来跨时间分区依赖。当前划分只保证窗口行足迹隔离，不能称作严格原始预处理隔离。发布检查点验证和新训练独立验证属于不同轨道。只有真实完整训练、验证选模、重载核验和最终测试均完成后，才可发布相应 legacy 基线的定量结果。

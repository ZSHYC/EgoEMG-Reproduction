# MANO→UmeTrack：原生 IK 函数与真实资产的 CPU 工程验证

本项推进全论文清单 D12，验证固定官方代码的实际 L-BFGS 路径。完整真实帧标签重生成仍未完成：本次输入是两个固定合成姿态，不是旧发布标签的再测，也不增加科学模型完成设置数。

## 实际完成

[完整证据](native-mano-umetrack-ik-cpu-20261029.json)记录直接调用未修改的 `f7cbd048` 官方 `optimize_batch`。使用现有合法 MANO-right 模型、原生 manotorch/UmeTrack FK，CPU/FP32、两个48维姿态：一个全零，另一个手指部分由 `linspace(-0.12,0.18,45)` 给出，两者全局轴角均为零。没有读取真实帧、写入任何标签数组或训练模型。

| 核验项 | 实际结果 |
|---|---:|
| 原始外层预算 / 每次 L-BFGS 最大内层迭代 | 200 / 50 |
| 按原停止规则实际外层调用 | 61 |
| 实际 closure 计算 | 1,399 |
| 初始 landmark 平方距离目标 | 3473.369140625 mm² |
| 最终目标 | 21.4661865234375 mm² |
| 两个姿态的平均 landmark 距离 | 3.900157928 / 4.200440407 mm |
| 两个姿态的 RMS landmark 距离 | 4.526518345 / 4.737405300 mm |
| 返回角度 | `[2,20]`，全有限且在原始限位内 |
| IK 优化进程实际用时 / 峰 RSS | 5.496 秒 / 439,767,040 bytes |
| 另一进程几何重算用时 / 新 IK 更新 | 0.834 秒 / 0 |

这里的 mm/mm² 沿用原函数的几何单位：manotorch 输出米，原 `mano_fk_batch` 乘1000，UmeTrack模型按原资产使用。上述两个合成例子的误差不是论文 annotator 的3.2mm统计，也不是实际数据标签质量结果。

对原 `torch.optim.LBFGS.step` 仅加了当前进程观察包装，逐次委托原方法，原返回值和更新保持。记录原参数和实际 closure 次数；最终返回角度与最后原始优化参数经 sigmoid 限位后的值精确一致。另一个已退出的进程只读最终参数，直接调用原 `skin_landmarks`、原 MANO 层并重新构造矩阵指数对齐，目标值及两姿态距离与生产者精确一致，没有重跑优化。两个进程实际 exit0，所属 PID/start 均确认退出，200份原始源码、模型文件和九项输入属性由退出后读者再次核对未变。没有CUDA初始化，模型训练更新为0；61次 IK 更新与模型训练更新分开记录。

## 必须遵循的源码差异

原实现：[batch_ik_mesh.py](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/ik/batch_ik_mesh.py)。本次保持这些实际行为：

- `side="right" / use_pca=False / flat_hand_mean=True`；`optimize_batch` **内部固定 β=0**。这不是缺资产后的代填，也不是实际参与者 β 的几何验证。后续遮挡网格仍需其独立、真实 β/姿态/相机链。
- MANO→UmeTrack 使用 X 轴翻转、固定尺度1.0843137502670288与固定平移，再优化全局轴角和平移。20个手指角经 sigmoid 限位；用于 FK 的两个腕角置零，不能用本算法宣称重新生成论文 FE/RU 腕角。
- 目标为20个固定对应 landmark 的平方欧氏距离均值；本次保留默认 landmark 分支、权重1，未改为 Chamfer 或只选更易拟合的对应点。
- 原 L-BFGS 为 LR0.1、strong_wolfe、history30、tolerance_grad1e-9、tolerance_change1e-11，以及原有 stale/patience/restart 规则；实际61次停止，不要求人为跑满200次。
- 历史完整 CLI 在此版本引用不存在的 `emg2pose/UmeTrack/dataset/generic_hand_model.json`；原 kinematics loader 实际使用存在的 `egoemg/UmeTrack/dataset/generic_hand_model.json`。本次 CPU 调用直接提供后者的原限位，未编辑上游、伪造目录或运行会写原标签的 CLI。这证明函数路径，不证明完整 GPU CLI 已可直接使用。

## 依赖与复现

本次固定外部 [manotorch](https://github.com/lixiny/manotorch/tree/a2a70c591f91551078b7bb2af9b5d9f275b626e0) 为 `a2a70c591f91551078b7bb2af9b5d9f275b626e0`，与其25项初始源码/说明清单一致，Git内容干净。EgoEMG没有固定作者当年的外部 checkout，因此该版本是明确记录的复现依赖，不宣称恢复作者环境。未执行 `setup.py` 或安装其可选可视化依赖，两个仓库及既有训练运行时均未修改。

Chumpy0.70的20份原始源码与已核验PyPI包一致，归档 SHA 为 `a0275c2018784ca1302875567dc81761f5fd469fab9f3ac0f3e7c39e9180350a`。仅在一次性进程恢复原 Chumpy 需要的已移除 NumPy 名称和 `inspect.getargspec`，与先前真实 MANO 资产验收一致；没有修改 Chumpy/manotorch/原 IK 的源码或数学计算。使用现有 MANO-right 的完整 SHA `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`，资产保持私有，不再分发。实际环境为 Python3.12.14 / NumPy1.26.4 / Torch2.2.2+cpu。

[公开调用程序](../scripts/paper/probe_native_mano_ik.py)在已有依赖环境中运行：

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python scripts/paper/probe_native_mano_ik.py \
    --upstream /path/to/pinned-egoemg \
    --manotorch /path/to/pinned-manotorch \
    --chumpy /path/to/original-chumpy-source \
    --mano-assets /path/to/private-mano-root \
    --output /path/to/new-private-output
```

MANO根目录需有 `models/MANO_RIGHT.pkl`。脚本拒绝已有输出、错误模型SHA、移动后的上游/依赖版本与输入变化；输出实际原始参数和几何复核记录。正式数据重生成仍欠左右完整 MANO pose、字段/有效性与左右 convention 的真实准入、隔离输出与全帧运行/独立核验；不能将本次两个合成姿态推广为 D12 完成，更不能代替缺失的 markers2mano 标注模型、腕角生成或遮挡分析。

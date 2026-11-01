# 原生密采样遮挡分析：索引与输入覆盖审计

本目录记录Figure13所需密采样的真实准备证据。原始标准网格几何已另在[`../native-point-geometry/`](../native-point-geometry/README.md)完成；本目录尚无密网格GT几何、模型预测误差或相关系数结果。

## 实际完整索引

固定官方代码 `f7cbd048b7092139852270491d3859cd472e0311`，实际执行原生数据集索引方法。每条路线各由两个全新进程重建全部索引，typed数组/身份完全一致；四阶段退出0，10个原拥有进程全部关闭，退出后来源与输出字节SHA一致。三个完整索引输入与此前实际完整数据准入一致；未读取EMG/目标/几何值、未构造模型、未更新优化器或初始化CUDA。

| 官方融合配置路线 | WL | 密步长 | User每手 | Gesture每手 | Both每手 | 唯一手请求合计 | 跨中心分区的手窗口 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vits_s | 7790 | 1947 | 3309 | 4627 | 648 | 17168 | 3518 |
| rn18_s | 12000 | 3000 | 2136 | 2997 | 424 | 11114 | 3504 |

ViT-S路线实际得到17,168个唯一手请求，与PDF Figure13标注数量一致；这只证明样本数的一致性，不证明预测、遮挡分数、物理投影或相关性一致。原脚本使用整数除法7790//4=1947，不是1948；窗口从episode起点生成、按中心帧split筛选，遍历user→gesture→both、各自left→right，jitterfalse/repeat1/max-samples0/eval_center_stride=None。跨分区窗口按官方保留。

## 实际来源输入交集

从已接受的完整融合报告追溯实际融合/视觉初始化/EMG初始化的训练身份，重建分区并与各自实际内容SHA逐一相符。另重建RN18已准入的未来融合数据分区；RN18正式融合及精确cotrain初始化尚未完成，此行仅为未来数据集。两个全新进程审计完全一致，24次完整布尔行掩码核对独立验证区间计算，6个原拥有进程关闭。此审计未重新加载模型张量，也不授予密网格模型推理准入。

| 密网格 | 对照来源 | 训练输入唯一行交集 | 与训练输入相交的手请求 | 中心帧落入训练输入范围的请求 |
| --- | --- | ---: | ---: | ---: |
| vits_s | emg_initializer | 917476 | 1196 | 0 |
| vits_s | fusion | 1111032 | 1280 | 0 |
| vits_s | vision_initializer | 1047052 | 1280 | 0 |
| rn18_s | future_fusion_dataset | 928778 | 756 | 0 |

行交集按共享时间轴只计一次；手请求分别计入左右手。中心在留出区不代表整个输入窗口与训练区隔离；输入重叠也不能直接等同于中心目标被训练使用。本复现保持官方选择并披露实际交集，不因已观察到的交集改变样本或重新调参。完整val/test等角色交集见JSON。

## 重建索引

在已配置官方依赖的Python环境，从仓库目录运行；指定物理路径，索引函数拒绝符号链接。

```python
from pathlib import Path
from egoemg_repro.fusion_literal import compose_fusion
from egoemg_repro.occlusion_dense_data import native_dense_points
upstream = Path("/path/to/pinned/official/repository")
data = Path("/path/to/legacy/memmap")
cfg = compose_fusion(upstream)
first = native_dense_points(data, upstream=upstream, config=cfg, recipe="vits_s")
second = native_dense_points(data, upstream=upstream, config=cfg, recipe="vits_s",
                             expected=first["identity"])
assert first["identity"] == second["identity"]
```

运行时设 `CUDA_VISIBLE_DEVICES=""`。RN18使用 `compose_rn18_fusion` 和 `recipe="rn18_s"`。返回值仅为只读索引及数据身份，不能当作模型、几何或完整科学实验的准入证据。22项原生索引检查和11项交集检查实际通过；两份完整运行证明分别见[索引](full-index-admission-20261101.json)与[来源交集](producer-footprint-audit-20261101.json)。

## 尚未完成

- 密网格所有实际姿态、世界/相机变换、标志与episodeβ提取、独立重建，以及原生GT MANO/遮挡的完整执行和重放。1947步长与标准7790步长不整除，不能用标准网格分数作近邻替换。
- 全部冻结视觉/EMG/融合模型的真实逐样本预测、无缺失的键连接、原生误差/转换与独立重放。RN18还需完成其正式训练及兼容初始化。
- 官方过滤、自适应bin、SEM、加权三次样条、Spearman及图表与底表验收。
- 实际episode视频尺寸、畸变与左右约定的物理投影一致性。1920×1080官方回退继续显式标注。

全论文继续进行；现有14个完整科学模型设置未增加，也未声称Figure13已复现。

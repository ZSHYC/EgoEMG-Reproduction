# 原生密采样遮挡分析：索引、GT与待完成预测

本目录记录Figure13所需密采样的真实证据。原始标准网格几何已另在[`../native-point-geometry/`](../native-point-geometry/README.md)完成；[RN18完整密GT](rn18-geometry-complete-20261101.json)及[ViT-S完整密GT](vits-geometry-complete-20261101.json)均已完成两遍原生计算、独立核验及全部数组重放。模型预测误差和相关系数尚未验收。

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

## 完整GT数值输入与原生执行

[完整GT数值输入证据](geometry-input-admission-20261101.json)已完成：两次独立全量准入，各自在提取前后重新核对全部37,778,344,975字节几何主体及元数据/β/标志来源，合计四次完整主体字节核对；全部导出typed数组和身份完全相同，6个拥有进程关闭。沿用先前完整有限值扫描的同一份已核验字节，未宣称重新扫描四次全部浮点数。读取采用流式SHA，未把12.7GB单文件整体读入Python字节对象。无Torch导入、学习模型/优化器/更新/CUDA，两个进程Python/NumPy随机状态未变。

| 路线 | 时间点 | 左手非零姿态 | 右手非零姿态 | 左手零姿态 | 右手零姿态 | 手请求总数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vits_s | 8584 | 8332 | 8318 | 252 | 266 | 17168 |
| rn18_s | 5557 | 5398 | 5389 | 159 | 168 | 11114 |

零姿态、相机tracked、marker有效性及image stale等标志均保留。每条原生source request映射到其真实时间点/手，不使用标准网格分数的近邻替换。身份`8f2685496dfbbd272acf286db996c950693a27162f0b4bf0d92692d8dbcc2b7a`与导出数据只保留在私有服务器目录；公开文件只有统计/来源/哈希。

[真实小样本与完整流程启动证明](native-probes-and-full-launch-20261101.json)：每路线实际完成一次双手原生执行及另一进程的原始包核验，共4个真实手点、2个实际右MANO几何层、4次原生mesh/遮挡调用；8个小样本阶段拥有进程全部关闭。随后每条路线的完整第一遍已实际开始；计划各两次完整原生执行与各自独立原始核验，完成后还须确认全部拥有进程退出、原生typed数组/行重放完全一致。当前未将这些计划计为完整几何结果。

## RN18完整密GT验收

[独立完成证明](rn18-geometry-complete-20261101.json)覆盖全部11,114个手请求、5,557个时间点。两遍完整原生运行各有10,787个非零姿态及327个保留的零姿态；另有真实两手小样本。完整运行及小样本各经过新进程的原始逐点核验，合计6阶段、14个原拥有进程全部退出0。实际3个右MANO层、22,230次mesh函数调用和21,576次遮挡调用均来自原始执行记录；没有构造学习模型、更新优化器或初始化CUDA。

退出后再次核对源码、运行时、行日志和全部压缩原始包，共4,312,291,005字节；两遍的全部11,114行及10,787对非零点包完全配对。140,231个typed数组的dtype、shape及数据字节一致，按单遍未压缩数组内容计179,692,283,862字节。逐个比较完整无损压缩的`.npy`成员载荷并检查类型头；相同载荷可推出完整数组字节相同，压缩载荷不同则回退到完整NumPy解码比较。本次140,231个成员全部载荷相同，无回退；另用6个真实点、78个数组做了完整NumPy解码交叉核对。这一判断基于[NumPy的NPZ/NPY格式](https://numpy.org/doc/1.26/reference/generated/numpy.savez_compressed.html)与[Python ZIP成员格式](https://docs.python.org/3.11/library/zipfile.html)，不是额外解压读取179GB的计时。该独立读取流程16.88秒结束，其2个拥有进程也已关闭。

本项完成的是RN18密GT几何；ViT-S现也另行验收，见下文。继续保留1920×1080官方回退、原始左右镜像约定及实际视频/物理投影未证明等边界；冻结模型预测、paired gain和Figure13仍待完成。

## ViT-S完整密GT验收

[实际独立完成证明](vits-geometry-complete-20261101.json)覆盖17,168个手请求、8,584个时间点，两遍各16,650个非零姿态与518个原样保留的零姿态。两遍完整原生执行、各自新进程原始核验及先前真实双手小样本/独立核验合计6阶段；14个原拥有进程全部退出0，原流程耗时9,328.96秒。实际3个右MANO层、34,338次mesh函数调用及33,302次遮挡调用，0学习模型/优化器更新/CUDA。

退出后再次逐字节核对冻结源码、运行时、全部行日志及6,653,445,416字节压缩原始包；完整两遍的全部行/非零点包配对一致。216,450个数组的dtype/shape/数据字节精确相同，按单遍未压缩typed内容计277,359,462,900字节。使用上文已验证的完整无损NPY成员比较和完整NumPy回退方法；本次216,450个成员全部相同、回退0。此处没有另行物理解压读取277GB，也未重新计算已关闭的几何数学。独立比较26.10秒完成，2个比较拥有进程也已退出0；方法的真实NumPy解码交叉验证引用此前已关闭RN18流程，未冒充新ViT点的解码执行。

两条路线现在均具备完整密GT：共28,282手请求、27,437非零姿态、845零姿态；各自两次完整原生计算、三次原始核验以及数组重放完成，28个原生拥有进程与4个后验比较拥有进程全部关闭。与PDF Figure13相同的ViT请求数仍不证明预测误差、相关性或物理投影等价；后续不得用完整GT证明代替真实冻结模型推理。

新入口调用未改动的已验收标准原生helper（源SHA`c2507b6d6308f3c9609a4931391a7863e7c4f3811a206b1bd485ead5a15f1634`）。独立验证直接编译该固定源码原`verify_geometry`的逐点验证循环，保留原投影/面积/可见性计算与严格行模式，未重写或优化已关闭的几何数学；只适配密网格IO、请求映射与完成范围。数据层30项检查、入口20项检查实际通过；零姿态单元测试使用模拟mesh函数，只证明IO/范围，真实MANO证明来自上述独立服务器小样本。

原生入口提供`validate`、`run`与`verify`子命令。例如完整运行及独立核验：

```bash
CUDA_VISIBLE_DEVICES='' python -m egoemg_repro.occlusion_dense_geometry run \
  --export /path/to/private/dense-export \
  --evidence results/native-dense-occlusion/geometry-input-admission-20261101.json \
  --recipe vits_s --mode complete_dense \
  --metadata /path/to/legacy/metadata.npz \
  --calibration /path/to/meta/GX010023_standard_calibration.json \
  --upstream /path/to/pinned/official/repository \
  --mano-assets /path/to/admitted/mano/assets --chumpy-dir /path/to/admitted/chumpy \
  --output /path/to/new/native-dense-run
CUDA_VISIBLE_DEVICES='' python -m egoemg_repro.occlusion_dense_geometry verify \
  --output /path/to/new/native-dense-run
```

输入证明SHA固定为真实全量提取结果；任意自行重新封签的数据不获准。`validate`仅载入和选择，不运行几何；`bilateral_probe`只检验实际非零双手小样本，不能取得`complete_dense_geometry`。公开证明不会让缺失的私有数据或授权MANO模型自动出现。

## 官方误差转换与统计接口

新增[`occlusion_dense_analysis.py`](../../egoemg_repro/occlusion_dense_analysis.py)，直接编译并执行固定官方版本的CSV字典和绘图统计语句；只增添原始行号/分箱成员观测，不改计算。调用前后核对源码、CSV与模块字节；没有模型、数据或Figure13准入权限。[16项实际检查](source-analysis-tests-20261101.json)全部通过，包含独立进程运行**未修改的完整官方绘图main**并生成443,707字节真实PNG：7个不等样本数的非线性bin，所有均值、SEM、200点样条、Spearman与散点索引逐项完全相同。该PNG来自合成工程数据，不是论文实验图。

保持官方57.3度换算/误差4位与几何6位小数；delta由原始弧度之差计算后取4位，而非减去已取整的两列。保留全部原始行和NaN几何；分析使用`occlusion_score.notna() & (mae_vision_deg > 0.01)`。0.05基础bin按`[lo,hi)`合并至默认100样本，末bin不足10丢弃；被bin丢弃或恰在最大右边界的行仍参与逐样本Spearman。SEM采用样本标准差；实际趋势为按bin样本数加权的三次`UnivariateSpline`，保留官方“LOESS trend”标签与实际算法的差异。常量输入的原生NaN相关系数显式输出null；无法拟合时拒绝生成虚构曲线。

```python
from egoemg_repro.occlusion_dense_analysis import source_csv_rows, analyze_source_csv
# records必须包含真实三分支误差、精确样本键、几何与标志；不会补零分支。
rows = source_csv_rows(records, upstream=upstream)
pd.DataFrame(rows).to_csv("private-results.csv", index=False)
statistics = analyze_source_csv("private-results.csv", upstream=upstream)
```

运行版本numpy1.26.4/scipy1.13.1/pandas2.2.3；独立官方PNG使用现有report extra的Matplotlib3.9.4（官方environment列3.9.2）。绘图依赖单独安装到任务目录，不改动正在训练的共享环境。首轮缺模块RED16/16，再完成全部GREEN16/16、无跳过；原生常量输入警告保留。独立重新读取测试日志、源文件和PNG字节后出具上述证明。

[实际评价入口兼容诊断](source-inference-compatibility-20261101.json)还证明：官方脚本引用的EMG实验配置已移入`_archive`，归档配置仍依赖缺失dataset；同脚本旧attention补丁与当前`TemporalAttentionPool`接口产生真实矩阵形状报错，原生pool则成功。后续推理须使用同一固定版本中可执行的原生中心前向及实际完成的模型来源，显式记录兼容适配；尚未据此执行或准入完整密网格预测。误差继续按官方逐样本22角度未mask均值，不替换成标准报告的masked指标。

## 预测与几何键连接

[`occlusion_dense_predictions.py`](../../egoemg_repro/occlusion_dense_predictions.py)新增严格有序连接接口，[26项检查](prediction-join-tests-20261101.json)实际通过：每个预期点必须同时具有三分支真实弧度误差与同序几何行。缺失、多出、重复、重排、左右手/中心/split或窗口不符、浮点冒充整数键均拒绝；ViT7790与RN18的12000窗口单独校验。几何零姿态保留NaN和真实模型误差；tracked/marker无效、image stale均不作为额外删除条件。官方CSV中`mocap_stale`实际取`image_head_stale`，连接时保留该命名映射；非零几何的原始分数和官方6位小数列须一致。

这个纯连接函数不载入模型，也不证明来源、完整人口、权重或推理已准入；检查用合成记录，不是新科学成绩。真实推理还必须分别提供同一窗口上的两种EMG输入：已完成的standalone cotrain-S为8通道、逐手向量归一化；ViT融合为16槽、unified标量归一化。不能从已标量归一化的16槽张量直接抽8槽冒充原模型输入，亦不能把7790窗口改成12000后删除不适配的中心。

## 尚未完成

- 两路线完整密GT均已验收；后续必须用相同精确请求键连接真实预测。1947步长与标准7790步长不整除，不能用标准网格分数作近邻替换。
- 全部冻结视觉/EMG/融合模型的真实逐样本预测、无缺失的键连接、原生误差/转换与独立重放。RN18还需完成其正式训练及兼容初始化。
- 统计接口虽已对齐未修改官方main，但真实完整预测上的过滤/bin/SEM/样条/Spearman、图表与底表验收仍待完成。
- 实际episode视频尺寸、畸变与左右约定的物理投影一致性。1920×1080官方回退继续显式标注。

全论文继续进行；现有14个完整科学模型设置未增加，也未声称Figure13已复现。

# 原生融合联合数据读取

`FusionLiteralDataset`把同一窗口的真实EMG数组与中心RGB、22角标签交给固定官方`_getitem_center_supervised`。它是完整ViT-S+EMGFormer-S实验的数据读取部分；不是完整数据核验、GPU资格或训练结果。之前视觉入口的零长度EMG占位没有用于此处。

## 原语义与明确适配

原16槽位置`[10,12,0,1,2,4,5,6]`、7,790点窗口、训练stride400/评价stride7,790保留。原函数先补零，再固定scalar归一化，再执行原训练旋转及tensor转换，最后转置为16×7,790；因此归一化后未占用槽含固定的小偏移，不能重新强制置零。RGB来自同一抖动中心的原图像读取，缺图继续给出归一化黑图与`vision_valid_mask=false`，不减少请求数。22角中心生成mask及原float64腕角换算后float32舍入保留，不追加旧协议的腕角清理。

发布训练配方声明`egoemg_unified/filtered_paper`，但其引用的`per_dataset_norm_stats.json`中没有所需键，实际原方法抛KeyError；repro别名文件也没有该namespace。入口显式绑定同一固定官方版本的`per_dataset_norm_stats_unified.json`，调用原`_load_norm_stats`，实际得到mean0.00011658444197544563/std10.839283499203331/epsilon1e-6。原配方本身未被改写，这个数据执行适配在`fusion_normalization()`中记录文件SHA、namespace、实际参数和原失败。它不是根据本项目保留训练区间新估计的统计量，也不能证明训练集独立归一化。

物理旧版`emg_left_filtered/emg_right_filtered`显式映射为原函数请求的`filtered_paper`；每样本来源仍为EgoEMG。它不冒充声明的EgoEMG/ShowEE/Incre物理合并数据。若检测到`dataset_source_id`则拒绝，合并来源需要独立适配。原head帧字段优先；只有它不存在时才用旧版webcam帧字段作为明示别名。完整8通道float32 EMG形状与文件大小、标签/帧字段和所需图像库在构造时检查。

## 窗口与使用

```python
from egoemg_repro.data import LegacyStore
from egoemg_repro.fusion_literal import compose_fusion
from egoemg_repro.fusion_literal_data import (
    FusionLiteralDataset, build_fusion_partitions, fusion_normalization,
)

config = compose_fusion(upstream)
normalization = fusion_normalization(upstream)
partitions = build_fusion_partitions(LegacyStore(data))
train = FusionLiteralDataset(data, partitions, crops, upstream=upstream,
    config=config, normalization=normalization, role="train")
sample = train[0]
```

训练保留原`randint(0,min(400,episode_end-anchor_end))`范围。若任何可能抖动后的完整EMG窗口接触验证/测试区间或离开训练split，提前排除整个锚点，不裁剪随机范围。验证只在官方训练区间内取独立连续留出；评价保留既有官方中心网格。该时间划分不能消除之前整episode滤波产生的依赖，也不是新建完全无重叠参与者测试。

每个角色按“左手全部窗口、右手全部窗口，再完整重复2遍”排序。返回`fusion_sample_index`、`fusion_unique_sample_index`、`fusion_repeat_index`、锚点/实际窗口/中心/抖动偏移等身份；重复消费不能计为独立手样本。

## 当前检查与仍欠内容

CPU检查使用真正的小型memmap、LMDB/JPEG文件，内容是明确的合成工程夹具。两手/三角色与直接原数据集逐字节比较EMG、RGB、标签、mask和随机状态；覆盖0/1/17/400尾部、窗口隔离、flat旧库、缺图、坏JPEG、帧字段优先级和损坏输入拒绝。检查不训练模型、不初始化CUDA，也不计算论文精度。

完整融合仍须：全部实际输入哈希、完整EMG和所有可能标签中心的有限性扫描、全量样本身份；真实验证best分支来源；原batch200/bf16的GPU训练/恢复验收；150轮完整训练、验证选模、冻结测试及独立报告。当前读取器允许读取的文件不能仅凭构造检查就用于声明科学训练合格。

服务器随后完成[实际旧版点读与人口索引检查](../results/native-fusion-real-data-points-20261028.json)：三个角色各左右手共6个真实窗口有限且形状正确，原进程实际exit0并独立确认退出。完整元数据索引消费train/val/test为395,276/3,152/8,612，唯一手样本197,638/1,576/4,306。它补充真实IO证据，仍未完成所有实际payload的哈希或有限性扫描。[一次整体审查](../results/native-fusion-data-review-20261028.json)无阻塞发现，另有一项直接损坏中心字段/namespace的回归覆盖待补充。

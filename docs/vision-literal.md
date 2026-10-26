# DINOv2：官方执行语义与可用数据读取层

目前实现了配置契约和数据读取层，**尚未构造这套原生Lightning训练模块、执行真实cohort优化或完整训练**。已有DINOv2预训练权重／合成CPU检查仍保留原证据，不能代替本入口的实际训练。[执行计划](superpowers/plans/2026-10-27-literal-vision.md)继续覆盖原生模块、优化／重载、完整训练、测试和报告；这不是全文完成声明。

## 固定源码裁决

官方版本固定`f7cbd048b7092139852270491d3859cd472e0311`。S使用实际`fusion/vision_vit_small`；B/L目前仅有明确声明的S配方backbone派生，不假称存在作者独立YAML。

| 项目 | 固定源码的实际行为 |
| --- | --- |
| 损失 | 两个指尖别名逐项相加，有效系数0.02 |
| 原生批量 | train／val／test均360；`make_data_module`只传`batch_size`，配置的`val_batch_size=256`未被读取 |
| 重复 | `_build_dataset`在左右手拼接后对所有角色repeat2，包括验证和测试 |
| 输入／采样 | 256px、7790点窗口、train stride780、eval stride7790；train原始randint jitter，val/test固定中心 |
| 框架 | bf16-mixed、clip1、AdamW、150轮cosine上限、val_mae/min、patience30、top3和last |
| 声明数据 | 当前vision lineage指向EgoEMG／ShowEE／Incre物理合并数据；可获得的旧版EgoEMG不能证明等同该训练集 |

原生数据模块已经在受控CPU输入上实际运行，核对repeat和批量行为；不是只读取单份YAML。未支持的backbone、损失／批量／精度／callback／数据声明修改和私有绑定替换会被精确配置比较拒绝。配置读取不创建模型或探测CUDA。

来源：[官方vision配置](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/fusion/vision_vit_small.yaml)、[实际数据模块](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/datamodule.py)、[声明的合并数据配置](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/dataset/egoemg_unified_angle_regression.yaml)。后续loader直接消费桥接层已有的repeat2，不能再套一次原模块重复。

## 旧版数据桥接

新`VisionLiteralDataset`执行未改的官方`_getitem_vision_only_crops`和`_read_episode_crops`，只桥接旧版memmap字段、独立窗口索引及目录／平铺LMDB。优先原`image_head_frame_index`；缺少时明确采用旧`image_webcam_frame_index`。左右手不再次翻转；RGB归一化、腕角的原始浮点转换、生成标签mask和缺失key处理均保留。

原生成mask作用于22角，不沿用旧训练器的逐角腕部清理。缺失key返回原归一化黑图和`vision_valid_mask=false`，保持原生成标签mask；不因为缺图删除样本。缺失整份episode数据库则阻止完整数据准备，损坏JPEG在读取时报错。当前桥接明确拒绝带混合source-id的数据，需要另行使用其真实来源语义。

独立验证仍来自保留的官方train区间。每个训练锚点保留原`randint(0,min(780,episode_end-anchor_end))`范围；任何可能的7790点抖动足迹触及验证／测试或非train行时，预先排除整个锚点，不裁短随机范围。左右手按原拼接顺序排列，再重复两次；唯一索引、重复轮次和实际消费索引分别保存。真实小memmap／JPEG与原读取器的数值、dtype、shape和RNG逐字节一致，包含episode尾部剩余0／1／17／780行。

## 服务器实际只读扫描

[原始脱敏扫描](../results/vision-literal-data-census.json)来自实际41份episode记录，尚不证明独立自然人数。完整抖动足迹规则排除了352个训练锚点：

| 角色 | 锚点窗口 | 唯一手样本 | repeat2消费手样本 | 全部可能中心行 | 缺失／请求唯一视频帧手key |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 50,505 | 101,010 | 202,020 | 39,393,900 | 138,429／2,362,960 |
| val | 788 | 1,576 | 3,152 | 788 | 101／1,576 |
| test | 2,153 | 4,306 | 8,612 | 2,153 | 173／4,306 |

所有可能中心行的手指／腕目标均未发现非有限值；负视频索引为0。key计数是唯一`(episode,video_frame,hand)`请求，**不是窗口、有效角度或模型误差分母**。训练中心抖动范围按并集扫描，不将未加权的可能中心分布冒充实际训练epoch。验证／测试重复消费数也不能当作新增独立样本。

扫描实际解码并核对18个原始点读样本，GPU未初始化，模型构造和优化次数均为0；52个实际数据／metadata／LMDB文件的前后stat保持一致。只核对所有可能key的存在性，未逐key解码全部JPEG，也未重新计算所有payload整文件hash；后续完整训练仍需独立不可变数据准入，不能仅凭这份扫描授权训练。

实际执行的[只读扫描源码](../tools/vision_literal_census.py)按原字节公开，摘要与CPU证据中的`audit_source_sha256`一致。在已配置依赖、原始数据、裁剪和固定官方源码的环境中，可从项目根目录运行：

```bash
PYTHONPATH=. python tools/vision_literal_census.py "$DATA" "$CROPS" "$UPSTREAM" > census.json
```

该工具只读数据并输出汇总，不构造模型；扫描中的时限和逐episode内存范围保留。84成员pytest快照覆盖模块和测试，这个额外公开工具另以同字节真实服务器执行核验，不能混称全部属于pytest。

## CPU集成与后续边界

[结构化CPU证据](../results/vision-literal-foundation-cpu.json)记录配置22项缺模块RED和数据9项RED后实现。最终37项定向通过（12.21秒）；固定84成员完整1,480 passed／3真实CUDA skip／106 warnings（295.00秒），无Torch轻量841 passed／595依赖skip／1 warning（22.60秒）。此前80个源码成员和七项legacy结果保持原字节；完整vision训练分支的最终独立审查仍未进行，本次托管CI尚未观察。

下一步使用真实官方Lightning factory核验预训练是否保留、原生损失／component optimizer及严格重载，再实现完整runner并在最终审查后做真实GPU容量与优化检查。原生bf16不能套用classic FP16 scaler契约；官方单rank/global batch和六rank差异、独立验证与缺失合并数据均须保留。没有新模型性能结果，论文表图和全文目标继续按[清单](full-paper-status.md)验收。

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

## 原生模块与短训练

新增`vision_literal_runtime`直接执行官方`setup_runtime`后调用原`make_lightning_module`。新建模型要求提前准备并校验真实HF资产，且在导入前启用离线模式。原生timm的加载路线保持，主干的每项张量与真实权重经过原位置插值后的结果比较；没有随机主干fallback。自包含恢复只关闭重复的预训练传输，随后严格加载整个原生Lightning模块状态。构造身份本身不证明完整训练。

真实DINO-S的20项CPU检查已完成，包括174项主干张量、178项完整模块状态、seed42、22角输出、完整权重重载与预测一致。原生损失仍使用生成标签mask，不因缺图标记而清除；两个指尖别名按原顺序分别加入0.01。原生AdamW保留`head`／`backbone`两组，学习率均为1e-4、weight decay为0.01；余弦调度仍为150轮、最低1e-5。

`vision_literal_check`执行原生Lightning的两批短训练、完整独立验证，再由另一个进程恢复并训练一批。CPU为明确的32-true工程适配，batch2、workers0；CUDA入口保留bf16-mixed且不创建FP16 GradScaler，实际GPU执行仍等待完整训练分支审查。测试用真实小memmap/JPEG文件，不能充当真实完整队列的优化结果。恢复核对模型、优化器、调度器、循环、RNG和原生可恢复callback状态；PL2.5保存但不恢复的`ModelCheckpoint.current_score`明确排除，其他选模／top-k字段保持逐项比较。两个阶段保留各自CSV日志，共用固定的自有检查点目录。

[实际原生CPU证据](../results/vision-literal-native-cpu.json)现已包含真实旧版队列的完整工程运行：S先更新2次、消费4手训练样本并验证全部3,152手；不同PID/start的新进程完整重推该验证集合，预测／目标／mask／顺序一致，再实际更新1次并再次完整验证3,152手。恢复时模型、AdamW状态、余弦调度、native可恢复callback、loop和Python／NumPy／Torch及loader RNG一致。这里的3,152是1,576个唯一手样本的repeat2，训练数据集长度202,020不代表短训已消费全部样本。两个loader各用seed42的独立generator，是明确的工程适配，不冒充原完整数据顺序。

实际文件身份绑定47个train/val所需文件，排除未读取、未提供的raw EMG声明；完整payload内容准入仍未完成。S两阶段均未初始化CUDA。B/L也通过本次原生factory、真实权重保留、原component optimizer、22角前向及严格禁传输重载；实际参数分别86,182,166／303,833,622，保留主干张量174／342项，尚未做真实cohort优化。

固定89成员完整CPU检查为1,506 passed／3实际CUDA skip／106 warnings（340.64秒），轻量841 passed／621依赖skip／1 warning（22.57秒）；shell和workflow run块语法检查通过。本次托管CI尚未观察。首次真实队列缺失raw文件预检失败及过长basetemp导致IPC检查失败均保留，修正后重新完成整体检查。

在提前配置好的真实权重缓存和数据环境中，可依次运行：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_HUB_CACHE="$CACHE" \
  python -m egoemg_repro.vision_literal_check prefix \
  --data "$DATA" --crops "$CROPS" --upstream "$UPSTREAM" --output "$NEW_OUTPUT" --device cpu
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_HUB_CACHE="$CACHE" \
  python -m egoemg_repro.vision_literal_check resume \
  --data "$DATA" --crops "$CROPS" --upstream "$UPSTREAM" --output "$NEW_OUTPUT" --device cpu
```

缓存中的原生`main`引用须解析到已核验的固定权重提交；禁止在模型构造期间补下载。输出是私有工程产物，包含可信本地checkpoint和运行信息，不应整目录公开。完整runner、最终审查、GPU容量和完整训练仍未完成。官方单rank/global batch和六rank差异、独立验证与缺失合并数据均须保留；没有新DINO性能成绩，全文继续按[清单](full-paper-status.md)验收。

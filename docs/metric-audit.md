# 已锁定检查点的指标审计

审计入口是 [`egoemg_repro.audit`](../egoemg_repro/audit.py)。它只对**完整训练、完整独立验证已选定 best、训练器已执行最终测试**的 run 做一次额外测试复算，验证冻结训练器记录的中心指标，并补充逐手／逐 split／逐用户交叉统计与跨模态相同的有图中心集合。它是已声明的测试结果核验和指标分解，不能用来选模型、换 checkpoint、调整超参数或依据测试精度反复试验；也不是新的独立 benchmark。

审计核对保存的来源、预测与统计，不能证明 legacy 数据与论文版本等价，也不保证发布 split 名称所暗示的用户独立性。[队列覆盖记录](../results/cohort-coverage.json)已确认实际测试窗口覆盖 40 人，User、Both 请求窗口各与训练窗口有 1 名用户交集。后续逐 split 结果沿用这些发布标签，并保留此限定；不能在测试后删改队列或重新选模来消除交集。跨角色输入行足迹无重叠也不消除整 episode 预过滤的时间依赖。

工具存在与夹具测试通过，不表示 E1–E4 的真实完整训练或最终审计已经通过。真实结果仍需各自完成的 run 与审计产物，参见[覆盖矩阵](reproduction-matrix.md)。运行中的训练代码和配方保持冻结。

## 调用

在已安装训练、视觉依赖的同一环境中，使用私有位置的真实产物：

```bash
python -m egoemg_repro.audit \
  --run /path/to/completed-run \
  --data /path/to/legacy-data \
  --upstream /path/to/pinned-official-source \
  --crops /path/to/verified-center-crops \
  --output /path/to/new-audit-directory \
  --device cuda:0
```

`--crops` 对 EMG 也必填，因为其额外的有图子集必须来自与视觉、融合相同的真实发布 crop。输出目录应为空。发生身份、完整性或数值不符时退出失败，不生成可发布的 `audit.json`；已有审计目录不会被覆盖。只有人工排查到可解释的执行问题后，才可重新进行显式核验，不得根据测试分数选择较好的审计结果。

## 核验边界

- `summary.json` 必须为 `status=training_complete`、`test_evaluated=true`、`test.bounded=false`。权威文件是 `last.pt` 内嵌的 schema 2、`validation_bounded=false` 的 best；工具不在兄弟 checkpoint 中搜索，不接受命令行选步数。
- 核对保存的 run ID、训练 epoch 预算完成、全局步数、选中 best 步数和验证分数，以及 summary、checkpoint 和单独保存的 provenance 是否一致。
- 原始 recipe 字节、有效 config、partition、normalization、初始化身份、实际数据身份、固定官方源码和执行时适配代码必须与保存身份一致。文档提交变化可接受，运行时代码字节变化不可接受。
- 重建模型使用冻结的官方构建函数和 `resume_state`，严格加载 best，不需要原始支路 checkpoint，也不触发 ImageNet 权重下载。保留保存的 batch size、worker 配置、precision、matmul precision、固定测试网格及双手顺序，不接受审计时覆盖超参数。若原最终测试使用 bf16，改用 CPU float32 将被拒绝。
- 数据／crop 身份遵循冻结训练器的检查范围：metadata 和映射字段内容、array 大小及已验证 receipt 摘要。工具不会逐次重新读取数百 GB payload 来证明发布者真实性；receipt 的来源与覆盖仍须单独核验。

额外锁定核验中的 EMG 前向仍先产生时间序列预测，再使用冻结的 `aligned_training_batch` 完成 510 点上下文裁剪和官方预测插值，取 `emg_window_length // 2 - model.left_context` 的真实中心。审计只重新累计中心指标，**没有额外复算整段轨迹指标**；公开轨迹数值仍来自原训练器最终测试，锁定产物检查只核对其保存的轨迹分母。这不表示 EMG 核验没有计算时序预测。视觉／融合取索引 0 的唯一中心目标。三种模型都使用权威 `angle_valid_mask`；手指 20 维与腕角 2 维独立有效，弧度误差只在报告 MAE 时乘一次 `180/π`。

## 预先声明的数值门槛

全中心结果与 `summary.test.sampling.center` 对比，包含 overall、逐手、逐 split、手×split 以及 fingers／wrist。分组集合必须相同，`valid_angle_count` 必须精确相等，MAE 的绝对容差固定为 **`1e-4` 度**，弧度误差和的容差相应为 `count × radians(1e-4)`。保存的每条 MAE 自身也必须能由其弧度误差和与分母重算。CLI 不提供放宽容差的选项。

视觉与融合还核对训练器已有的 `paired_population.all`、`vision_valid` 和缺图计数。若完全没有有图样本，输出零分母／`null` MAE；冻结累计器的空 component 字典按等价的零计数处理，不虚构可比较精度。

## 统计与 population

`all` 和 `crop_present` 分开报告以下内容：

| 字段 | 含义 |
| --- | --- |
| `overall`、`by_component` | 所有有效角度的弧度误差和、有效角度数及按该分母计算的 pooled MAE |
| `by_user` | 匿名用户各自的 overall／fingers／wrist 误差和、计数及 MAE |
| `user_summary`、`user_summary_by_component` | 各用户 MAE 的非加权均值与总体标准差，`ddof=0`；该 component 无有效角度的用户不参与其均值／SD |
| `by_hand`、`by_split`、`by_hand_split` | 每个条件内独立进行同样的 pooled 与逐用户计算；不从跨 split 的用户总体值推导条件内 SD |

`populations` 在 overall、逐手、逐 split 和手×split 下保留请求中心数、有图数、缺图数，以及 `all_sha256`、`crop_present_sha256`、`missing_sha256`。摘要先绑定中心全局行号、手与 partition hash，再对排序后的不透明 token 多重集做 SHA256；重复请求保留重数，输入顺序不影响摘要。跨模型比较必须检查相同 partition、实际 crop 身份与对应条件的集合摘要／计数，不能仅因样本数相同就认定集合相同。

EMG 通过冻结 episode／中心和发布的 `image_webcam_frame_index`（或 `image_head_frame_index`）查 `CropStore` 的原始 LMDB key；不从 EMG 行号猜视频帧号，不重新产生真值裁剪、不重编码图像。视觉／融合使用冻结 dataset 已产生的 `vision_valid_mask`。缺 key 留在 all 中并计为缺图；缺整个 episode LMDB 会失败。

## 可发布文件与隐私

成功输出 `audit.json`，schema 为 `audit_schema=1`、`status=verified`、`purpose=locked_checkpoint_verification`。`baseline_verification` 记录实际一致性检查和固定容差；`training_test_passes=1` 与 `additional_verification_passes=1` 记录这个产物对应的训练最终测试及额外核验，不能解释为整个项目历史总共只运行过一次审计。

公开字段使用明确白名单：指标、计数、匿名 `user-001` 标签、分组集合摘要，以及 config／原始 recipe／partition／normalization／source／data／reproduction／初始化／crop／实际检查点／原 summary／审计运行时代码的 SHA256。匿名标签按同一测试参与者集合排序分配，仅在相同数据／partition 身份下比较；这不是可逆参与者名表。

不复制原 provenance、私有路径、机器地址、参与者姓名、episode ID、逐中心 token、图像、原始数组或权重。`audit.json` 中的摘要绑定私有原始证据，但不替代这些原始证据的保管与独立审阅。

## 本地验证

```bash
python -m pytest tests/test_audit.py -q
EGOEMG_UPSTREAM=/path/to/pinned-official-source \
  EGOEMG_NO_PRETRAINED_DOWNLOAD=1 python -m pytest -q
```

无官方源码环境时，真实官方模型夹具测试会明确跳过；其余统计与拒绝条件仍执行。官方夹具使用小型合成 memmap、发布格式的真实 JPEG／LMDB 与实际官方模型前向，验证 EMG、视觉、融合的冻结对齐、严格自包含重载、缺图 population 与数值一致性。夹具中的完成形状证据仅服务于软件测试，不是实际训练完成证明。

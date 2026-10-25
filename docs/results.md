# 结果与训练曲线

截至 2026-10-25，EMGFormer-S 与 EMGFormer-M 均已完成完整训练、独立验证选模、训练器最终测试、严格结果导出及独立复算，完整报告、审计与曲线见[已完成实验](../results/completed/README.md)。全部 41 份裁剪与中心覆盖已通过 G1 就绪门槛，见[裁剪就绪证据](../results/crop-readiness.json)。原审计父流程的进程绑定失败已通过独立续跑解决：S 原成功产物经核验接纳，没有重复推理；M 实际复算退出码为 0 并通过接纳，两者共同中心人口摘要一致。续跑队列现等待原 RN18 训练完成。RN18 真实有界训练及严格重载通过，完整训练运行中；融合尚未启动。**全项目定量复现尚未完成；完整论文表格尚未复现。** 实际运行进度见[运行状态](run-status.md)，模型与论文口径差异见[复现对照](reproduction-matrix.md)和[指标审计](metric-audit.md)。

| 模型 | 最终测试结果 | 发布条件 |
| --- | --- | --- |
| EMGFormer-S | 严格导出与独立复算通过，见[完整产物](../results/completed/README.md) | 200 轮、25,400 次更新已完成；验证选择第 635 次更新；续跑核验接纳原成功报告及审计，没有重复推理 |
| EMGFormer-M | 严格导出与独立复算通过，见[完整产物](../results/completed/README.md) | 独立运行，相同冻结数据／划分／归一化；200 轮、25,400 次更新，验证选择第 635 次更新；实际复算成功，匹配中心人口与 S 一致 |
| ResNet-18 | 完整训练运行中，待验证选模后的测试 | 完整 crop 人口与缺失统计已核验，真实有界训练和严格重载已通过；仍须完整训练、最终测试及审计 |
| ResNet-18 + EMGFormer-S | 尚未启动，待完整训练和验证选模后的测试 | 合规的独立验证分支初始化、依赖验收及完整测试 |

本页不以短跑、集成检查或进行中的验证 MAE 填充最终基准表。完成后的表格应由下面的导出器生成，保留误差和、有效计数及协议身份。

本次冻结测试窗口覆盖 40 人，按 legacy 发布中心标签报告 Gesture / User / Both。User 与 Both 各有 1 名用户也出现在训练窗口中，不能据分组名宣称严格未见用户测试。该发现不改变已冻结的标签、窗口和所选检查点；计数范围、短 episode 与来源身份见[队列覆盖记录](../results/cohort-coverage.json)。

## 安装与状态快照

在既有训练环境中安装可选绘图依赖：

```bash
python -m pip install -e '.[report]'
```

该 extra 固定 Matplotlib 3.9.4，并约束兼容的 contourpy；项目继续要求 NumPy `<2`。它不修改运行中的训练源码或配方。

```bash
python -m egoemg_repro.report status \
  --run runs/emg_s_full \
  --output outputs/public/emg-s-status
```

状态输出固定为：

- `status.json`：白名单化的配置、优化器更新数及训练／验证曲线数据；
- `status.md`：明确标记“未完成／未核验”的说明；
- `training-curves.svg`、`training-curves.png`：标准 Matplotlib 图，醒目标记 `INCOMPLETE / UNVERIFIED`。

`status` 只读取 `config.json` 和 `progress.jsonl`，不读取测试数据、`summary.json` 或模型检查点。进行中的最后一行若尚未写完会被忽略并计数标记；完整但损坏的 JSON 行会报错。训练曲线显示原始目标函数，验证曲线单独以度为单位；bounded 验证有不同标识。没有验证记录时不会制造曲线点。

没有绘图库时可使用 `--no-plots`；状态解析和 JSON／Markdown 输出仅依赖标准库。这样的输出会明确说明图表未生成。状态文件的 `benchmark_eligible` 始终为 `false`，即使某个源目录可能已经训练结束，也必须经过完整结果导出核验。

## 完整结果导出

确认训练完成后，在能够读取训练检查点的 Torch 环境中执行：

```bash
python -m egoemg_repro.report export \
  --run runs/emg_s_full \
  --output outputs/public/emg-s-final \
  --model-label EMGFormer-S
```

完整输出固定为 `metrics.json`、`metrics.md`、`training-curves.svg` 和 `training-curves.png`。`--model-label` 使用简短公开模型名。状态和完整结果使用不同输出目录，防止残留的最终成绩文件把进行中的快照伪装为已完成基准。

导出器要求并交叉核对：

1. `summary.json` 实际记录 `training_complete`、`test_evaluated: true`，且测试不是 bounded；
2. 原始配方、有效配置、划分、训练归一化及 provenance 身份一致；划分足迹不跨角色重叠；
3. `last.pt` 自包含的最佳状态与 `best.pt` 权重、选择分数、步数、完整验证策略及身份相符；
4. 训练和完整验证记录支持所选模型，测试批次及 22 角度的有效／无效分母覆盖预期窗口；S/M 的固定左上下文为 510 点，12,000 点输入的完整轨迹必须包含 11,490 点，测试和验证均核验此分母，未知 EMG 架构拒绝导出；
5. 分手、split、手指／手腕、匿名用户误差和与有效计数能够重算 pooled MAE；度只从弧度误差和换算一次；
6. 视觉结果明确区分 all-label 与 `vision_valid` crop 人口，缺失数量与各人口计数一致。

bounded 短跑、未完成状态、部分测试／验证、错误的最优检查点或不一致的误差分母均不能导出为最终成绩。导出器核验已有运行证据，不重新执行测试集推理；独立检查点复评与数值重现仍是完成审计的一部分。若迁移后的目录只保留后续验证日志，会显式记录历史不完整，而不会补造训练曲线。

## 指标口径与隐私

`metrics.json` 分别保留 center 与轨迹指标、手、split、手指／手腕分项、手×split 表、弧度绝对误差和、有效计数和无效计数。视觉模型只有 center 标签，不会从单帧生成“轨迹成绩”。缺失 crop 的黑图仍保留在官方 all-label 人口中，`vision_valid` 是相同样本定义下的 crop 可用子集；该子集的左右手数量可以不同。

逐用户统计仅使用 `user_001` 等导出内匿名编号，不发布原始参与者名、episode 记录或身份映射。**`user_summary` 的均值和总体标准差来自跨所有手、所有 split 合并的逐用户 MAE，并非每个条件的 σ。** 条件内统计应读取各模型实际完成的[独立指标审计](metric-audit.md)产物，不能由 pooled 用户统计反推；S/M 已有审计，视觉与融合尚待完成，论文队列和统计等价性仍须单独判断。

公开 JSON 使用字段白名单，只保留公开模型参数、计数、匿名统计和 SHA256／固定上游提交身份。原始路径、原始 provenance、参与者名、窗口行记录、图像、数据集和权重均不复制到公开输出。请只选择经过核验的上述输出文件发布，勿发布整个私有运行目录。原始数据的整段滤波依赖、生成标签来源差异和未证实的论文等价性仍然适用。

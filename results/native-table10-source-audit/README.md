# 原生 Table 10 的同帧来源核验

**现有原生中心报告不能直接作为同帧 Table 10 完成证明。** 三个 EMGFormer 使用 12,000 点窗口的 2,792 个测试手样本；已完成的 ViT-S 使用 7,790 点窗口的 4,306 个唯一手样本、8,612 次原生重复消费。实际读取选定测试预测后，按全局中心、手、split 和私有用户键连接，三个 EMG 点网格与视觉网格的交集均为 **0**。这不是标签文件版本不同：两者共有的十份冻结标签、腕角、split 和元数据内容身份完全相同。

原始 S/M/L 点报告、完整轨迹报告及视觉报告保持原字节。6000 点目标、轨迹对齐中心原6255和视觉窗口内3895点分别保留，不能交换分数或仅凭样本数声明同一人口。本核验没有新增预测、训练更新、检查点选优或模型家族。

## 官方入口与真实完整网格

固定官方版本为 `f7cbd048b7092139852270491d3859cd472e0311`。完整执行未改动的 [`egoemg/center_frame.py`](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/center_frame.py) 的 `collect_val_centers`，使用真实41段数据，并在前后核对官方源码、manifest、episode_index、frame_split及读取的原始包SHA与五项文件属性。

| 清单 | Gesture手请求 | User手请求 | Both手请求 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 官方7790参考网格 | 2324 | 1660 | 322 | 4306 |
| 统一要求可容纳12000点窗口 | 2314 | 1646 | 322 | **4282** |

后者是 `required_window_length=12000` 的实际结果：2141个双手窗口、40段，统一排除24个无法容纳最大模型窗口的手请求。它完全包含于已有视觉预测，缺少旧视觉预测的请求为0。选择12000来自模型窗口长度，与测试误差无关；边界和官方函数对episode最后行的处理保持原样。**这尚不是新的4282请求模型评价，也不证明作者使用了这一共同人口。**

第二个独立CPU进程从 `metadata.npz` 的episode边界构造算术oracle，实读split和episode边界，并重新核对26份来源，所得两个完整网格及逐split计数均与真实官方调用一致。它还用不依赖用户键的“中心＋手”连接，独立确认旧网格交集为0，三个旧点模型的顺序、目标和掩码完全相同。两个审计、输入探针和独立读取进程均实际退出。

## 取样路径的真实工程探针

在共同网格首尾分别读取左右手，共四个真实输入。现有已核验的legacy EMG view调用官方 `_getitem_center_supervised_emg`，得到有限的float32 `8×12000`输入，22角目标及bool掩码与同中心的实际视觉记录逐字节相同。它不需要图像，也没有模型前向。

同一纯EMG view调用统一helper使用的融合分支 `_getitem_center_supervised`，四次均报 `KeyError: 'image_head_frame_index'`。这是**当前纯EMG适配视图上的实测接口问题**，不扩大为所有官方数据版本均失败的结论。后续适配应调用已有官方纯EMG取样分支，保留归一化、标签、mask和raw-token中点预测；取样错误必须报错，不能依赖helper的捕获异常后静默跳过来形成一个较小表格。四探针不替代完整4282请求输入验证或预测。

## 证据与边界

- [完整原始目标／身份连接](targets.json) · [两次官方完整网格调用](grid.json) · [四个真实输入探针](input-probe.json)
- [独立来源／网格核验及进程关闭](independent-verifier.json) · [来源绑定](source-bindings.json) · [产物清单](completion.json)
- 实际执行的CPU worker源码：[目标](targets-source.py)、[官方网格](grid-source.py)、[输入探针](input-probe-source.py)、[独立核验](independent-verifier-source.py)。各文件完整保留实际执行的字符串字节，不是修改后的模型实现。
- [重放配置](execution-configs.json)来自未变化的运行记录；配置包含已经公开匿名化的报告和原始文件SHA，不含私有用户键。原始预测包及原数据仍在服务器私有目录，不能仅下载此目录便重放原始数据核验。worker从stdin读取对应配置JSON，在既有服务器环境执行，不能把其成功解释为新模型评价成功。
- [原生200轮点结果](../original-emgformer/small/README.md) · [已完成的完整轨迹结果](../native-emgformer-trajectory/README.md) · [原始视觉结果](../original-vision/vit-small/README.md) · [后续同帧评价计划](../../docs/superpowers/plans/2026-11-02-native-center-reference.md)

待完成：冻结共同参考清单与全部输入；S/M/L原200轮、原验证BEST在该清单上的FP32原生raw-token中点评价和新进程重放；同视觉目标逐请求证明；逐用户mean±SD、总体样本MAE及独立退出后报告。BF16历史结果保持其身份。旧版数据、生成标签、整episode预过滤、单rank及作者人口/混合来源的缺口仍在；Table 10与全论文均未完成。

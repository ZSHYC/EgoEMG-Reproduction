# Classic：验证集预测复算与报告准备

新增 `egoemg_repro.classic_emg_report` 提供原始22角算术、逐批输入／预测摘要和训练结束后的完整验证集复算入口。**目前没有实际GPU验证集复算、冻结profile或新的最终测试成绩。** VEMG的完整训练仍在进行；NeuroPose的旧验收失败保持失败。下列实现及CPU检查不构成模型精度或全文完成证据。

[修复后实际CPU证据](../results/classic-report-validation-cpu.md)：定向110项通过，固定80文件完整组1,250 passed／3真实CUDA skip，轻量762 passed／444依赖skip。独立静态审查的3项发布问题均有先失败后通过的回归；最末原始FD仅作不抛错的资源清理，系统关闭错误可能留下直至进程退出的描述符。本次源码的托管CI尚未观察。

## 固定评价语义

| 项目 | VEMG2Pose | NeuroPose |
| --- | ---: | ---: |
| 原始输入长度 | 7,790 | 7,790 |
| 左／右上下文 | 1,790／0 | 0／0 |
| 原tuple输出长度 | 6,000 | 7,790 |
| 原窗口中心在输出中的位置 | **2,105** | **3,895** |
| 固定单卡batch | 600 | 256 |
| 完整验证手样本／末批 | 1,576／376 | 1,576／40 |

消费原模型实际返回的 `(prediction, target, generated_mask)`，不再次裁剪或插值。目标为20个手指角和2个腕角，统一弧度；原生成标签mask保持 `B×T`，仅在报告算术中广播至22角。轨迹、原窗口中心、逐角、左右手、split、用户、手指和proximal/mid/distal分组分别保留误差和与有效分母，聚合后才转换为度。

报告MAE按有效角度加权。原生 `test_loss`、FK及其他原指标保留原始batch归约语义，不能用报告角度MAE替换。此模块尚未实现原生测试结果的报告闭合与公开导出。

## 两次独立的完整验证集预测

入口先重新验证真实formal训练已达到原100轮或有限patience50终止、完整验证选模及所选best。工程短跑、外部终止、调用者提供的通过字典不能进入校准。原训练和已有sidecar保持只读；输出目录必须是新的独立目录。

以下命令须分别启动新的Python进程，并使用已配置好的真实服务器运行时、同一设备和冻结源码。`TRAIN_RUN`指向真正终止的formal训练目录；`CALIBRATION_ROOT`是新的独立校准目录：

```bash
mkdir -p "$CALIBRATION_ROOT"

python -m egoemg_repro.classic_emg_report validation-pass \
  --run "$TRAIN_RUN" --output "$CALIBRATION_ROOT/validation-reference"

python -m egoemg_repro.classic_emg_report validation-pass \
  --run "$TRAIN_RUN" --output "$CALIBRATION_ROOT/validation-replay" \
  --reference "$CALIBRATION_ROOT/validation-reference"

python -m egoemg_repro.classic_emg_report freeze-validation \
  --reference "$CALIBRATION_ROOT/validation-reference" \
  --replay "$CALIBRATION_ROOT/validation-replay" \
  --output "$CALIBRATION_ROOT/validation-profile.json"

python -m egoemg_repro.classic_emg_report verify-validation \
  --profile "$CALIBRATION_ROOT/validation-profile.json"
```

真实运行仍为单卡、workers0、原batch、16-mixed及原算子；模型以eval／inference模式调用原forward。每批核对所选参数、buffers和scaler不变且没有梯度。验证集shuffleFalse、drop_lastFalse、专用loader seed46。每个准备和消费的批次记录目标、2D mask、样本顺序、EMG、role/split/subject及原dtype预测的独立摘要链；原生目标／mask／样本链编码保持兼容。

第二次运行必须有不同的实际PID/start_ticks，并精确重现每个完整批次的输入和原dtype预测摘要。**这里采用精确相等，没有调用者容差，也不复用工程验收的epsilon规则。** 后端可以因此未通过；失败要保留，不能改变算子、精度或在观察测试结果后放宽门槛。

## 冻结与重新核验

profile冻结前、读回后及关闭输出句柄后重查原始证据；新文件不覆盖已有文件，晚期错误通过保留的目录句柄撤销本次文件，目录被移动或替换时保护外来文件。

`verify-validation`重新打开两次原始sidecar、训练终态、所选状态、数据队列、源码和框架摘要，并重建逐批链及整个profile。单独改写 `passed`、profile摘要、样本数、状态字段或选择信息都不能替代这些证据。profile仅表示**这两次完整验证预测观察到精确一致**，不是普遍确定性、完整训练轨迹一致或科研完成证明。

sidecar和profile含私有训练路径及数据来源信息，应留在私有输出目录；不应直接复制到公开仓库。

## 尚待完成的后续链路

原生测试启动前的父子进程profile绑定、同一所选best的原生完整测试、测试预测参考与另一次独立复算、脱敏JSON／Markdown及最后的科学完成标记仍待实现和实际执行。当前三个CLI子命令均不运行test，不产生科学完成标记。

预定测试链包含原生测试、预测参考、独立预测复算三次完整遍历；不能以两次验证校准代替测试，也不能从测试表现重新选模。原生FK依赖batch，后续必须保留原测试batch；验证profile本身不允许绕过尚未完成的启动绑定。

CPU测试中的手算和JSON读取正例仅验证机制，部分读取fixture显式替换终止状态权威，不构成真实CUDA、完整队列或GPU校准通过。全文其他模型、数据、标注和表图仍按[全论文清单](full-paper-status.md)验收。

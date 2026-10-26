# Classic：验证集复算与绑定后的原生测试

新增 `egoemg_repro.classic_emg_report` 提供原始22角算术、逐批输入／预测摘要、训练结束后的完整验证集复算和绑定后的原生测试入口。**目前没有实际GPU验证集复算、冻结profile或新的最终测试成绩。** VEMG的完整训练仍在进行；NeuroPose的旧验收失败保持失败。下列实现及CPU检查不构成模型精度或全文完成证据。

[验证阶段CPU证据](../results/classic-report-validation-cpu.md)保留原110项定向、1,250项完整与762项轻量结果。该阶段托管CI曾因`/tmp` tmpfs容量不足失败，后续仅将pytest临时目录移至工作磁盘；[实际修复运行](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40347752496)官方CPU1,250 passed／3真实CUDA skip、轻量762 passed／444依赖skip。该CI不覆盖随后新增的原生测试启动绑定。

[原生绑定阶段修后CPU证据](../results/classic-report-native-gate-cpu.md)为定向211 passed、固定80成员完整1,351 passed／3真实CUDA skip、轻量806 passed／501依赖skip。独立审查发现的同目录文件替换回滚问题已用4项先失败回归修正；未做第二次独立审查，新的托管CI尚未观察。

## 固定评价语义

| 项目 | VEMG2Pose | NeuroPose |
| --- | ---: | ---: |
| 原始输入长度 | 7,790 | 7,790 |
| 左／右上下文 | 1,790／0 | 0／0 |
| 原tuple输出长度 | 6,000 | 7,790 |
| 原窗口中心在输出中的位置 | **2,105** | **3,895** |
| 固定单卡batch | 600 | 256 |
| 完整验证手样本／末批 | 1,576／376 | 1,576／40 |
| 完整测试手样本／末批 | 4,306／106 | 4,306／210 |

消费原模型实际返回的 `(prediction, target, generated_mask)`，不再次裁剪或插值。目标为20个手指角和2个腕角，统一弧度；原生成标签mask保持 `B×T`，仅在报告算术中广播至22角。轨迹、原窗口中心、逐角、左右手、split、用户、手指和proximal/mid/distal分组分别保留误差和与有效分母，聚合后才转换为度。

报告MAE按有效角度加权。原生 `test_loss`、FK及其他原指标保留原始batch归约语义，不能用报告角度MAE替换。原生测试入口沿用未修改的evaluator；独立预测报告的最终闭合与公开导出尚欠。

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

## 原生测试启动绑定

只有真实训练终止且上述完整验证profile已冻结并重新核验后，才能运行下列入口。`TEST_ROOT`必须与训练和两次验证sidecar分离；native和launch分别使用新的、不相互嵌套的目录：

```bash
mkdir -p "$TEST_ROOT"

python -m egoemg_repro.classic_emg_report native-test \
  --run "$TRAIN_RUN" \
  --validation-profile "$CALIBRATION_ROOT/validation-profile.json" \
  --output "$TEST_ROOT/native" --launch-output "$TEST_ROOT/native-launch"

python -m egoemg_repro.classic_emg_report verify-native-test \
  --launch-output "$TEST_ROOT/native-launch"
```

父进程在创建child之前完成完整profile／队列／所选best准入。随后用真实`Popen`取得child的PID、`/proc`启动tick和继承pipe身份，将实际命令、父子身份、所选状态、源码、运行时、profile与原测试batch写入并fsync `launch.json`，读回后才通过pipe发送固定33字节的GO与摘要并关闭写端。child在导入评价运行时之前等待此消息，核对自身／仍存活的父进程／原始命令／pipe／manifest摘要，独立重开完整证据，再沿用原`run_selected_best_test`。

父进程要求child成功退出，重新读取原五份测试产物，核对role-best快照与seal、完整4,306手的顺序／批量、原目标／mask／样本摘要、原生损失与模型／scaler不变，并绑定到实际放行的child。`verify-native-test`从原始文件重新构建整条绑定；单独提供通过字典、只写一个结果摘要、使用last代替best或补写一个未绑定的profile均不满足此入口。

`progress.json`仅表示启动状态。最终`result.json`和`status.json`均为一次性不覆盖发布，在公开文件名之前保留各自创建时的FD／inode。末期写入、源检查或目录关闭失败会撤销本次文件，包括已移走的本次inode；同一目录里后来替换的外来文件保持原字节。最后保留的原始FD只做不抛错清理，关闭错误可能留下直至进程退出的描述符。该机制不改变冻结的训练／evaluator／artifact源码。

完成的launch仍明确为`scientific_eligible=false`。它只支持已绑定的原生测试；独立测试预测和公开科学报告另有完成条件。此阶段CPU的真实pipe子进程证明消息顺序，实际父进程故障探针则刻意以17退出；这些检查均没有执行或模拟CUDA正例。

## 尚待完成的后续链路

原生测试启动绑定及raw读取已实现并有CPU检查，实际GPU校准／profile／原生完整测试尚未执行。测试预测参考与另一次独立复算、只读状态API、脱敏JSON／Markdown及最后的科学完成标记仍待实现和实际执行。`native-test`会在严格准入后运行原生test；验证子命令均不运行test，任何当前入口都不产生科学完成标记。

预定测试链包含原生测试、预测参考、独立预测复算三次完整遍历；不能以两次验证校准代替测试，也不能从测试表现重新选模。原生FK依赖batch，后续必须保留原测试batch；验证profile本身不允许绕过尚未完成的启动绑定。

CPU测试中的手算和JSON读取正例仅验证机制，部分读取fixture显式替换终止状态权威，不构成真实CUDA、完整队列或GPU校准通过。全文其他模型、数据、标注和表图仍按[全论文清单](full-paper-status.md)验收。

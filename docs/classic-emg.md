# vemg2pose 与 NeuroPose：当前实现范围

当前已接入固定官方模型、原始数据变换和真实 Lightning CPU 训练及 epoch 边界恢复。**正式 GPU 训练、完整验证选模后的冻结测试及科研结果导出尚未完成。** 这些工程检查不能填入论文性能表。

最新状态：6adc源码的VEMG连续两轮、prefix单轮及fresh恢复三条GPU路径均成功，汇总验收失败保留。只读诊断确认模型配置身份的唯一表示差异是原tuple与JSON读回list，其他容量、队列、runtime、recipe及binding吻合；现改用完整JSON metadata摘要比较，原配置/通道顺序/参数数变化仍拒绝，typed checkpoint摘要不变。另原生恢复后若下一轮没有新best，ModelCheckpoint会保存current_score=None；新增真实stock Trainer恢复回归已复现，last仅在best来自早先完整epoch时允许此值，best及刚选中的last仍拒绝None，其他选择字段继续严格验证。

两项修正后的formal/artifact/evaluator定向161 passed/1实际GPU skip；77文件整体1125 passed/3实际GPU skip（281.02s），轻量739 passed/389依赖skip（21.79s）。新源码尚未取得GPU验收。此前6adc三路径的只读后续诊断实测最终模型/optimizer浮点状态不同；首步后预测最大差6.1035e-5rad、下一步后9.765625e-4rad，scheduler/scaler和执行计数一致。全局Python/NumPy/Torch/CUDA RNG与sampler/val-loader状态一致，train-loader状态在resumed端多一次原生int64 base-seed draw。当前逐位比较门槛仍拒绝这组数据，尚须独立解决原生生命周期及预测试数值协议；没有把这些差异改写为GPU通过。

最新修正组包含[终止后 selected-best 测试入口](../egoemg_repro/classic_emg_evaluate.py)及原生 AdamW 配置校验。77文件冻结组实际完整CPU **1118 passed/3 skipped**（271.95s），轻量 **739 passed/382依赖skip**（21.40s），源码和原七项结果均核对不变。三个完整环境skip仍是实际CUDA检查，不能据CPU通过授予GPU或科研资格。

首次最终源码GPU工程运行在两次训练尝试和完整1,576手验证后，于首轮检查点保存失败。原因是官方Hydra保留`ListConfig`形式的betas，校验器临时期望参考却用了Torch默认tuple；没有显存耗尽或保存成功。修正只给临时参考显式传入相同数值的列表，不改真实优化器、checkpoint或typed digest；两原配方真实helper的旧拒绝/新接纳及错误值拒绝均独立核验。旧失败保留，修改后的源码集合必须在新目录重新实际GPU验收。

修后新集合的实际GPU运行已完成VEMG连续两轮和单轮prefix，保存成功；第三个新进程在首次forward前的完整状态恢复核对失败。该失败及前两阶段原checkpoint保留，尚未执行Neuro阶段或证明最终源码验收。恢复差异另行调查，不能据两阶段成功开放完整训练。

该恢复拒绝现已用真实CPU原生Trainer复现：ModelCheckpoint把`current_score`写入checkpoint，但原生`load_state_dict`不恢复它，fresh callback保留None。实际GPU prefix也确实保存非空score；此前没有记录GPU resumed逐组件快照，不能称其余GPU组件已逐项排除。修正只在用于恢复比较的副本中将这一非恢复字段规范为None，不改原callback或checkpoint；原保存校验仍严格验证真实current_score，best/kth与EarlyStopping各字段继续比较。新增原生恢复测试真实RED→GREEN，formal/artifact/evaluator定向155 passed/1真实GPU skip；新77文件组完整1119 passed/3真实GPU skip（263.56s），轻量739 passed/383依赖skip（20.74s）。新formal源码6adcadfe与artifact187b的GPU验收另行启动，旧失败不追溯改成通过。

测试入口只接纳真实完整预算或有限patience正常终止的训练，严格载入验证选中的best及scaler，使用原`Trainer.test/test_step/_step`遍历4,306手测试人口。它不优化参数、不重选模型，结果仅为独立测试sidecar；原调用探针确认selected参数和buffer确实载入，失败回滚只撤销本次拥有的目录状态。33项定向CPU检查和独立复审通过，尚无完整训练后的真实GPU测试。输入/目标/mask指纹不冒称预测指纹，独立预测报告与复算仍欠。

2026-10-26新增[正式训练入口](../egoemg_repro/classic_emg_formal.py)，已通过CPU契约检查、完整集成及独立审查，实际源码SHA为`5285f6c6bdcb310162fce3e1973b1ff258ae9c446c3888f47b291f04ee9368c1`。它严格读取原数值检查、三梯级容量、完整队列及最终源码GPU验收产物；调用方的通过标记或旧工程结果不能开启完整训练。当前尚无该入口的三进程GPU验收bundle，不能据此称正式GPU训练已开始。

入口保留原模型、自动AdamW、T_max100、指尖合计0.02、val_loss/min和patience50；正式预算100轮、完整train/val，同批量600/256。实际单卡/workers0与作者六卡/12workers声明分开记录。独立工程模式仅取前两个train batch、完整1,576手val，比较连续两轮与一轮后新进程恢复；它不能作为完整训练或最终测试。完整typed状态/RNG恢复与连续GPU数值比较分别核验，不保证非确定性后端的逐位训练轨迹。

新增入口定向CPU检查为48 passed/1项实际CUDA验收缺失skip。曾发现收尾close异常可遗留成功标记的问题，现已用原两故障探针及四项实际资源关闭回归修复并独立复审；原失败记录保留。包含该入口及CI修复的75文件冻结组，本机完整1076 passed/3 skipped、轻量727 passed/352依赖skip；两个CUDA限定正例和本入口GPUbundle用例均未由CPU执行。完整训练后的所选best测试与独立报告仍须后续实现和实际运行。

| 项目 | vemg2pose | NeuroPose |
| --- | --- | --- |
| 真实参数数 | 5,963,948 | 6,354,969 |
| 实际输入通道 | 目标手 8 通道 | 目标手放入 16 槽 |
| 输入长度 | 7,790 | 7,790 |
| 原模型左上下文 | 1,790 | 0 |
| 原 tuple 的预测/目标长度 | 6,000 | 7,790 |
| 实际角度数 | 22 | 22 |

模型保留官方返回的 `(prediction, target, generated_valid_mask)`，其中 mask 为二维，不再裁剪、插值或替换成旧复现层的逐角 mask。固定官方 scalar 归一化、NeuroPose 先填原始零槽再归一化、原 rotation/with_aug 顺序均经过完整官方 reader/Compose 的实际对照。VEMG 的配置声明 16 槽但卷积要求 8 通道，以及原 reader 漏选 legacy filtered 字段，两项兼容裁决都记录在身份中。

新分区是独立的 `window_length=7790 / train_stride=780 / eval_stride=7790`、验证比例 0.12；不是已有四模型的 12,000 点队列。原始标签非有限值会明确拒绝，不悄悄补零。模型/数据接口和两份基础配方见 [classic_emg.py](../egoemg_repro/classic_emg.py)、[classic_emg_data.py](../egoemg_repro/classic_emg_data.py)、[VEMG 配置](../configs/vemg2pose.json)及 [NeuroPose 配置](../configs/neuropose.json)。

## 已验证的训练行为

[classic_emg_train.py](../egoemg_repro/classic_emg_train.py)直接构造原 `EmgPredictionModule`，使用原自动优化、AdamW、cosine、clip0、LRMonitor、ModelCheckpoint 和 EarlyStopping；保留原 raw loss 双别名及有效指尖系数 0.02。选模量是原 Lightning 按 batch-size 加权的 `val_loss/min`，不是 degree MAE 或全局有效角数量加权。原 cosine T_max100、EarlyStopping patience50 在缩短的工程运行中仍保持原值。CSVLogger 是明确记录的日志适配。

两项真实模型各以小 memmap 完成两轮连续运行，并与暂停一轮后只携带 last、搬迁到新目录的离线恢复对照。每条完整路径有四次实际优化更新和两次完整**夹具**验证；模型、AdamW、scheduler、loop、callback、增强输入以及 Python/NumPy/Torch/loader/sampler RNG 均匹配。新进程实际读取自己的 JSON 身份和 checkpoint 后可严格恢复。检查点保存携带不递归的 selected-best，支持离线重建原 best。

该子阶段的 35 项定向测试实际通过；无 Torch 层为 10 passed / 25 个运行时依赖 skip。独立审查另外执行五项关键恢复/破损状态测试，并实际注入最终状态 rename 后的 fsync 失败及低容量，确认不会遗留成功标志或破坏 checkpoint、源数据和邻目录。曾发现的 JSON tuple/list 身份及失败终态问题已修复，原失败证据保留。

最终 CPU 源码 SHA256 为 `52b1ce655f3c4e107e3d809e403cd507fbcfd655f7ac9ea7bc0b9a592b18b0d8`，测试为 `6ecc55104b24fdbcc8379e50897152f1b94a695da683d6ed80ba52a14c176875`。这些摘要绑定实际被测文件；后续文档提交不能替换真实执行身份。

## 实际完整队列检查

[新队列CPU证据](../results/classic-cohort-cpu-engineering.md)已完成两配方实际准入及各自完整重算。train/val/test分别为50,857/788/2,153窗口，即101,714/1,576/4,306单手样本；12数组整文件身份覆盖66,161,725源行，finite扫描覆盖冻结足迹内65,046,370行。固定布局/归一化及22D单位转换均finite，个别空标签样本不删除；没有导入Torch或计算模型误差。这是CPU数据准入，其receipt的CUDA/容量字段保持false；后续实际GPU数值与容量证据独立发布，不追改该CPU身份。

另新增独立[CUDA限定检查入口](../egoemg_repro/classic_emg_cuda_check.py)，只执行固定四个train手样本和两个val手样本的prefix/resume/verify，不开放正式训练。最初强确定性版本在真实NeuroPose反向中因Torch2.2.2+cu118的linear upsample不支持该模式而退出失败，原记录保留。后续固定canonical_backend_nondeterministic_v1，使用原算子、默认GradScaler和16-mixed，明确不保证位级轨迹一致；这份新源码已完成[实际GPU数值检查](../results/classic-cuda-numeric-engineering.md)：各8次尝试、7次真实optimizer执行及1次注入Inf跳步，默认scale65536→32768，新进程完整typed状态/RNG恢复一致；只证明batch1固定子集，未验证后续轨迹逐位一致。

包含新数据准入、CUDA限定入口、预取和派生TaskA的66文件冻结组，本机统一检查为879 passed/3 skipped（284.01s），轻量层609 passed/273依赖skip（20.12s）。其范围排除仍在开发的正式artifact与派生worker，两个实际CUDA正例未由CPU执行；后续托管CI须按新提交另验。

## 原批量真实容量已通过

[六梯级实际检查](../results/classic-cuda-capacity-engineering.md)已完成：VEMG32→128→600、Neuro16→64→256，每个新进程两次真实更新和完整1,576手validation。无自然overflow，目标600/256的train与val峰reserved分别13.488/16.928GiB，均满足85%总量及至少4GiB剩余门槛。原train/val同B、原batch-union FK及最后partial batch保持；这是单卡/workers0短程工程容量，不是完整训练或原六rank等价。

实际工具现按原字节公开为[tools/classic_capacity.py](../tools/classic_capacity.py)，SHA仍为b890de97c1ebafcb5ddb88dde91785a40a439dce904e919e433f82f8389f7d87；只改变发布位置，未重写当时执行身份。[公开定向测试](../tests/test_classic_emg_capacity.py)与私有17项语义一致。

## 正式状态与日志契约

新增[classic_emg_artifacts.py](../egoemg_repro/classic_emg_artifacts.py)实现有界追加journal、typed checkpoint/非递归carried-best、原optimizer/scheduler/callback/loop/scaler/RNG检查与独立selected-best读取。64项定向CPU契约测试和独立原故障回归通过；真实stockPL两轮改善→不改善证明ModelCheckpoint.current_score保留已选minimum，本模块按原语义校验，不刷新原生callback状态来迁就校验器。

这些是状态/存储契约，尚无该最终formal入口真实GPU或科研训练结论。CPU RNG检查使用隔离Generator实际恢复，不改全局随机流；目录获取明确采用可信parent、私有随机staging和held-fd/NOREPLACE假设。最终文件与namespace检查是当时的准入见证，不承诺返回后不受同UID恶意修改。

新增公开capacity工具、worker与原生callback语义修正后的72文件冻结组，源码快照SHA9f879422b483206c0133b1bd095ce405503d14f962b5b183abfec1ea9f1d005d，本机完整CPU **1025 passed/3 skipped**（271.86s，两个真实CUDA正例及一个既有ViT-L内存准入skip），轻量 **690 passed/338依赖skip**（23.83s）。最终文件SHA逐一复核不变；该组明确排除仍在实现的formal runner及其测试，托管CI待对应发布提交实读。

## 仍需完成

- 新正式训练入口的最终源码实际GPU验收；已完成的batch1数值及600/256容量不能自动授权新增runner。
- 新正式入口的状态恢复/下一步数值检查，以及单卡/原六rank全局批量和NeuroPose BatchNorm差异记录；不承诺尚未证明的位级轨迹。
- 原 max100、patience50 的完整训练/每轮完整验证，以及按有限 val_loss 选定 best 后执行冻结 test。100 是上限，不能为了凑轮数删除早停。
- 严格正式导出、逐关节/手/split/用户统计及独立复算。旧 legacy 的 0.01 目标不能改名为官方合成的 0.02。

原训练入口只开放 `--engineering --device cpu`，独立CUDA入口仅供限定验收；正式入口会明确拒绝，不自动降级为CPU。已验证的恢复范围为 workers0 的 epoch 边界，没有多 worker 或中途 batch 的位级续序承诺。容量保护、源码/配置/分区/数据/归一化/检查点身份检查继续生效。[实施计划](superpowers/plans/2026-10-26-classic-emg.md)中的整个 Task3 和 Task4 保持未完成。

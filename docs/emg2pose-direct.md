# EMG2Pose：原始磁盘数据上的原生模型入口

该入口消费[原始数据准备流程](emg2pose-original-disk.md)实际完成的普通memmap文件。使用原始`Emg2PoseDataset`类、原模型factory和训练配置；没有TAR描述符、封印、虚拟文件或新数据集子类。完整源仍在下载。原CPU构造基础和新训练恢复入口的资格分别记录；仍不能填写论文Tables11/13。

[实际证据](../results/emg2pose-direct-native-runtime-engineering.json)记录37项新检查与145项原入口/源回归：本机182通过/23warning/53.85秒，服务器182通过/23warning/69.16秒；完整轻量1,355通过/727可选框架或工具跳过/1warning/78.57秒，全部exit0。单独真实失败回归发现原canonicalizer会丢弃另一个格式的私有context；修正只作用于新入口，在seed/model构造前拒绝该context，没有改原validator或全局对象。

`emg2pose_direct_source.open_source`要求另行冻结的pipeline完成标记SHA，绑定准备、回收、获取、转换、源清单、CSV、归一化、固定官方源码以及四个实际字段文件。它调用一次未修改的`emg2pose_runtime.validate_completed_data`完整扫描字段，随后只有inode/size/mtime/ctime与相关配置均保持当前时才可使用缓存的内容身份；关闭或修改源后拒绝继续。公开属性返回身份副本，不能通过修改返回字典把mini提升为full。

`emg2pose_direct_runtime`提供`compose_config`、`build_runtime`、`runtime_identity`、`reload_runtime`和`close_data_module`。精确的原factory代码只在私有globals副本中替换源准入函数；原模块、网络、数据集、超参和变换不改。初始化沿用已声明的先调用原`setup_runtime`再构造模型策略，是对上游启动顺序的显式修正。原生S/M/L保留16通道/20角、7,790点输入、510点左上下文，实际参数为3,481,636/6,638,884/16,342,308；AdamW/cosine250及原损失/采样配置保持一致。

完整真实mini另外实跑27.59秒exit0，深度字段准入0.98秒且发生在导入Torch之前。全部526窗口的输入/目标/mask/元数据摘要与独立原参考相同，有效window mask为3,493,232。三模型初始权重、实际CPU前向、原优化器/调度器配置和严格权重重载与原入口逐张量精确一致；输出形状为`[1,20,7280]`、`[1,20,7280]`、`[1,7280]`。所有拥有的普通memmap视图关闭，源关闭后不能再准入，两个实际PID/start拥有者独立确认退出。真实运行没有注入资源观测或使用CUDA。

mini只有原val角色。三个独立角色的微型工程用例另检查了原训练增强/jitter在恢复Python/NumPy/Torch RNG后的精确一致。这不是mini完整训练，更不是完整源训练；本阶段没有优化器更新，也没有优化器/调度器/循环/RNG的训练状态恢复资格。后续仍须原批量320/12workers/bf16的实际GPU恢复、完整250轮fit/选best、FP32最终评价、官方字面选择及文件行角色隔离对照，最后生成可核验的科学报告。

## 原生训练状态恢复入口：CPU检查通过，真实mini验收运行中

[CPU恢复证据](../results/emg2pose-direct-native-recovery-cpu-engineering.json)记录本机65项/126.00秒、服务器65项/152.81秒，各27warning且actual exit0；完整轻量1,355通过/728可选依赖或工具跳过/1warning89.66秒actual exit0。15项新增检查先实际因缺少模块失败，再通过。CPU工程夹具保留原格式独立三角色，实际恢复测试使用其中完整1窗口val：S/M/L各两次更新、独立新进程完整重推、模型/AdamW/cosine/回调/循环/RNG六项精确恢复，再实际更新一次到step3。CPU为batch1/workers0/FP32，单元夹具容量观测被隔离，不能当作真实容量或GPU资格。

`emg2pose_direct_recovery`提供相同的`run_recovery_phase`和受管理checkpoint IO；算法代码对象与已核验旧恢复入口完全相同，只在私有globals副本绑定普通文件source/runtime、新格式及同时覆盖复用/新增文件的producer。原模块全局、原模型/数据集/训练循环不变。CLI要求普通`--source-root`和另行冻结的`--expected-source-completion-sha256`，拒绝旧archive描述符环境且不关闭调用者的fd。已有输出、不同模型/来源/恢复格式、损坏或未冻结的prefix、同进程恢复、改变预测及不完整最终发布均有真实拒绝检查；失败与尝试标记保留，不自动重试。

真实官方完整mini的12个受管理CPU/GPU阶段已经实际一次启动，拥有者PID/start另核对活动；此快照确认small CPU前缀与新进程恢复均完成：两次更新、两轮完整526窗口验证，新进程526窗口精确复推、六状态恢复、再一次更新与全部526窗口验证；medium正在执行，large与GPU阶段仍欠。原GPU验收必须实际batch320/12worker/bf16、完整320+206验证批量及六状态新进程恢复，并确认所有拥有的消费者退出后才能宣称通过。mini复用val作为工程训练夹具，没有独立科学验证或连续训练轨迹结论；原始内存后端的GPU成绩不转授给这个普通文件入口。

完整源/独立train-val正式fit/原250轮预算/验证选best/FP32最终测试、官方字面选择与角色隔离对照/科学控制器报告仍欠，全文目标不变。

## 完整真实mini CPU/GPU恢复：实际验收完成

[完整恢复证据](../results/emg2pose-direct-native-recovery-engineering.json)记录12个受管理阶段、702.24秒actual exit0。S/M/L均完成CPU与实际A800 GPU前缀/新进程恢复：每模型两次实际更新、两轮全部526窗口验证，新进程完整526窗口输入/预测/顺序/指标精确复推，六类状态精确恢复，再一次真实更新至step3并完整验证。GPU实际batch320/12workers/bf16、验证320+206，原clip1/AdamW/cosine250/top3+last/noearlystop和TensorBoard保持；CPU为1/0/FP32。全部24个worker/guardian实际PID/start及外层两个拥有者独立确认退出，原prefix字节与全部冻结源码未变，资源观测未注入，父进程未导入Torch。

这是普通原文件入口自己的实际资格，没有转用旧内存后端结果。上节“运行中”为此前CPU发布时快照，现由此完成证据更新；mini复用val工程训练、连续轨迹和科学结果的边界仍保留。接下来独立角色fit有自己的完整epoch/GPU选best资格，不能用本恢复检查代替。

## 独立角色原生fit入口：CPU核验完成，GPU完整epoch检查运行中

[CPU证据](../results/emg2pose-direct-native-fit-cpu-engineering.json)包含22项新检查及原fit/recovery/runtime回归：本机105通过/39warning215.91秒、服务器105通过/39warning285.08秒；完整轻量1,355通过/729可选跳过/1warning84.77秒，全部actual exit0。21项原先因缺少新模块实际失败，补充真实val-only普通源拒绝检查后22项通过。

`emg2pose_direct_train`精确复用已有fit函数和每个`FitWitness`方法代码对象，在同一私有globals与新类绑定direct source/runtime/recovery，新格式为`emg2pose_native_direct_fit_v1`。原类/函数全局不改。CLI只接受普通源与冻结完成SHA，不消费archive fd。明确的三文件/25,710行合成三角色工程夹具实际执行两完整CPU轮次/32次更新、原Sampler每轮16次消费（1唯一train锚点）、每轮完整1窗口val，按实际验证最小值选best。独立新进程从完整epoch恢复六状态后完成第二轮；last/best/journal/假full/旧格式/错模型/冻结SHA/锁和完整预算均有实际拒绝检查。

full仍要求真正完整25,253源、train17,136/val1,950/test6,167、462,824,048,640原始字节，并读取这个新入口自己实际完成的CUDA工程资格。原250轮/320batch/12worker/bf16/clip1/cosine250/top3+last/noearlystop/全量train-val不改。CPU两轮不是full或科学结果；fit本身也不会声明最终test完成。

在上节真实mini CPU/GPU恢复实际完成后，新独立冻结GPU检查已一次启动：通过真实受管理loopback aria2获取显式合成工程TAR、原始文件准备/转换/普通source准入，随后运行三模型独立角色完整两epoch/新进程恢复与最后只读选best核验；当前源阶段已完成，small前缀worker/guardian实际活动，尚未取得此fit的GPU资格。没有改动活动full下载、准备控制器或DINO模型，原始内存fit资格不转授。之后仍需完整源250轮及FP32最终评价/报告。

## 独立角色GPU完整epoch/选best检查：实际完成

[实际GPU fit证据](../results/emg2pose-direct-native-fit-engineering.json)为9个受管理阶段/213.35秒actual exit0，2026-10-27T19:43:58.486292UTC终态。这是显式合成3文件/25,710行三角色源：真实aria2 loopback接收、原源准备/转换/普通数据准入后，S/M/L各完成原batch320/12workers/bf16两完整工程epoch，共32次实际更新。原Sampler每轮5,120次消费（S仅1、M/L仅2个唯一train锚点）、完整1窗口val分列；不能把补齐消费当独立数据。

每模型第1轮暂停后，新进程在原完整epoch边界精确恢复六状态并完成第2轮。真正的验证min选best、终态last、冻结history前缀与全部epoch日志由另外CPU只读阶段重开核验，实际profile/原生循环源码摘要保存。独立收集再次核对所有检查点与源码SHA、18个worker/guardian和外层两个拥有者PID/start已退出。全过程普通原文件、无archive fd/封印、无资源观测注入，父进程无Torch/CUDA；CPU“GPU运行中”为历史快照，现由此更新。

该fit入口现在具备自己的实际GPU完整epoch资格；旧内存资格没有转授。full依然需要真正全部25,253来源与原250轮预算；这里只完成工程两轮，没有新模型精度、完整源、FP32最终评价/容量或科学controller/report资格。下一步是普通原文件上验证选择best的FP32评价、官方字面与角色隔离对照、完整科学报告链，并在全量源真正完成后进行三个完整实验。

## 原始普通文件评价数据：全选窗实测完成

[实际数据准入证据](../results/emg2pose-direct-evaluation-data-engineering.json)包含36新检查+原数据准入/运行时回归：本机104/178.46秒、服务器104/247.36秒，各8warning且actual exit0；完整轻量1,355/730可选跳过/1warning77.52秒actual exit0。新`emg2pose_direct_evaluation_data`保留官方`Emg2PoseDataset`精确类型及原group/frame/loader算法，在私有绑定中关闭原NPZ临时句柄；普通字段路径/dtype/shape和已打开的只读map绑定到实际源，不使用TAR fd/封印/新dataset子类。metadata准入不构造模型或读取信号窗口，真正的窗口比较另在准入后执行。

真实完整mini实跑28.30秒actual exit0，两份独立获取/转换的原始字段深度核对一致，全部690次选窗输入/目标/mask/元数据typed摘要精确相同：官方字面选择user/user_stage各230窗、30文件行交集；显式文件行隔离为167/63窗、交集0。所有原map及source关闭，RNG不变；worker/guardian与两个外层拥有者实际PID/start独立确认退出。第一次私有探针先导入原dataset却未激活上游路径，实际exit1及全部退出记录保留；只修复探针激活顺序后另起冻结成功运行，生产算法未改。以上是数据资格，模型推理另行核验。

## 验证选择best的原生FP32评价与独立复算：工程检查完成

[CPU入口证据](../results/emg2pose-direct-selected-evaluation-cpu-engineering.json)记录22新检查+旧评价/新数据/独立算术回归，本机107/432.13秒、服务器107/564.64秒，各11warning；完整轻量1,355/731可选跳过/1warning80.47秒均actual exit0。[实际GPU证据](../results/emg2pose-direct-selected-evaluation-engineering.json)为17个受管理阶段/286.11秒actual exit0。复用之前真正训练并按验证min选出的普通源S/M/L工程best，分别完成原生val、新进程完整预测复推、测试前profile冻结、原生test及新进程test复推、最后独立CPU重开。所有typed输入/输出和统计量完全复推，原按用户float32与独立float64误差和/有效分母一致，原20角/左510/mask位对齐不改，模型参数及fit的best/last/日志保持原SHA。34个worker/guardian及外层两个拥有者实际PID/start另核对退出。

新`emg2pose_direct_evaluate`使用独立pass/profile格式和普通源pipeline完成SHA+完整数据身份SHA；每次重读独立核对source-file绑定，拒绝旧kernel identity。实际owned证明要求匹配真实worker、空descriptor map和已退出的worker/guardian；源检查贯穿实际原生评价。原读者/journal/pair/profile/算术算法私有精确复用，原模块全局和固定数值容差不改。

GPU工程源仍是明确的三文件/25,710行合成三角色、重复录制ID队列，仅预先声明的`metadata_indices`对照可用于独立val，官方字面混角色val已实际拒绝。各pass实际1窗口，虽配置320/12worker/FP32，**尚未取得真实batch320或320+1资格**。CPU独立录制夹具覆盖两种选择。完整321窗/两模式/12worker FP32容量需要自己的较大独立录制源与新fit，不能转用旧内存后端资格；完整源/250轮/科学控制器报告和全文仍欠。

## 原始普通文件完整FP32批量与两种选择：实际容量已核验

[实际证据](../results/emg2pose-direct-evaluation-capacity-engineering.json)为显式合成三角色/独立录制5,009,750行，原格式HDF仅使用无损gzip4+shuffle，实际TAR299,345,920B，不放宽647MB工程源上限。原接收/原保存和转换/原生准入均真实执行；另独立全量核对生成HDF与普通字段的time、joint_angles、emg及5,009,750个有效mask，全部一致。S/M/L各新的两轮原生工程fit/32更新与验证min best，两个选择模式的val/test native及新进程复算、测试前profile均完成；每pass321窗、实际320+1、12worker、FP32，预测和充分统计精确复推。

原GPU链完成后私有最后读取误用恢复专用资格API，653.04秒实际exit1保留；仅纠正最后CPU读取，66.38秒actual exit0，未重复训练或预测、未改原模型/算法/容差。37个成功阶段另有1失败读取，76个worker/guardian及两组外层拥有者实际PID/start确认退出，代码/权重/epoch日志不变。原恢复资格继续使用已完成的独立恢复运行。本工程结果没有完整25253源/250轮或科学成绩；最终科学controller/report与全量实验仍须完成。

## 最终报告接口与发布范围

`emg2pose_direct_report.export_report`接收实际借用的普通源、单一模型的冻结fit结果SHA、最后fit进程拥有者，以及`official_literal`与`metadata_indices`两套val/test native/replay及其拥有者目录。它重开验证最小值best/last/epoch日志、实际原人口与预先冻结profile，并核对所有实际worker/guardian退出；不更换官方人口、模型或指标。`read_report`从私有输入证据重新准入源与全部结果，重算公开字段，不能靠重签一个JSON把工程实验升级为完整科学实验。

full要求真正25,253来源和原train/val/test角色数量、250完整轮次、两模式完整三条件和full profile。工程报告可以显式保留部分条件，始终`scientific_complete=False`。Table11采用逐用户无权均值/总体SD，Table13采用条件窗口数加权；有效角度pooled另列。匿名用户编号只在单一选择模式内对齐，不作为跨模式身份映射。

公开只选择`metrics.json`、`metrics.md`和`completion.json`；`evidence.json`保留私有源/检查点/拥有者路径供独立重开，不能随整个目录公开。完整控制器与真实full/250轮仍是后续验收范围。

报告API现已通过[本机/服务器58项与真实复核](../results/emg2pose-direct-final-report-engineering.json)：S/M/L各实际工程GPU fit与两模式320+1 FP32结果，三份报告生成和新进程重开共6个受管理CPU阶段303.39秒actual exit0，所有消费者/外层拥有者独立确认退出，未重复训练或推理。原输入权重/日志/pass保持SHA。此项资格只覆盖工程报告流程，完整科学控制器和真正全源250轮仍欠。

## 共享实验控制器：14个独立受管理阶段

`emg2pose_direct_experiment.run_experiment`接受普通源的冻结完成SHA、模型size、独立输出与三类资格：逐模型原生训练恢复fit、逐模型真实321窗FP32报告、full额外需要共享编排自身CUDA工程完成标记。一个真正CUDA工程流程资格覆盖模型参数化的共同控制逻辑；每个正式S/M/L仍分别需要自己的训练与评价资格。原训练和评价函数不改写；不导入Torch的父进程只负责源/资源/进程/文件绑定。

14阶段为：CPU准入、native fit，随后两种选择分别执行完整val/native-replay、profile冻结、test/native-replay，最后CPU报告导出和新进程核验。每个CUDA阶段前需64GiB余量，运行中保持实际256GiB主机/15GiB磁盘/8GiBGPU检查。`driver-completion.json`只能在全部实际阶段成功、阶段消费者退出和源/资格/源码不变后产生；`read_experiment`还要求原控制器实际退出并重开报告。失败保留、不自动重试，工程不能提升为full。CPU工程仅1/0/FP32、两轮，GPU工程320/12/bf16、两轮；full始终250轮/完整源，并保留两个原选择口径。

[实际证据](../results/emg2pose-direct-experiment-engineering.json)记录43本机/43服务器检查及一个S模型的14阶段真实GPU流程312.91秒，额外独立受管理CPU重读57.27秒；30个worker/guardian和外层拥有者均独立确认退出。实际使用三角色合成5M行源，不是benchmark精度。单元测试明确注入内存观测，真实服务器GPU及独立重读未注入。

CLI为`python -m egoemg_repro.emg2pose_direct_experiment run`，参数与上方API同名并使用连字符；full必须分别提供`--qualification`、`--evaluation-qualification`、`--controller-qualification`及各自独立冻结SHA。`verify`子命令默认只接受full，工程重读须明确`--allow-engineering`。完整运行目录含私有request/evidence/拥有者信息，公开仅选择脱敏的最终报告文件。真正完整源的队列接通和250轮结果仍欠。

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

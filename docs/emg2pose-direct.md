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

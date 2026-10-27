# EMG2Pose 完整源数据的内存存储接入

本路线用于解决完整源包约462.8GB、服务器磁盘不能同时保存原TAR和转换数组的问题。**当前已核验全部526个真实mini窗口、三模型CPU恢复、原批量GPU容量，以及原生检查点的新进程GPU完整状态恢复；完整源和三个完整实验尚未完成。** 保留本机分块xz接收进度，原始数据、旧mini产物及DINO训练快照保持独立。下文历史工程快照各有独立范围，最新恢复证据见本文最后一节。

## 已实现的接收组件

持久化xz接收另有独立的[响应体重试组件](../egoemg_repro/emg2pose_archive_retry.py)，不修改原归档核心或已提交块。它只对原读取器明确包装的连接、读取超时或响应中断重试同一范围，最多三次响应体尝试，等待1/2秒；每次仍由原代码核验206、范围、强ETag、Last-Modified、长度和内容。TLS、协议、身份、内容、资源和其他错误立即传播。原请求建立阶段最多三次尝试另行保留，因此不能把三个响应体尝试误写为所有情况下最多三个HTTP请求。

容量guard在每次尝试及退避前检查，退避后再次检查才能请求；独立私有持久化入口继续保留340GiB预算和15GiB磁盘余量，不自动重启进程或训练。[实际证据](../results/emg2pose-persistent-body-retry-engineering.json)覆盖真实HTTP半途断开、后续身份变化拒绝、三次耗尽不提交失败块、先前已提交块保留及显式恢复后完整SHA一致。本机Python3.12和服务器Python3.11各76项通过。这些是传输与归档工程证据，不能作为完整源就绪或模型成绩。

`egoemg_repro.emg2pose_memory_archive.acquire_archive`沿用已有下载器的强ETag、Last-Modified、精确206范围、总长度和响应体检查。默认8个并发、每块16MiB，仅保留有界的在途响应；原HTTP对象的每一个字节都写入拥有的Linux memfd，不提取或删去字段、文件、TAR头或填充。

开始前检查完整对象与缓冲所需内存，默认额外保留256GiB；接收和最终读回期间持续检查主机MemAvailable及可见cgroup-v2各级限额。完整接收后重新核对HEAD，再加上WRITE/GROW/SHRINK/SEAL封印；逐块读回比较实际收到的SHA256，并计算整个原对象SHA256。任何失败都关闭本组件拥有的fd，不返回可用缓存。

这里的检查是容量观察，不是操作系统预留承诺。根cgroup与非根接口有区别：内核文档把`memory.current`/`memory.max`定义为非根接口。实际全局根节点可无这两个文件，必须核对其memory控制器；带有限额的命名空间根节点仍须检查限额。未知或损坏的层级不能直接当成无限额。[Linux cgroup-v2文档](https://docs.kernel.org/6.6/admin-guide/cgroup-v2.html#memory)

Python可能未暴露seal常量或`os.memfd_create`，即使实际Linux内核支持。组件使用相同Linux UAPI；缺少Python包装时调用主机libc的同一个memfd_create操作。它不退回普通磁盘文件或可写源。若内核/libc不支持则失败。

## API及生命周期

需要Linux、requests，以及足够的实际可用容量；mini接收也默认保留256GiB。下面只接收mini，不会启动训练：

```python
from egoemg_repro.emg2pose_memory_archive import acquire_archive

with acquire_archive('mini') as cache:
    descriptor = cache.fileno()
    receipt = cache.receipt
    # 消费者在此作用域中按范围读取，或通过subprocess的pass_fds显式继承。
    # 此fd是原始TAR，不是官方memmap数据目录。
```

调用者必须使用上下文管理器或显式`close()`；这只释放当前所有者的fd。如果将fd复制或传给子进程，最后一个持有者关闭后才释放整个对象。因此后续控制器还必须管理自己创建的消费者及资源压力，不能只关闭父进程fd就宣称缓存已释放。RAM缓存是易失的，持久化xz进度不依赖它。

`receipt.complete=true`仅代表全部原HTTP字节已经接收、封印并核验；`full_corpus_complete`始终为false。ETag标识传输对象，收到的SHA256标识这些字节，二者都不冒充发布者提供的SHA256。数据集完整性还需精确文件人口、CSV/HDF5属性和原生字段/mask检查。

## 实际证据及未完成步骤

[接收工程证据](../results/emg2pose-memory-acquisition-engineering.json)记录匹配源码的实际服务器mini检查：647,198,208字节、39块、完整SHA256与已有mini源一致、真实内核拒绝修改、检查结束后关闭拥有的fd。没有分配完整462GB源，也没有使用GPU或运行新的模型训练。

同一99成员源码的完整项目检查为1,606 passed/4 skipped；三项跳过需实际CUDA验收，另一项为本机内存不足而未执行的旧ViT-L构造。轻量检查为874 passed/689依赖skip；接收与既有归档的75项定向检查通过。这些结果分别证明代码范围，不能代替未运行的完整数据或模型实验。

新增[原TAR/HDF5盘点证据](../results/emg2pose-memory-data-engineering.json)已在服务器真实mini上核验：全部32个TAR成员中的31文件、30个HDF5的对象和属性、全部4,218,756行的四个使用字段、每个会话的字段摘要，以及原metadata.npz全部数组，均与既有直接转换结果一致。原始valid行3,559,584、有效区间1,015；这些是原始mask统计，不是模型窗口应用all-false fallback后的有效分母。检查结束后所有内存描述符均关闭，没有进行新模型优化。

该新增盘点组件对应101成员源码的完整项目检查为1,651 passed/4 skipped，轻量919 passed/689依赖skip；45项新读取/盘点检查与37项原转换检查合计82项通过。上一接收组件99成员源码的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40429258366)也已成功：官方CPU1,607/3真实CUDA skip、轻量874/689；其证据不覆盖随后新增盘点代码。

## 原TAR/HDF5盘点API

`emg2pose_memory_data.FdSlice`为一个TAR成员提供独立游标和严格字节边界，拥有自己的只读封印对象描述符副本。HDF5访问按原成员偏移读取，不提取到磁盘；HDF5先关闭、slice后关闭。原始额外字段和属性仍完整保存在TAR中。

`index_archive`核对全部字节SHA256，每个成员记录数据/头部摘要及物理位置；拒绝不安全路径、重复、链接、稀疏成员、截断及隐藏追加归档。至少两个零TAR结束记录必须完整存在。长名字的GNU/PAX头部也绑定摘要，不把第一个结束记录后隐藏的数据忽略掉。

`inspect_archive`自行核验上述源，使用固定官方helpers排序会话、构造IK/finite mask和连续有效区间，以最多262,144行逐块计算字段摘要。完整/mini模式额外要求对应已完成接收receipt、固定官方CSV和精确HDF5人口；结果为独立的私有`emg2pose_sealed_tar_inventory_v1`格式，包含实际源字段和参与者元数据，须留在私有数据目录。

```python
from pathlib import Path
from egoemg_repro.emg2pose_memory_archive import acquire_archive
from egoemg_repro.emg2pose_memory_data import inspect_archive

with acquire_archive('mini') as cache:
    source = cache.receipt
    inventory = inspect_archive(
        cache.fileno(), expected_bytes=source['source_bytes_received'],
        expected_sha256=source['full_sha256'], cohort='mini',
        acquisition_receipt=source, upstream=Path('upstream'),
        metadata_csv=Path('data/emg2pose/emg2pose_metadata.csv'),
    )
```

## 原生数据集与训练入口

`emg2pose_memory_index.prepare_index`独立复核源盘点，再原子写入四个小文件：source-inventory.json、metadata.npz、manifest.json和最后的completion.json；不生成源大小的dat数组，不覆盖已有或失败的索引。`open_source`必须接收调用者另外保存的completion SHA256，核验索引和实际只读TAR字节后返回拥有独立fd的`VerifiedSource`。它在准入时完整hash一次，运行中检查封印和文件绑定，不在每个batch重新hash数百GB源。

`emg2pose_memory_dataset.SealedTarDataset`继承固定官方`Emg2PoseDataset`，只替换底层字段读取。窗口、过滤、jitter、增强、归一化、mask fallback及会话分组沿用原方法。每个窗口的HDF5上下文及时关闭；fork重置进程内缓存，spawn通过`DupFd`转移实际描述符权限，不能把某个fd整数当作跨进程身份。

`emg2pose_memory_runtime.compose_config`和`build_runtime`要求仍然有效的`VerifiedSource`。在两个数据配置集合中显式替换dataset target和描述符，其他模型、损失、优化器、归一化及科学配方不变；先逐项检查实际私有配置，再生成可跨相同内容缓存重建的身份。原生setup_runtime在模型factory之前调用，沿用已声明的seed顺序修正。`reload_runtime`严格恢复权重；完整优化器/调度器/RNG/Lightning循环恢复由调用方另行验证，不能以权重加载替代。

调用方须让源在数据集构造及使用期间有效，并在消费者结束后调用`close_data_module`关闭各唯一叶数据集的fd，包括共享ConcatDataset/Subset。该函数不会停止其他进程；控制器还须等待自己创建的worker退出。索引、配置和完整身份包含私有路径或会话信息，应留在私有目录，公开报告只发布脱敏摘要。

当前18项索引、19项数据集、28项运行时和45项既有盘点检查合计110项定向通过。三个真实模型的随机初始化权重、前向和官方变换与原入口一致；这不是新的性能结果。实际mini的全部526窗口、样本元数据、RNG消耗和会话分组逐项一致，累计3,493,232个true输入mask元素。此分母不是原始valid行数，也不是对齐后的模型输出mask分母。

[实际原生工程证据](../results/emg2pose-memory-native-engineering.json)记录S/M/L各两次真实CPU更新，全部526窗口在新进程中的输入/预测/目标/mask/顺序及指标摘要精确一致；AdamW矩、cosine和显式RNG恢复一致，原生Trainer在step2加载权重、优化器、调度器和循环计数后再执行一次实际更新到step3。所有拥有的内存fd最终关闭，未初始化CUDA。原生框架对中途checkpoint的dataloader顺序提示保留，因此不声称连续训练轨迹逐位一致。

首次未隐藏GPU的CPU检查在最终资源检查失败；原生Lightning即使CPU fit也会通过isolate_rng读取可见CUDA RNG。该失败保留，最终检查在进程启动时显式隐藏CUDA，生产读取器/模型/优化器源码不变。三个模型均为mini原val、float32、batch1/workers0工程检查，不证明原GPU批量或论文精度。

一次实现审查发现：最后的HTTP HEAD响应阻塞时，已完成缓存仍持有内存但未持续检查压力。两项真实阻塞/压力用例先失败，修正后接收与原归档77项通过；失败路径在等待HTTP退出前先关闭缓存fd。修后重新执行全部526窗口与三个模型的实际检查，最终107成员完整项目检查为1,719 passed/3实际CUDA skip（863.83秒），轻量939 passed/691依赖skip（29.71秒），均已捕获实际exit0。此前1,717通过但缺退出码的历史运行仍单列，不追认为成功；新的本机结果不冒充托管CI。

后续必须完成三模型实际检查、完整源准入、资源/进程生命周期验收和三项250轮实验。不能直接把新TAR交给现有memmap入口，也不修改旧转换器的完成标记规则。

实现与验证顺序见[内存接入计划](superpowers/plans/2026-10-27-emg2pose-memory-storage.md)。这项工程结果不能填入论文Tables11/13。

## 拥有的进程与缓存消费者

新增`emg2pose_memory_process.run_owned_stage(argv, output, *, descriptors, guard, environment=None, poll_seconds=0.25, terminate_grace=5.0)`用于管理一次读取、盘点或模型阶段。调用者保留原fd，在阶段结束后再释放缓存；`descriptors={'ARCHIVE': fd}`通过私有Unix socket的SCM_RIGHTS转移实际权限，消费者从`EGOEMG_STAGE_FD_ARCHIVE`读取自己的fd编号。不能把父进程的某个数字直接当作子进程fd。

独立guardian是消费者的父进程；启动消费者后立即关闭自己的缓存副本，只保留控制socket和真实PID/start/UID/session身份。持有缓存的调用者被SIGKILL、收到TERM/INT、资源检查失败，或消费者正常/异常退出时，guardian先回收同一session里的剩余消费者，再释放未回收的leader身份。逐个打开pidfd并重新核对身份后发送TERM，超过宽限期才发送KILL；不对未验证的数字进程组发送信号。所有相关消费者实际退出前不返回干净结果，内核不可中断进程也不能被假定已经退出。

服务器Python和libc没有提供pidfd包装，但实际Linux6.8/x86_64头文件提供同一内核操作。组件在这种环境使用434/open与424/send_signal syscall；启动前真实open自身pidfd并执行signal0，不把缺少包装误判为内核不支持，也不退回数字PID信号。其他未支持平台直接拒绝。

[进程工程证据](../results/emg2pose-memory-process-engineering.json)记录本机20项、真实服务器20项检查，包含实际父进程SIGKILL/TERM/INT、仍持有缓存的孙进程、拒绝TERM、资源压力/未知容量、错误身份和只读权限。最终109成员整体1,739通过/3实际CUDA skip、轻量959/691依赖skip，均捕获实际exit0。完整mini还实际经过新进程管理、严格加载既有small工程检查点及未改的官方逐用户评价；原转换数据与新存储的两组指标完全一致，父进程、消费者缓存fd均关闭，运行中实际观察guardian不持有缓存副本。

这次逐用户评价沿用原`skip_ik_failures=True`，得到460窗口（两组各230，各只有1个mini用户），与此前`skip_ik_failures=False`的526验证窗口分列。它仍是mini原val、CPU float32、batch1/workers0，没有新训练、原GPU批量或模型性能结论。消费者必须保留所属session；同时失去guardian与调用者的恢复没有通过此检查。完整源/索引控制器、三模型原生GPU验收、完整训练和最后报告仍需实现与运行；本模块只发布私有阶段receipt，不产生科学完成标记。

## 受管理的源数据准备

`emg2pose_memory_prepare.prepare_owned_source(cache, output, *, upstream, metadata_csv, cohort='full', reserve_bytes=274877906944)`消费一个仍由调用者拥有的`SealedArchive`。父进程冻结接收receipt的文件SHA，把长时间盘点、索引和实际准入交给独立消费者，持续检查主机/cgroup内存、封印源、代码和输入文件。full/mini保持至少256GiB余量，显式engineering夹具不能获得full资格。调用者须在后续模型阶段继续保留缓存。

消费者的`prepare_source`借用其fd，独立重验所有原字节、HDF5字段/属性/会话/有效区间，复用现有索引生产器并真正重新打开索引。`prepared.incomplete/{index/,preparation.json,status.json}`原子无覆盖发布为`prepared/`；其中index仍是独立的四文件格式，没有源大小dat或HDF5提取。失败保留不完整目录、删除准备成功标记，不能覆盖已有输出；进程结束关闭继承的fd，直接函数不关闭调用者的fd。

完整模式须满足原25,253会话及train17,136/val1,950/test6,167；mini始终是工程数据。父进程只有在消费者实际exit0、拥有的进程均退出、来源/生产者一致且最终资源检查通过后，才写`owned-preparation.json`；它是私有准备receipt，不是科学完成标记。工作目录里存在同名Python包也不能换掉已核验代码：guardian按其实际文件启动，准备消费者显式从当前bundle加载，生产者摘要另外核对。

[实际证据](../results/emg2pose-memory-preparation-engineering.json)记录本机/服务器各66项检查，全部647MB mini准备83.47秒、30文件/4,218,756行/四字段与原转换参考一致，仅写紧凑索引、没有导入Torch。最终111成员完整1,767通过/3实际CUDA skip，轻量987/691依赖skip，均实际exit0；实际完整源仍未准备。

## 原批量GPU容量工程检查

[实际三模型证据](../results/emg2pose-memory-native-gpu-capacity.json)保留原module/0.01损失/AdamW/cosine250/clip1和原train sampler/collate、val loader，在A800上逐模型执行两次batch320、12worker、bf16更新与两轮完整526窗口验证。S的原eval spawn、M/L的默认上下文均保留。全部拥有的消费者正常退出，缓存所有者隐藏CUDA且未导入Torch。

| 模型 | 实际更新 | 每轮原生验证窗口 | 峰allocated GiB | 峰reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| S | 2 | 526 | 8.75 | 9.89 |
| M | 2 | 526 | 10.70 | 11.28 |
| L | 2 | 526 | 14.07 | 15.27 |

这里仍复用mini原val及其验证变换作为训练夹具；不是原完整训练人口、增强或独立验证。每epoch只消费一个完整320训练batch、保留250周期的优化器日程而在两步后停止。该检查没有保存/恢复GPU完整状态、选best或最终测试；仅证明观测设备上该批量/读取进程/模型前后向及完整mini原生验证可执行，不能填论文Tables11/13或称为完整训练。

## 原生GPU检查点恢复

`emg2pose_memory_recovery.run_recovery_phase(upstream, source, metadata_csv, size, output, *, phase, device='cpu', expected_prefix_sha256=None)`借用一个实际准入的`VerifiedSource`，始终只产生工程结果。安装`.[emg2pose-runtime]`包含固定TensorBoard2.18，以调用原`build_callbacks`、`build_logger`和`build_trainer`。GPU保留batch320、12workers、bf16、clip1、cosine250、top3+last、val_mae/min且没有EarlyStopping；仅日志/检查点路径及两步/三步的工程停止预算不同。CPU明确为batch1/workers0/float32。

`prefix`执行两次真实优化器更新及两轮完整验证，随后在自动topK目录外冻结`prefix.ckpt`和`prefix.json`。`restore`必须在另一个真实进程中提供另外冻结的prefix JSON SHA，严格核验数据、原配方、运行时、生产者、GPU配置和检查点SHA；原目录及日志version保持一致。先严格加载权重并重跑全部验证，再让原生Trainer.fit加载完整检查点，最后执行一次真正的参数更新及完整验证。输入/输出/顺序采用流式摘要，观察器不累积全队列预测。

[实际三模型证据](../results/emg2pose-memory-native-gpu-recovery.json)记录一块A800上的六个受管理阶段、总计636.92秒，各阶段实际exit0且拥有的消费者全部退出：

| 模型 | 前缀实际更新 | 新进程完整重验窗口 | 恢复后实际更新 | 恢复后完整验证窗口 | 精确状态核对 |
| --- | ---: | ---: | ---: | ---: | --- |
| S | 2 | 526 | 1 | 526 | 六项全部一致 |
| M | 2 | 526 | 1 | 526 | 六项全部一致 |
| L | 2 | 526 | 1 | 526 | 六项全部一致 |

六项为模型权重、AdamW完整状态、cosine调度器、原生callbacks、原生fit循环和Python/NumPy/Torch/所选CUDA的随机状态。观察位置是未改的`restore_training_state`返回后、迭代器重建前。PL2.5保存但故意不恢复`ModelCheckpoint.current_score`，只排除这个暂态字段；其他选模、分数、路径与topK状态逐项比较。原bf16没有FP16 GradScaler，不制造scaler恢复结论。完整新进程验证与前缀末轮的全部原生指标及输入/输出/顺序摘要精确一致。

检查点IO支持原生保存、加载及自动topK删除；仅允许拥有的目录，保留15GiB磁盘余量、同步原子写入，冻结检查点不允许覆盖或删除。独立CLI通过`EGOEMG_STAGE_FD_ARCHIVE`接收真实继承权限并在退出时关闭；调用方仍须使用已有guardian管理12worker后代和资源压力。失败保留开始/检查点/错误记录、删除该阶段的成功receipt，并拒绝自动重试。

本次仍复用mini原val和验证变换作为工程训练夹具，没有独立验证、完整训练人口或增强验收，没有验证选best后的最终测试。完整526窗口的`skipIKfalse`验证与原逐用户分析的460个`skipIKtrue`窗口不同。检查点状态恢复不证明中断前后训练轨迹或数据顺序逐位一致；原生提示保留。完整源、正式250轮训练入口及选模/测试/报告仍需完成，不能用本节填Tables11/13。

## 正式训练入口及受控工程验收

`emg2pose_memory_train.fit_source(upstream, source, metadata_csv, size, output, *, scope='full', resume=False, expected_last_sha256=None, stop_after_epoch=None, qualification=None, expected_qualification_sha256=None)`只运行原生Trainer.fit；不会自动打开测试集。full要求实际准入的全部25,253会话、462,824,048,640个源字节及原train17,136/val1,950/test6,167角色，保留250轮、batch320、12workers、bf16、clip1、cosine250、top3+last/val_mae/min且无EarlyStopping。full没有子集、步数上限或工程早停参数。原val-only mini没有train角色，因此不能通过此入口启动训练。

full还要求该规模、相同科学配方/代码/运行时/硬件的实际CUDA工程资格：`read_qualification`重新打开完整两轮的原生best/last/恢复前缀和epoch journal，核对真实GPU配置及新进程六状态恢复，外部须另行冻结fit-result文件SHA。CPU结果、旧mini容量报告或单个成功布尔字段均不构成这份资格。完整数据的身份另外独立绑定，不能用合成数据的角色和字节数替代。

官方`SessionShuffleSampler`会按每rank的16个batch块循环补齐，而不是只迭代`ceil(len(dataset)/batch)`。入口保留原采样器类型、单rank、块大小和每轮补齐行为，分别记录`train_unique_windows`（数据集锚点数）、`train_samples`（实际消费数）与实际batch/更新次数。完整epoch hash链绑定完整train/val消费数、原生验证MAE、流式输入/输出/指标摘要及全部模型state；原模型损失和选模方法不重写。

`read_fit(output, require_full=True)`独立检查终态last、每轮覆盖、优化器计数、cosine250和原生验证最小值所选best的文件/权重/epoch。只有完整250轮及真实full源才是full训练结束；fit-only结果仍不代表最终科学测试或全文完成。恢复显式要求last.ckpt的另外冻结SHA，在相同输出目录保留不可变恢复前缀和旧结果；原callback路径不变。错误模型/输入的预检拒绝不改写旧成功凭据，并发写入拒绝不遗留锁fd。未提交journal尾部在恢复前另存，不自动重试。日志和数据集清理完成后才发布最后的fit-only receipt。

[实际GPU工程证据](../results/emg2pose-memory-native-fit-engineering.json)为3个**合成HDF5文件**、25,710行和显式分开的train/val/test角色，不是官方mini或完整benchmark。S/M/L各完成两轮原320/12/bf16训练，共32次实际更新（首进程16，另一个进程恢复后再16），每轮消费5,120个原采样器补齐窗口，唯一锚点分别1/2/2；每轮验证只有1窗。原训练增强、验证角色、native best/last/前缀、epoch hash链和六状态恢复全部实际检查，另一个CPU消费者重新接受三份CUDA资格，全部七个受管理阶段exit0，总计457.00秒。合成误差不发布为性能结论；此前真实mini526窗口GPU重推证据仍独立保留。

完整源、正式250轮实验、仅验证集校准、原逐用户最终评价/独立统计复推、拥有的完整控制器及Task4最后实现审查仍未完成。本节只是入口实现与受控工程验收，不能填Tables11/13。

## 独立20角统计与原生评价人口

[独立统计组件](../egoemg_repro/emg2pose_memory_statistics.py)按样本流式累计float64逐角误差和、有效时间点及消费次数，分条件和私有用户保留20角、五指和近/中/远端八组；逐用户均值和ddof0标准差与有效角度加权MAE分列，不加入腕角。完整批次的类型、有限值和元数据先通过校验才能改计数。原始20×7790标签/生成mask必须在left510/right0后与20×7280原生tuple逐位一致。原FP32指标与独立float64只使用预先固定的2e-6 rad绝对/相对数值界；这不等于预测重推资格。

[实际工程证据](../results/emg2pose-memory-statistics-engineering.json)：本机/服务器各33项通过，含新进程执行未修改的官方逐用户函数及全部常量组映射。真实647MB mini随后在两个独立受管理CPU进程中重载既有两次更新的工程前缀，每次按原函数完成460次消费、56项原生指标核对；完整输入/预测摘要和独立统计精确重现，190.52秒、父进程exit0，全部拥有的消费者和fd退出。batch1/workers0/FP32、全部原val、没有正式选best或GPU/full资格，也不发布工程MAE为模型成绩。

这次真实执行发现[原生选择人口问题](../results/emg2pose-native-evaluation-selection-audit.json)：mini有30个阶段/手文件行，但只有1个记录session ID。原`get_dataloaders`把这个重复ID传给`allowed_sessions`，于是user/user_stage原本22/8行都选成30行；各230次消费的有序窗口以及输入/预测摘要完全相同，合计460不能解释为两个独立人群。首次私有观察器按记录ID推断条件而失败，exit1保留；后改为随原loader迭代标记，**保留原筛选行为并测量重叠**，没有静默纠正官方人口或放宽数值界。

另用同一未修改筛选函数对官方完整CSV实际执行元数据层检查：

| 请求角色/条件 | 请求文件行 | 原筛选选中文件行 | 选中角色 |
| --- | ---: | ---: | --- |
| val/user | 1,618 | 1,950 | val 1,950 |
| val/user_stage | 332 | 1,934 | val 1,934 |
| test/stage | 3,539 | 20,370 | train 16,831；test 3,539 |
| test/user | 2,172 | 2,628 | test 2,628 |
| test/user_stage | 456 | 2,610 | test 2,610 |

这些是25,253行CSV上的原筛选函数结果，**不是完整HDF5实物或窗口推理结果**。后续完整评价须预先声明`official_literal`为原样路径，并单列通过官方已有`allowed_session_indices`/`allowed_splits`隔离文件行的`metadata_indices`对照；两者不混表、不在测试后选择协议或容差。仅验证集校准、所选best及实际完整源仍须各自准入；本节算术证据不能授予这些资格。

[持久化续传实查](../results/emg2pose-persistent-body-retry-live-observation.json)在08:54UTC已完成旧76.974GB前缀核验并新增128MiB至77.108GB。一次实际响应体ConnectionError后，同范围第二次尝试成功提交，另独立解压核对该16MiB块的原始及压缩SHA。接收进程当时仍存活，完整源仍未完成。[托管CI证据](../results/emg2pose-memory-hosted-ci.json)记录7b8911f与8a05ae9两次成功；后者完整CPU1,832/3实际CUDA skip、轻量1,021/693依赖skip，**不覆盖随后新增的统计源码**。

## 评价数据的元数据准入

`emg2pose_memory_evaluation_data.build_evaluation_data(upstream, source, metadata_csv, size, *, role, scope='full', selection='official_literal')`先使用原数据集筛选函数核验请求与实际选择，再通过私有存储构造绑定执行原`get_dataloaders`代码。full要求完整25,253文件及独立原角色；显式工程scope允许验证存储和人口。它不构造模型、不推理、不写结果文件，也不授予正式选best、校准或测试资格。

返回对象的`config`/`context`供后续受管理评价消费，`population`与`identity`是私有副本。`check()`核对实际源、代码、元数据、角色/逐用户设置、配置、筛选、变换、归一化、loader顺序与窗口计划；`close()`及上下文管理只关闭本对象复制的数据集fd，借入源仍由调用者持有。必须另外等待拥有的消费者退出，不能把关闭父fd当成进程已退出。

验证选择包含非val文件时，在创建存储读取器前拒绝。原样test保留官方实际选择并披露额外角色；`metadata_indices`明确使用原有文件行与split过滤，不能与原样结果混淆。原元数据函数未显式关闭NPZ文件，实测拒绝路径留下文件fd；现在仅在当前调用的私有numpy绑定中显式关闭所打开NPZ，临时实例方法随即移除。原函数代码、模块全局变量、数组值及筛选算法保持不变，测试在禁用循环垃圾回收时也能立即释放这些文件。

[实际准入证据](../results/emg2pose-memory-evaluation-data-engineering.json)记录本机31项/115.57秒、服务器同31项/172.47秒通过。真实647MB mini另在一个受管理CPU消费者中检查两种模式：原样两组各30文件/230窗；控制模式为22文件/167窗与8文件/63窗，文件行重叠为0。全程未读取信号窗口、未构造模型、未改变Torch RNG；调用者源保留至作用域退出，拥有的消费者和fd均退出，父进程实际exit0，总计102.82秒。后续完整推理、仅验证集校准、最终测试/复推/报告仍未实现完成。

335348b的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40478615135)已成功：完整CPU1,865 passed/3实际CUDA skip、轻量1,053 passed/694依赖skip，pip check通过。这覆盖独立统计源码，**不覆盖本节随后新增的准入源码**。

## 最佳检查点的流式评价与测试前校准

此前元数据准入提交430109a的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40487240153)现已成功：完整CPU1,896 passed/3实际CUDA skip（1,388.15秒），轻量1,053 passed/695依赖skip（46.01秒），pip check通过。这覆盖前一节元数据准入，尚不覆盖本节新增的推理/profile代码；对应范围见[CI记录](../results/emg2pose-memory-hosted-ci.json)。

`emg2pose_memory_evaluate.run_evaluation_pass`消费实际`VerifiedSource`和已完成的原生fit目录，先由`read_fit`重新核验原验证最小值选出的best文件与完整权重，再运行未修改的`_evaluate_emg2pose_single_pass`。显式声明`role='val'|'test'`、`kind='native'|'replay'`、scope和selection；full要求完整源及真正完成250轮的full fit。工程fit、恢复前缀或mini不能取得full资格。

评价保留原7790输入、left510/right0后的7280输出、20角、FP32与无autocast。full/CUDA工程配置batch320、12workers，CPU工程为1/0；最后一个不完整batch照常消费。metadata-only计划绑定原条件顺序、每批样本数、用户和窗口元数据摘要；实际forward逐批核对输入、标签/mask对齐及模型状态。GPU启动前保留64GiB，自己的模型及工作空间分配后持续保留8GiB，不把启动前门槛重复套在已分配的模型上。

私有输出包含`identity.json`、`plan.json`、有界`batches.jsonl`、`statistics.json`、`native-metrics.json`和最后的`pass.json`。逐批hash链保存带类型的输入/输出摘要及每用户20角float64误差和/有效计数，不保存整套预测数组。`read_evaluation_pass`重新读取实际检查点和全部sidecar，检查完整窗口消费、原始日志统计、固定2e-6原生数值核对及前后权重一致；全部无效样本不能带非零误差和，每个有有效标签的样本至少贡献一个有效时间点。

`freeze_evaluation_profile(native_val, replay_val, output, ...)`要求两份另外冻结SHA的完整val结果、不同实际PID/start、输入/预测摘要和充分统计量精确一致。full还要求与两个实际worker及源封印绑定匹配的受管理exit0和消费者全部退出。`read_evaluation_profile(output, expected_sha256, ...)`深读所绑定的原结果；测试入口必须先通过这个同模式、同源、同best和同推理后端的profile，才会构造测试数据或推理。原样和文件行隔离对照分别校准，不能交叉使用。这里的“校准”只固定一致性与来源，没有拟合阈值或根据测试成绩调参。

成功pass在日志、指标及数据集清理后才发布；任何失败保留已有诊断文件但不保留成功pass，已有目录及fit/source/reference/profile不得覆盖。所有写入继续保留15GiB磁盘余量，控制器必须另行等待消费者退出。pass/profile始终只证明一个阶段，`scientific_complete`和`full_paper_complete`均为false；完整科学报告、控制器及Task4最后审查仍需完成。


[实际评价工程证据](../results/emg2pose-memory-selected-evaluation-engineering.json)记录服务器16项/362.52秒通过；本机另有三项真实RED→GREEN，初始13项日志与退出记录缺失的边界在私有ledger中保留，不拼成一次新的本机16项运行。真实A800 S/M/L共17个受管理阶段、713.39秒、实际exit0，包括原样混角色val拒绝、12次GPU推理、三份仅验证集profile冻结与一次独立CPU全重读；全部worker/guardian实际退出。前后best权重不变，val/test新进程输入/预测和统计精确一致。

GPU使用已有原生两轮/32更新的合成三角色fit及相同源，3文件/25,710行，重复记录ID；因此仅运行声明的metadata_indices对照。配置320/12/FP32，但每次实际1个窗口，**不能当成完整320样本FP32推理容量或benchmark**。CPU独立唯一记录ID夹具覆盖两种模式。实际完整批量评价资格、最终科学控制器/报告、完整源及三个250轮实验仍欠；本组件验收不等于Task4或全文完成。


## 表11/13的独立算术与匿名汇总

`emg2pose_memory_report.summarize_tables(statistics, native_rows, native_users, *, require_all_conditions=True)`从每个条件/私有用户的20角原始误差和、有效时间点、消费数及全无效数重建全部E1统计，检查原角顺序/名称、计数守恒和所有缓存汇总，再按原固定2e-6数值界核对未改官方逐用户评价的FP32指标。它只做算术，不准入来源、模型、预测、profile或拥有的进程，输出始终不能声明科学或全文完成。

表11保留原生逐用户均值、ddof0标准差和用户数（度），独立float64值另列。表13按PDF p24表注，以各条件实际消费的窗口数加权三个条件的组MAE：`sum(n_condition * group_MAE_condition) / sum(n_condition)`；原生FP32组MAE和独立float64值分别生成。固定官方single-pass在每个条件内部按有效角度汇总；本次在固定源码的scripts/analysis中未找到跨条件表13生成脚本，因此这里把表注要求的跨条件样本权重明确写出。所有条件的有效角度加权总量是另一种单列的视图，不能静默替代样本权重。原样条件有重叠时保留真实重复消费，不冒称唯一窗口。

默认要求stage/user/user_stage三个条件且各有有效标签；显式`require_all_conditions=False`只得到标为不完整的算术结果。公开用户键替换为本次汇总内一致的匿名序号，不跨独立报告宣称同一人。20角、八组原始误差和及有效分母保留，不加入腕角或用已舍入MAE逆推原始量。

[实际算术证据](../results/emg2pose-memory-table-arithmetic-engineering.json)：15项缺模块RED后通过；本机旧统计+新汇总48项/5.19秒、服务器同48项/5.23秒、无模型运行时轻量15项/0.43秒均实际exit0。另一个无Torch进程实际重读S在两种模式下的val/test各321窗GPU统计共四份，汇总守恒、原生数值一致、用户已匿名，输入文件SHA不变。它们是合成工程数据，不是表11/13的正式成绩。

5fbd897的[实际托管CPU运行](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40494382039)失败：1,899通过、3实际CUDA skip、13个真实fit夹具在严格CPU环境门槛拒绝。该job原先未在Python启动前显式设置空CUDA_VISIBLE_DEVICES；轻量job1,053通过/696依赖skip。现在仅在official-cpu任务环境补上该变量，生产检查和skip条件保持；从实际YAML任务配置启动的新本机进程64项/275.47秒通过。修正后托管运行仍须另行观察，旧失败保留在[CI记录](../results/emg2pose-memory-hosted-ci.json)。


## 实际FP32评价批量320与最后不足批次

[完整批量工程证据](../results/emg2pose-memory-evaluation-capacity-engineering.json)使用明确合成的原HDF5结构三角色源：train8,570行，val/test各2,500,590行，共5,009,750行/761,518,080字节；记录ID唯一。原helpers盘点所有字段/mask并建立新索引。S/M/L分别使用与该源严格匹配的原生两轮/32次更新fit，train每轮原采样器补齐5,120次消费，实际val每轮321窗；不把训练stride误套在验证上。

每模型每模式均完成实际FP32 val-native/fresh replay/profile冻结/test-native/fresh replay，原12worker、实际批次[320,1]，每pass321窗。所有原始输入/预测摘要和独立统计精确复推，原验证选best前后权重不变。24次GPU评价共7,704次窗口消费；它们含模式及复推的重复，不能说成7,704个唯一数据窗口。父进程未导入Torch，35个受管理阶段全部actual exit0，独立CPU最终重开全部fit/pass/profile/owned证据；所有worker/guardian及缓存所有者另核对实际退出，总计1,783.53秒。

| 模型 | 最大allocated GiB | 最大reserved GiB | 实际评价批次 |
| --- | ---: | ---: | --- |
| S | 2.249 | 2.430 | 320+1 |
| M | 2.261 | 2.441 | 320+1 |
| L | 2.297 | 2.469 | 320+1 |

峰值来自同一真实预测进程的CUDA计数器，并绑定pass与进程身份；不改变原forward、模型或评价源码。首次私有探针在CUDA未初始化时重置计数，worker actual exit1且未进入推理；成功源准备和Sfit保留。新GPU进程先复现该错误、显式初始化后计数通过；修正后的独立新调用核对旧父进程已失败退出、源/CSV/Sfit/原owned退出记录的另外冻结SHA，复用只读源和真实Sfit，重新封印并盘点/建立新索引，另外训练M/L。旧失败不会变为成功或被覆盖，没有自动重试。

这是原FP32完整批量及尾批流程资格。合成数据、两轮工程fit、相同源行的两种模式不证明完整benchmark人口、250轮、科学成绩或原样完整测试的无交叉；真实full源、最终科学控制器/报告及Task4最后审查仍须完成。

# EMG2Pose 完整源数据的内存存储接入

本路线用于解决完整源包约462.8GB、服务器磁盘不能同时保存原TAR和转换数组的问题。**当前已核验全部526个真实mini窗口，以及新入口三个模型的实际CPU优化/新进程恢复；最终完整项目检查和实现审查仍欠，完整源和三个完整实验尚未完成。** 保留本机分块xz接收进度，原始数据、旧mini产物及DINO训练快照保持独立。

## 已实现的接收组件

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

后续必须完成三模型实际检查、最终完整项目检查和实现审查，再执行完整源准入、资源/进程生命周期验收和三项250轮实验。不能直接把新TAR交给现有memmap入口，也不修改旧转换器的完成标记规则。

实现与验证顺序见[内存接入计划](superpowers/plans/2026-10-27-emg2pose-memory-storage.md)。这项工程结果不能填入论文Tables11/13。

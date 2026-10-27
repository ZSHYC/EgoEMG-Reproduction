# EMG2Pose 完整源数据的内存存储接入

本路线用于解决完整源包约462.8GB、服务器磁盘不能同时保存原TAR和转换数组的问题。**当前只完成无损内存接收组件；完整源、原生数据集适配和三个完整实验仍未完成。** 现有本机分块xz接收继续运行，原始数据、旧mini产物及DINO训练快照保持独立。

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

下一步须实现有界TAR成员/HDF5读接口、完整源人口和字段盘点，以及只替换底层存储的官方数据集适配。必须比较全部30文件/526个mini窗口的输入、标签、mask、顺序与恢复状态；通过后才可执行完整源准入和三项250轮实验。不能直接把新TAR交给现有memmap入口，也不修改旧转换器的完成标记规则。

实现与验证顺序见[内存接入计划](superpowers/plans/2026-10-27-emg2pose-memory-storage.md)。这项工程结果不能填入论文Tables11/13。

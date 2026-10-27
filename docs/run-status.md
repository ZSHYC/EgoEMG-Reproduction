# 运行状态

## 2026-10-27：最佳模型的流式评价完成三模型GPU工程复推

[新评价代码与实际证据](../results/emg2pose-memory-selected-evaluation-engineering.json)完成原生selected-best重载、未改逐用户函数、逐批输入/输出摘要与20角统计、验证native/fresh replay冻结profile、测试native/fresh replay。服务器16项/362.52秒实际exit0；另有三项计数/资源边界先RED后GREEN。full source/fit、模式、单独冻结SHA、原始sidecar损坏、同进程伪复推和无profile测试均有拒绝检查，失败不留下成功pass或改写fit。

A800上的S/M/L另经17个真实受管理阶段，713.39秒、父进程actual exit0；12次GPU评价、3次CPU冻结、1次真实mixed-role验证拒绝和1次独立CPU全重读均exit0，所有worker/guardian另核对实际退出。各模型val/test的新进程输入/预测摘要及统计精确一致，selected-best前后权重不变，父进程没有导入Torch。配置为320/12/FP32，但该合成3文件/25,710行夹具每次实际只有1窗；只运行声明的metadata_indices对照，原样val因重复记录ID混入其他角色而实际拒绝。CPU测试另覆盖唯一记录ID下的两种模式。**这不证明实际320窗FP32批量，也不产生benchmark成绩或250轮结果。** 后续仍需完整评价批量资格、完整科学控制器/报告、真实完整源、三项250轮及Task4最终审查。

前一430109a元数据准入源码的[托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40487240153)现已成功：CPU1,896/3实际CUDA skip/134warnings（1388.15秒），轻量1,053/695依赖skip/1warning（46.01秒），pip check通过；它不覆盖本次新增的selected-best评价/profile代码。

## 2026-10-27：评价数据准入完成真实mini检查

[新准入组件与证据](../results/emg2pose-memory-evaluation-data-engineering.json)在原样/文件行隔离两种明确模式下核对实际角色、条件及窗口计划；val混入其他角色先拒绝，full只接受真实完整25,253源。原数据集与评价函数代码保持，私有绑定仅提供封印源、声明的索引过滤与当前NPZ生命周期。新增回归实测发现角色/评价类型/逐用户设置改变未拒绝，以及原NPZ未及时关闭；保留RED后修正，本机31项115.57秒、服务器31项172.47秒通过，含禁用循环回收的实际fd检查。

真实647MB mini准入实际exit0，102.82秒、拥有的消费者/源fd均退出，父进程无Torch。原样user/user_stage各30文件/230窗；显式隔离模式22文件/167窗与8文件/63窗，行重叠0。没有读取信号窗口、构造模型或推理，Torch RNG不变；这是元数据准入，不能作为校准或benchmark成绩。10:08UTC三个DINO训练仍full-fit/error=None且子PID/start匹配；10:01UTC下载器仍存活，已至81,352,720,384源字节，重试计数1。完整源和三个250轮实验、完整推理/校准/测试/报告/控制器及Task4最终审查仍欠。

335348b的[托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40478615135)已成功：完整CPU1,865/3实际CUDA skip/134warnings（1064.92秒）、轻量1,053/694依赖skip/1warning（37.96秒），pip check通过；覆盖独立统计，尚不覆盖本次新准入源码。

## 2026-10-27：独立20角统计完成真实mini复推，发现原生条件人口重叠

[统计与实际证据](../results/emg2pose-memory-statistics-engineering.json)：本机/服务器各33项、两次真实mini受管理CPU推理190.52秒，父进程及两个阶段均exit0，所有消费者/fd退出。每次按未改官方函数消费460窗、核对56项原指标，完整输入/预测摘要与独立float64统计在新进程精确一致；原20角/left510/生成mask保持，私有逐用户记录不公开。仅工程前缀、原val、CPU/batch1/workers0，不是正式选best、校准、GPU或benchmark成绩。

[原人口审计](../results/emg2pose-native-evaluation-selection-audit.json)实测mini两个条件各230次消费来自相同30文件行与相同有序窗口；记录session ID重复，不能把它当文件唯一ID。私有观察器首次映射失败exit1保留，改为原loader迭代标签后重推通过，未改原筛选。进一步对完整CSV实际执行原函数，test/stage的3,539请求行会选中20,370行，包含16,831个train文件条目；这是元数据执行证据，不是全量HDF5窗口结果。完整评价必须分列原样`official_literal`和明确文件行隔离的`metadata_indices`对照，保留各自实际人口及事先冻结的验证校准，不能静默修正或混表。

[持久化接收](../results/emg2pose-persistent-body-retry-live-observation.json)08:54UTC已核验原76.974GB前缀并新增128MiB至77.108GB；一次真实响应体中断经同范围重试后提交，该块另独立解压/SHA通过，进程存活。09:00UTC三个DINO训练仍为full-fit、PID/start匹配、error=None。[托管CI](../results/emg2pose-memory-hosted-ci.json)7b8911f与8a05ae9已成功；8a05ae9完整CPU1,832/3及轻量1,021/693实际通过，覆盖重试组件但不覆盖新统计源码。完整源、三个250轮实验、完整评价/控制器/报告及Task4最终审查仍欠。

## 2026-10-27：持久化接收的同范围响应体重试通过本机与服务器检查

[独立组件与实际证据](../results/emg2pose-persistent-body-retry-engineering.json)针对07:29真实ReadTimeout中断：原核心仅在请求建立阶段重试，响应内容读取失败直接退出。新helper只重试原核心明确包装的网络响应体错误，最多三次body尝试；每次保留原协议/身份/长度核验，guard覆盖每次请求及退避两侧，失败块从不提交，协议/TLS/内容/资源错误不重试。真实loopback半途关闭、三次耗尽、后续对象变化和显式恢复完整SHA已检查，本机/服务器各76项实际exit0，原核心SHA9fb668不变。未将此传输检查计为完整数据、250轮训练或新增模型成绩。新的持久化续传须另行记录实际进程和前缀验证状态；旧失败退出保留。

## 2026-10-27：原预算fit/选模入口完成受控GPU工程验收

[正式fit入口的工程证据](../results/emg2pose-memory-native-fit-engineering.json)记录原factory/训练增强/验证角色、原320批量/12workers/bf16/clip1/cosine250及top3+last/val_mae/min/noearlystop。S/M/L各在显式合成的3个HDF5文件、25,710行及分开的角色上执行一个完整工程epoch，另一个真实进程恢复至两轮；每轮16次实际更新、5,120次消费，合计每模型32次更新。六类原生恢复状态全部一致，原生best/last/恢复前缀/epoch journal另由独立CPU消费者重新接受，七个受管理阶段实际exit0，457.00秒。合成数据不是官方mini或benchmark，每轮验证只有1窗；没有模型精度、完整源或250轮结果。此前真实mini526窗口GPU重推证据独立保留。

原`SessionShuffleSampler`按每rank的16个batch块循环补齐。本次真实执行先揭示了“ceil(unique_windows/batch)”的错误计数假设，现严格保留原采样器，单列唯一锚点（S1/M2/L2）和实际消费（每轮5,120）。CPU工程同样保留补齐：唯一锚点1、每轮16更新。新入口full默认要求完整25,253源、原角色及该入口实际CUDA资格的另行冻结SHA；full只允许250完整轮次，无子集/步数上限/工程早停。训练和原生验证选best后，独立reader核验实际检查点和完整journal；不自动打开test。错误模型恢复不改旧凭据，并发拒绝不泄漏锁fd，完成预算不能通过resume延长。

本机18项72.14秒、服务器同18项104.71秒通过；116成员固定源码整体1,798 passed/3实际CUDA skip/137warnings（1042.97秒），轻量987 passed/693依赖skip/1warning（40.35秒），均实际exit0且前后源码一致。仅剩三项skip为另行提供实际CUDA证据的classic专项。此前114成员7ed5be0的[真实托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40462128853)两项成功：官方1,780/3/128warnings（1285.42秒）、pip check通过；轻量987/692/1warning（43.43秒），已验证其磁盘测试目录修正，**不覆盖本次116成员fit代码**。

07:44UTC原DINO-S/B/L训练子进程身份均匹配，正在第57/39/17轮，已完成56/38/16轮完整验证，无终态错误或最终测试。主机可用527,075,725,312 bytes仍低于完整源737,836,173,312 bytes准入需求。持久化接收器已完成75.3GB旧前缀校验并新增至76,973,867,008源字节（压缩52,231,762,008）；07:29真实exit1，错误为当前范围响应体ReadTimeout、未提交该不完整块，已提交前缀保留。后续需处理当前范围的有界重试，不能因观察超时重启存活进程。完整源、三项250轮实验、验证校准/最终原逐用户评价/独立复推报告、完整控制器和Task4最后实现审查仍欠；本节不能填Tables11/13或验收全文。

## 2026-10-27：三模型原生GPU完整状态恢复及新进程完整验证通过

[新恢复入口与证据](../results/emg2pose-memory-native-gpu-recovery.json)使用原生factory、callbacks、TensorBoard和Trainer：S/M/L各两次batch320/12worker/bf16前缀更新及两轮完整526窗口验证，另一个真实进程严格重载后完整526窗口的输入/输出/顺序与全部原生指标精确一致，原生完整检查点的权重、AdamW、cosine250、callbacks、fit循环和Python/NumPy/Torch/CUDA随机状态六项一致，再实际更新一次至step3并完整验证。六个受管理阶段实际exit0，全部拥有的消费者退出；总计636.92秒。只排除PL2.5明确不加载的ModelCheckpoint.current_score暂态字段；原top3+last/val_mae/min/noearlystop/clip1保留，bf16没有FP16 scaler。

本机13项真实恢复/IO/拒绝/失败发布检查60.53秒、服务器同13项85.18秒通过；首轮真实prefix暴露导入顺序错误、额外真实故障注入暴露失败成功receipt未删除，均保留RED后修正。完整114成员冻结源码检查1,780 passed/3实际CUDA skip/131warnings（988.82秒），轻量987 passed/692依赖skip/1warning（35.95秒），均捕获实际exit0且前后源码一致。这不是250轮训练、独立验证或Tables11/13结果：仍复用mini原val及验证变换作为工程训练夹具，不证明完整人口/增强或连续训练轨迹；完整源、正式fit/选模/test/replay/report及Task4最终审查仍欠。

另核对上一111成员4d18fbd的[托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40455566293)：官方CPU实际1,767 passed/3实际CUDA skip（1205.46秒）、pip check通过；轻量4 failed/983 passed/691 skip（38.22秒），整次workflow失败。四个源准备子进程正确拒绝只有8GiB的/tmp。当前仅把轻量测试目录移至已有官方CPU层同样采用的磁盘RUNNER_TEMP，保留生产15GiB余量；新来源托管结果须另外观察，不把本机绿色替代云端结果。

06:34UTC原DINO-S/B/L训练子进程身份均匹配，无终态错误，正在第50/33/15轮，已完成49/32/14次完整验证，无最终测试。完整源RAM观察518,454,767,616 bytes仍低于固定737,836,173,312 bytes准入需求，未分配完整缓存。持久化接收器06:24真实exit3/stop_reason=projected_capacity，保留75,329,699,840源字节（压缩51,073,675,560字节）。核验三份本任务已通过、已退出测试的可再生成临时目录后释放约21.8GB，仅涉及测试副本；日志、原数据、真实模型结果和活动测试保留。空间重新满足剩余投影加15GiB余量后，原固定接收代码在独立进程续传，正在重验75.3GB前缀，尚未声明新增字节；没有自动重试或降低资源门槛。

## 2026-10-27：受管理的源准备与S/M/L原批量GPU容量检查完成

[源准备入口](../results/emg2pose-memory-preparation-engineering.json)把完整源盘点、索引生成和实际重新准入放在独立CPU消费者中，父进程持续检查内存/源/代码与输入文件。完整647MB mini实际运行83.47秒，30HDF5/4,218,756行及四字段与独立原转换参考一致，只生成四个小索引文件；所有消费者退出，父缓存仅由调用者关闭。full模式仍要求全部25,253会话和原train17,136/val1,950/test6,167划分；这些完整数据尚未取得，不能用mini成功代替。本机66项/9.86秒、服务器66项/12.87秒通过；当前111成员完整1,767 passed/3实际CUDA skip/125warnings（917.72秒），轻量987/691依赖skip/1warning（38.63秒），均捕获exit0。不同工作目录中的同名包误加载、子进程生产者身份不符均先以真实RED复现后修正。

[三模型GPU容量检查](../results/emg2pose-memory-native-gpu-capacity.json)使用实际A800、原320批量、12读取进程、bf16/clip1/AdamW/cosine250。每模型实际更新两次，每轮完整526窗口原生验证；全部三项实际exit0，完整流程309.37秒。S/M/L的Torch峰reserved为9.89/11.28/15.27GiB，设备最低空闲观测均高于8GiB；容量观测不等于OS/GPU预留。仅复用mini原val作为工程训练夹具，采用原训练sampler/collate及验证loader；完整训练人口、训练增强、GPU完整状态恢复、检查点选模和正式测试尚未核验，不发布这些短程指标为论文成绩。

05:20UTC实查原DINO-S/B/L控制器及子进程身份均匹配，仍在full-fit，日志零起算epoch41/27/11、step23,569/15,329/6,648，无终态错误。完整源RAM观察521,379,311,616 bytes低于固定737,836,173,312 bytes准入需求，未启动全量缓存。持久化接收器在独立进程中完成旧前缀校验后已新增数据，当前71,219,281,920/462,824,048,640 bytes，仍incomplete。上一109成员745a864的托管CI40450278844已真实成功（1,739/3及959/691）；它不覆盖新111成员。

## 2026-10-27：EMG2Pose拥有的进程管理通过实际服务器与真实mini流程

[新工程证据](../results/emg2pose-memory-process-engineering.json)覆盖父进程被SIGKILL/TERM/INT、仍持有缓存的孙进程、TERM拒绝、资源压力/未知容量、错误进程身份与真实封印字节。独立guardian通过SCM_RIGHTS把缓存权限交给自己创建的消费者后立即关闭副本，等待整个拥有的session实际退出；不会把单个父fd关闭当成缓存已释放。服务器Python/libc缺pidfd包装的真实失败保留，按主机x86_64内核UAPI执行同一操作的兼容层经两项RED→GREEN和实际20项检查；本机20项通过。

完整647MB mini随后实际通过新进程管理和既有small工程检查点的严格重载，未修改的官方逐用户函数在原转换数据与新存储上结果完全一致。该路径开启IK过滤，460窗口（230/230）与此前526验证窗口分列；CPU/batch1/workers0、两组各1个mini用户、没有新优化或科研成绩。缓存消费者、父fd均关闭，运行中实际观察guardian无缓存fd；完整流程126.50秒。最终109成员整体1,739 passed/3实际CUDA skip（863.65秒）、轻量959/691依赖skip（31.22秒）均实际exit0。Task4完整源/原生GPU/三项完整实验及其最终实现审查仍欠，不能填Tables11/13。

上一107成员c46c345的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40447571670)也已成功：官方CPU1,719/3实际CUDA skip及轻量939/691，pip check通过；它不覆盖随后新增的109成员进程模块。

## 2026-10-27：EMG2Pose新内存读取器通过全部mini窗口和三模型原生CPU检查

[实际证据](../results/emg2pose-memory-native-engineering.json)记录新索引/官方存储子类/原生factory绑定。全部526窗口的输入、标签、mask、样本元数据、RNG消耗及会话分组与原读取器一致；S/M/L各2次真实更新，全部526预测在新进程重载后精确一致，原生Trainer恢复权重/AdamW矩/cosine/循环计数后再执行一次实际更新到step3。所有拥有的memfd关闭；CPU工程进程显式隐藏CUDA，保留此前可见GPU时最后资源检查失败。一次实现审查发现最终HEAD阻塞时缓存未持续检查内存压力，真实两项RED→77项GREEN已修正。最终107成员完整1,719通过/3实际CUDA skip（863.83秒）、轻量939/691依赖skip（29.71秒）均捕获实际exit0；修后全部526窗口与三模型原生检查再次通过。此前1,717通过但缺退出码的日志仍保留为历史，未追认exit0。全部mini为原val，不能作为独立验证或论文成绩。

原101成员盘点代码的[托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40433812371)已实际完成：官方CPU1,652 passed/3实际CUDA skip/104warnings（963.93s）及轻量919/689依赖skip/1warning（28.30s），pip check通过。它不覆盖后续新适配器。完整462.8GB源、资源/进程控制器、原GPU批量准入和三项250轮科学实验仍欠；11:56北京时间实读持久化前缀70.4GB（15.21%）且原续传进程PID/start仍匹配。同期DINO-S/B/L控制器及训练子进程均存活、无终态错误，日志分别在零起算epoch33/20/8；仍没有新DINO最终测试成绩。

## 2026-10-27：显式重计算L七阶段GPU验收通过，S/B/L均进入各自完整训练

[实际L GPU证据](../results/vision-literal-large-checkpoint-gpu-engineering.json)已重新打开原始检查点／验证预测核验：原360／bf16、两轮／4次实际更新，每轮720训练消费与完整3,152验证消费；模型、优化器、调度器、回调、循环、RNG六项恢复一致。原生验证与新进程独立复推的3,152个预测精确一致，七阶段实际exit0。原生timm激活重计算显式开启，原模型／损失／优化器／批量保留。前两阶段设备最低观测空闲60,736,667,648／61,915,267,072bytes；这是全设备观测，不冒充本进程峰值显存。

L随后从真实预训练权重另起full；2026-10-26T23:59:08UTC控制器及full子进程PID/start均存活，身份中的scope=full、max150／patience30、训练限制=None、重计算=True，已记录训练step87，尚无完整epoch。原L未启用重计算的显存失败保留。S、B继续各自原冻结源训练，三个新DINO实验均未完成最终测试，不发布工程验证MAE为模型成绩。

匹配代码2129accc的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40420859582)两项成功：[原始日志摘要](../results/vision-literal-activation-hosted-ci.json)为完整1,574 passed／3实际CUDA skip／104warnings（1003.73s），轻量841／689依赖skip（20.85s），pip check通过。本机1,573／4及服务器单独补验仍按原范围记录，不能改写成本机1,574／3。

## 2026-10-27：L显式激活重计算实现完成CPU检查，S/B训练继续

[显式适配与实际证据](../results/vision-literal-activation-checkpoint-cpu.json)保留原360／bf16／损失／优化器／回调，只启用固定timm的原生激活重计算。真实DINO-S前向、全部梯度和反向RNG一致，原生训练恢复六项一致；38项原生fit／评价通过，一次独立实现审查无问题。完整97成员检查1,573 passed／4 skipped（767.49s），轻量841／689依赖skip（25.50s）；本机ViT-L内存skip已在同源码服务器用原测试脚本补做成功，另三项需要实际CUDA。最初漏传官方源码目录的检查在确认范围不足后仅停止拥有的测试进程组，exit143和原日志保留，不计成功。

新L代码／真实权重已在独立目录暂存，完整52输入再次hash及有限性准入72.00秒，匹配原L冻结receipt。代码2129accc已实际推送；独立GPU6验收已启动，2026-10-26T23:38:36UTC核对控制器和子进程PID/start存活，处于capacity-prefix，尚无完成容量或full成绩。原L显存失败、S/B源快照和七项legacy成绩保持原身份。2026-10-26T23:32:06UTC实查S完成6轮／3,372更新、B完成2轮／1,124更新，同一训练PID/start仍存活；均无最终测试成绩。

同时补齐[旧四模型派生GPU限定证据](../results/legacy-derived-bounded-gpu.json)：原四阶段均exit0，各4手中心，13份服务器receipt与保存归档一致。这不是新增模型训练或完整手势表；工具仍需完整batch准入及2,792手全队列pass，不把限定结果写成Table14/15。

## 2026-10-27：DINO-S/B实际GPU验收通过并全量训练，L显存准入失败

后续实查：B已独立通过全部七个GPU验收阶段，[实际B证据](../results/vision-literal-base-gpu-engineering.json)为两轮／4次更新、每轮720训练与完整3,152验证、六项恢复一致及完整验证预测精确复推；新full-fit已启动。2026-10-26T23:05:58UTC再次核对S/B各自控制器及训练子进程的PID/start均存活，均处于full-fit，无终态错误。L失败仍保留，显式原生激活重计算适配另行验证，不改写原S/B源快照。

[实际S GPU证据](../results/vision-literal-gpu-engineering.json)记录原batch360／bf16短程两轮、4次实际更新，每轮720次训练消费及完整3,152次验证消费；新进程原生恢复的模型、优化器、调度器、回调、循环和RNG六项均一致。完整验证的原生评价与另一进程独立预测复推精确一致，七个容量／恢复／验证阶段均实际exit0。另起的真实预训练S完整训练PID/start已核实存活；证据快照为完成2轮／1,124次更新，原150轮上限与patience30保留，最终测试尚未进行，不发布短程验证为模型成绩。

B/L分别完成独立协议的完整输入receipt冻结，实际输入和分区与S一致，各在新输出启动原batch360验收。B第一轮成功，正在检查新进程恢复；L于2026-10-26T22:52:26UTC触发既定8GiB显存余量门槛，最低观测设备空闲45,088,768bytes，未完成首轮，原失败输出保留，控制器和子进程均已退出。该失败不是成功的容量验收，也未自动缩batch或重试；后续单独核验原生激活重计算适配。S/B运行目录保持冻结。

匹配e7c20b9的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40412185942)两项成功：[日志摘要](../results/vision-literal-e7-hosted-ci.json)为完整1,564 passed／3实际CUDA skip／104warnings（873.77s），轻量841 passed／679依赖skip（26.80s），pip check通过。这不替代实际模型训练与测试。下文保留早期快照，七项legacy结果保持原字节。

## 2026-10-27：完整vision训练／评价实现与实际输入准入

52个实际视觉输入文件已完成整内容hash及全部可能中心目标有限性扫描；原生loader的202,020／3,152／8,612重复消费人口、360批量和原全局RNG已实读核对。新增完整原生fit、epoch恢复、验证最优检查点读取以及预测试验证复推／冻结规则、原生测试、独立复算、匿名逐角报告入口。93成员fit阶段整体验证为1,535 passed／3实际CUDA skip（406.64秒），轻量841／650依赖skip；随后完整评价与报告的95成员整套检查已通过1,547／3实际CUDA skip；一次完整视觉实现审查发现两项Important，分别用真实RED复现后修正：fresh full必须匹配事先冻结的整内容receipt，原生／复推模型摘要必须绑定实际选中检查点。修正后43项定向通过，96成员整体为1,563 passed／4 skipped／106 warnings（694.08秒），轻量841／679依赖skip（22.85秒）。第四项skip为本机内存不足的独立legacy ViT-L随机CPU构造检查；另三项需要真实CUDA。该项已在同96源码服务器目录按同测试脚本补做L随机构造／参数schema并通过，本机1,563／4的原记录不改写。此时新DINO GPU验收及完整训练仍未运行。

VEMG报告12阶段现已全部实际exit0，最后verify-completed于2026-10-26T21:40:03UTC结束；[新公开结果](../results/original-classic/vemg2pose/README.md)的轨迹／中心MAE为16.452932°／16.347705°，与旧七项分列。EMG2Pose原接收器因HTTP错误真实结束；已核对原下载源身份和下一段206范围响应，另以原完整prefix校验、340GiB预算和15GiB余量规则显式续传，旧失败保留。DINO一次完整代码审查及根修正已完成；新GPU容量和完整训练尚欠，七项legacy结果保持原字节。

## 2026-10-27：VEMG训练结束，原生评估已启动；DINO原生模块与CPU短训

VEMG正式训练已真实终止：62轮完整验证、10,540次尝试、10,535次实际优化器更新，5次自然overflow，原patience50触发early stopping。原训练PID/start已不存在；identity/result/status/last四项终态内容摘要与82项冻结报告源码重新核对。新报告流程已完成终态检查、两次完整验证推理及预测试一致性策略冻结／复核，正在运行原生最终测试。最终测试、独立复算和公开完成报告尚未全部结束；不把验证损失当作测试MAE。

DINO原生构造／重载已实现：直接调用原setup_runtime和factory，实际174项主干张量与真实预训练权重经过原位置插值后的结果一致；完整178项状态严格重载和预测一致，原生head/backbone AdamW与cosine150保留。真实小memmap/JPEG上的两批原生CPU训练、新进程恢复后再更新一批已核对模型／优化器／调度器／callback／loop／RNG，完整小验证预测一致；这是工程证据，没有新DINO性能成绩。62项定向检查通过56.03秒；随后真实文件缺口回归1项RED→GREEN（2.64秒）。新89成员整体检查已通过1,506 passed／3实际CUDA skip／106 warnings（340.64秒）。真实旧版队列S先更新2次、完整验证3,152手，新进程完整重推一致后再更新1次并重新验证3,152手；六类状态恢复一致。B/L原生构造与严格重载也已完成，尚无其cohort优化。[脱敏CPU证据](../results/vision-literal-native-cpu.json)保持工程范围。

首次真实队列CPU预检因错误要求清单中声明、纯视觉却不读取的缺失raw EMG而在构造前停止，原目录和失败记录保留。修正后文件身份取自实际点读取器已打开的memmap、split、metadata和crop；不补造不存在的文件。另一次整体检查为1,504 passed／3实际CUDA skip／1失败，原因是本次过长basetemp超过AF_UNIX地址长度，已修正调用路径；未修改旧IPC测试或冻结classic源码。新轻量层841 passed／621依赖skip／1 warning（22.57秒），新完整层已在短路径下重新通过。

此前74e3f4d的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40376769468)已确认两项success：官方1,480 passed／3实际CUDA skip／104 warnings（314.74秒）、轻量841／595（25.44秒），pip check通过。该CI对应上一配置／数据基础提交，不覆盖当前原生模块。新完整vision runner、一次完整分支审查、GPU容量和完整训练仍欠；七项legacy结果保持原样。

## 2026-10-27：DINOv2官方配置与真实数据读取准备

**新literal vision配置／数据层完成CPU集成和实际服务器只读扫描；尚无原生模块优化或新模型成绩。** [语义、分母和缺口](vision-literal.md)、[CPU证据](../results/vision-literal-foundation-cpu.json)及[原始脱敏数据扫描](../results/vision-literal-data-census.json)记录实际原生数据模块的三个重要行为：所有loader均用batch360，声明的val_batch_size256未读取；所有角色repeat2；当前lineage声明EgoEMG/ShowEE/Incre合并数据，不能把可用legacy EgoEMG视作该完整来源。S为原YAML，B/L只作明确S派生，尚无作者独立配方等价证明。

新的7790/780/7790窗口保留原train jitter完整随机范围；预先排除352个可能跨入独立val/test足迹的锚点，不裁剪随机范围。实际唯一／repeat2消费手样本为train101,010／202,020，val1,576／3,152，test4,306／8,612。原point方法的RGB、腕角舍入、生成mask及缺失key语义保持，真实小memmap/JPEG和原读取器数值／RNG逐字节核对。37项定向通过（12.21s）；固定84成员完整1,480 passed／3真实CUDA skip／106 warnings（295.00s），轻量841 passed／595依赖skip／1 warning（22.60s），此前80源码及七项legacy结果未变。新分支最终独立审查与托管CI尚未完成，未以CPU检查授权GPU训练。

服务器84文件逐一hash核对；52个实际数据/metadata/LMDB文件前后stat稳定。扫描全部39,393,900个可能训练中心、788个验证中心和2,153个测试中心，目标均有限，检查所有可能视频帧手key并记录缺失；仅18个点读做JPEG解码与原始目标核对，未整文件重hash所有payload。这是数据准备，不是模型推理或性能评价，GPU未初始化。后续原生Lightning／预训练保留／优化器／重载和完整runner仍需实现。

上一报告实现e53e86a的[实际CI40366479641](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40366479641)两项success：官方1,443 passed／3真实CUDA skip／104 warnings（385.70s）、轻量835 passed／564依赖skip（24.11s），pip check通过。该CI不覆盖本次vision新增源码。

VEMG2026-10-26T18:08:59UTC（北京时间10月27日02:08）实读8,547次真实执行、50次完整1,576手验证，相同PID/start仍存活，原controller session仍运行。报告代码及12阶段私有执行器已经准备，尚无真实终态、GPU校准／原生测试／独立复算或新科学成绩；全文目标继续进行。

## 2026-10-27：测试独立复算、只读状态与完整报告实现

**完整报告实现已完成一次审查修正和CPU集成，实际GPU执行仍欠。** [使用方式](classic-report.md)与[新CPU证据](../results/classic-report-complete-cpu.md)覆盖原4,306手测试预测参考／独立复算、类型与统计守恒、真实metadata匿名映射及用户交集、只读日志状态、JSON／Markdown和最后完成标记。审查的2项Important由19项实际RED复现，一次修复后定向303 passed／2 warnings（12.95s）；固定80成员完整1,443 passed／3真实CUDA skip／106 warnings（284.78s），轻量835 passed／564依赖skip／1 warning（22.74s）。全部源摘要和七项旧结果不变，没有第二次独立复审。

训练统计现在与严格读取的终态last日志和实际运行时策略逐类型核对，适配字段匹配固定原声明及实际策略；不能重写result／status便导出错误计数或嵌套私有值。Markdown附完整未舍入公开映射，保留协议、停止、无效分母、手×split、用户离散度、单位和来源。服务器新独立目录已逐一核验80文件；真实只读API实读6,613次执行、38次完整验证，CUDA读取前后均未初始化，没有更改运行中的源码。

[82109cd实际托管CI40356183999](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40356183999)两项成功：官方1,351 passed／3真实CUDA skip（372.13s）、轻量806 passed／501依赖skip（19.87s），pip check通过；它覆盖前一原生绑定版本，不覆盖本次新增源码。本次托管CI尚未观察。

VEMG于2026-10-26T17:10:49UTC（北京时间10月27日01:10）实读6,677次真实执行、39次完整1,576手验证，相同PID/start仍存活，controller为training。尚无终态、实际GPU校准／profile／原生测试／测试独立复算或新科学完成标记。七项legacy成绩不变，全文目标继续进行；下文均保留各自历史观察时间。

## 2026-10-27：原生测试绑定的CPU实现与CI修复实读

**原生测试父子进程／profile／pipe绑定及raw读取已实现，实际GPU测试尚未执行。** [入口](classic-report.md)与[CPU证据](../results/classic-report-native-gate-cpu.md)记录实际child身份在GO前持久化、child独立准入、原五份测试产物／role-best快照和4,306手覆盖读取。独立审查的1项Important通过4个真实文件替换RED回归修复；最终result／status保留创建时FD／inode，失败时撤销本次文件并保护外来替换文件。修后固定80成员完整1,351 passed／3真实CUDA skip（283.35s），轻量806 passed／501依赖skip（26.97s），定向211 passed；没有第二次独立审查。实际pipe探针不执行Torch，父进程故障探针刻意exit17，不当成CUDA正例。

**b4608f8的实际托管CI两项成功：** [40347752496](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40347752496)官方CPU1,250 passed／3实际CUDA skip（257.27s）、轻量762 passed／444依赖skip（24.56s），pip check通过。仅移动pytest临时目录解决原tmpfs容量问题，生产15GiB门槛未改；下文586失败仍为原始历史。该CI不覆盖上述原生绑定新源码，新源码托管结果尚未观察。

VEMG于2026-10-26T16:02:14UTC（北京时间10月27日00:02）实读4,469次真实更新、26次完整1,576手验证，相同PID/start仍存活，controller仍为training；尚未完成训练或最终测试。测试预测参考／独立复算、只读状态API、脱敏科学导出及实际GPU校准仍欠。七项legacy成绩不变，全文目标继续进行。

## 2026-10-26：全论文目标继续执行

用户将目标扩展为论文全部内容，歧义优先官方代码。下文 2026-10-25 四基线完成记录保留为历史证据，不能作为新目标完成声明。[全文清单](full-paper-status.md)覆盖全部 15 表、13 图、17 项主要姿态模型设置及标注/预处理流程。

**586c36f的真实托管结果已读取：完整CPU检查失败，旧记录保留。** [CI40345052715](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40345052715)的官方CPU为1244 passed／3真实CUDA skip／6 failed（361.86s），轻量762 passed／444依赖skip（23.73s）。6项新文件发布fixture均触发原15GiB保留门槛；日志同时记录`/tmp`为7.9GiB tmpfs、剩余7.5GiB，而工作磁盘剩余86GiB。新修正仅将官方CPU的pytest basetemp移到`RUNNER_TEMP/egoemg-pytest`并补对应磁盘诊断，原运行代码、15GiB门槛和真实低容量拒绝检查保持不变。新提交的实际托管结果待实读；下文1250通过是本机结果，不能覆盖这次托管失败。尚在开发的原生测试启动绑定不属于该公开修正。

**新增验证集报告准备实现，未完成实际GPU校准。** [入口与边界](classic-report.md)提供原22角／原中心算术、完整1576手独立预测、逐批原dtype精确复算和原始证据绑定profile；测试启动绑定、测试预测参考／独立复算及最后科学导出仍欠。[修复后本机证据](../results/classic-report-validation-cpu.md)为定向110 passed，固定80文件完整组1250 passed／3真实CUDA skip（276.93s），轻量762 passed／444依赖skip（21.63s）。独立静态审查的3项发布问题经7个RED回归修复，没有第二次复审；未改formal/evaluator/artifacts或旧七项结果。本次源码托管CI尚未观察。VEMG于2026-10-26T14:51:54UTC实读2213次真实更新、13次完整1576手验证，相同PID/start仍存活；这是历史快照，尚未完成训练或最终测试。下文各次早期状态按原时间保留。

**当前新增：VEMG已通过最终源码GPU限定验收并进入完整训练；NeuroPose预测检查失败。** [实际证据](../results/classic-formal-gpu-engineering.md)记录六个GPU进程均成功，VEMG汇总与fresh复核成功，Neuro汇总在8/171,380个预测元素处超出[固定规则](classic-canonical-comparison.md)，原失败保留、门槛未放宽。两者所有工程train/val批次输入一致、checkpoint-to-live严格恢复，最终浮点状态与完整RNG不逐位一致。VEMG另起full cohort训练，train/val为101,714/1,576手、max100/patience50/原同B600；2026-10-26T13:52:31UTC实读309次attempt及真实执行、1次完整验证，原PID/start仍存活。这是运行快照，未完成训练或最终测试。匹配658d8c6源码[实际CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40330846529)为1173 passed/3实际CUDA skip及739 passed/390依赖skip；本机78成员及七项旧结果未变。下文历史检查不覆盖后续报告实现。

**最新结果：WiLoR真实预训练与禁网CPU检查已完成**，见[完整证据](../results/wilor-pretrained-cpu.md)。原下载终止后，新独立attempt保留并复制原片段，以精确16MiB Range/ETag和整文件SHA完成2,564,989,533-byte官方权重。随后原WiLoRViTPose两次真实构造各404个主干tensor匹配；实测631,631,657总参数、52,499冻结，原高度裁剪192×256、1×1280×12×16特征和1×22×1有限输出、严格模型重载预测逐位一致。网络尝试0、CUDA未初始化；这是合成CPU接入，尚无WiLoR完整训练。

classic最新6adc三条VEMG路径（连续两轮、prefix单轮、fresh恢复）实际全部成功；汇总因模型tuple/JSON-list metadata比较失败。该窄修及原生恢复后未选新best时current_score=None的合法last保存回归，完整77文件1125 passed/3实际GPU skip（281.02s），轻量739 passed/389依赖skip（21.79s），原结果和源hash核对一致。只读后续诊断另揭示浮点轨迹不逐位相同、train-loader base-seed多一次draw；当前验收仍拒绝，尚无最终GPU资格或完整classic训练。原全局RNG/采样器状态及scheduler/scaler计数相同，差异与下一步工作详见[classic范围](classic-emg.md)。

最新回调恢复修正：原生CPU Trainer的checkpoint→fresh Trainer在首次恢复forward前真实表明，ModelCheckpoint.current_score虽保存却不恢复，旧完整callback digest因此拒绝；actual GPU prefix为非空tensor，原失败保留。仅在比较副本规范这一非恢复字段，原checkpoint/保存校验和其他可恢复字段不变。新增回归真实RED→GREEN、定向155 passed/1实际GPU skip；77文件冻结组完整1119 passed/3实际GPU skip（263.56s），轻量739 passed/383依赖skip（20.74s），所有冻结hash再核一致。新6adc源码已在独立目录暂存并重验55个输入pin，GPU验收已再次启动；完整训练尚未开放。

`4925caada9b13acacc38cf347185f8eaf7be1f17`已正常公开，[实际CI40319331373](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40319331373)两项成功：官方CPU1118 passed/3真实CUDA skip（277.90s），轻量739 passed/382依赖skip（19.72s），pip check通过。它覆盖上一artifact/evaluator组，不覆盖此后6adc回调比较修正。

最新修正组77文件实际完整CPU **1118 passed/3 skipped**（271.95s），轻量 **739 passed/382依赖skip**（21.40s）；三个完整环境skip仍为真实CUDA检查，七项已完成结果和所有冻结成员hash不变。新增selected-best独立测试入口已修复实际权重载入及目录移走后本次完成标记撤销问题，原探针与33项定向测试、独立复审通过；尚未做正常终止完整训练后的GPU测试或报告。

此前5285正式入口的首次GPU工程运行真实到达两次训练尝试和完整1,576手val，随后首轮保存被原生AdamW校验拒绝：唯一差异为Hydra betas的ListConfig与临时期望参考默认tuple。原失败输出保留，未保存checkpoint、未进行后续阶段，也不是OOM。仅修正临时参考的列表语义后，两原配置真实helper、严格错误值拒绝与独立复审通过；新独立源码集合5285+187b逐一实读55个输入pin后再次实际GPU验收：VEMG连续两轮及单轮prefix均成功，第三个新进程在训练前的完整状态恢复核对失败，原输出与checkpoint保留。尚未执行Neuro阶段或完整训练；新的恢复差异另行诊断，不把前两阶段成功当整个验收通过。共享固定GPU每阶段启动需64GiB空闲、运行监测需8GiB，停止仅作用本次child；耗时不当作论文独占GPU-hours。

上一公开头`2927aabfbe5a7a8e2f7003cc276ba729d0e24745`的[实际CI40310839399](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40310839399)已成功：官方CPU1076 passed/3真实CUDA skip（300.14s），轻量727 passed/352依赖skip（17.34s），pip check及固定旧commit有界fetch实际通过。该CI不覆盖随后这组artifact/evaluator修正。

WiLoR另取得固定官方提交的26个源码/配置/许可文件，逐文件Git blob hash一致；隔离依赖与mean参数校验后，原get_config/backbones.vit在禁网、隐藏CUDA的新进程导入通过，无模型构造。大权重下载因ChunkedEncodingError已终止，93,323,264字节partial保留，尚未完整取得或加载，不能称预训练模型验收。完整EMG2Pose接收器原PID/start_ticks仍匹配，最近实读23,605,542,912原始字节（约5.1%），complete=false。

2026-10-26 18:50新增核验：[classic正式训练入口](classic-emg.md)的5285源码完成独立复审及75文件统一CPU集成，**1076 passed/3 skipped**（267.26s），轻量**727 passed/352依赖skip**（21.59s）。此次ViT-L CPU构造实际通过；三个skip均为真实CUDA检查，不能当作GPU验收。训练收尾的两种真实故障已修复并保留原RED记录。服务器新独立目录的13个源文件、依赖、两配方原数值证据和各三梯级容量证据实读重验通过，CUDA未初始化；尚未运行最终源码三进程验收或完整classic训练。后终止selected-best完整测试和报告仍欠。

公开提交`6599e05da0edbec2adac570c9f75cc43f927c69b`的[CI40306913437](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40306913437)实际失败：官方CPU1013 passed/2 skipped/11 failed/2 errors，轻量678 passed/337 skipped/11 failed/2 errors。原namespace夹具在浅checkout中缺少固定8dc提交，另有事务测试受小于15GiB的临时文件系统影响。新修复只给CI有界fetch该固定commit，并固定事务夹具的足够空间前提、补充真实低容量/下降容量拒绝用例；生产15GiB及worker源码不变。修复已独立复审并纳入上述本机集成，新提交托管结果仍须另读，旧6599失败不改写。

旧四模型派生评价的实际服务器预检也已通过：原十个运行时文件、原始last/best及协议身份完全匹配，四项均在未导入Torch的条件下准入。随后尝试启动有界检查时，所选GPU已被其他进程占用，资源门槛在创建attempt/claim前拒绝；未进行新推理或完整派生pass，未改旧数据、权重或结果。

截至北京时间2026-10-26 17:57的实际观察及完成记录：

| 新实验 | 已通过的真实门槛 | 当前阶段 | 冻结运行代码 |
| --- | --- | --- | --- |
| EMGFormer-L | batch500 两次真实优化；严格重载；两次各 1,052 样本完整验证一致 | **200 epochs/25,400更新/200次完整轨迹验证已完成**；最终测试、严格导出与独立中心复算均成功，[结果](../results/completed/emgformer-l/README.md) | `7b805d428957e62cc93ff48f34e954c64159691e` |
| ResNet-50 | 真实 ImageNet1K_V1 权重及哈希；batch400 两次真实 crop 优化；两次完整验证重载 | **150 epochs/23,700更新/150次full val已完成**；最终测试、严格导出、锁定复算均成功，[结果](../results/completed/resnet50/README.md) | `a357423b722b8df3e15b78b88b897c6087a04884` |
| ResNet-152 | 真实 ImageNet1K_V1 权重及哈希；batch180 两次真实 crop 优化；两次完整验证重载 | **150 epochs/52,650更新/150次full val已完成**；最终测试、严格导出与锁定复算均成功，[结果](../results/completed/resnet152/README.md) | `a357423b722b8df3e15b78b88b897c6087a04884` |

三项使用独立目录与不同 A800；均保留原四基线的 legacy 队列作为这一实验族。RN50/152 的独立作者 YAML 未随固定代码发布，本项目显式组合官方 RN18 experiment 并覆盖 backbone；批量来自 PDF Table8，其余优化/数据配方继承已声明的官方 RN18 和 legacy 适配，不能称作者完整原始配方。每个单次控制流程只有短验收通过后才开启正式训练，正式训练结束后才导出并做一次锁定中心审计；三项后续阶段均已完成，仍不构成官方字面配方等价。

历史13:27读取三项生产进度分别处于零起始epoch 89/200、131/150、69/150，step为11,349/20,806/24,371；原controller/child身份均匹配。随后13:51的实际读取：RN50的controller已在13:48:16成功完成，完整训练、严格导出与锁定复算三个阶段均exit0；L/RN152仍由相同controller/child运行，零起始epoch100/200与80/150、step12,783与28,297。三项仍是已声明的legacy 0.01损失协议，不能据完成状态升级成官方0.02等价。

RN50的best由完整独立验证选出第19,118步（第121轮）；全部中心MAE6.640058°、有图子集6.535572°。2,792请求/59,572有效角与原冻结测试人口一致，独立审计14个分组在0.0001°/计数精确一致的门槛内通过，有图子集也一致。原JSON/MD/audit保持字节哈希，另由同一metrics生成曲线与[独立发布清单](../results/completed/resnet50/publication-manifest.json)，不改写旧四模型清单。该成绩保留数据/标签/窗口/损失/单卡训练适配的边界。

RN152控制流程于16:20:22完成，五个阶段均exit0。完整独立验证最小值为5.900450°，best在第52,650步（第150轮）；全部/有图中心测试MAE为5.982738°/5.869706°，人口与RN50相同，14组与有图子集均通过独立锁定复算。新增原始JSON/MD/audit、曲线和[发布清单](../results/completed/resnet152/publication-manifest.json)已独立核验；原四项与RN50发布清单不改。单次seed42、0.01目标、缺独立作者YAML和legacy人口等限制继续有效。

新增[classic CPU训练/恢复](classic-emg.md)已提交`ec2d5b127828578dd3c125c44d7293dfab105c4c`并正常公开推送。两原模型真实小夹具连续运行与只携last的离线epoch恢复一致；原clip0、batch加权val_loss/min、双别名0.02、T_max100与patience50均有CPU执行证据。整体本机672 passed/1既有ViT-L内存skip（215.40s），无Torch层490 passed/183依赖skip（16.72s）；这是57文件冻结快照，不含后续实现，托管CI需另读。整个classic Task3尚欠实际CUDA/GradScaler和正式full train/val-best/test，Task4仍未完成。

服务器classic依赖已在新独立目录真实安装并导入PL2.5.0.post0、torchmetrics1.4.3、utilities0.11.9、pandas2.2.3和joblib1.5.2；共享Torch2.2.2+cu118/NumPy1.26.4未改变。ec2d5b1的Git归档34个源码/配置文件已核SHA并独立暂存、两配方实际合成成功；没有据此声称GPU验收或正式classic训练已开启。

新[手形统计](../results/dataset-statistics/shape-statistics.md)已从实际左右 beta 重算并独立核验，41 个引用行、82 手，官方 float32/ddof0/range 的 50 个统计重算误差为零。与 PDF Table5 的 50 个显示数比较，48 个按三位小数吻合；beta6 range 和 beta10 mean 两个微差照实保留。官方 range 与论文列名 P95−P5 不同，实际线性分位差另列；箱线图是聚合展示适配，不冒称官方逐 episode 柱状图。

数据获取：额外 beta/校准共 4,646 bytes 已校验；EMG2Pose metadata 已下载并核验大小、25,253 行与分组。官方公开 S3 mini 已完整取得，647,198,208 bytes、30 个 HDF5 全为原 val；已核验 tar 与各成员哈希并安全提取，只作工程验证，直接转换与官方 reader 核验已完成。完整 S3 tar 的无损分块接收已启动，使用显式 SSH 路由、逐块 SHA256 与严格 Range/源身份校验，独立存储预算340 GiB、动态空闲空间保留15 GiB；已观察到首块之后的新块成功提交，仍为 incomplete。legacy raw/几何补充传输尚未完成；两个 MANO pose 与相机 transform 收到明确 Google Drive quota HTML，已由大小门槛拒绝，未计为数组数据。用户无法下载百度当前发布，现阶段使用已有数据与公开 S3 推进，不等待该手动步骤，见[下载指南](download.md)。用户确认手头没有缺失的 markers2mano 包/训练资源；继续查找公开实现与明确缺失规格，不以未训练替身验收。

legacy raw的直连下载因TLS握手中断退出。随后通过显式SSH路由单次恢复，左右已有前缀末尾各1 MiB与远端严格206响应逐位一致，left新增两个完整16 MiB块后再次TLS中断并停止，未连续重试。当前left/right保留956,301,312/368,050,176 bytes，均没有完整获取receipt；不影响独立进行的S3全量接收或已有训练。

新增代码已通过用户为该仓库配置的专用 SSH 写入身份正常推送到[全论文分支](https://github.com/ZSHYC/EgoEMG-Reproduction/tree/codex/full-paper-reproduction)，冻结训练提交保持原始身份，见[发布来源](publication-provenance.md)。`2a779dfccb0efec952c28c014d23c8aedc325826` 的[实际托管 CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40264031015)两项均成功；草稿 PR 尚未创建。

此前 RN/统计组合记录为238 passed。后续已接入 DINOv2 S/B/L、完成mini数据转换与可恢复xz归档，各任务独立审阅通过。统一本地运行曾为397 passed、1个ViT-L内存准入skip，随后该ViT-L用例在独立新进程通过；历史skip不改写。公开CI先暴露轻量环境缺官方源码/绘图依赖，再暴露测试fixture依赖runner实际空闲空间，均已修复。最新实测 `2a779df` 的官方 CPU job **398 passed、无skip**，无Torch轻量 job **350 passed、48个可选运行时skip**，两项shell语法检查均通过。容量fixture只固定测试磁盘快照，生产15 GiB reserve及低/变化容量测试未削弱。该CI成绩只证明对应提交的工程检查，不证明完整训练或论文数值。

新增mini数据[工程报告](../results/emg2pose-mini-data.md)记录修后独立目录的30文件、4,218,756行、645,469,668字段字节、1,015 valid blocks以及360官方reader窗口位级核验。旧转换来源与新来源分开保留；没有优化模型、没有独立val性能、没有完整EMG2Pose实验。

Task3的[官方模型入口与真实mini CPU证据](../results/emg2pose-mini-runtime.md)已经完成：S/M/L各检查1个实际窗口，16ch/20D、7790→7280、归一化、构造前seed、严格禁网重载均核验；526是可用窗口数，未做完整526评价。独立审查的字段hash后变更漏洞已用真实RED修复，11个新变更用例通过，Spec/Quality最终均PASS。修后核心统一检查为445 passed、1个未改ViT-L内存准入skip；真实无Torch环境379 passed、67个可选依赖skip。两项运行均明确排除仍在实现的semantic模块测试，不将未完任务计入。新提交托管CI仍需对应实际运行另验。

该Task3代码已公开为`79153e1e3895af2f216cebcc927528cff8e0b1a4`。[对应托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40268110037)实际成功：官方CPU **446 passed、无skip**，轻量379 passed/67可选依赖skip，两项shell检查通过。新增官方Lightning依赖在CI真实安装并通过pip check；本机历史内存skip仍保留，不改写为本地通过。Task4实现与小夹具的新增验证单独列于下文，不由旧CI计作完成。

[60类词表](../results/dataset-semantics/gesture-vocabulary.md)、[22维关节语义](../results/dataset-semantics/joint-angle-semantics.md)与[实际同步帧覆盖](../results/dataset-semantics/coverage.md)现已独立审核通过：66,161,725行中全部60类确有观测，raw50有86,952行，active/inactive与四split/三族群总数守恒。原帧审计实读并重哈希三个全数组，当前导出使用该审计并核对来源与聚合，不声称再次扫描。窗口中心、手计数、有效角度和模型误差仍为not_computed。

[classic模型入口](../egoemg_repro/classic_emg.py)也已独立审核通过：四个基础/with_aug配置严格白名单，原始双loss key、原模型tuple及2D mask保留；VEMG明确8ch/WL7790→6000、NeuroPose16槽/WL7790→7790，真实参数和错误布局负例通过。仅为模型接口工程验收，未进行优化或完整训练，loader/mask/固定scalar归一化与FP16仍待实现。

当时新增上述模块的53文件快照，本机CPU整体检查为**518 passed、1个既有ViT-L内存skip**（66.09秒）；无Torch层为**433 passed、86个可选依赖skip**（15.92秒），采用与轻量CI相同的不设置EGOEMG_UPSTREAM方式。两项均排除当时未完成的Task4；后来的托管结论另列如下，旧Task3成功不代替新来源。

上述semantic/classic contract已公开为`a203fa6dca67c0cb3eac879e8e406e0cb2ffd3a4`，[对应CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40270328170)两项实际成功：官方CPU **519 passed、无skip**（85.55秒），轻量433 passed/86可选依赖skip（14.09秒），pip check及shell检查通过。这是该来源的托管结果，不替换本机内存skip，也不包含之后的Task4。

Task4的[两步优化/全窗口重载入口](../egoemg_repro/emg2pose_check.py)已修正独立审查发现的容量和失败终态问题，并复审通过。此前小H5夹具完成2更新/3窗×2；随后真实mini的S/M/L各完成**2次实际更新、全部526窗×2检查、严格模型/AdamW/cosine/RNG恢复一致**，见[独立核验的聚合证据](../results/emg2pose-mini-optimization.md)。各pass有3,263,464有效帧，输入/目标/预测/mask/顺序/实际指标的流式hash前后均一致。全部mini是原val，优化后在同split检查不构成独立验证，未选best、无GPU或科研完整训练结论。

运行源`07019609583640d685f724041b022d10b4831e61`的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40271080945)为官方CPU **551 passed、无skip**（80.63秒），轻量440 passed/111可选依赖skip（16.79秒），pip check/shell检查通过。该源本机历史550 passed/1个ViT-L内存skip仍保留，与后来的托管结果区分。

后续classic数据入口与冻结中心审计实现也分别通过独立Spec/Quality审查。classic完整reader/Compose/RNG对照已通过，原modality缺legacy filtered字段的KeyError及显式字段裁决均保留；来源`f6c293d6308dc0ead082b7afd2c70936901d31dd`的[实际CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40273189915)为官方CPU **638 passed、无skip**（144.22秒），轻量480 passed/158可选依赖skip（19.77秒），新OpenCV参考依赖、pip check与shell检查通过。本机637 passed/1个内存skip是另一实际执行，不改写。

随后已从同一冻结来源实读[全部保存中心](../results/dataset-semantics/center-coverage.md)：33,508窗口/67,016手中心，1,474,352可能角/1,420,084有效角。测试1396窗口/2792手中心/59,572有效角；train/val/test实际手势覆盖51/32/59，整体60。raw50有66窗口，未知-1有241窗口；59类定义视图为33,201窗口，不能把二者混为raw。仅12字段中心点读6,232,488bytes，本次未全payload重hash，也没有模型误差。原legacy窗口为12000、训练stride1200；公开MD的2000笔误经独立RED发现并修为1200，JSON与实际计算始终是正确冻结值，未重生成或修改分区。

新增[损失与选模审计](../results/official-loss-audit.md)：实际Hydra合成、未改官方Lightning归约AST和独立复核发现，EgoEMG S/M/L、RN18、RN18+S、DINO-S及classic两项的两个指尖别名合计系数0.02；既有复现层实际为0.01。旧四结果和当前L/RN50/RN152源身份不改，继续明确为该legacy配方；不得把完成记录升级为官方字面协议等价。classic实际监控val_loss并有patience50，RN18/DINO-S有patience30；今后须保留真实回调行为。EMG2Pose三项仅有base指尖项0.01，不受该叠加影响。

DINOv2-S已从匿名HF实际取得88,240,510 bytes预训练文件，实际文件与主干tensor摘要、官方位置嵌入重采样、合成256px前向和严格离线重载逐位相同均记录于[独立工程报告](../results/dinov2-small-pretrained.md)。该验收使用冻结`8dabdaf`、隔离小型依赖且禁用CUDA，不涉及真实cohort优化/性能。B/L仍仅实际CPU构造/schema，真实预训练资产、GPU与完整预算实验仍未完成。

15:11实际读取确认L/RN152的原controller及当前child身份仍匹配，零起始epoch分别138/200与118/150、step17,643与41,510。完整EMG2Pose归档原receiver/tunnel身份仍匹配，已提交14,445,182,976原始字节，压缩10,082,882,504字节；complete=false，不能据此开展完整数据实验。

`8ef6aeec24d7f9f38bf67bb93db83b89187e0355`的[实际托管CI40278377967](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40278377967)成功：官方CPU **673 passed、无skip**（268.78s），轻量490 passed/183依赖skip（19.38s），pip check无损坏依赖。原生成功case临时目录清理后，runner的/tmp已用48MiB、可用7.8GiB，保留pytest目录92KiB；旧失败记录保留。这一成绩覆盖classic CPU实现，不包括随后新增的预取、派生或CUDA代码。

可选[预取适配器](emg2pose-prefetch.md)的104项原归档+新适配器测试通过；真实两次32MiB检查字节/xz6记录一致，单/双worker分别约1.050/0.931MiB/s，没有显示提速，因此原完整接收器未切换。[派生评价工具](legacy-derived.md)目前只完成元数据预检与纯归约，新增模型推理、派生表和定性图仍欠。

15:58的实际服务器观察：L仍在原controller/child下运行，零起始epoch160/200、step20,426；RN152为139/150、step49,140。随后新CUDA检查的前置门槛仍观察到目标GPU占用，未创建新attempt或启动模型；不会重复已失败的强确定性运行。

新增[完整classic队列CPU检查](../results/classic-cohort-cpu-engineering.md)已真实完成，两配方各准入再完整重算一致，Torch导入被禁止。冻结足迹65,046,370行、test4,306手样本等边界独立复核通过；CUDA/容量/production资格均false。包含该实现、canonical CUDA限定入口、预取及派生TaskA的66文件快照SHA为6ca23e7191d33cafee1e21a02f1ac83854e8a6cf9ea3af283695af6ab93e1380，本机统一879 passed/3 skipped（284.01s），轻量609 passed/273依赖skip（20.12s）。明确排除派生worker和随后开始的正式artifact WIP；旧673托管成功不能代替此组新提交的CI。

上述源码组已公开为`9b5f4931d53db0d536e0ebe64e386e8048872ab8`。[实际托管CI40292505908](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40292505908)成功：官方CPU **880 passed/2 skipped**（208.53s，两个skip仅为需要真实CUDA的正例），轻量609 passed/273依赖skip（21.25s），pip check与shell检查通过。它不覆盖后续未集成的派生worker、正式artifact或真实GPU容量。

16:31–16:33的独立classic canonical CUDA限定运行已完成：两模型各8次训练尝试、7次真实optimizer执行、1次刻意注入Inf后的正确跳步/scale回退，无自然overflow；新进程完整状态/RNG恢复一致。实际检查源码为cfbccea940016fd1981aeebdfa7144399ba804ced7c66adec81cd7a32e8e8c54，采用显式非确定性官方backend profile；保留此前严格确定性后向拒绝的失败。当前只验证batch1的固定小子集，不证明轨迹逐位一致、600/256容量、正式训练或科研成绩。17:06原EMG2Pose receiver仍为同一进程，已提交20,686,307,328原始字节（约4.47%），complete=false。

17:11，DINOv2-B实际预训练CPU检查完成：匿名HF文件346,334,872 bytes、SHA55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b；174个过滤后主干tensor逐字节一致，位置嵌入1370→325与官方插值一致，256px合成输入输出1×22×1有限。冻结8dabdaf原builder下网络调用为0，完整模型与精确输入/预测严格离线恢复逐位一致；只支持CPU正确性，不是cohort优化、GPU容量或完整训练成绩。L尚未通过真实预训练检查。

17:21:57，L控制流程五阶段均成功完成；验证最低15.245916°在第508步（第4轮），全部/有图中心16.612111°/16.571878°。200轮完整训练未消除验证曲线后期上升的过拟合迹象。原测试轨迹16.591133°仅由训练器计算，独立audit只复算中心；13份原导出/协议的remote/local SHA与size一致，发布草稿另经独立算术、人口、身份及曲线检查。实际source7b805不变。

DINOv2-B/L的[实际预训练与CPU证据](../results/dinov2-base-large-pretrained.md)已完成；两者都在原builder内使用真实HF权重、官方位置插值，并通过合成256px前向与严格禁网恢复。此阶段没有cohort优化、GPU容量或完整训练。

17:39–17:47，classic六个[原批量容量梯级](../results/classic-cuda-capacity-engineering.md)各两次实际更新及完整1576手val成功，12次实际执行无自然overflow。VEMG600/Neuro256峰reserved13.488/16.928GiB，train/val同B保持原FK归约。原工具b890按字节公开，所有层级仍scientific/production=false；正式入口、全训练与冻结test仍未完成。原controller/child都已成功退出，未因观察超时重启。

独立修复后的worker和formal-artifacts在70文件冻结快照05e50aecf8af8c079020e85261fcfd951ed8cef8b1d97f346cb2dd30c58db78c下，本机完整CPU1006 passed/3 skipped（270.05s），轻量673 passed/336可选运行时skip（19.27s）。原长basetemp触发AF_UNIX路径长度的两次失败完整保留；仅缩短协调器临时目录，未改测试断言或源码。该快照不包含后来版本化的capacity工具、新发现native callback语义修正或仍在开发的正式runner；后续来源另验，不挪用旧绿结果。

已发布RN152的ae3031c对应[实际CI40298074155](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40298074155)成功：官方CPU880 passed/2真实CUDA skip（222.86s），轻量609 passed/273依赖skip（22.15s），pip check与shell通过。新worker/artifact/capacity源码不由此CI覆盖。

新增公开capacity工具、worker与原生callback语义修正后的72文件冻结组，源码快照SHA9f879422b483206c0133b1bd095ce405503d14f962b5b183abfec1ea9f1d005d，本机完整CPU **1025 passed/3 skipped**（271.86s，两个真实CUDA正例及一个既有ViT-L内存准入skip），轻量 **690 passed/338依赖skip**（23.83s）。最终文件SHA逐一复核不变；该组明确排除仍在实现的formal runner及其测试，托管CI待对应发布提交实读。

## 2026-10-25：四基线历史完成记录

更新时间：2026-10-25。**获批 legacy 四基线的完整训练与审计已完成**：S/M 各 200 轮，RN18 与 RN18+S 融合各 150 轮，均通过独立验证选模、原训练器最终测试、严格导出及锁定复算，见[完整报告、环境和曲线](../results/completed/README.md)。全部 41 份裁剪已通过就绪核验；独立续跑接纳四份结果，审计队列于北京时间 **21:18:54**记录 `complete`。原下载失败、进程绑定阻断和训练中快照保留为历史证据。论文 v1 精确数值等价仍未证明，其他论文扩展未运行。

## 已观察到的状态

| 阶段 | 状态 | 证据范围 |
| --- | --- | --- |
| 独立公共仓库 | 已建立 | 原创复现层与官方源码分别管理 |
| 官方源码固定 | 已完成 | `f7cbd048b7092139852270491d3859cd472e0311`；训练入口校验版本及源码身份 |
| 独立运行环境 | 已就绪 | Python 3.11.12、torch 2.2.2+cu118、torchvision 0.17.2+cu118；CUDA / bf16 检查已执行 |
| legacy 小元数据 | 已下载、校验 | 真实 v2 manifest、metadata.npz、summary；固定大小与 SHA256 |
| EMG 与标签 | 全部下载、验证完成，下载退出码 0 | 21 文件，共 18,392,993,230 bytes；大小与 SHA256 receipt 已记录 |
| 视觉对齐字段 | 合并核验完成 | 3 文件；与 EMG/标签合计 24 文件的数据 receipt 已冻结 |
| 视觉 crop | 41/41 已验证，就绪门槛通过 | 45,037,686,784 bytes；第三批恢复退出码 0，2026-10-25 16:14:04（北京时间）生成 ready；库存、ledger、coverage 关联摘要一致 |
| 真实 crop 读取与中心覆盖 | 全部文件及保存划分的中心覆盖核验完成 | 完整 payload 哈希已核验；中心缺图为训练 3,739/63,172、验证 67/1,052、测试 104/2,792，文件齐全不代表所有中心有图 |
| 真实 split 划分 | 已执行 | 训练 63,172、验证 1,052、测试 2,792 个单手样本；跨角色输入行足迹无重叠 |
| EMGFormer-S GPU 合成检查 | 已执行 | batch 500，有限值前向/反向，峰值显存约 20.865 GiB |
| 自动测试 | 冻结代码本地 162 项通过；托管 CI 按发布提交核验 | 实际测试执行于 `f481f5f0b19fa0a06b9599880cccc24678ed5485`；30 个已测源／配置／测试文件内容保持相同。本地 CPU 夹具与生产 GPU 验收分开记录，最终发布 CI 不能由旧提交成功替代 |
| 视觉 / 融合训练集成 | CPU 夹具与真实裁剪 GPU 步已通过 | RN18 batch 900 约13.968 GiB；融合 batch200 约8.633 GiB；重复双手样本只用于内存/有限值检查 |
| 真实 EMG 数据有界 GPU 集成 | 最终版本已通过 | batch500，两步优化，有限 loss / 梯度；`test_evaluated=false` |
| 严格检查点重载与验证重算 | 最终版本已通过 | 两次验证报告一致；保存的有效配置支持迁移目录恢复至第3步 |
| EMGFormer-M 真实集成 | 已通过 | 官方 middle 实际6层、6,617,382参数；batch500两步训练、严格重载与重复验证，峰值约25.552GiB；集成阶段未评估最终测试 |
| EMGFormer-S 完整训练、训练器最终测试 | 完整训练、严格导出及独立复算通过 | 200 epochs、25,400 次更新、200 次完整验证；验证所选 best 为第 635 次更新（第 5 轮）。原训练退出码 0；原成功审计由续跑于 16:56:39（北京时间）核验接纳，未重复推理 |
| EMGFormer-M 完整训练、训练器最终测试 | 完整训练、严格导出及独立复算通过 | 200 epochs、25,400 次更新、200 次完整验证；独立验证所选 best 同为第 635 次更新（第 5 轮）。实际审计于 16:57:53（北京时间）退出码 0、无 guard 错误，16:57:54 核验接纳 |
| ResNet-18 完整训练、训练器最终测试 | 完整训练、严格导出及独立复算通过 | 150 epochs、10,650 次连续更新、150 次完整验证；验证选择第 6,319 次更新。18:06:56（北京时间）训练退出码 0，18:08:12 审计退出码 0、无 guard 错误，18:08:14 接纳通过；全部中心 MAE 7.7849°，有图子集 7.6908°，无轨迹成绩 |
| 融合完整训练、训练器最终测试 | 完整训练、严格导出及独立复算通过 | 150 epochs、47,400 次更新、150 次完整验证；验证选择第 36,024 次更新（第 114 轮）。21:17:03（北京时间）原训练退出码 0，21:18:12 严格导出完成，21:18:50 审计退出码 0、无 guard 错误，21:18:52 接纳通过；全部中心 MAE 6.7935°，有图子集 6.6923°，无轨迹成绩 |

样本数按“窗口 × 手”统计。相应窗口数为训练 31,586、验证 526、测试 1,396；它们不是独立参与者数量。验证是已有训练用户的时间留出，不是额外跨用户验证队列。

保存的窗口覆盖训练 / 验证 / 测试 **36 / 36 / 40** 名用户；发布 Gesture / User / Both 测试标签分别覆盖 **36 / 5 / 5** 人。训练、验证、Gesture 的用户集合相同，User 与 Both 的用户集合相同，其中各有 **1 人**也出现在训练窗口中。以上按请求窗口、合并双手统计，不是逐条件具有有效监督的训练用户数。41 个 episode 中有 1 个仅 10,109 行，短于 12,000 点输入，因此未产生完整窗口。脱敏数量与来源哈希见[队列覆盖记录](../results/cohort-coverage.json)。

## 集成检查的解释

最终版本 EMG 有界检查使用 batch500、两步真实数据优化，未评估最终测试集。严格重载后两次验证报告一致，SHA256 为 `5e956e530b9ace31f80a1d220c2073dce76957a551d3faf40762e4b3da2a73c3`；随后使用保存的配置和检查点迁移目录，成功恢复至第3步。两三步更新不支持收敛或泛化结论。

[脱敏集成证据](../results/integration-evidence.json)保存了合成 GPU profile、6 个真实单手窗口的有限值审计、实际 EMG 重载检查、真实裁剪的视觉/融合显存检查及协议数量。重复双手裁剪用于确认输入规模和梯度，不是完整 cohort 评估。这些集成快照的 `quantitative_reproduction_complete=false` / 最终测试未完成字段只描述当时状态，原件不作追改。此后的四模型完整训练、训练器最终测试、严格导出及独立复算均已完成，现态以[已完成实验](../results/completed/README.md)和共同人口清单为准。

## 视觉数据的恢复与就绪记录

**历史首批恢复：** 此前下载曾收到配额提示。服务器的首批延迟恢复于 **2026-10-24 20:58:50（北京时间）**按计划启动，重新核验已有文件后，于 21:00:20 以退出码 1 结束：端点没有返回断点下载所需的 HTTP 206 响应，在写入新传输字节前停止，已有合法二进制前缀保留。当时完整裁剪为 **24/41**，本批没有新增通过校验的完整文件。本批日志没有记录确切响应状态码或正文，因此不能据此确认它仍是配额 HTML。

**历史第二批恢复：** 第二批于 **2026-10-25 03:00:20（北京时间）**启动，03:00:50 以退出码 1 结束，同样在非 206 响应门槛处停止，没有新增完整文件；日志仍未包含确切响应状态码或正文。当时已有校验完成的裁剪保持 **24/41**，完整数据就绪检查及视觉训练尚未启动。

**历史本机补充下载检查（2026-10-25）：** 同一官方文件的 1 MiB 小段请求返回了精确匹配的 HTTP 206 二进制响应，但该批后续完整文件与分段恢复未成功。当时 17 个缺失文件各有一份已保存的错误响应，均为 HTTP 200、`text/html`、2,040 bytes，标题为 `Google Drive - Quota exceeded`，并含下载量过高提示；其中其余 16 个文件分别只作了一次完整下载尝试。这批本机尝试没有新增完整 LMDB 或完整文件 receipt，也没有向服务器传输数据或覆盖其已有文件。这些本机响应不能用于补推前两批服务器请求未记录的具体状态码或正文。

**第三批恢复与就绪：** 最后一批于 **2026-10-25 15:00:50（北京时间）**启动，即第二批结束后等待 12 小时，16:12:54 以退出码 0 结束。全部 **41/41** 份裁剪完成校验，共 **45,037,686,784 bytes**；不再需要补齐此前的 17 个缺失文件。完整文件与目标中心覆盖就绪记录于 **16:14:04**生成，恢复父进程于 **16:14:24**进入 `ready` 状态。41 项 receipt 的名称、大小和 Drive ID 与库存逐项一致，ready、ledger、coverage 及恢复父进程记录的关联 SHA256 一致。payload 摘要依据为本地传输与完整文件核验，不是发布方认证摘要。就绪记录同时确认来源身份及当时的资源条件。

保存划分的中心覆盖如下，按“窗口中心 × 手”计数，不是独立视频帧或参与者数量：

| 划分 | 请求中心 | 有图中心 | 缺图中心 |
| --- | ---: | ---: | ---: |
| 训练 | 63,172 | 59,433 | 3,739 |
| 验证 | 1,052 | 985 | 67 |
| 测试 | 2,792 | 2,688 | 104 |

测试缺图中左手 53、右手 51；Gesture 73、User 24、Both 7。全部文件就绪不代表所有中心有图；缺图样本继续保留并单独报告，有图匹配子集不能无声明地替代全部请求中心。裁剪 `ready` 只表明数据就绪门槛通过，不代表视觉/融合完整训练、最终测试或指标审计已经完成。

## 已落实的协议

独立 sidecar 保留固定 episode 网格上的官方测试中心，并按真实 `frame_split_id` 名称解析 User / Gesture / Both。验证区间从足够长的官方训练连续区间尾部确定性留出；训练窗口必须完整落在保留区间，短区间不会被强行切成不可用验证集。双手共享该 sidecar。

既有legacy实验按最终训练窗口覆盖的唯一行计算归一化，并用独立验证选择检查点；新classic/EMG2Pose采用各自已声明的固定官方norm资产，不能套用此前的train-only表述。检查点包含继续训练状态与所选最佳状态；有界验证结果不能直接参加完整验证的选模竞争。训练会记录配置、源码、数据receipt、划分和归一化身份，避免在恢复时默默改变输入。

## 尚未解决的科学边界

1. **数据与标签版本：** legacy release、论文 v1 与当前上游描述未证明逐项等价。41 人数量一致不足以证明生成标签或预处理一致。
2. **滤波依赖：** 初始实验使用整 episode 预过滤 EMG；无窗口交叠不能消除 FFT 引入的跨时间分区依赖。严格隔离需要原始 EMG 分区过滤及边界丢弃。
3. **架构与配置：** 主 EMG 采用发布兼容 12,000 点输入；论文 7,790 点设置另行研究。清洁融合采用 8 通道分支，并记录与发布 16 通道检查点及零残差初始化之间的差异。
4. **视觉完整性：** 全部裁剪文件及保存划分的中心覆盖已核验，真实缺图数量见上表；这不证明发布的视频帧对齐质量已经校正。适配器保留缺失 key 样本并单独报告，不能只挑有图样本作无声明对比。
5. **定量结论：** 合成 loss 有限或显存可容纳不证明训练收敛；本次成绩已有四份完整真实运行与独立审计支持，但仅适用于声明的 legacy 协议。不能据此宣称论文 v1 数值等价、其他未运行模型或跨随机种子的结论。
6. **发布标签与用户独立性：** 旧版 User、Both 各与训练窗口有 1 名用户交集；不能称为严格未见用户测试。保存的 33,508 个窗口中心与发布的 `frame_split_id` 类别一致，未发现适配器换错类别；上游交集成因尚未确定。窗口行足迹无重叠不证明用户独立性，也不消除整 episode FFT 依赖。本次冻结实验保持原标签与选定检查点，不依据最终测试结果删改队列。

运行命令见[训练指南](training.md)；下载校验语义见[下载说明](download.md)。公开记录仅保留脱敏的配置、聚合指标与可复核身份，不包含运行主机地址、账户或私有路径。

## 训练过程的历史快照

[真实训练曲线与时间戳](../results/live/README.md)保留 S/M 训练过程中的脱敏静态快照，只含当时的训练与独立验证，不代表当前完成状态，也不含最终测试数据或检查点权重。快照始终为 `benchmark_eligible=false`，不能填入论文对照结果表。完整结果导出及锁定检查点复算的要求见[结果指南](results.md)和[指标审计](metric-audit.md)。

## 已完成的实验与历史执行记录

初始生产实现提交为 `5c2712385ecf97010e2a8fce41ad9291f15daf03`；每个 run 的数值源码、所选配方及初始化内容 SHA 独立保留，当前文档提交不覆盖其原运行身份。S 与扩展 M 分别在前两张 A800 完成原有完整训练；两者均未重启或调整配方。完整裁剪与目标中心覆盖通过核验后，第三张 A800 已完成 RN18 两步真实短训练和严格重载检查，退出码均为 0。RN18 完整训练于 **2026-10-25 16:15:14（北京时间）**启动，**18:06:56**以退出码 0 完成，原训练进程已成功回收。此前 **16:58:48** 的 `epoch=58`、`step=4156` 是历史进度快照。完整记录包含 150 次验证与 10,650 次连续更新；验证选中第 6,319 次更新，验证 MAE 为 7.825144941°，随后执行最终测试。

融合在 S/RN18 分支选模与依赖验收后，通过两步真实有界训练和严格重载；完整训练于 **18:09:04（北京时间）**启动，配置为 150 epochs、batch200。**18:10:50**观察到的 `epoch=1`、`step=402` 保留为历史进度快照。原协调流程按既定程序退休旧 wrapper 并记录 `superseded`，未重启 RN18。融合于 **21:17:03**以退出码 0 完成，保留 47,400 次更新、150 次完整验证及原最终测试；最佳验证中心 MAE 为 **6.832980371891606°**，选中第 **36,024** 次更新。原八个初始化协议文件完成后仍与初始证明的 SHA 一致：S635、RN6319 均来自完整独立验证，随后按已记录逻辑重置 EMG 最终层并精确零残差初始化。

M 使用独立配方和输出目录，保留同一冻结数据、划分与归一化身份。[M 集成证据](../results/emgformer-m-integration.json)只证明短训练与重载通过，不代表完整结果。

**原审计流程的历史阻断：** 裁剪就绪后，原审计流程实际执行 S 的严格结果导出和锁定检查点复算，两个子进程退出码均为 0；复算日志报告 `verified`、`baseline_agreement=true`。但原父进程随后于 **2026-10-25 16:15:34（北京时间）**因待执行完整训练的进程绑定校验失败（`pending full-training producer binding invalid`）停止，记录为 `failed`、`completed_models=[]`。该历史空列表不表示 S 未执行实际复算，原成功产物和失败记录均保留。

**独立续跑已恢复队列：** 新续跑于 **2026-10-25 16:56:35（北京时间）**启动，保留原来源、配置、锁和输出，经核验于 **16:56:39**接纳 S 的原成功报告与审计，`inference_replayed=false`，没有再次执行 S 推理。M 的实际审计于 **16:57:53**以退出码 0 结束且 `guard_error=null`，于 **16:57:54**通过接纳。S/M 接纳状态均为 `verified`，共同中心人口摘要均为 `4cf005e68d2c3db3af7ac67028779c8535c1075218a04995662ca55b8f0ee371`。RN18 随后于 **18:07:37**完成严格导出，于 **18:08:12**完成锁定检查点审计（退出码 0、`guard_error=null`），并于 **18:08:14**获 `verified` 接纳，共同中心人口摘要与 S/M 一致。此前 `completed_models=['s','m','rn18']`、`waiting_original_fusion` 是等待融合时的历史状态。融合严格导出于 **21:18:12**完成，实际审计于 **21:18:50**以退出码 0、`guard_error=null` 完成，**21:18:52**获 `verified` 接纳；共同人口摘要与前三者一致。**21:18:54**队列记录 `status=complete`、`models=['s','m','rn18','fusion']`、每模型额外核验一次，S 为接纳原成功产物而未重放。

四模型现已按既定要求，在各自训练成功退出后完成严格导出及锁定复算，核对相同中心集合、来源身份和有效角度分母；这一额外验证的用途与数值门槛见[指标审计](metric-audit.md)。它没有参与选模或调参，不补做未纳入获批范围的论文模型。

## 代码验证与发布范围

本机完整测试实际在提交 `f481f5f0b19fa0a06b9599880cccc24678ed5485` 执行：**162 passed、17 warnings、0 skipped**，含固定官方模型的 CPU 夹具。30 个已测试的源码、配置、测试、环境与工作流文件内容摘要保持不变，因此后续仅文档及结果产物的提交沿用这份代码证据；不能把实际执行提交改写为当前文档 HEAD。安全摘要及对应文件身份见[发布清单](../results/completed/publication-manifest.json)。

该本机环境为 Python 3.12.14、Torch 2.2.2+cpu、torchvision 0.17.2+cpu、NumPy 1.26.4、Matplotlib 3.9.4；生产 GPU 环境由四份 run 的独立记录证明，不能相互替代。最终发布提交的托管 CI、公开产物内容和页面展示须按对应提交实际核验，旧提交成功不能充作未来提交通过。

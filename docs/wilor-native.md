# WiLoR 原生22角实验

完整30轮流程已于2026-10-29 02:55:24 UTC退出1：[磁盘不足与保存状态证据](../results/wilor-native-full-disk-failure-20261029.json)。已完成6轮、67,344次实际更新；保存last时触发原15GiB加临时检查点空间保留量，未改小batch或精度。第6轮原始检查点已由CPU调用生产校验器完整通过（15.096秒），模型/393项优化器状态/调度器/原生循环/RNG及原字节九属性保留；last仍只含5轮，不能盲目从last接续。原fit接口不提供隐式恢复，显式续训方案和足够持久存储仍待完成，尚未恢复或完成30轮/最终测试。已验证完整科学设置仍11。

[显式恢复的父状态准入](wilor-recovery.md)已由新进程通过，选择真实第6轮并校验四份回调依赖；尚未执行新的优化更新或恢复完整训练。

[首个完整轮次的真实原始检查点](../results/wilor-native-full-first-epoch-20261029.json)已由独立CPU进程读取，调用同一生产校验器通过：7,579,991,738字节原生保存、完整scope/sanity2、176验证批、模型和冻结摘要、393项AdamW状态/11,224步及16字节CUDA随机状态均核验，14.80秒exit0，无模型构造/恢复更新/CUDA。额外退出后读者核对113冻结成员与原始包九项属性未变、四个记录的读者/SSH进程已退出。首次私有读者误从训练模块顶层导入内部局部依赖而exit1，实际在检查点加载前；保留失败目录，仅在新隔离目录修正导入路径，不修改原训练或生产代码。这项证明覆盖首轮真实完整训练保存边界，不是完整30轮或验证选模结果。

[首次真实GPU验证](../results/wilor-native-gpu-qualification-20261029.json)绑定修复前f6fab48：原631,631,657参数/404预训练张量，batch18/BF16，两轮实际4次更新，每轮36训练和全部3,152验证消费。独立进程重算的typed预测精确一致，末批2；训练、原生验证、独立重算、资格写入和另进程CPU读取五阶段均实际exit0，18个相关进程退出，224份GPU及输入冻结成员、三个检查点和预测完整重验。此为明确的旧源码工程证据，不是当前源码的许可或完整训练成绩。

唯一整体审查发现0项Critical、2项Important、1项Minor。已补齐原生Lightning训练循环/批次/优化器/调度器/验证计数及16字节CUDA随机状态检查，并将验证与重算双方的fit身份检查移入共享资格读取边界。保留真实保存时序：第2轮检查点的epoch completed为1，不能误判为2。[修复检查](../results/wilor-native-final-review-validation-20261029.json)记录36失败/1通过的初始回归、37项定向通过，最终321项原生/相关通过和轻量1,632通过/1,235可选依赖跳过。[真实原始包验证](../results/wilor-native-final-review-raw-20261029.json)确认新校验接受原CPU/GPU完整检查点、拒绝缺失loops/错误GPU随机状态和三种错配fit，未训练、构造模型或初始化CUDA。单卡限制的报告措辞问题留为Minor，不另开整体审查。

[修复后完整输入两次重验](../results/wilor-native-data-full-review-20261029.json)绑定8ee760f，52文件/57.15GB、全部可能中心有限，两个进程分别重新读取并得到相同身份；不是给旧GPU资格改标签。训练文件整份哈希改变会更新数据及训练身份，因此当前源码仍需新的实际GPU验证。[存储优化证据](../results/owned-output-storage-sharing-20261029.json)记录内核比较完全相同文件后共享22.789GB物理块，全部文件内容和inode/大小/mtime/ctime等九项属性保持；没有删除或重命名文件、修改配方或降低空间门槛。当前空间仍须满足原47GiB工程/55GiB完整运行要求。

[修复版实际GPU资格流程](../results/wilor-native-gpu-qualification-review-launch-20261029.json)已于19:29UTC启动，冻结f993556/113成员，生产源码保持8ee760f内容，新输入身份adca6170…60693；当前实际拥有者活动，尚无资格完成或完整30轮声明。[输入物理块共享](../results/unchanged-input-storage-sharing-20261029.json)额外共享30.518GB重复内容：243.464GB关节角及3.043GB有效性文件的完整前后SHA、九项身份均相同，现有只读训练消费者保留，没有改写、删除、重命名、缩小数据或改配方。启动前真实可用75.459GB；完整运行仍在启动时重查原55GiB空间及GPU/内存门槛。

[当前修复版GPU资格](../results/wilor-native-gpu-qualification-review-20261029.json)为9a9973a2…64e411，绑定当前8ee760f生产源码、完整新输入身份adca6170…60693。两轮4更新/全部3,152验证/typed独立重算及CPU原始记录重开五阶段均exit0，18个拥有者退出，225成员/检查点/预测由独立读者重验。[完整30轮原生流程](../results/wilor-native-full-gpu-start-20261029.json)历史启动证据：原B18/BF16、202,020/3,152完整人口，GPU实际fit拥有者与完整身份已核验；当时尚无完整epoch或最终成绩；当前6轮及空间失败状态以上述最新证据为准。[已关闭检查点的物理块共享](../results/wilor-reviewed-checkpoint-storage-sharing-20261029.json)保留全部字节/九项属性，启动前60.27GB满足原55GiB门槛；未删除权重、改动配方或重复整体审查。

Task4评价源码7910238。[真实CPU评价链](../results/wilor-native-eval-cpu-20261028.json)复用已完成的原631,631,657参数模型和原生best；没有重新训练。验证和测试各选择每手首窗及原repeat2，共4消费/2唯一手样本，测试原索引明确为`[0,3,6,9]`；不是完整legacy测试集。原生Trainer和新进程独立前向的typed预测均精确一致，22角/生成标签mask/缺图样本/重复消费与唯一样本统计保持完整；每角色含2个缺图消费，不能只在有图子集上冒充全部结果。验证先冻结，再测试；PNG/SVG曲线由另一个进程从真实epoch日志逐字节重生一致，实际曲线已目视检查。报告显著标记工程验证，合成样本的误差不能作为科学成绩。

原7个评价/报告阶段成功。最后负例测试的私有驱动因尝试覆盖已复制sidecar而exit1；保留失败，只在独立目录修正测试副本写入并重做记录校验，未修改生产源码、重跑模型评价或训练。恢复阶段exit0，11种被改写记录均拒绝，输入在绘图后变化时拒绝发布且没有completion。24个相关进程全部实际退出，原始检查点、预测、报告与226份冻结成员由独立读者完整hash重验未变。[检查证据](../results/wilor-native-eval-cpu-validation-20261028.json)：本机284原生/相关检查通过，服务器2项真实保留链集成检查通过，完整轻量1,601通过/1,229可选依赖跳过；均exit0且来源未变。完整GPU/30轮入口仍然关闭，科学完成设置数不增加。

[工程曲线](../results/wilor-native-eval-cpu-20261028/training-curves.png)、[完整22角统计](../results/wilor-native-eval-cpu-20261028/metrics.json)与[报告](../results/wilor-native-eval-cpu-20261028/metrics.md)均公开。后续正式运行使用`run_wilor_pass`分别进行val/native、val/replay、test/native和test/replay；每个pass独立进程运行，replay显式引用原生pass。`freeze_wilor_validation`必须在测试前完成，`publish_wilor_report`发布后由新进程`verify_wilor_report`重开原始链核验。默认`require_full=True`拒绝工程fit，只有明确的工程验收才使用`False`。GPU工程/full角色必须消费完整3,152验证与8,612测试，原batch18，末批分别2和8；CPU四项选择不能升级为GPU或全量资格。

Task3最终绑定源码8013690。[两轮真实CPU验收](../results/wilor-native-fit-cpu-20261028.json)使用原631,631,657参数模型、真实404张量初始化和普通合成memmap/LMDB：B1/FP32，每轮2训练/2验证、共4次实际更新；三个完整原生检查点共22,739,889,583字节，模型/优化器/调度器/全局RNG/覆盖及冻结状态均核验。另进程严格重载原生best后，全部两个验证样本的typed预测和原生MAE精确一致，不重开预训练权重、不增加更新；第三个独立读者重开原始链，11种实际检查点损坏均拒绝。三个阶段实际exit0、八个所属进程退出，原始文件和111份冻结成员再次hash未变。这个合成工程过程不是准确率结果。

[完整实际输入的两次独立重验](../results/wilor-native-data-full-20261028.json)各79.38/78.13秒actualexit0：52文件/57,145,308,966字节，全部可能训练抖动中心的目标均有限、RNG未变、没有模型或CUDA。注册legacy实际202,020/3,152/8,612消费及101,010/1,576/4,306唯一人口一致，六个所属进程退出，连续输入/源码generation guard通过；[完整内容身份](../results/wilor-native-data-full-identity-20261028.json)保持数据资格范围。先前30项数据边界检查已经包含在最终80项fit/data检查内。

[Task3检查身份](../results/wilor-native-fit-cpu-validation-20261028.json)记录当时208项原生/相关检查和完整轻量1,552通过/1,200真实可选依赖跳过，均actualexit0、12份来源与实际拥有者核验通过。`fit_wilor`没有续训/覆盖或任意配方覆盖接口，完整模式在读取数据/模型前要求当前原始CUDA资格；`read_wilor_fit`重开原始epoch链及native best，`reopen_wilor_best`用自包含完整状态严格重载。数据/CPU实证不能打开完整30轮入口；首次GPU证据及修复后所需资格见上文。

[配方来源审计](../results/wilor-angle-recipe-audit-20261028.md)核对了所附PDF第22页与固定官方源码。原仓库有WiLoR模块，独立视觉WiLoR训练YAML缺失；此处明确派生自官方视觉ViT-S配方，整份替换模块参数，再应用论文WiLoR的batch18、初始LR1e-5和30epochs。BF16、weight_decay0.01、梯度裁剪1、val_mae选模/top3+last/patience30等分别标为继承字段。

```python
from egoemg_repro.wilor_literal import compose_wilor, wilor_protocol
cfg = compose_wilor(upstream)
contract = wilor_protocol(upstream, cfg)
```

该API仅生成并验证配方，不构造模型、读取真实样本或准入权重。绑定的完整配置被更改时会拒绝；公开元数据排除数据目录和权重路径。`pretrained_path=None`是尚未准入资产的配置状态，实际原生模型入口必须另行核验真实完整权重，不能直接调用原类的随机初始化回退。

固定代码使用1280维特征/GELU/22角head，保留256×256输入的原高度裁剪192×256；原文1576维不覆盖代码。先前[真实CPU权重实证](../results/wilor-pretrained-cpu.md)已测631,631,657参数，其中52,499冻结、404个主干张量与真实预训练精确匹配；这些结果不证明本次新运行入口已训练或取得GPU资格。

新增入口`admit_wilor_assets`先完整核验固定目录源码、配置、mean与2.56GB原权重，再允许`build_wilor_runtime`调用原生构造器；捕获前已加载的错误`WILOR_PATH`会拒绝。模型契约校验实际结构、冻结状态和全部404个预训练主干张量，`wilor_optimizer_identity`校验原生head/backbone两组AdamW与恒定cosine行为。独立恢复要求完整初始化来源和严格模型状态，不重开原预训练权重，也不宣称训练轨迹续接。

[当前真实CPU工程实证](../results/wilor-native-model-cpu-20261028.json)绑定源码a52a0ba：真实模型完成一次原生AdamW更新，非零梯度有限、冻结参数未变；完整保存7,579,750,462字节私有模型/优化器/调度器包。另一个进程严格重载全部模型状态，两个普通合成memmap验证样本的原生预测精确一致，新增训练更新为零。第三个独立CPU读者重开原始包、来源与状态核验成功，两个阶段220份冻结成员未变，十个原进程与独立读者均已实际退出；没有CUDA或完整数据资格。

[最终检查](../results/wilor-native-model-cpu-validation-20261028.json)记录128项原生/相关检查通过（78模型入口、27WiLoR配置、23原视觉配置），完整轻量1,501通过/1,171真实可选依赖跳过，均实际exit0、十来源摘要未变。[两次工程驱动失败](../results/wilor-native-model-cpu-failures-20261028.json)保留：官方直接调用`model.forward`使hook漏记、Torch2.2的mmap要求字符串路径；只修正私有观测/加载驱动。第二次已成功的更新和原包保留，仅新进程恢复，未重跑训练、改生产源码或放宽比较。这里只证明工程入口，不报告准确率或full/GPU准入。

原base的指尖项与实验别名在原生Lightning中合计0.02。继承的cosine floor1e-5等于初始LR，实际学习率恒定；报告明确记录此行为。数据仍保留原7790点、stride780、评价stride7790、repeat2与生成中心mask、缺图的归一化黑输入语义。原生加载器在三个角色都使用batch18；配置中val_batch_size256未被make_data_module传入，不能据此声称验证批量256。

源码契约保留原六设备声明作为来源；未来实际单卡执行另有身份，不能称为相同的六卡优化轨迹。可用legacy数据与声明的合并训练数据、PDF原表的等价性仍未解决。完整数据预期消费202,020/3,152/8,612，唯一101,010/1,576/4,306，与新融合的395,276训练消费分开。所有实际原始资产和派生模型权重保持私有，公开仓库保存实现、配置和聚合证据。

[当前配置实证](../results/wilor-native-config-20261028.json)绑定源码028250c：独立新进程重新合成/校验成功并实际退出，Torch未导入，未构造模型/数据或读取权重。[检查身份](../results/wilor-native-config-validation-20261028.json)记录27项新配置及23项原视觉配置检查共50通过，以及完整轻量1,475通过/1,119可选依赖跳过，均actualexit0；七份来源摘要逐字节未变、四个检查拥有者和独立校验子进程均确认退出。原生数据模块用受控三行源验证三角色各repeat为六消费、批量18；不是实际完整队列。初次汇总对嵌套上游使用错误Git仓库而退出1，修正汇总器后重新校验成功，保留失败且未重跑模型/预测。

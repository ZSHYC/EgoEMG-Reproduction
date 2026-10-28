# 全论文复现状态与证据清单

当前进度（北京时间2026-10-28 21:44）：[NeuroPose实际GPU资格](../results/neuropose-native-gpu-qualified-20261028.json)和[唯一一次新实现审查](../results/neuropose-native-final-review-20261028.json)完成，0Critical/0Important/2Minor；[完整原生训练已启动](../results/neuropose-native-full-launch-20261028.json)。完整101,714/1,576/4,306输入准入、已资格源码/运行时/来源绑定及107成员SHA核验通过，四个所属PID/start实际存活，训练worker确认为GPU消费者。保留原B256/16-mixed/max100/val_loss/patience50；完整终止、选best、预测试freeze和全测试独立复算仍待完成，科学完成数仍10。兼容融合的EMG初始化最近实查111/150轮，DINO-L80轮；EMG2Pose-S已实际GPU训练但首轮journal尚未完成，M/L仍在原队列后续。右手有效性文件继续下载，几何仍4/8。

[Table1后续原始来源核对](../results/dataset-statistics/dataset-comparison-audit.md)现已确认EMG2Pose原论文的50个提示姿态类别定义、双腕各16通道及不追踪腕关节；NinaPro分库人数/rest/DB6采集差异分别记录。原表不改写，其他列与EgoEMG完整模态盘点仍欠。[WiLoR配方只读核对](../results/wilor-angle-recipe-audit-20261028.md)进一步确认缺独立视觉YAML、代码1280维与原文1576维差异，以及继承eta_min1e-5会导致候选LR恒定；合成配置/CPU标量探测exit0，不是模型优化或新成绩。以下为各时点历史记录，不能将早期“尚欠”的步骤读作当前状态。

2026-10-28新增：[NeuroPose原生连续训练入口](neuropose-native.md)已完成Task1源码a0e4f4f与CPU工程验收，保留原B256/16-mixed/max100/val_loss/patience50完整配方和原厂网络；新路径不提供中断续训，旧失败恢复轨迹门槛/源码/产物保持。[独立CPU复算](../results/neuropose-native-fit-cpu-20261028.json)真实6,354,969参数/两轮两更新/2训练2验证，另一个进程严格best重载后全部验证预测精确一致，actualexit0/无CUDA/新增更新0/原fit字节不变。[33+60原生和1,474轻量检查](../results/neuropose-native-fit-cpu-validation-20261028.json)最终均通过，保留并修正轻量runner环境误开Torch可选开关的旧失败，未改断言。完整验证/测试新进程复算、实际CUDA资格、一次新实现审查及NeuroPose完整100轮上限/原早停仍欠；没有新的科学成绩，完成设置数仍10，全文目标继续。

北京时间2026-10-28本次融合[唯一一次最终整体实现审查](../results/native-fusion-training-final-review-20261028.json)已完成：0Critical/0Important/1Minor，结论可继续真实EMG150交接与计划中的full150。8个源码/测试文件与实际资格c14e6f8逐字节相同，审查HEAD7ab6e43，工作区与固定上游均干净；唯一Minor是EOF额外空行，核对range diff确有提示，保留已资格源码并登记格式欠项，无生产修改、无必要作者C/I修复或重复整套检查。所有declined行为都逐项裁决：完整交接/150轮/最终指标仍是Task4Steps4–6，先前模块不重复整体审查，标注与其他论文要求仍保留在全文目标内。原兼容EMG150最近实查70轮/17,010更新，右手下载127.7MB仍增长；新融合正式150轮未开始，科学完成数仍10。

北京时间2026-10-28 19:26补充：[融合原batch200/bf16工程资格](../results/native-fusion-training-gpu-qualified-20261028.json)现已完整通过。前9个实际GPU/输入阶段保留，缺绘图库的旧报告exit1也保留；11个固定版本安装包经本机下载、传输完整SHA核验及隔离离线安装exit0后，两个新CPU阶段分别生成报告、独立重开原始链并逐字节重生核对PNG/SVG，恢复流程实际exit0。24个阶段worker/guardian和4个流程拥有者独立确认退出，所有原86成员、新恢复3356成员及11安装包再次hash未变；修复新增训练/预测0，实际曲线已检查。它只证明两轮/4更新工程链，仍用明确未训练EMG工程分支，科学完成数仍10。下一步为本次新融合训练实现唯一一次最终整体审查；完整双分支交接和新融合150轮仍等待兼容EMG150的实际完成与独立best验收。

北京时间2026-10-28 19:04更新：[左手marker有效性文件](../results/geometry-browser-left-valid-complete-20261028.json)经浏览器正常完成，实际1,389,396,225字节/66,161,725×21个bool，全字节0/1及完整SHA核验通过，原下载保留；现4/8几何文件完整。[四文件服务器传输](../results/geometry-four-files-server-transfer-20261028.json)实际exit0，共7,807,083,550字节，两次完整远端SHA校验后再次独立重开均一致，实际拥有者退出、无未提交片段；几何整体准入及遮挡仍欠。右手有效性文件由用户确认正在下载，继续等待，无重复请求。[融合GPU资格流程](../results/native-fusion-gpu-report-failure-20261028.json)9阶段实际成功：原batch200/bf16两轮/4更新、六状态恢复及完整验证精确一致，完整3,152验证与8,612测试消费的新进程预测均精确一致；最后CPU报告因缺Matplotlib实际exit1，20个阶段拥有者已退出，86冻结成员未变。保留原失败，只在新任务目录补齐已声明的报告依赖并重做报告/独立verify；整个资格尚未完成，未增加科学完成数或启动新融合150轮。原兼容EMG150最近实查62轮/15,066更新，EMG2Pose S进入full-experiment-running，均原进程继续。

北京时间2026-10-28 18:29补充：[新融合当前源码的两次全量输入重验](../results/native-fusion-training-readmission-20261028.json)各96.61/85.49秒actualexit0，54文件/61.38GB及完整有限足迹精确一致；对应四拥有者实际退出、86冻结成员未变。原资格控制器已进入prefix阶段，实际worker在指定GPU有CUDA上下文，完整资格仍未完成；新的融合正式150轮尚未开始。代码/CPU证据已推送8bf2a2c并独立核对GitHub ref。


北京时间2026-10-28 18:23更新：[ViT-S融合原生评价与报告](fusion-native-training.md)在提交c14e6f8完成：最终31项评价、122项原生相关及完整轻量1,460通过/1,051可选跳过均actualexit0；6测试拥有者独立核对退出、10源码/测试摘要未变。[独立CPU证据](../results/native-fusion-eval-cpu-20261028.json)另以3个新子进程完成全12消费/6唯一/6缺图的预测重放、报告及verify，实际预测/选best/原mask一致；原父进程用错解释器的汇总失败保留，匹配环境最终核验未重复子阶段或增加训练。[完整输入/GPU资格单次流程](../results/native-fusion-training-gpu-launch-20261028.json)已冻结86成员启动，初始四拥有者实际存活，当前仅输入核验快照，不表示GPU通过或新融合150轮。Tasks1–3完成，Task4真实资格、一次整体审查及完整实验继续；科学完成数仍10。


北京时间2026-10-28 17:29后补充：几何第七次续传[实际exit1](../results/legacy-geometry-bounded-seventh-interruption-20261028.json)，新增83,886,080字节后同偏移三次临时错误用尽，旧前缀保留、拥有者退出、仍3/8完整。连接的Drive工具能核对文件名/大小，但存储二进制下载上限268,435,456字节，1.389GB有效性文件实际413拒绝。随后经官方浏览器正常入口启动该文件下载，[两次观察](../results/geometry-browser-download-progress-20261028.json)确认本机临时文件从48,305,891增长至142,065,593字节，源前64KiB与原严格探测一致、实际打开文件的写入进程PID/start存活。文件尚为临时未确认名称，未宣称完成或几何准入；等待正常下载完成及最终文件核验，不以临时文件替代正式数据。新的融合训练实现与证据已推送`bac86a4`并由GitHub ref核对；全文目标继续。

北京时间2026-10-28 17:03更新：[原生融合完整训练/恢复入口](fusion-native-training.md)已提交`aaadf52`，保留官方batch200/bf16/150轮/无早停/原生八优化器组，full要求真实完整双分支交接和注册输入。新43项原生及完整轻量1,460通过/1,020可选跳过均actualexit0；[独立CPU工程证据](../results/native-fusion-fit-cpu-20261028.json)重开两轮/4更新、全部4个合成验证消费、298项状态与六恢复状态，完整验证预测精确一致；已完成预算的结果丢失显式模拟恢复新增更新0，原检查点/轮次/日志字节不变。[发布核对](../results/native-fusion-fit-cpu-validation-20261028.json)保留中断、过长测试目录与旧临时fixture失效证据，成功后4测试拥有者退出，无CUDA/最终测试/新成绩。原EMG初始化实查25/150轮、6075更新且同worker活动，DINO-L69轮同进程活动；EMG2Pose S仍在控制流程前置来源核验，无新完成成绩。几何[第七次单次续传](../results/legacy-geometry-bounded-seventh-renewal-20261028.json)冷却4260秒后原策略五探测通过、21份旧失败文件保留且原拥有者实际存活，仍3/8完整；不表示几何准入。Task3评价报告、Task4 GPU/完整实验与新实现最终审查仍欠，已完成科学设置仍10，全文目标继续。

北京时间2026-10-28 16:18更新：[原生融合训练实施计划](superpowers/plans/2026-10-28-native-fusion-training.md)已推进权重来源交接及原生八组优化器检查；[真实视觉半链](../results/native-fusion-initializer-vision-half-20261028.json)在冻结91a879a新目录实际100.05秒exit0，重新核验54文件/61.38GB，与已完成ViT-S124轮/原生last及52共同payload配对，完整可能训练足迹对整个新融合验证区间/测试输入均0重叠，四个实际拥有者独立确认退出、84固定成员未变，无模型/CUDA/优化更新。76新/构造原生及88相关检查、完整轻量1460通过1000可选跳过；完整融合fit/eval与一次最终整体审查仍欠。原EMG初始化同一任务已完成13/150轮、3159实际更新，DINO-L68轮，均原进程活动；EMG2Pose进入S控制流程但尚无已完成轮次或成绩。几何第六次续传[实际退出1](../results/legacy-geometry-bounded-sixth-interruption-20261028.json)，新增503,316,480字节/仍3/8完整、断点保留。已完成科学设置仍10，全文目标继续。

北京时间2026-10-28 15:37更新：新融合所需的[官方EMG初始化完整150轮已实际启动](../results/emg-initializer-full-launch-20261028.json)，全部63,026训练/582验证，原260/bf16/不早停；86固定成员、四拥有进程实际存活且训练worker确认为GPU消费者，[首个完整轮次](../results/emg-initializer-full-first-epoch-20261028.json)已完成243次实际更新及全部582验证，原进程继续。其前置[一次整体审查与作者修正](../results/emg-initializer-final-review-20261028.json)处理3项Important，13新增RED→GREEN，118原生及完整轻量1,449通过974可选跳过；[修正来源完整数值](../results/emg-initializer-cohort-fixed-full-20261028.json)71.57秒两新进程精确一致，[修正后真实GPU资格](../results/emg-initializer-runtime-fixed-gpu-20261028.json)99.23秒/全部八进程退出。GPU3被其他任务占用时启动预检拒绝且未启动任何进程，另用空闲GPU7单次运行，未改模型或配方。旧证据保留、不算新测试成绩。DINO-L最近实查66轮，EMG2Pose完整源深度核验仍原四进程活动/0项正式开始。几何[第五次网络中断](../results/legacy-geometry-bounded-fifth-interruption-20261028.json)新增922,746,880bytes后实际exit1，3/8/断点保留；冷却683秒后[第六次明确续传](../results/legacy-geometry-bounded-sixth-renewal-20261028.json)五项新探测通过且原拥有进程存活，已新增134,217,728bytes。全文目标仍进行，完成科学设置仍10。

北京时间2026-10-28 14:25更新：[完整EMG2Pose源准备](../results/emg2pose-full-source-preparation-20261028.json)已实际退出0，25,253原文件/3,043,302,266行、全部原始字节保留及465,625,246,698字节转换字段完成。旧等待队列在交接阶段实际exit1且0模型开始，根因是对原始identity提前调用深度核验后的archive gate；[边界修复与新隔离队列](../results/emg2pose-handoff-recovery-20261028.json)保持原模型/训练/资格源码不变，20定向和1,449轻量检查通过，一次独立审查0/0/0、真实只读交接重放exit0。新队列四个实际进程存活，当前深度来源核验，0模型正式开始。新融合所需EMG初始化的[真实16文件/16.54GB完整数值准入](../results/emg-initializer-cohort-full-20261028.json)及新进程重开已确认63,026/582手样本；[原生模型与训练/恢复入口](emg-initializer.md)20模型+14训练检查通过，真实GPU/150轮仍待完成。DINO-L最近实查63轮活动。几何[第四次网络中断](../results/legacy-geometry-bounded-fourth-interruption-20261028.json)实际exit1，仍3/8，断点保留。完成科学设置仍10，全文目标继续。

北京时间2026-10-28 13:24更新：[融合完整输入的一次整体审查与修复](../results/native-fusion-cohort-review-20261028.json)完成，原2项Important均通过一次作者修复：映射步长/offset/只读布局绑定、数值文件读取前登记及共享图像缓存代次检查。9定向/82相关原生与完整轻量1,437通过861可选跳过；[修复源码32f10bd服务器全量重验](../results/native-fusion-cohort-fixed-full-20261028.json)实际219.83秒exit0，两次54文件/61,379,659,366bytes及完整有限足迹检查记录逐字节相同，全部6进程独立确认退出，旧源码证据保留、无模型/CUDA、新成绩仍无。原75119de的[完整托管CI](../results/full-paper-ci-75119de-20261028.json)实际成功：原生2,624/3CUDA跳过、轻量1,452/804可选跳过，只覆盖该历史提交。[初始化划分核对](../results/fusion-initializer-footprint-compatibility-20261028.json)发现旧EMG-S训练与新融合验证/测试输入重叠，须另训兼容初始化；[官方配方只读核对](../results/fusion-compatible-emg-recipe-feasibility-20261028.json)确认12,000点/stride1,200/完整轨迹/batch260/150轮/repeat1，每手8维原生归一化，前瞻可用63,026训练/582验证手样本，尚未训练。几何[第三次终止](../results/legacy-geometry-bounded-third-interruption-20261028.json)实际exit1但右手world文件完成并独立复核，现3/8；冷却1,095秒后[第四次续传](../results/legacy-geometry-bounded-fourth-renewal-20261028.json)实际活动且新增234,881,024bytes。最近13:19实查DINO-L61轮、完整EMG2Pose源准备与等待队列原进程活动、0项EMG2Pose正式开始；完成模型数仍10，全文目标保持进行。

北京时间2026-10-28 12:39更新：[融合完整输入核验](fusion-literal-cohort.md)实际完成54文件/61,379,659,366bytes的hash及全部可能抖动数值扫描，新进程完整重算记录逐字节相同。EMG并集65,171,929行/1,042,750,864元素、可能中心39,530,541行/1,739,343,804目标元素均有限；消费395,276/3,152/8,612与唯一197,638/1,576/4,306分别记录。两阶段及外层共6进程独立确认退出、77固定成员不变、无模型/CUDA，[实际证据](../results/native-fusion-cohort-full-20261028.json)。本机73项相关原生/完整轻量1,437通过852可选跳过；当前等待一次整体审查，GPU/完整150轮及新精度仍欠。未来初始化文字已按原配置纠正为视觉last、EMGbest，不推断作者未公开文件内部来源；最终融合仍独立验证选best。完成模型数仍10。

北京时间2026-10-28 12:09更新：[联合读取一次整体审查](../results/native-fusion-data-review-20261028.json)为0Critical/0Important/1Minor，复查43项实际通过；额外namespace/asset及损坏中心字段的直接回归用例为非阻塞覆盖欠项，现有完整descriptor/共享schema拒绝逻辑保留。随后[服务器真实旧版点读](../results/native-fusion-real-data-points-20261028.json)实际exit0/拥有进程独立确认退出，73个隔离源码文件前后核验，三个角色各左右手共6个真实联合输入有限/形状正确、无模型/CUDA。全元数据索引的重复消费train/val/test为395,276/3,152/8,612，唯一手样本197,638/1,576/4,306；仍非完整输入哈希/有限性准入或新训练。新[几何第三次续传证据](../results/legacy-geometry-bounded-third-renewal-20261028.json)在原2/8基础新增83,886,080bytes后拥有者活动，保留九份失败文件、六项新探测通过；该快照不是完整接收。全文仍进行，完成科学设置仍10。

北京时间2026-10-28 12:01更新：[原生融合联合读取](fusion-literal-data.md)已完成25项新检查及视觉/分区回归共43通过，完整轻量1,437通过/819可选跳过，独立collector六个合成真实文件样本有限且无CUDA。原16槽/7,790点/stride400、原EMG+RGB中心读取、原增强/RNG和全可能抖动足迹隔离均保留；原归一化文件缺键实际KeyError，以同版本官方unified统计文件作明示数据适配，[CPU证据](../results/native-fusion-data-cpu-20261028.json)。当前等待一次整体审查，真实完整内容/GPU/训练/新成绩仍欠。先前e7aaa5b的[完整托管CI](../results/full-paper-ci-e7aaa5b-20261028.json)已实际成功：CPU2,540通过/3真实CUDA跳过，pip check通过；轻量1,450通过/722可选跳过，仅覆盖该旧提交。第二次几何续传实际exit1，新增788,529,152bytes，右手片段2,153,775,104bytes保留；898秒冷却后第三次显式续传已启动，仍2/8且几何未准入。

北京时间2026-10-28 11:51更新：原生ViT-S融合构造的一次独立整体审查完成，原2项Important已在一次TDD修复中处理：完整110项EMG来源清单/原加载决策/逐项覆盖核验，以及fresh/restore工厂均禁预训练联网回退并在setup后复查文件身份。15项定向检查、82项相关原生检查、完整轻量1,436通过/795可选跳过均actual exit0；新进程collector严格恢复预测精确一致、无CUDA。[审查与修复证据](../results/native-fusion-construction-review-20261028.json)、[更新CPU证据](../results/native-fusion-construction-cpu-20261028.json)。这是构造工程完成，真实联合数据/GPU/150轮训练与测试仍欠，完成模型数仍10。11:46实查DINO-L58轮、原训练进程活动，完整EMG2Pose源准备与等待队列均活动且0模型正式开始。

北京时间2026-10-28 11:30更新：[原生ViT-S融合构造](fusion-literal.md)已完成CPU工程验证：25,855,037参数、174视觉backbone+4head张量完整传输、105项EMG匹配/5项跳过，原训练模式loss/backward/一次AdamW更新、全部参数及时间注意力组、FP16初始化→FP32原生转换、新进程严格重载/预测精确一致；64项相关原生及完整轻量1,436通过/780可选跳过，独立collector实际exit0，[证据](../results/native-fusion-construction-cpu-20261028.json)。一次整体审查待完成；初始化是明确的工程状态，无数据/GPU/完整训练或新增精度，完成模型数仍10。几何[第一次有限续传实际退出1](../results/legacy-geometry-bounded-interruption-20261028.json)：新增1,040,337,776bytes，一次实网重试后成功16MiB，再在后续偏移连续三次EOF停止。冷却938.8秒后[第二次显式续传](../results/legacy-geometry-bounded-second-renewal-20261028.json)六项预检/一份旧尾部对照通过，原六份失败文件保留，新拥有者实际活动并新增167,772,160bytes，2/8完成；完整几何/遮挡仍欠。11:24实查DINO-L57轮、EMG2Pose源准备与三模型等待队列原进程均活动、0正式训练开始。

北京时间2026-10-28 10:55补充：[几何续传实际进度](../results/legacy-geometry-bounded-progress-20261028.json)已新增369,249,136bytes，左手world transform完整3,175,762,800bytes通过原最终门槛及独立SHA复算，连同tracked共2/8文件完成，原拥有者仍活动、正在接收右手；其余数组/几何准入/遮挡未完成。[ViT-S融合配方审计](../results/fusion-recipe-audit-20261028.md)实际合成五配置：发布评价为16槽补零/7,790点，8通道S到16槽融合的首卷积不匹配会由原strict=False加载器跳过；cotrain-S有效指尖系数0.01，独立S与融合0.02，不能用一个配方概括。该审计无新forward、加载或训练结果。

北京时间2026-10-28 10:51更新：测试断言修正29041be的[完整托管检查](../results/paper-analysis-validation.json)已实际终态成功：原生CPU2,513通过/3项仅CUDA跳过/198warning/2,719.34秒，pip check通过；轻量1,423通过/722可选跳过/1warning。原失败记录保留，原GPU分析产物不变；该旧提交不覆盖后来新增的有限续传模块。后者已完成[一次独立整体审查](../results/legacy-geometry-bounded-recovery-review.json)，0Critical/0Important/0Minor，另66项fresh与8项额外离线故障检查通过。10:50实查DINO-L56轮且原控制/训练进程活动；EMG2Pose源准备实际IO继续推进，队列仍0项正式训练。以上均不改变当前完成模型数10项或全文未完成的结论。

北京时间2026-10-28 10:45补充：[几何数据有限续传已实际重新启动](../results/legacy-geometry-bounded-renewal-20261028.json)。保持原八文件37,778,344,975bytes选择与原下载器不变，七项新范围探测/两份旧片段尾部对照通过，原失败进程实际退出、新拥有者PID/start匹配存活并已提交至少一个完整16MiB新块；原失败状态/日志/脚本和quota文档保留。外层固定每停滞偏移最多三次/1、2秒，原始协议/证书/容量错误不重试；27项缺模块RED、66项相关GREEN及完整轻量1,435通过/737可选跳过。全几何文件、几何准入和遮挡实验仍未完成。10:44实查原EMG2Pose源准备与三模型等待队列各四进程均活动，0项正式训练开始。

北京时间2026-10-28 10:14更新：[旧版双手原官方FFT重生成与独立复核](../results/dataset-statistics/emg-fft-bilateral-full.md)完成：左右各41段、每手66,161,725行/8通道，合计4.234GB新filtered全部有限，四次实际执行/独立读取均exit0；[双手raw接收](../results/raw-emg-full-acquisition-20261028.json)也已完成。EMG2Pose原文件保存后verify已读完462.8GB，原四进程最近实查仍活动，源准备/转换准入未完成，三模型0项正式训练开始。DINO-L北京时间10:17实查55轮、原控制器/训练进程活动。浮点断言修正后，相关原生120通过、完整轻量1408通过/737可选跳过；新托管轻量1423通过/722跳过，完整CPU检查仍运行。[几何补取](../results/legacy-geometry-acquisition-progress-20261028.json)七个范围预检通过、提交约755MB新二进制后因TLS EOF实际退出1，进度保留，需有限重试恢复。全文目标仍未完成。

北京时间2026-10-28 08:44历史快照：[逐关节与逐手势细分分析](../results/paper-analysis/README.md)已用五个冻结best完成新推理和退出后独立核验。S/M/L整体及手/split/组件/用户完整轨迹通过原baseline；RN18/S/融合共2,792中心，排18未分配后2,774请求，59类中58类融合优于视觉。保留有效角mask/插值中心/180pi与无mask/token中点/57.3两公式、四个预定义人口；全部推理仍为原bf16/high，官方公式视图不等于原默认FP32入口。Table12/14/15、Figure12参考与清晰展示共63产物字节重复一致，旧版0.01/数据与PDF等价边界保留；完整托管检查与一次最终整体审查仍待新提交观察。实查08:44：EMG2Pose全部462,824,048,640字节已到齐，但完整receipt尚未落盘，原下载两个拥有者实际存活且在读archive.tar做哈希，源码准备和三模型队列仍等待；尚未开始250轮科学训练。

北京时间2026-10-28 07:48[实际快照](../results/progress-snapshot-2026-10-28.json)：已有10项声明协议下的完整训练/验证选模/测试结果（7项legacy、VEMG、DINO-S/B），不等于10项PDF原表等价复现；两组测试为2,792和4,306唯一手中心，不能混成受控排行榜。DINO-L原训练/控制进程活动、完成50轮验证。服务器完整EMG2Pose实际收到416,402,112,512/462,824,048,640字节（约89.97%），原四进程活动、尚无完整receipt；原源准备控制器仍等待。三模型等待队列已实际运行但0科学训练开始。全文、官方配方补齐、逐手势/关节/遮挡/定性及标注链仍未完成。

2026-10-28新增：[全量队列与最终实现审查](../results/emg2pose-final-implementation-review-engineering.json)已完成一次独立代码审查，修复监护进程单独丢失遗留工作进程、原始保留文件缺损未被准入发现、独立入口输出污染上游三个问题。10项真实RED→GREEN，补充登记竞态/旧PID检查；本机125/436.87秒、服务器125/523.89秒、完整轻量1,379通过/733可选跳过均actual exit0。新版本还在服务器4.21秒只读核验已有5,009,750行工程源及原始保留字节，所有拥有者独立确认退出。全量固定S/M/L队列已提交且13项检查在两机通过；因源码身份改变，[受影响的GPU资格更新](../results/emg2pose-reviewed-runtime-qualification-engineering.json)已实际完成：S/M/L原320/12/bf16两轮及六状态恢复、两模式每pass321窗/320+1/12/FP32、报告与共享14阶段编排及退出后独立重读全部通过，61个实际受管理阶段/1,645.13秒exit0，122个worker/guardian和外层拥有者独立确认退出。对应a6a9f2e的[完整托管检查](../results/emg2pose-reviewed-whole-ci.json)现已实际成功：CPU2,480通过/3项真实CUDA跳过、pip check通过；轻量1,394通过/718可选跳过。全量队列于北京时间07:48一次实际启动，四个拥有进程PID/start均匹配，当前waiting-existing-full-source、0模型开始；它等待原下载与原转换完成，再执行S/M/L各250轮，不构成新论文精度。

状态快照：2026-10-28（北京时间）。**当前目标是复现论文全文，仍在进行中。** 有歧义时优先采用固定官方源码，并公开说明与 PDF 的差异；七项legacy实验完成不构成全文完成。范围以[全文设计](superpowers/specs/2026-10-26-full-paper-reproduction-design.md)为准，动态进度见[运行状态](run-status.md)，阶段动作见[阶段计划](superpowers/plans/2026-10-26-full-paper-stage1.md)。

原文为 [EgoEMG v1](https://arxiv.org/abs/2605.05712v1)，共 34 页。下文页码使用 PDF 印刷页码。官方源码固定为 [f7cbd048b7092139852270491d3859cd472e0311][official]；源码、resolved 配置、实际发布资产与运行证据分别核对，注释和文件名不能覆盖实际执行行为。本文不把论文数值填入本项目结果。

## 当前实际完成到哪里

- **普通原文件共享实验控制器已通过真实GPU工程验收：** [控制器证据](../results/emg2pose-direct-experiment-engineering.json)为22新检查与原报告回归，本机43/378.64秒、服务器43/467.82秒actual exit0；轻量1,355/733可选跳过。一个S模型在真实原格式合成5,009,750行源上完成14受管理阶段312.91秒、原320/12/bf16两轮32更新、两种选择各321窗/320+1/12/FP32、验证profile与报告；控制器实际退出后另受管理CPU独立重读57.27秒actual exit0，全部30个worker/guardian实际PID/start确认退出。父进程无Torch、实际GPU/读取资源观测未注入。共享编排代码支持三模型，模型训练/评价资格仍逐模型独立；此处未重做三模型数值资格。真正完整源及三项250轮、自动衔接队列和整篇论文仍欠。

- **普通原文件最终报告API通过真实GPU结果的只读工程复核：** [报告证据](../results/emg2pose-direct-final-report-engineering.json)为21新检查+原表格/fit回归，本机58/244.12秒、服务器58/295.62秒，各7warning且actual exit0；轻量1,355/732可选跳过。复用已完成5,009,750行源的S/M/L工程fit和两模式实际320+1 FP32预测，三份报告生成与三个新进程重开共6受管理CPU阶段/303.39秒actual exit0，12个worker/guardian及外层拥有者实际PID/start独立确认退出，原权重/日志/pass/侧文件不变。两种人口分开、原Table11/13与独立算术/匿名边界重核，工程无法升级full，未重复GPU工作。完整源/250轮/科学控制器及全文仍欠。

- **普通原文件完整FP32批量与两模式已实测通过：** [容量证据](../results/emg2pose-direct-evaluation-capacity-engineering.json)为S/M/L各新的两轮工程fit/32更新，官方字面与文件行对照各val/test原生及新进程完整复推，每pass321窗/实际320+1/12worker/FP32。显式合成三角色5,009,750行，经gzip4+shuffle无损HDF保存为299,345,920B，原接收/转换后全部字段另核对逐字节一致，未放宽647MB工程源上限。37个成功受管理阶段，另保留1个私有最后读取失败；原653.04秒链exit1，纠正仅该CPU读取后66.38秒exit0，无GPU重训/重测。76个实际worker/guardian及两组外层拥有者独立确认退出，权重/日志/源码不变。完整源/250轮/最终科学controller/report与全文仍欠，不能填Tables11/13。

- **DINOv2 ViT-B完整实验及独立报告核验完成：** [完整结果与曲线](../results/original-vision/vit-base/README.md)记录94轮、52,828次实际更新、原patience30早停，验证选择零起算epoch63（第64轮）。全部/有图中心MAE为6.761261°/6.623680°，4,306唯一手样本（repeat2共8,612消费），原生test/新进程完整复算/报告/最后核验共16阶段actual exit0，外层及全部阶段进程独立确认退出；另在正确冻结环境重开核验通过，无CUDA。实际86,182,166参数与768维B骨干/真预训练来源已核验，checkpoint沿用S模板名称不改变模型身份。作者缺独立B YAML，明确记录为S配方派生、0.02有效损失/原360/bf16/单rank/workers0/旧队列；不是PDF或多seed架构优劣结论。L仍完整训练，全文仍未完成。

- **普通原文件选best原生FP32评价完成三模型GPU工程复推：** [数据证据](../results/emg2pose-direct-evaluation-data-engineering.json)为104本机/104服务器检查及真实mini28.30秒actual exit0，690次选窗与独立原参考相同（字面230+230/行交集30，文件行对照167+63/交集0）。[评价证据](../results/emg2pose-direct-selected-evaluation-engineering.json)为107本机/107服务器检查、17受管理阶段/286.11秒actual exit0；S/M/L真正工程best完成val/test和新进程typed预测/统计精确复推、测试前profile及最后CPU重开，所有34个worker/guardian和两个外层拥有者独立确认退出，fit及权重未变。显式合成三角色每pass仅1实际窗口，配置320/12/FP32不能升级为真实batch320容量；字面混角色val拒绝、GPU只资格化文件行对照。普通源321窗FP32容量/完整源250轮/科学报告仍欠，不能填写Tables11/13。

- **原始普通文件独立角色GPU fit/选best工程资格已完成：** [实际证据](../results/emg2pose-direct-native-fit-engineering.json)为9个受管理阶段/213.35秒actual exit0。显式合成3角色/25,710行原格式源经过真正loopback aria2接收/原准备转换后，S/M/L各原320/12/bf16两完整工程epoch/32实际更新，原Sampler每轮5,120消费与1/2唯一train锚点分列；新进程完整epoch六状态恢复及真正验证min选best、last/history/日志另行CPU重开通过。全部18个worker/guardian和两个外层拥有者独立确认退出，源码/检查点SHA不变、无实际容量观测注入。完整源及250轮、普通源FP32最终评价/科学报告和全文仍欠，工程结果不填Tables11/13。

- **原始普通文件独立角色fit入口CPU核验完成，自己的GPU资格正在执行：** [证据](../results/emg2pose-direct-native-fit-cpu-engineering.json)为22新检查及原回归，本机105/215.91秒、服务器105/285.08秒，各39warning；完整轻量1,355/729/1warning84.77秒均actual exit0。合成三角色普通源两完整CPUepoch/32更新/新进程六状态恢复和验证min选best通过，实际val-only源拒绝。完整full250/320/12/bf16及真正25253源+新GPU完整epoch资格门槛不改。前述真实mini恢复完成后，新GPU独立角色检查已一次启动，源准备完成/small前缀活动，尚未qualified；完整训练/最终评价/科学报告仍欠。

- **原始普通文件三模型CPU/GPU完整状态恢复已实际完成：** [证据](../results/emg2pose-direct-native-recovery-engineering.json)为12个受管理阶段/702.24秒actual exit0。S/M/L各CPU和实际GPU原320/12/bf16前缀两次更新，两轮完整526验证；新进程526窗输入/预测/顺序/指标精确复推、六状态恢复、再更新至step3并完整验证。GPU验证实际320+206，全部24个worker/guardian与两个外层拥有者独立确认退出，原prefix及源码不变、无资源观测注入。mini原val仅复用工程训练，没有独立科学验证/250轮或Tables11/13成绩；普通源独立角色fit/选best/最终评价仍欠。

- **原始普通文件训练恢复入口的CPU检查通过，真实完整mini仍在运行：** [证据](../results/emg2pose-direct-native-recovery-cpu-engineering.json)为本机65/126.00秒、服务器65/152.81秒，各27warning；完整轻量1,355/728/1warning89.66秒均actual exit0。三种模型在明确CPU工程夹具上完成六状态新进程恢复及step3；新入口只私有绑定已有精确算法。真实mini12阶段已实际启动，当前已确认small CPU前缀/新进程526窗复推/六状态恢复及step3全验证，medium正在运行，其余未完成，不能借用旧内存后端GPU资格。原始普通源GPU恢复、完整源/250轮fit和科学结果仍欠。

- **DINOv2 ViT-S完整实验及最终独立核验完成：** [完整结果](../results/original-vision/vit-small/README.md)为124轮、69,688次实际更新、验证选择零起算epoch93；4,306唯一测试手样本（原配置repeat2共8,612次消费），全部/有图中心MAE为6.464179°/6.324737°。完整原生测试、独立完整预测复算、报告及最后核验共16阶段实际exit0，外层及阶段进程确认退出，另在原冻结环境重新打开原始链条核验通过。保留服务器三个核心报告原字节，补充精确源码e7c20b9/配置/依赖/124轮曲线清单；原生0.02配方、独立验证、单rank/workers0及旧版数据/缺失合并来源/PDF不等价边界保留。L仍在完整训练；B已完成，见上方新证据。

- **原始普通文件上的S/M/L原生CPU入口已完成独立核验：** [入口与边界](emg2pose-direct.md)、[实际证据](../results/emg2pose-direct-native-runtime-engineering.json)为182本机/182服务器以及完整轻量1,355/727 actual exit0。完整mini27.59秒实际exit0，全部526窗口与独立原参考相同；原始数据集类、三模型初始权重/CPU前向/原AdamW-cosine250配置及严格权重重载精确一致。一次深度字段准入后保持具体文件绑定，不使用archive FD或封印、不改变原validator/global。没有优化器更新、训练状态恢复、CUDA/full-source/250轮或新精度结果，这些仍欠。

- **原始数据全流程入口与等待现有下载的控制器已接通：** [证据](../results/emg2pose-original-source-pipeline-engineering.json)为179本机/179服务器与完整轻量1,332/713 actual exit0；新的完整官方mini从下载到受管理的原准备/回收/转换/准入131.52秒actual exit0，四字段/22数组一致、实际拥有者退出。相同冻结控制器已实际启动full模式，当前等待既有下载完成及其拥有者退出，尚未准备完整源；不会重复获取全量包。原始直接数据的模型/GPU恢复/250轮/最终评价和全文仍欠。

- **原始磁盘源准备与冗余TAR回收已通过完整mini实测：** [证据](../results/emg2pose-original-disk-preparation-engineering.json)为170本机/170服务器及完整轻量1,323/713 actual exit0；新下载的官方mini实际142.30秒exit0。全部原文件与额外字节保留后，重新核验全包及原HDF5盘点，实际回收本任务这份冗余TAR，再在原包不在场时精确重建；原转换四字段/22数组及原生准入一致。此前参考包、活动全量下载和原转换器不变。完整25,253会话的真实准备/转换和三项250轮科学实验仍欠。

- **原始TAR全部字节保存与独立重建已实测：** [证据](../results/emg2pose-original-tar-preservation-engineering.json)为130本机/130服务器及完整轻量1,283/713 actual exit0；官方647MB mini实际22.39秒exit0，从全部31文件及25,394额外字节重建原包SHA一致，原盘点/转换/准入和四字段/22数组全等。六项真实失败回归修正了布尔/小数偏移准入；没有改原转换器、删除原始TAR/HDF5或执行新模型训练。冗余TAR受管理回收、完整源和科学实验仍欠。

- **已启动完整原始EMG2Pose并行获取：** 新磁盘容量支持优先使用[原始文件路线](emg2pose-original-disk.md)。[实际核验](../results/emg2pose-original-disk-engineering.json)为15服务器真实aria2测试和91.88秒完整mini下载/解包/原转换/全部526窗口及三模型CPU前向exit0，原字段/22数组全等；本机完整轻量1,235/713actual exit0。全量462.8GB已在独立进程启动、实际8连接，进度取RPC接收量而非预分配文件大小；尚未收全/盘点/训练。之前122361e[托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40535564494)实际2,168/3及1,235/698成功，但只覆盖此前原生压缩基础，不覆盖新磁盘模块。全文/原生完整科学实验仍未完成。

- **压缩源原生窗口与三模型CPU构造基础已完成独立核验：** [实际证据](../results/emg2pose-compact-native-engineering.json)为288本机/288服务器、完整轻量1,235/698actual exit0；完整647MB mini 383.84秒实际exit0、所有拥有的fd和进程退出。全部526官方配置窗口的字段/RNG/完整摘要、原采样5120消费/526锚点以及32次原训练变换+jitter一致。新独立runtime的S/M/L初始权重、实际CPU前向、原AdamW/cosine250构造与严格权重重载一致；旧读取/校验不改。首次mini报告变量冲突导致actual exit1，修正私有探针后另起冻结成功运行，原失败保留。**尚无压缩源实际优化/六状态恢复/GPU/250轮或新精度成绩**；这些仍是F3C2/F4及完整Task4欠项。

- **完整mini压缩源清单与原生元数据索引已独立核验：** [实际证据](../results/emg2pose-compact-source-engineering.json)为167本机／167服务器、完整轻量1,235/696actual exit0及342.65秒整包实跑。全部30文件／4,218,756行／3,559,584原始有效行／1,015块、四字段哈希和22个原元数据数组与原转换相同；双封印源在原持有者关闭后仍可读，全部拥有的fd释放。官方窗口／变换／RNG／采样、原生模型CPU/GPU及完整源／250轮／科学报告仍欠。ef0825[完整托管](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40519871088)已实际2,032/3及1,173/696成功，但只覆盖此前载体／评价测试，未覆盖这份新源索引代码。

- **评价测试改为与正式阶段相同的独立进程调用，完整托管结果仍待观察：** 94a1托管CPU实际1,964/3/11ERROR，单独归一标志不足。已实际复现父1线程／子2线程的身份拒绝；原推理和复推均改为独立CLI，保留同进程必须拒绝的真实检查。污染父环境下16项／291.63秒actual exit0且父线程及cuDNN设置未变，生产评价／模型／门槛不改；这不是新的完整托管成功。

- **完整mini的独立封印压缩字节载体／通用HDF读取已实测：** [完整证据](../results/emg2pose-compact-archive-engineering.json)为105本机／105服务器、完整轻量1,173/696actual exit0，以及647MB全包294.16秒实际exit0。278.4MB载体+49.4KB索引逐字节还原，全30HDF／90对象／4,218,756行所有字段与独立原包相同，真实资源余量和全部fd释放已核验。尚未授予官方数据集／模型准入，F3原生窗口变换与采样、完整源和科学实验仍欠。

- **当前完整源瓶颈有新的无损压缩可行性证据，但尚未接入训练：** [有界编码证据](../results/emg2pose-compact-codec-engineering.json)为48项本机／48项服务器、64个真实1MiB块及64次窗口大小读取，全部原字节和摘要一致；样本估计约196.7GB而非全量容量保证。完整封印载体、原生HDF/数据一致性和全源仍欠。最新f1托管CPU实际1,916/3/11ERROR，cuDNN全局测试状态根因已真实复现并以测试夹具修正，污染环境16项actual exit0；下次完整托管结果仍待观察。

- **EMG2Pose实际FP32完整批量与表格算术已通过工程核验：** [批量证据](../results/emg2pose-memory-evaluation-capacity-engineering.json)为S/M/L、两模式、每pass321窗/实际320+1、原12worker/FP32，35个受管理阶段/1,783.53秒及父进程actual exit0，原选best权重未变、跨进程预测/统计精确一致，全部worker/guardian实际退出。合成三角色5,009,750行与两轮工程fit，初始计数探针失败及源/Sfit只读复用均保留；不是benchmark或250轮。[表格算术](../results/emg2pose-memory-table-arithmetic-engineering.json)另通过48本机/48服务器/15轻量及四份真实GPU统计重读，明确用户/样本/有效角度三种权重与匿名边界。新CPU job配置以实际YAML驱动64项通过，旧5f托管失败不追认；完整源、科学控制器/报告和Task4最终审查仍欠。

- **EMG2Pose最佳模型的原生流式评价完成三模型GPU工程复推：** [实际证据](../results/emg2pose-memory-selected-evaluation-engineering.json)包含服务器16项检查及17个受管理阶段/713.39秒实际exit0；S/M/L均重载真正由原验证最小值选出的工程best，val/test与新进程输入/预测摘要及充分统计量精确一致，校准在测试前冻结，模型权重未变，所有拥有的worker/guardian另核对退出。合成3文件/25,710行，每次仅1个实际窗口（配置320/12/FP32），GPU只运行预先声明的文件行对照，原样混角色val已实际拒绝；CPU覆盖两种模式。没有实际320窗FP32批量资格、完整源、250轮或benchmark成绩，最终科学控制器/报告及Task4最后审查仍欠。

- **EMG2Pose评价数据准入已实际核验两种选择模式：** [新证据](../results/emg2pose-memory-evaluation-data-engineering.json)记录31本机/31服务器检查与真实mini元数据阶段exit0。原样两个条件各230窗；明确文件行隔离后167/63窗、22/8文件、重叠0。val跨角色在读取信号前拒绝，full要求实际完整源；原读取/筛选代码保留，仅作用域内关闭NPZ资源。未构造模型或执行推理，这不是正式校准、最终测试或三项250轮成绩。

- **EMG2Pose独立统计通过真实mini新进程复推，原条件选择问题已实测并分列：** [工程证据](../results/emg2pose-memory-statistics-engineering.json)为33本机/33服务器检查、两个独立受管理CPU阶段、每次460次原生消费和56项FP32/float64指标核对，输入/预测摘要及统计精确复推。mini两个条件的230次消费来自相同30文件行及相同有序窗口；[原筛选审计](../results/emg2pose-native-evaluation-selection-audit.json)另在完整CSV上执行官方函数，test/stage会选入16,831个train文件条目，完整HDF5窗口尚未实测。完整科学评价须分别声明原样和文件行隔离对照，不能静默替换官方人口；尚无正式benchmark成绩。[接收实查](../results/emg2pose-persistent-body-retry-live-observation.json)已完成旧前缀核验并在一次真实中断重试后新增128MiB，完整源仍欠。

- **EMG2Pose原预算fit/恢复/选best入口已通过受控GPU工程验收：** [实际证据](../results/emg2pose-memory-native-fit-engineering.json)为显式合成3HDF/25,710行、分开train/val角色，S/M/L各两完整工程轮次/32实际更新，六状态新进程恢复与独立best/last/前缀/journal读取通过；七受管理阶段exit0，457.00秒。原Sampler按16batch循环补齐，唯一锚点1/2/2与每轮5,120消费分列；原320/12/bf16/clip1/cosine250/top3+last/noearlystop及训练增强保留。full入口要求全部25,253真实源和实际CUDA资格，不能以工程结果代替；本机/服务器各18定向、116成员完整1,798/3及轻量987/693均实际exit0。**没有benchmark成绩或250轮实验**；完整源、最终评价/校准/控制器/报告和Task4最后审查仍欠。

- **EMG2Pose S/M/L原生GPU完整检查点恢复已通过：** [实际证据](../results/emg2pose-memory-native-gpu-recovery.json)记录A800上的六个独立受管理阶段，636.92秒全部实际exit0且拥有的消费者退出。每模型保留原320/12worker/bf16/clip1/cosine250及原top3+last/TensorBoard/noearlystop：两次前缀更新和两轮完整526窗口验证，新进程完整526窗口精确复推，权重/AdamW/cosine/callbacks/循环/RNG六项精确恢复，再实际更新一次至step3并完整验证。本机/服务器各13项真实检查通过；整体软件检查按独立JSON的实际退出记录列示。mini原val仍仅是工程训练夹具，没有独立验证、完整训练人口/增强、连续轨迹或科研成绩；完整源、正式fit/选模/test/报告及Task4最终审查仍欠。

- **EMG2Pose源准备与三模型原批量GPU容量已通过实际检查：** [源准备证据](../results/emg2pose-memory-preparation-engineering.json)为完整647MB mini的30HDF5/4,218,756行独立盘点、索引重验及受管理的准备进程；全部字段与原转换参考一致，本机/服务器各66项检查通过。111成员整体1,767/3实际CUDA skip、轻量987/691依赖skip均exit0。[GPU容量证据](../results/emg2pose-memory-native-gpu-capacity.json)为S/M/L各两次实际batch320/12worker/bf16更新、两轮完整526窗口原生验证，峰reserved约9.89/11.28/15.27GiB，全部拥有的消费者退出。mini原val复用为工程训练夹具，不是独立验证或模型精度；完整源、GPU状态恢复、正式训练/选模/测试与报告仍欠。

- **EMG2Pose缓存消费者的进程管理已通过真实检查：** [实际证据](../results/emg2pose-memory-process-engineering.json)为本机/服务器各20项实际进程检查和完整647MB mini流程；父进程强制退出、孙进程持有fd、资源压力及TERM拒绝均回收，guardian不保留缓存fd。严格重载已有small工程模型后，未改官方逐用户函数在原转换数据/新存储上结果精确一致；IK过滤路径460窗口（230/230，各1用户），CPU/batch1/workers0，无新训练成绩。最终109成员整体1,739/3实际CUDA skip及轻量959/691依赖skip均exit0；完整源控制器、GPU准入、三项完整训练及Task4最终审查仍欠。

- **EMG2Pose只读内存接入已完成真实mini原生检查和一次实现审查修正：** [接入与边界](emg2pose-memory.md)、[全字段盘点](../results/emg2pose-memory-data-engineering.json)及[新原生工程证据](../results/emg2pose-memory-native-engineering.json)覆盖完整647,198,208字节/30HDF5/4,218,756行、全部526窗口与原读取器的样本/RNG一致。S/M/L各2次实际CPU更新、526窗口新进程精确复推、原生权重/AdamW/cosine/循环/RNG加载后再执行一次更新到step3，全部拥有的fd关闭。审查发现最终HEAD阻塞时未持续检查资源压力，真实两项RED→77项GREEN修正；最终107成员整体1,719/3实际CUDA skip及轻量939/691依赖skip均捕获exit0，修后实际526窗口和三模型检查再次完成。33fc756托管CI仍只覆盖此前101成员盘点代码。完整源、资源/进程控制器、GPU准入及三项完整训练仍未完成；mini工程检查不能填Tables11/13。

- **DINO-S/B及显式激活重计算L的实际GPU验收均通过，三者已进入各自全量训练：** [实际S证据](../results/vision-literal-gpu-engineering.json)、[B证据](../results/vision-literal-base-gpu-engineering.json)和[新L证据](../results/vision-literal-large-checkpoint-gpu-engineering.json)各为原360／bf16两轮、4次实际更新，每轮完整3,152手验证；六项原生恢复一致，完整验证预测跨进程精确复推，各七阶段exit0。L保留原模型／损失／优化器／批量，显式启用固定timm激活重计算；原未启用版本的显存失败保留。三项full均保留150上限／patience30，无最终测试成绩。匹配新2129accc[真实托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40420859582)为1,574／3实际CUDA skip及841／689依赖skip。下文CPU阶段“GPU尚欠”为历史记录，不覆盖此新快照。

- **VEMG完整实验与最终核验已完成：** [实际公开报告](../results/original-classic/vemg2pose/README.md)记录62轮、10,535次实际更新，原patience50早停；原生最终测试、独立完整预测复算、统计复核和最后完成核验共12阶段实际exit0。新4,306测试手样本的轨迹／中心MAE为16.452932°／16.347705°，有效角度551,176,956／91,916。保留7,790点、原0.02损失及单rank适配，与旧七项12,000点legacy成绩分列；不是PDF等价或全文完成。

- **DINO原生模块与真实队列CPU优化／新进程恢复已完成工程检查：** [入口与边界](vision-literal.md)保留真实预训练来源、原factory、174项主干保留与178项模块严格恢复、原生0.02损失和component AdamW/cosine。62项定向通过后，真实队列预检发现身份错误要求未读取的缺失raw EMG，已用回归复现并修正为绑定实际点读文件；新89成员1,506／3实际CUDA skip、轻量841／621依赖skip，以及S真实旧版队列两次更新、完整3,152手验证、新进程完整重推与再更新一次后的完整验证均通过。[实际CPU证据](../results/vision-literal-native-cpu.json)保留全部边界；B/L也已通过原生构造／重载，但尚无其cohort优化。完整输入准入和原生fit／验证选模／原生评价及独立复算代码已实现；一次完整实现审查的两项Important已RED→GREEN修正，43定向、96成员整体1,563／4 skip及轻量841／679通过。第四项skip为本机ViT-L随机构造内存不足，另外三项需实际CUDA；新DINO GPU／完整训练尚欠，不产生新模型性能结论。

以下保留此前各阶段的工程快照；其中“尚欠／未完成”是各自记录时的状态，当前执行进展以上文与[运行状态](run-status.md)为准。

- **DINOv2配置／literal数据基础阶段：** [使用与边界](vision-literal.md)、[实际只读扫描](../results/vision-literal-data-census.json)和[CPU证据](../results/vision-literal-foundation-cpu.json)新增原数据模块实读：所有角色batch360（val_batch_size256未消费）、repeat2、原lineage合并来源。保留原jitter且排除352个不安全锚点；可用legacy唯一／消费手样本为101,010／202,020、1,576／3,152、4,306／8,612。所有可能中心目标和crop key已扫描，18个真实点读，52文件stat稳定，无模型或CUDA。37定向、84成员完整1,480／3CUDA skip及轻量841／595依赖skip；新源码最终审查／托管CI仍欠。S/B/L真实cohort优化、完整训练及与缺失合并源/PDF等价仍未完成。


- **验证／原生绑定／测试独立复算／只读状态／完整报告已实现，实际GPU链尚欠：** [使用与边界](classic-report.md)和[新CPU证据](../results/classic-report-complete-cpu.md)记录真实原始证据重开、完整4,306手预测参考／新进程精确复算、原22角／中心统计、真实metadata匿名用户与交集及最后拥有的完成标记。独立审查2项Important以19项RED复现后一次修正：训练统计绑定终态last日志与固定运行时／适配策略；Markdown包含全部公开JSON证据。定向303通过，固定80成员完整1,443／3真实CUDA skip，轻量835／564依赖skip；没有第二次独立复审，本次托管CI尚未观察。上一82109cd真实托管1,351／3及806／501已通过。服务器新目录80文件核验一致，只读API实读6,613次执行／38次完整验证且未初始化CUDA。VEMG于2026-10-26T17:10:49UTC实读6,677次执行、39次完整验证，相同PID/start仍存活；训练终态、实际GPU校准／测试／复算及科学完成产物仍欠。

- **VEMG已通过最终源码GPU限定验收并进入完整训练；NeuroPose未通过：** [实际证据](../results/classic-formal-gpu-engineering.md)记录全部工程输入一致及严格恢复，VEMG汇总与fresh复核成功，Neuro的8/171,380个预测元素超出固定门槛，失败保留。VEMG另起101,714手full训练、完整1,576手val、max100/patience50；13:52:31UTC快照为309次真实更新、1次完整验证，尚未完成或测试。78文件组本机CPU1173/3实际CUDA skip与轻量739/390依赖skip、匹配658d8c6托管CI成功，不替代真实GPU证据或报告校准。

- **七项 legacy 实验已完整运行并审计：** EMGFormer-S/M/L、ResNet-18、RN18+S及独立发布的RN50/RN152。原结果、配方、源码和初始化身份保留在[已完成实验](../results/completed/README.md)及[发布清单](../results/completed/publication-manifest.json)，不因新目标改写。它们与 PDF 原表的数据、标签、输入和训练协议仍有差异。
- **新增损失合成差异已确认：** [11配置实际审计](../results/official-loss-audit.md)证明固定官方EgoEMG配置的两个指尖别名最终相加为0.02，既有legacy训练器/配方为0.01。此前漏核的差异现已明确；旧四实验及当前L/RN50/RN152仍只属于既定legacy协议，不是当前官方合成损失的等价复现。新官方字面实验须另行完成，不能由旧结果补算得到。EMG2Pose直接官方流程仍为0.01。
- **EMGFormer-L已完整结束并审计：** [独立结果](../results/completed/emgformer-l/README.md)为200轮/25,400更新/200次完整轨迹验证，验证选择第508步（第4轮）。全部/有图中心16.612111°/16.571878°，最终测试、严格导出及独立中心复算成功；轨迹16.591133°仅来自原训练器测试，未额外重推。运行源码7b805d4与legacy限制保留，不用后续文档头替换。
- **RN50与RN152均已完整完成：** [RN50独立结果](../results/completed/resnet50/README.md)完成150轮/23,700更新/150次完整独立验证，第19,118步best的全部中心MAE为6.640058°；最终测试、严格导出和锁定复算均成功。[RN152独立结果](../results/completed/resnet152/README.md)完成150轮/52,650更新/150次完整独立验证，验证最小值选择第52,650步，全部/有图中心MAE为5.982738°/5.869706°；最终测试、严格导出和锁定复算均成功。两者运行来源均为a357423，固定官方版本缺两份独立YAML，属于明确声明的legacy派生对照，保留0.01目标与全部数据限制。
- **DINOv2-S/B/L真实预训练与CPU检查均通过：** [S证据](../results/dinov2-small-pretrained.md)和[B/L证据](../results/dinov2-base-large-pretrained.md)记录实际HF权重身份、主干tensor字节比较、官方位置插值、合成256px前向与严格禁网重载。尚未进行真实cohort优化、GPU容量准入或完整训练。
- **classic历史失败与差异保留：** [入口与范围](classic-emg.md)保留原0.02目标、val_loss/min、100轮/原patience50及同批量600/256。早期6adc三条VEMG路径均成功，但汇总的tuple/JSON-list metadata比较和跨运行逐位要求未通过；修正及后续新源码的实际结果分别记录，不追认旧失败。两者当前均未完成正式训练、最终测试或报告，单卡/workers0也不代表原六rank全局batch及BatchNorm等价。
- **WiLoR真实预训练与禁网CPU检查已完成：** [实际证据](../results/wilor-pretrained-cpu.md)记录完整2,564,989,533-byte权重及官方SHA、26个固定源码文件、404个主干tensor全匹配、实际631,631,657参数和52,499冻结参数。原22角前向及完整模型严格离线重载逐位一致，没有随机主干fallback或CUDA初始化；尚无真实cohort优化、GPU容量或完整训练。
- **手形统计与聚合图已生成并独立核验：** [实际统计与分布图](../results/dataset-statistics/shape-statistics.md)覆盖 41 个引用 beta 行/82 手，50 项官方 float32/ddof0/range 重算误差为零。PDF Table5 在三位小数有 48/50 项吻合，两处微差保留；真正 P5/P95 单列。聚合箱线图的展示适配、独立参与者计数限制均明示。校准为 3840×3360，包含 rational_polynomial 与 fisheye 模型，与 PDF 720p 的关系尚待核验。
- **60类词表、22角语义、帧与冻结中心覆盖已核验：** [手势定义](../results/dataset-semantics/gesture-vocabulary.md)、[关节语义](../results/dataset-semantics/joint-angle-semantics.md)、[帧覆盖](../results/dataset-semantics/coverage.md)及[独立中心点读](../results/dataset-semantics/center-coverage.md)区分定义与各分母。66,161,725个共享同步帧有全部60类；已另实读33,508个冻结窗口/67,016手中心，1,420,084个有效角度。train/val/test中心分别观测51/32/59类；不能从词表或全帧覆盖推成每个角色60类。模型误差和全帧有效性仍未由此计算。
- **classic官方模型、数据与CPU训练/恢复及完整队列数值准入已核验，未完成正式训练：** [当前范围](classic-emg.md)保留原tuple/2D mask、固定scalar与新7790网格。新增[真实CPU队列证据](../results/classic-cohort-cpu-engineering.md)对两配方各完整准入及重新hash/扫描一致，train/val/test为101,714/1,576/4,306手样本；finite覆盖冻结足迹65,046,370行。原双别名0.02、val_loss/min、clip0与patience50保留；兼容裁决入身份。独立CUDA限定入口已实现/CPU审查，最终cfbc源码真实16-mixed/scaler的batch1限定检查已完成，各8次尝试/7次实际更新/1次注入跳步及完整状态恢复一致；[原600/256容量](../results/classic-cuda-capacity-engineering.md)六个梯级也已通过，各2次真实更新及完整1576手val；正式full train/val-best/test与Task4仍欠。
- **EMG2Pose metadata、mini与S/M/L真实CPU优化/全窗口重载已核验。** [数据证据](../results/emg2pose-mini-data.md)记录30文件、4,218,756行全字段/原mask位级复核；[早期模型证据](../results/emg2pose-mini-runtime.md)为各单窗口随机初始化检查，[新增真实mini证据](../results/emg2pose-mini-optimization.md)为各2次更新、526窗口×2完整工程评价和严格恢复一致。mini全部原val，优化与工程检查复用它，不能视作独立验证或论文性能；Table11/13仍需完整数据及三个完整训练。
- **公开CPU工程实现已独立审查与整体检查。** `9b5f4931d53db0d536e0ebe64e386e8048872ab8`的[实际托管CI](https://github.com/ZSHYC/EgoEMG-Reproduction/actions/runs/40292505908)为官方CPU **880 passed/2 skipped**（仅真实CUDA正例），轻量609 passed/273可选运行时skip，pip check通过。它覆盖classic cohort、canonical CUDA入口的CPU检查、预取及派生TaskA；不覆盖后续未集成worker/artifact、GPU容量或完整训练。随后72文件本机整合1025 passed/3 skipped、轻量690 passed/338依赖skip；不含仍在开发的formal runner，详见运行状态。历史源与实际执行身份继续保留。
- **扩展代码与聚合证据已公开，原运行身份保留。** [全论文分支](https://github.com/ZSHYC/EgoEMG-Reproduction/tree/codex/full-paper-reproduction)保留各训练提交，阶段CI和CPU/实际数据证据分别记录在[发布来源](publication-provenance.md)与[运行状态](run-status.md)。完整数据接收和多个必需模型仍未完成；绿测试、mini或中心统计不改变正式实验的完成状态。

EgoEMG 有 14 个主要模型设置（5 EMG、7 视觉、2 融合），EMG2Pose 另有 S/M/L 三项，markers2mano 还需独立训练与质量验证。现有七项只是这份清单中的legacy协议实验子集，不能写成“原表已复现7/17”。全文还包括数据统计、预处理和标注重生成、派生评价及全部定性图。

## 状态、证据与验收含义

| 状态 | 具体含义 |
| --- | --- |
| 未实现 | 本项目尚无满足该项要求的实现或产物；官方文件存在不改变此状态 |
| 工程验证 | 已执行真实构造、有限训练、读取或重载等明确检查；只支持所列范围 |
| 完整运行 | 已达到独立实验预算并完成验证选模、最终测试；仍需对应导出与审计 |
| 分析产物 | 已有由真实数据生成的底表、统计或图，尚需核验其协议、身份和计数 |
| 已核验 | 指定产物及其来源、单位、分母、样本集合、聚合规则已经核验；仅对声明的协议成立 |
| 外部依赖 | 缺少指定资产、访问条件、训练规格或存储；应继续推进不依赖它的工作 |

一项可同时有多个状态。下表“官方入口”只证明所列源码或配置已发布；“尚欠”才是本项目仍须完成的工作。数据统计要读取实际数组，派生分析要固定 checkpoint 和样本键，训练结果要有完整结束、仅用验证选模、冻结测试及可重算误差和/有效分母。公开文档不包含原始参与者标识、机器私有路径、账号文件或受限模型。

| 证据编号 | 实际证据 | 支持范围及限制 |
| --- | --- | --- |
| E1 | [四模型完整结果、曲线与身份](../results/completed/README.md) | S/M 各 200 epochs，RN18/融合各 150 epochs；原训练器轨迹结果与锁定中心复算分开，未额外复算整段轨迹 |
| E2 | [共同人口与环境清单](../results/completed/publication-manifest.json) | 同一冻结中心及 crop-present 人口、初始化来源、配置与运行身份；不是其他模型的证据 |
| E3 | [队列覆盖](../results/cohort-coverage.json)、[crop 就绪核验](../results/crop-readiness.json) | 41 份 crop、2,792 单手测试中心、2,688 有图中心；实际测试 40 名用户，User/Both 各有 1 人与训练交集 |
| E4 | [原始选择性资产清单](../assets_manifest.json)、[下载指南](download.md) | 原 21 文件的 EMG/标签与元数据范围；manifest 声明其他字段不代表那些数组已下载 |
| E5 | [本项目数据适配](../egoemg_repro/data.py)、[指标](../egoemg_repro/metrics.py)、[训练](../egoemg_repro/train.py)、[视觉与融合](../egoemg_repro/visual.py) | 已有 20+2 有效性、整体/逐手/split/user、整体手指/腕角；不能从这些汇总恢复逐关节或逐手势误差 |
| E6 | [动态运行记录](run-status.md)、[L 配方](../configs/emgformer_l.json) | 新 L 的短验收与生产运行、额外资产获取分别登记；配置和运行中状态均不是完整成绩 |
| E7 | [词表](../results/dataset-semantics/gesture-vocabulary.json)、[22角语义](../results/dataset-semantics/joint-angle-semantics.json)、[实际帧覆盖](../results/dataset-semantics/coverage.json) | 60类/30·18·12族群、22维单位/mask/名义限位、157条class/split/activity聚合记录；帧审计实读并重哈希三个完整数组，导出层核对审计与守恒；不推断窗口或模型误差 |
| E8 | [冻结中心覆盖](../results/dataset-semantics/center-coverage.json)与[范围/分母](../results/dataset-semantics/center-coverage.md) | 原12000窗口、train stride1200/eval12000；实际点读12字段共6,232,488bytes，33,508窗口/67,016手中心，178个role/split/class/activity cells；本次未全payload重hash，模型误差仍空 |
| E9 | [逐关节/手势真实细分分析](../results/paper-analysis/README.md)、[验证记录](../results/paper-analysis-validation.json) | 五个保留best各完整2,792手中心；S/M/L完整轨迹baseline另行重推通过，22角/10组、59类/族群、四视图/两公式及Figure12已生成。7实际受管理阶段/139.31秒exit0，14消费者和外层拥有者独立确认退出；旧数据/0.01/bf16边界保留 |
| E10 | [Table1原表CSV/JSON与原始来源审计](../results/dataset-statistics/dataset-comparison-audit.md) | 六行原值及六数据源调查，PDF原表已逐行视觉核对；MyoKi/PiMForce/NinaPro脚注差异明示，剩余模态/计数定义未核实 |
| E11 | [原官方FFT首段实际重算](../results/dataset-statistics/emg-fft-first-episode.md) | 固定原函数及12频率点/首段对照，另有[双手全量](../results/dataset-statistics/emg-fft-bilateral-full.md)各41段/66,161,725行的实际重生成与独立读取。全部新filtered4.234GB有限；无新训练或当前发布等价声明 |

## 全部主要模型设置

| 数据/模型 | 官方发布内容与已核验接口 | 本项目状态与真正欠缺 |
| --- | --- | --- |
| EgoEMG EMGFormer-S | 官方 EMGFormer 与 S 配置 | **完整运行、已核验（legacy）**，见 E1/E2；原表精确等价未证明 |
| EgoEMG EMGFormer-M | 6 层、256 维、8 heads、FFN 1,024 | **完整运行、已核验（legacy）**，见 E1/E2；不替代 L |
| EgoEMG EMGFormer-L | [L YAML][large]：8 层、384 维、12 heads、FFN 1,536；实际 16,320,806 参数、22 输出、510/0 上下文 | **完整运行、已核验（legacy）**，见[独立结果](../results/completed/emgformer-l/README.md)；200轮/25,400更新，全部中心16.612111°，轨迹审计边界与配方差异保留 |
| EgoEMG vemg2pose | [官方配置][vemg]及真实 LSTM 模型；实际预测22角、tuple输出、1790/0上下文，5,963,948参数，基类元数据仍写20 | **完整运行、已核验（原生配方／旧版数据单rank适配）**；[实际结果](../results/original-classic/vemg2pose/README.md)为62轮早停、10,535更新、4,306测试手样本，轨迹／中心16.452932°／16.347705°。目标手8ch/WL7790输出6000；原批量600、完整验证选模、原生test和独立复算／报告完成。六rank、原表数据和标注等价未证明 |
| EgoEMG NeuroPose | [官方配置][neuropose]及真实 CNN 模型；16槽、tuple输出、0/0上下文、22角、6,354,969参数 | **CPU恢复和原批量容量通过，正式GPU准入未通过**；WL7790、原零填16槽再归一化/全槽旋转、clip0/val_loss保留。正式限定检查的171,380个预测元素中8个超出预先固定门槛，失败保留、未降低容差；尚未正式完整训练 |
| EgoEMG ResNet-18 | [官方 RN18 配置][rn18] | **完整运行、已核验（legacy）**，见 E1/E2；非其他视觉模型的替代 |
| EgoEMG ResNet-50 | [ResNetVisionPose][resnet]支持该 backbone；[派生配方](../configs/resnet50.json)，含角度头总参数24,568,406 | **150 epochs完整legacy实验与锁定复算完成**；23,700更新/150次full val，best第19,118步，全部中心6.640058°。独立YAML未发布，损失/队列/单卡配方与官方字面/PDF等价仍未证明，见[结果](../results/completed/resnet50/README.md) |
| EgoEMG ResNet-152 | 同一真实类；[派生配方](../configs/resnet152.json)，总参数59,204,182 | **150 epochs完整legacy实验与锁定复算完成**；52,650更新/150次full val，验证选择第52,650步，全部/有图中心5.982738°/5.869706°。独立作者YAML未发布，保留损失/队列/派生配方限制，见[结果](../results/completed/resnet152/README.md) |
| EgoEMG ViT-S/14 | [官方独立配置][vits]、[DINOv2 实现][vit]；真实预训练与严格来源记录 | **完整运行、已核验（原生配方/旧版队列单rank适配）**；[实际结果](../results/original-vision/vit-small/README.md)为124轮/69,688更新、4,306唯一测试手样本，全部/有图中心6.464179°/6.324737°。原360/bf16/repeat2与验证选模、原生测试/独立复算/报告完成；合并来源、六rank及PDF等价未证明 |
| EgoEMG ViT-B/14 | 真实 DINOv2 类；[明确S派生的配方](../configs/vit_base.json)，实际86,182,166总参数 | **完整运行、已核验（S配方显式派生/旧版队列）**；[实际结果](../results/original-vision/vit-base/README.md)为94轮/52,828更新、4,306唯一测试手样本，全部/有图中心6.761261°/6.623680°。原360/bf16/0.02与验证选模、原生test/新进程复算/报告及额外独立重开完成；作者独立B YAML、合并来源、六rank和PDF等价未证明 |
| EgoEMG ViT-L/14 | 真实 DINOv2 类；[明确S派生的配方](../configs/vit_large.json)，实际303,833,622总参数 | **显式原生激活重计算GPU七阶段通过，完整训练运行中**；[证据](../results/vision-literal-large-checkpoint-gpu-engineering.json)保留原360/bf16及模型／目标／优化器，原未启用版本显存失败保留。独立作者YAML未发布，最终测试和完整报告仍欠 |
| EgoEMG WiLoR | [WiLoRViTPose][wilor]真实22角类、原高度裁剪及GELU头；无独立fusion/vision_wilor YAML | **真实权重与CPU接入已完成**，见[证据](../results/wilor-pretrained-cpu.md)：固定源码/配置/mean/完整权重、404主干tensor及严格离线重载已实际核验；未做cohort优化、GPU容量和完整训练，独立22角训练配方仍需明确 |
| EgoEMG RN18+S | 真实残差融合类、清洁分支来源及精确零残差适配 | **完整运行、已核验（legacy）**，见 E1/E2；保留原 S635/RN6319 身份 |
| EgoEMG ViT-S+S | [原生16槽/WL7790构造](fusion-literal.md)、[五配置来源审计](../results/fusion-recipe-audit-20261028.md) | **CPU构造/初始化/严格重载工程验证完成，尚未完整训练**；25,855,037参数，174+4视觉张量完整传输、105EMG匹配/5跳过，原训练模式loss/backward/AdamW及新进程复推通过。当前两分支是明确未训练的工程状态，不能替代真实best初始化；原联合数据/stride400隔离划分和[完整54文件内容/数值准入](fusion-literal-cohort.md)已实际完成并新进程复算一致，GPU/150轮完整fit和最终报告仍欠，见[构造证据](../results/native-fusion-construction-cpu-20261028.json) |
| EMG2Pose EMGFormer-S | [官方 EMG2Pose S 配置][e2p-s]，16通道、20角、独立数据协议 | **真实mini恢复、普通源原批量fit/FP32两模式与报告工程核验通过**；[入口证据](../results/emg2pose-direct-native-fit-engineering.json)保留原320/12/bf16/250cosine及原Sampler补齐，合成分角色两轮/32更新/六状态恢复和原生选best已核验。完整源、真正250轮训练及最终测试仍欠 |
| EMG2Pose EMGFormer-M | [官方 EMG2Pose M 配置][e2p-m] | **同样完成mini原批量恢复和新fit入口工程验收**；合成角色没有benchmark或完整人口意义，正式250轮/最终评价仍未完成 |
| EMG2Pose EMGFormer-L | [官方 EMG2Pose L 配置][e2p-l] | **同样完成mini原批量恢复和新fit入口工程验收**；没有完整科研结果，不能用EgoEMG-L或工程数值填表 |

classic 配方的 precision=16-mixed 和 gradient_clip_val=0 还需要真实实现检查：现有复现训练器只为 bfloat16 启用 autocast，裁剪值直接设 0 会清零非零梯度，不能照搬数值宣称忠实训练。两类模型 tuple 已对齐，不能重复裁剪。RN50/152、ViT-B/L/WiLoR 缺独立 YAML 是固定提交内的已核验发布缺口，不是模型类不存在；新增派生配方必须明确其来源和差异。[官方配置说明][config-readme]也将配置定位为研究记录。

## Tables 1–15

| ID / PDF 位置 | 性质与必须产物 | 状态、实际证据 | 尚欠与验收条件 |
| --- | --- | --- | --- |
| Table 1，§3.3 p.4 | 数据集比较；EgoEMG 为作者数据定义，其余行为外部引用 | **六行原表CSV/JSON及逐源审计已生成**；[来源与差异](../results/dataset-statistics/dataset-comparison-audit.md)保留PDF原值，MyoKi三项、PiMForce交互数/参考文献、NinaPro脚注差异单列；未找到官方独立生成脚本 | 剩余列的逐项原始来源核验、NinaPro原表范围的作者口径与跨库身份、EgoEMG全模态实物盘点。已有原值表不证明全部事实已核实；无需重训这些引用研究 |
| Table 2，§5.1 p.8 | 作者 EMG 主结果；S/M/L、vemg2pose、NeuroPose 的轨迹 Gesture/User/Both 与加权 Avg、参数量 | **S/M/L有完整legacy结果，VEMG有原生配方/旧版队列完整结果**，E1/E2及各独立报告；NeuroPose尚未完整运行 | 五模型各自完整证据；逐用户 mean±SD、样本权重、轨迹 mask/分母；模型同名不能证明与 v1 原表等价 |
| Table 3，§5.2–5.3 p.8 | 作者单帧视觉/融合主结果；7 视觉与 2 融合 | **RN18、RN50、RN152、RN18+S有完整legacy运行，ViT-S/B有原生/显式派生配方的旧版队列完整运行**，E1/E2/E3及[RN50](../results/completed/resnet50/README.md)；仍不等价于原表 | ViT-L、WiLoR、ViT-S+S尚无完整结果；ViT-B见[实际报告](../results/original-vision/vit-base/README.md)，ViT-S见[实际报告](../results/original-vision/vit-small/README.md)；所有模型逐条件用户SD、真实初始化/选择、匹配人口与官方字面配方仍须逐项核验 |
| Table 4，A.1 p.15 | 60类完整词表与描述、30/18/12族群，数据定义 | **定义、实际帧与冻结中心覆盖已核验（legacy）**，E7/E8；完整词表保留raw50；中心有241未分配、66 raw，59类视图33,201窗口，与帧统计分开 | 全帧逐手mask统计仍欠；不同新grid须另报，中心标签统计不能填Table14/15模型误差N。PDF措辞差异单列 |
| Table 5，A.2 p.16 | 作者82手、10维β统计 | **分析产物、已核验（legacy）**；[统计与来源](../results/dataset-statistics/shape-statistics.md)含50项官方重算和实际P5/P95，PDF显示48/50吻合 | 两个微小显示差异保留，原因未定；episode/beta行数不能独立证明人口身份。官方std(ddof0)和range优先，不能把range混称真正P95−P5 |
| Table 6，B.3 p.20 | 22 DoF语义、单位、左右约定和有效性，数据定义 | **语义表与冻结中心逐角mask已核验**，E7/E8；0–19手指rad、20腕FE/21腕RU磁盘deg，模型统一rad；实际中心1,420,084/1,474,352角有效。原literal与legacy20+2分列，腕名义限位未知 | 全帧逐角有效性与标签重生成仍欠；语义相同不证明legacy轴角投影等价PDF landmark IK；不从通用模型末两项臆造腕限位 |
| Table 7，C.1 p.21 | 作者 S/M/L 架构、参数与训练 GPU-hours | **三模型架构及总/可训练参数已统一实测**；[原工厂核验](../results/dataset-statistics/emgformer-architecture.md)：S3,460,134/M6,617,382/L16,320,806，三者均全参数可训练，六次CPU前向/状态不变及M独立对账通过；legacy完整运行另见E1/E6 | 官方新配方的实际完整训练用时、GPU数/型号/精度/验证开销仍欠；CPU前向或A800墙钟不能填原文4090 GPU-hours |
| Table 8，C.2 p.22 | 7 个视觉 backbone 的配置与参数，非另加 7 次实验 | **RN18/RN50/RN152及DINO-S/B部分已核验**；S/B有实际源/真权重/参数与完整运行证据，L和WiLoR进展见模型表；其余配方/初始化未闭合；上表及[模型源码][resnet] | 每种实际 resolved 配置、256 输入/patch 策略、回归头、冻结状态、权重身份及参数量；PDF 超参和缺失独立 YAML 必须分栏 |
| Table 9，D.1 p.23 | Table 2 五模型的逐手/逐 split 轨迹评价 | **S/M/L细分证据已有，完整表未生成**；E1 原训练器轨迹 by_hand_split；额外审计只复算中心 | 固定 best 生成五模型完整左右表与分母；无需另训左右两套模型，中心审计不能代替轨迹 |
| Table 10，D.2 p.23 | S/M/L 中心帧 EMG 参照，为轨迹模型的派生评价 | **S/M/L中心结果已核验（legacy）**，E1/E2/E3及L独立结果 | 固定来源 checkpoint、TDS 中心对齐和同视觉 population、逐用户/样本计数；原文关于 L 最优条件的文字与显示数值不符，不作为验收约束 |
| Table 11，D.3 pp.23–24 | 作者 EMG2Pose S/M/L 实测；† 基线是第三方原论文引用 | **作者三行未完整运行**；普通源S/M/L原batch320 fit/FP32两模式与报告已工程核验，仍没有全源250轮成绩 | 完整数据及 User/Stage/User+Stage 协议、20 维、16 通道、三训练及独立 val/test；引用 SensingDynamics/NeuroPose/vemg2pose/CLDM 的 † 数值应明确来源，不能声称本项目重训 |
| Table 12，D.4 p.24 | EgoEMG S/M/L 的10个逐指/关节组派生误差 | **完整legacy轨迹逐角/10组已生成并独立核验**；[真实分析](../results/paper-analysis/README.md)由三个验证best各2,792手窗口新推理，核对原整体/手/split/组件/用户轨迹baseline，保存22角及重叠组的误差和/分母 | 当前是legacy有效角时间pooled视图；官方0.02/unified训练、原表Avg/跨用户SD等价仍未证明，不能将组均值相加 |
| Table 13，D.4 p.24 | EMG2Pose S/M/L 的 8 组派生误差，无两腕组 | **尚无正式结果，依赖 Table 11**；逐组/逐用户方法及三种权重的独立表格算术已实际核验，不替代完整数据/模型 | 三个完整模型与标准三 split 的逐角计数、同官方组映射及独立复算；无需为该表再训练三遍 |
| Table 14，D.5 p.25 | RN18、S、RN18+S 在59类的3族群同中心比较 | **旧版匹配人口分析已生成并核验**；[底表/族群/图](../results/paper-analysis/README.md)为三独立best共2,792请求，排18未分配后2,774请求/2,690至少一角有效，59类；all/crop及all-test/gesture-only四视图、两公式分开，族群官方加权独立对账 | 旧0.01/bf16/训练区间归一化与PDF/current不等价；官方公式保留无mask/token中点/57.3，但不是默认FP32原入口；旧N非4,278，84全角无效请求明确记录 |
| Table 15，D.5 p.27 | 同Table14底表生成59类ID/name/N/三MAE/Δ与排序 | **完整59类定义表及四视图/两公式CSV、JSON已生成并核验**；[实际表](../results/paper-analysis/README.md)在族内按未舍入Δ排序，空类null，raw50/未分配排除数保留；本次all-test两公式均58/59类融合改善 | 58/59是当前旧版单seed实测，不代填论文54/59；同一受控官方新配方及原人口等价仍欠 |

Table 12 的官方组映射：proximal=[0,1,4,5,8,9,12,13,16,17]，mid=[2,6,10,14,18]，distal=[3,7,11,15,19]；五指各占连续四维。各组重叠，不能把组均值相加还原总体。

## Figures 1–13

| ID / PDF 位置 | 性质与必须产物 | 状态、实际证据 | 尚欠与验收条件 |
| --- | --- | --- | --- |
| Figure 1，p.3 | 原采集设备、布置、marker/MANO 与穿戴示意 | **来源说明部分已有，采集未重做**；[论文指南](paper-guide.md)与原创示意只支持说明 | 原系统设备/拓扑/同步/标定记录与可核验图；公开数据 benchmark 不要求重新招募，但不能将原照片说成本项目新采集 |
| Figure 2，p.4 | 3 类同步样例，每行 3.9s 的 GT、双腕 EMG、ego RGB、外部 RGB-D | **未实现完整面板**；E3/E4 有 EMG/crop 中心链 | 固定匿名 episode/time/gesture、原始全帧/外部 RGB-D、GT mesh 与时间偏差/有效性；crop+信号曲线不足以证明完整多模态同步 |
| Figure 3，p.5 | (a)82手β箱线图；(b)参与者×手势split矩阵 | **(a)聚合展示图、(b)协议示意与真实旧发布矩阵已生成并核验**；[手形统计](../results/dataset-statistics/shape-statistics.md)、[41键×60类矩阵及计数](../results/dataset-statistics/split-matrix.md)。三标签全396,970,350字节重哈希及逐项对账，556多标签格、全帧Train∩User/Both各5键；与模型中心队列1/1分开 | (a)隐藏单点/离群点是展示适配，官方[β脚本][beta]另画左右柱图。(b)左为PDF概念、右为真实旧版帧标签；保留raw/未分配与交集，不证明PDF人口等价，不声称逐像素重画 |
| Figure 4，p.6 | EMGFormer remapping/TDS/RoPE/head/增强架构 | **按实际源码的架构图及三模型形状已生成并核验**；[图/记录](../results/dataset-statistics/emgformer-architecture.md)为原样12,000点/8目标手通道、两TDS stage/globalSE/非因果RoPE/ReLU头；三模型各另测7,790诊断长度，重复绘图逐字节一致 | 仅声明配置中的Rotation/ToTensor，不等于完成PDF全部增强、实际归一化或合并训练数据；PDF7790ms/通道/头激活差异分列，图不是新精度结果 |
| Figure 5，p.6 | 256 crop→ResNet/ViT→池化/角度头 | **RN18 部分完整运行；图未重生**，E1；[真实实现][vit] | 全部 7 backbone 构造与初始化；具体头激活由实际类裁决，不能以一张通用图代表七项实验 |
| Figure 6，p.7 | 视觉预测+EMG 时间池化残差融合 | **RN18+S 已核验（legacy）；ViT-S+S 原生CPU构造/真实点读已检查，完整训练未完成、图未重生**，E1/E2 | 两对独立来源、真实形状、零残差等价性与架构图；官方极小正态初始化与本项目精确零适配分别记 |
| Figure 7，p.9 | 自遮挡、深度歧义、运动模糊三类同例 GT/EMG/Vision/Fusion 定性图 | **未实现**；E1 已有三模型，尚无关联误差的渲染图 | 固定样本与选择规则、同一角度/mesh 空间、合法模型与相机/视角、逐例未舍入 MAE/有效分母；论文单例数值不填本项目图 |
| Figure 8，p.17 | markers2mano 图 Transformer 架构 | **未复现；外部依赖**；[官方推理入口][m2m]存在，模型包未随仓库发布 | 实际图谱/PE/参数/权重、forward 与 778 顶点重建；脚本 fallback 宽度/层数不是已核验论文训练配方 |
| Figure 9，p.18 | marker 训练的 8 类增强可视化 | **未实现** | 实际变换概率、参数、组合顺序、固定 seed 前后 marker/mesh；EMG channel rotation 不能替代 marker 增强 |
| Figure 10，p.18 | 11 源原始规模与约 712K 平衡混合物 | **来源调查已有，样本混合未重做** | 各源版本/许可/计数、确定性采样/去重/划分 manifest 与真实饼图；H2O/H2O-3D 身份待作者清单消歧 |
| Figure 11，p.21 | 原 RGB 上 MANO 重投影及 UmeTrack/MANO 重建质量 | **未实现；外部资产依赖**；[IK][ik]与重投影相关官方入口已发布 | 随机抽样清单、原全帧/标定/变换、MANO 与 UmeTrack 资产、实际 overlay 和质量统计；crop 不能代替全帧重投影 |
| Figure 12，p.26 | 59类Vision/Fusion散点，点大小=N、颜色=族群 | **旧版真实底表的原脚本参考及清晰展示PNG/SVG已生成并核验**；[图与口径](../results/paper-analysis/README.md)四视图×两公式共8组，原始绘图PDF中间物/颜色/点面积原样，另供完整范围/清晰标注；63产物重复逐字节一致 | 原脚本Top4字重叠、负收益标注循环、2.5度下界裁掉gesture-only的PinkyHook均披露并给补图；不证明PDF人口/数值精确等价 |
| Figure 13，p.26 | 密采样自遮挡与视觉/融合误差、收益趋势和相关 | **未实现；外部几何资产依赖**；[评价][occlusion-eval]、[几何算法][occlusion]、[绘图][occlusion-plot]存在 | 1/4 stride 的实际密网格、GT mesh/相机/分辨率与过滤、逐样本底表、bin/SEM/趋势/逐样本 Spearman；原文 17,168 个手样本不是现有 2,792 中心 |

## 数据、采集、标注与算法清单

这些项目与模型训练并列验收。消费公开标签不等于重生成标签；引用原采集元数据不等于重新采集。

| ID / 论文位置 | 必须覆盖内容 | 当前实际证据、状态与欠缺 |
| --- | --- | --- |
| D01，§3.1/A.3 | 原采集人数、人口、时长、prompt 与双手执行 | **部分来源核对**：E4 的 41 episodes、66,161,725 行只能说明发布记录，按 2kHz 约 9.189h；不能证明 >10h 原录制、性别/年龄或自然人去重。缺匿名人口/原采集记录；无需伪造重新招募 |
| D02，§3.1/A.3 | EMG/IMU/mocap/双视频的采样、时间戳软同步与线性插值、校准诊断 | **部分工程证据**：E3 有中心映射与 crop 覆盖，新校准 JSON 已读取；缺全部模态、视频原生尺寸、stale/delta 分布和同步核验。原文未报告统一同步误差数值，不凭空新增该论文结果 |
| D03，A.3 | release schema、投影框/左手翻转/256 crop、全模态盘点 | **legacy 获取与读取已核验**，E3/E4/E5；缺从 mocap+全帧重生 crop、当前 v3 manifest/checksums 与原 v1 队列对应。选择性清单不是完整 release |
| D04，A.4 | DC、50/100Hz notch、bandpass/rolloff、mV 与边界、原始 EMG | **旧版双手各41段的原官方FFT重生成与独立复核完成**；[完整证据](../results/dataset-statistics/emg-fft-bilateral-full.md)：每手66,161,725行/8通道，合计4.234GB输出全部有限，两个原入口与两个独立读取均exit0。[首段频率/数值对照](../results/dataset-statistics/emg-fft-first-episode.md)已完成；全41段与旧filtered逐值对照、独立单位标定及当前发布/PDF等价未证明，保留整episode非局部依赖 |
| D05，B.1.1 | 21 虚拟 marker、MANO 顶点索引、左右 convention、局部坐标和刚体恢复 | **官方推理规格部分可读，未重运行**，见[官方推理入口][m2m]；缺 raw markers/valid、合法 MANO 模型和实际重建产物 |
| D06，B.1.2 | 图 Transformer、8 维 Laplacian PE、最短路 attention bias、β/θ6D→MANO | **未实现、外部发布缺口**；推理脚本导入未发布的 markers2mano 包；缺真实训练架构/图谱/配置/权重。不得以未训练替身验收 |
| D07，B.1.3 | 11 源参数/mesh、tier 采样、约 712K 混合、去重和 train/val/test | **来源调查，不是采样完成**；下节逐源列访问情况。缺精确样本清单、采样 seed/权重、版本与来源划分；图中比例不是实采数据 |
| D08，B.1.3 | 50% 不增强；8 类为 bone length、global scale、近邻 swap、dropout 插值、neighbor blend、Gaussian noise、局部 drift、spike outlier | **未实现**；缺官方变换训练代码、精确参数/组合与概率、确定性检查和图9产物；不能用姿态模型 EMG 增强替代 |
| D09，B.1.3 Eq.2 | vertex MSE + 5×marker MSE 的 annotator 训练、选模与质量评价 | **未运行**；缺训练入口、schedule、选择规则、权重与完整曲线。姿态基线 L1+tip 损失不等于该训练 |
| D10，§3.2/B.1.3 | marker/vertex 误差、invalid rate、InterHand 子集与重投影质量 | **未重测**；论文 aggregate 只作参考，缺单位、有效帧判据、数据划分与误差分布。3.2mm 的正文 per-marker/附录 per-vertex 名称矛盾待裁决；不是角度真值精度 |
| D11，B.2 | 腕带 marker 构系、FE/RU 几何公式、左右符号、退化帧/有效性 | **仅消费 released 腕角**，E5 单位转换已核对；未从原 marker 重生成，缺原始腕带/手 marker、顺序/正交化/阈值及可确认生成入口 |
| D12，B.3 Eq.3 | MANO→UmeTrack landmark IK、对齐/限位/L-BFGS、收敛/残差 | **官方可读实现存在，未重生成**，见[batch_ik_mesh.py][ik]；其 mapper 训练不是 D06。缺 MANO pose/β/transform 与合法模型；legacy 元数据描述另一算法，不可当同一链路 |
| D13，§3.3/A.2 | 每参与者左右各一β、82手统计 | **实际统计与聚合图已核验（legacy）**，见[产物](../results/dataset-statistics/shape-statistics.md)；metadata引用行去重，每行左右各一次。episode/参与者自然人身份的独立证明仍缺，不能将逐帧重复值当独立手 |
| D14，§3.4/§4.2 Eq.1 | split、MAE、逐用户 SD、加权 Avg、轨迹/中心区别 | **legacy 部分已核验**，E1/E2/E3/E5；与 PDF 队列不等价；需所有新模型逐项复核。± 是跨用户 SD，不是多 seed 误差条 |
| D15，§4.3/C.1 | 左右 remapping、dropout/frequency mask/noise/jitter 等 EMG 增强 | **现有仅声明的官方 rotation 适配**；不能称覆盖 PDF 全部增强。classic 默认/with_aug 与实际归一化由 resolved 配置裁决，缺真实新配方训练 |
| D16，§4.3/C.1 | TDS/SE/RoPE/回归形状、输入与上下文对齐 | **S/M/L legacy完整运行已核验**，E1/E6；classic原模型tuple/真实shape已核验，数据和训练链尚待适配。按真实上下文校验分母，非因果/全窗行为不宣称在线延迟 |
| D17，§4.4/C.2 | 7 视觉模型的真实主干、初始化、角度头、损失和训练 | **RN18/RN50/RN152 legacy完整，DINO-S/B原生/派生配方旧队列完整**，见模型表；L仍完整训练，WiLoR尚无完整训练；真实预训练来源与缺失独立配方继续分列 |
| D18，§4.5/C.2 | 两对残差融合、时间池化、分支来源、末层初始化 | **RN18+S legacy 已核验，ViT-S+S 原生CPU构造/初始化/时间池化优化器已核验、真实best来源与完整训练仍欠**；E1/E2。精确零是明确适配，官方极小正态与 PDF zero 的区别保留 |
| D19，D.6 Eq.4 | camera-space 自遮挡 z-buffer、5×5邻域/5mm、三角面积加权 | **[官方算法][occlusion]存在，本项目未运行**；缺 GT pose/β/faces/world→camera/K/真实尺寸与 validity；不得以 crop 遮挡或 2D 框重叠替代 |
| D20，D.6 | 1/4 stride 密测、bin、SEM、paired gain、趋势与 Spearman | **未运行**，见 Figure13；[官方绘图][occlusion-plot]的实际过滤/样条与 PDF 不同，按下节裁决并保留全部底表 |

## 资产与外部依赖

| 资源 | 已确认事实与当前状态 | 尚欠/继续执行条件 |
| --- | --- | --- |
| legacy 已用数据 | E4 的原 21 文件约 18.39GB，另有视觉对齐字段；E3 的 41 份 crop 约 45.04GB 已核验 | 不包含所有 raw/mocap/MANO/camera/video；字段存在于 schema 不是获取证明 |
| β 与 calibration | [左β][left-beta]、[右β][right-beta]、[校准][calibration]共4,646bytes已取；beta统计/聚合图已核验 | 参与者身份独立证明与相机适用性待核验；统计的两处PDF微差和图形适配详见实际产物 |
| raw 与 markers | 官方 legacy 公共目录已定位；双手 raw 约 4.234GB，双手 marker+valid 约 36.124GB | 双手raw已[完整接收](../results/raw-emg-full-acquisition-20261028.json)，并完成[双手全部41段官方FFT](../results/dataset-statistics/emg-fft-bilateral-full.md)；marker位置与valid尚未完整取得，已有几何续传片段不算完成 |
| 遮挡几何 | 必要 pose、world/camera 变换、tracked、marker-valid、β 约 37.778GB，再加校准与尺寸证据 | 未获取完整必要字段；需合法 MANO 模型和 legacy webcam→current head 字段映射，不能静默 β=0 或尺寸 fallback |
| current v3 | [ASSET_SETUP][asset-setup]写当前百度路径与 preview/checksums；[LICENSE][official-license]仍写完整 current release 未发布 | 当前目录实物、完整 manifest/checksums 与 v1 队列未闭合。官方 CLI 路线需百度登录，不将“链接可打开”写成“全量已匿名下载” |
| EMG2Pose | [Meta 官方数据与代码][emg2pose]给匿名 S3；完整包 462,824,048,640 bytes（约431GiB），metadata 与30文件 mini 已完整取得并校验 | 完整原包现已取得并记录[接收证据](../results/emg2pose-full-archive-acquisition.json)，完整源准备与准入仍在进行。[直接转换](../egoemg_repro/emg2pose_data.py)已通过mini全字段验收；[分块xz归档](../egoemg_repro/emg2pose_archive.py)已通过独立审阅，正式全量接收已启动且首块后新块提交成功，仍明确incomplete。动态容量不足将停止并保留可恢复进度，不能缩小cohort；首块压缩比不保证全量容量。不能同时保存全tar/H5/Zarr/memmap |
| EMG2Pose 发布权重 | Meta 参考权重对应其 classic 模型；EgoEMG 作者 EMGFormer 权重是另一资产 | 当前代码 val/test 已修复不追溯清洗旧选模史；不得以旧 released checkpoint 或 mini 替代三个新完整实验 |
| MANO | 限定项目范围查找已发现左右 MANO 模型文件；[官方许可][mano]与具体格式兼容性尚待核验；安装 smplx 不等于完成资产验收 | 不能继续称模型完全不存在或必须重新下载；须核验现有文件的格式、版本、许可与 WiLoR/标注/遮挡接口，不再分发模型或公开私有位置 |
| markers2mano | [推理/模型加载入口][m2m]与[质量评价入口][m2m-eval]已发布，.gitignore 排除 markers2mano；默认 checkpoint 为作者私有运行 | 缺模型包、11 源训练预处理/采样清单、训练入口/配置/增强、train/val 来源划分、v51 权重公开来源。MANO→UmeTrack mapper 训练不能补这个缺口 |

下面是 annotator 的 11 个来源，不是“全部需要账号”，也不是“已经全部获取”。本页记录 2026-10-26 调查快照；公开链接与传输、许可检查、完整混合物是不同门槛。

| 来源 | 官方路径/可推进部分 | 实际缺口 |
| --- | --- | --- |
| GigaHands（PDF 写 GigaHand） | [作者仓库](https://github.com/brown-ivl/GigaHands)的 hand_poses HTTPS 包匿名 HEAD 可达，约11.30GB | 未下载；需包内许可与参数/抽样核验，不必为 mesh 训练获取全部多视角视频 |
| HOT3D | [官方 toolkit](https://github.com/facebookresearch/hot3d)；其 [HF clips](https://huggingface.co/datasets/bop-benchmark/hot3d)部分训练 GT 可匿名获取 | 全量 VRS 路线需邮箱领取限时链接；clips 不是作者完整采样证明，仍缺确切混合物与版本 |
| ARCTIC | [官方数据说明](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md)，raw_seqs/meta 可独立于数百 GB 图像获取 | 需本人注册与 research license；相关模型另有许可，未取得训练数据 |
| HOI4D | [官方页](https://hoi4d.github.io/)公开 Hand Pose/Camera/Annotations 链接 | 参数包传输与大小未验证，不笼统写必须申请账号 |
| HanCo | [官方数据页](https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html)提供 shape/calib/meta 分包 | 调查时官方旧下载端点 404，需恢复或核验正式镜像；research-only 条款与采样待闭合 |
| DexYCB | [官方页](https://dex-ycb.github.io/)有全包/按 subject 分包与 MANO 相关标签 | 未发现独立完整 MANO-only 小包；可单 subject 集成，不能当全量 mixture 完成 |
| H2O / H2O-3D | 文字引 [Kwon H2O](https://github.com/taeinkwon/h2odataset)，图写 H2O3D 76K、与 [H2O-3D](https://github.com/shreyashampali/ho3d)相近 | 数据集身份未决；前者需注册、后者有公开链接，不能混同；需作者精确来源清单 |
| InterHand2.6M | [官方页](https://mks0601.github.io/InterHand2.6M/)独立 annotations 含 MANO pose/shape/trans | 未获取/核验；无需仅为参数训练下载全部 RGB |
| FreiHAND | [官方页](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)有训练/eval 资源说明 | 调查时官方 v2 zip 端点 404，需正式端点恢复/镜像和 research-only 条款核验 |
| HO3Dv3 | [作者仓库](https://github.com/shreyashampali/ho3d)公开 v3 与 evaluation GT 链接 | 未验证完整传输；custom academic/noncommercial 及再分发限制，需参数提取/采样清单 |
| HO3Dv2 | 同一作者仓库公开 v2 与 evaluation GT | 同 v3；两版本录制重叠与训练权重须按作者清单核验，不能当独立新增帧直接相加 |

## PDF 与官方实现的裁决记录

采用“官方代码优先”时，应写明实际源码、resolved 配置和影响；无法由已发布内容裁决的项继续标为待核验。不能通过改写旧结果或挑选论文显示值来消除差异。

| 项目 | PDF / 旧数据陈述 | 固定官方源码与本项目处理 |
| --- | --- | --- |
| 输入/时间 | 7,790 samples；图4/6写7790ms；图示8通道而C.1写16 | [L][large]等当前配置为目标手8通道、12,000点；7,790samples在2kHz为3.895s。新实验逐配方声明；官方 unified 配置可能含多来源，不当 legacy 队列已等价 |
| 标签算法 | B.3为 landmark L-BFGS IK | [官方 IK][ik]存在；实际 legacy metadata_summary 明写 axis-angle projection，legacy README 又描述 anatomical Euler 路径。应保留版本/生成算法差异，不能凭同22维宣称同标签 |
| 滤波/归一化/优化 | PDF无幅值归一化；EMG/视觉/融合预算与部分学习率不同 | 以实际配置和运行身份为准；现四模型训练专用标准化、整 episode filtered、RN18与融合配方均已冻结，新运行不得追溯改写 |
| classic 内部矛盾 | vemg 注释称无归一化/原架构，数据却设16槽 | [实际 YAML][vemg]启用 per-dataset 归一化，首卷积仍8通道；NeuroPose需16槽。先记录显式数据/模型裁决，再真实验收；基类 out_channels=20不代表实际预测20角 |
| 视觉独立配方/头 | 表8给7模型配方；图示与文字激活不同 | RN50/152、ViT-B/L/WiLoR独立YAML未发布；ResNet/DINOv2头为ReLU，WiLoR角度头为GELU。不能将脚本引用缺失文件视为完整recipe已具备 |
| 融合初始化 | PDF末层zero | 当前官方为极小正态权重、零bias；本项目已声明精确零适配，旧 clean 分支身份保留。发布16槽配置不与8通道分支静默混载 |
| β统计/图 | Table5末列写P95–P5；Fig3a箱线图、PCA无固定语义 | [beta脚本][beta]实际std(ddof0)、末列ptp=max−min，画2×5左右柱图且附语义名；以range说明官方算法，真正分位差另算，图形/语义差异保留 |
| 手势评价人口/误差 | T14/15称所有split、59类；PDF族群Δ与逐类显示Δ加权不全相符 | [官方eval][gesture-eval]默认gesture split、取EMG token中点，逐样本MAE未按传入mask筛角；[paper脚本][gesture-paper]依赖输入先排raw。本项目masked/aligned中心不能静默当同一指标；须固定实际过滤、底表和差异 |
| Table10文字 | 段落称L最强Gesture/Both，显示表中S更低 | 保留原文内部矛盾，输出真实模型结果与选择规则，不以文字结论约束结果 |
| 遮挡工程前提 | 论文相机/原生分辨率与legacy webcam字段 | [eval入口][occlusion-eval]根路径可能指向scripts/config，硬编码校准、读current head字段，β缺失默认0、尺寸失败默认1920×1080；这些为静态风险，未运行不称实测失败。必须核验字段/路径/尺寸，不能默用fallback出图 |
| 遮挡统计/曲线 | 固定0.05 bin、LOESS/趋势置信带；正文与图内ρ不同 | [实际绘图][occlusion-plot]先滤occlusion非空且vision MAE>0.01，0.05基础bin合并至默认至少100样本，末不足10丢弃；使用加权三次UnivariateSpline，非LOESS；Spearman用过滤后逐样本。只(a)绘SEM带，趋势无对应置信带。按代码生成并明确差异，不抄ρ/p |
| 数据人数/时长 | 41参与者、6heldout、>10h | E3/E4实际41记录、40测试用户、User/Both各5用户且各1与train交集；发布行数约9.189h。标签数不自动证明自然人数，采样人数不等于窗口数 |

## 外部引用、未来工作与未展示项目

- Table 1 中其他数据集和 Figure 10 的原始来源规模属于外部资料；annotator 的实际抽样混合与训练属于作者工作，两者均需来源但不能混作本项目实测。
- Table 11 的 † 结果来自对应原论文。VQ-MyoPose 文中只提不同子集；CLDM/VQ 未在 EgoEMG 报告实验。无需创造论文没有的 EgoEMG 训练行。
- RGB-D、IMU 和时序多帧视觉是论文说明的数据模态及未来方向；应核验数据存在性、同步和 schema，不捏造原论文对应模型成绩。
- PDF 没有独立定量消融表、推理 latency/FPS/throughput/FLOPs 或多 seed 实验。C.1 被正文称有 ablations，实际未展示；B 被称有逐手势标签误差分布，实际只有方法与 aggregate。登记为“论文提及未展示，待官方资产”，不虚构数值。官方额外 joint-range/t-SNE 图也不误编号成 v1 的13幅图。
- consent/IRB/补偿/匿名化、许可、稳定托管/校验及 checklist 应映射到作者实际公开文件；原自述不能替代本项目证据，也不要求伪造新伦理审批。承诺未来发布的资源不当已取得资产。

## 继续顺序与全文完成门槛

1. L已完整结束并完成中心审计；继续正式classic入口的最终源码GPU验收、完整训练与冻结测试，同时推进DINO真实cohort训练。保留现有七项legacy身份，另行对齐官方0.02与原回调；不追溯改写旧成绩。
2. 已生成并核验β统计与聚合图、60类词表/22维语义及中心覆盖；继续补齐各模型逐手/中心子表和统一参数/实际用时统计。每项仍须保留协议、读取身份与实际计数。
3. 从固定 selected-best 做一次足够细的派生评价，保存匿名 sample key、逐角 error/valid、gesture/hand/split/crop，可共同支持 Table12/14/15、Fig7/12；不能从旧总体均值逆推细分结果。
4. 实施经典模型 tuple/context/16槽/精度适配，以及 ResNet/DINOv2 分派、真实初始化和明确派生配方；ViT-S完整结束后再以 clean 分支训练 ViT-S+S。WiLoR先闭合专用资产与22角配方。
5. 完成 EMG2Pose mini 工程验收，独立解决完整数据存储/转换与三训练；并获取遮挡必要几何、MANO/相机证据后先最小集成、再完整密采样分析。子集检查不替代全量。
6. 持续查找 markers2mano 缺失模型包、训练规格、权重和精确11源清单；先完成公开可得来源与标注下游检查，真实发布缺口逐条保留，不以未训练替身宣布完成。

全文仅在本页所有必需作者实验、数据/标注算法、分析与定性产物达到各自验收，引用/未来项正确归类，公开结果与同一证据一致，且无未完成或未验证必需项时完成。阶段计划结束、配置存在、mini、绿测试、短训练或四基线结果均不能替代这个门槛。

[official]: https://github.com/zhenqis123/EgoEMG/tree/f7cbd048b7092139852270491d3859cd472e0311
[large]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/egoemg_emgformer_large.yaml
[vemg]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emg2pose/regression_vemg2pose_egoemg.yaml
[neuropose]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emg2pose/regression_neuropose_egoemg.yaml
[rn18]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/fusion/vision_resnet18.yaml
[resnet]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/resnet_vision.py
[vits]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/fusion/vision_vit_small.yaml
[vit]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/vit_vision.py
[wilor]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/wilor_vit.py
[fusion-vits]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/fusion/fusion_vits_s_center_8ch.yaml
[e2p-s]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/emg2pose_emgformer_small.yaml
[e2p-m]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/emg2pose_emgformer_middle.yaml
[e2p-l]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/emgformer/emg2pose_emgformer_large.yaml
[config-readme]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/config/experiment/README.md
[gesture-paper]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/paper/generate_per_gesture_appendix.py
[gesture-eval]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/eval/analyze_per_gesture.py
[beta]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/paper/plot_beta_diversity.py
[constants]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/constants.py
[paper-figures]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/paper/plot_paper_figures.py
[m2m]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/mano/infer_mano_for_egoemg.py
[m2m-eval]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/eval/eval_m2m_checkpoints.py
[ik]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/ik/batch_ik_mesh.py
[filter]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/data/generate_egoemg_filtered_paper_memmap.py
[occlusion-eval]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/eval/evaluate_occlusion_fusion.py
[occlusion]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/occlusion.py
[occlusion-plot]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/paper/plot_occlusion_fusion.py
[asset-setup]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/docs/ASSET_SETUP.md
[official-license]: https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/LICENSE
[left-beta]: https://drive.google.com/file/d/1B8fhGsjNhy7-pkeqaA8Ru1kmMtAb-z2L/view
[right-beta]: https://drive.google.com/file/d/19SMv0e-aL1gNSpN5I5qcv2ROHxIxCJDv/view
[calibration]: https://drive.google.com/file/d/1atBjy4EgNLNzw87SiOvC4CCjNrljxfcc/view
[emg2pose]: https://github.com/facebookresearch/emg2pose
[mano]: https://mano.is.tue.mpg.de/

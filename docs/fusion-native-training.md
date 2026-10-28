# ViT-S + EMGFormer-S 原生融合训练

当前在实现[全文设计](superpowers/specs/2026-10-26-full-paper-reproduction-design.md)内的正式融合实验。[实施计划](superpowers/plans/2026-10-28-native-fusion-training.md)分开登记权重交接、完整训练、冻结评价和实际GPU/完整运行；本文的接口存在不表示完整150轮已完成，也不增加已完成模型数。

## 官方执行配方

固定上游`f7cbd048b7092139852270491d3859cd472e0311`，实际合成实验`fusion/fusion_vits_s_center_16ch_wl7790`。目标手8通道按原位置放入16槽，其余补零、不插值；输入7,790点、训练stride400，三个角色都repeat2、实际batch200。保留有效0.02指尖项、原增强与生成标签mask、bf16/clip1、AdamW1e-4/wd0.01和cosine150/eta_min1e-5；150轮、不早停、完整验证按`val_mae/min`选择最终融合best。

分支初始化遵循原配置：视觉取完成训练的**last**，EMG取完成训练后独立验证选出的**best**。融合模型自己的最终测试仍使用融合验证best。原生8到16通道加载器按精确名称和形状匹配：105项EMG张量加载、5项跳过，不扩展或重塑首卷积。最后残差权重保留原std1e-5正态初始化和零bias。

## 训练来源交接

`fusion_initializer_handoff.admit_fusion_initializers`要求完整注册融合输入，重新打开两个实际full producer的原生检查点、轮次记录、完整验证历史与选模。EMG导出还必须由`read_emg_initializer_export(..., source_fit=...)`核对原生所选best和全部110项张量，工程文件、自报训练完成或只有文件hash不能通过。

已完成ViT-S的训练早于后来为ViT-L增加的激活检查点功能，因此两者的fit读取源码不同。交接支持已经发布的旧读取器，其完整SHA256固定为`e44dc947111bec3aa858fb5fc36493650cda24f2093915bcd133a2d090d070da`；另核对其三个公共依赖的旧/当前实际字节一致，才在独立模块名中加载。原报告、当前视觉模块及运行中的EMG初始化源码不修改。不执行任意调用者提供的Python文件。

两个producer的分区从当前实际metadata、frame split重新构造；共同EMG、标签、图像和元数据payload必须同源。输入足迹按所有可能抖动范围合并，EMG/视觉训练与**新融合整个验证区间**和全部测试输入的交集都必须为零。兼容EMG的receipt、导出和重建分区还必须绑定同一个未来融合partition摘要。

```python
from egoemg_repro.fusion_initializer_handoff import admit_fusion_initializers
from egoemg_repro.fusion_literal_runtime import build_fusion_runtime, fusion_optimizer_identity

branches = admit_fusion_initializers(
    cohort, vision_fit=completed_vision_fit,
    vision_source_root=frozen_vision_source,
    emg_fit=completed_compatible_emg_fit,
    emg_export=compatible_emg_best_export,
)
branches.check(require_full=True)
prepared = build_fusion_runtime(upstream, config, **branches.runtime_kwargs)
configured = prepared.module.configure_optimizers()
optimizer_identity = fusion_optimizer_identity(
    prepared, configured['optimizer'], configured['lr_scheduler']['scheduler'],
)
```

`engineering_fusion_initializers`是独立、明确的工程入口，只绑定可信本地文件及当前cohort；它不声称权重经过训练，`check(require_full=True)`必须拒绝。不能通过改一个scope字段升级为正式交接。

## 优化器与人口

`fusion_optimizer_identity`检查原生八个参数组，特别保留`_unassigned`中的时间注意力参数；所有可训练参数必须各出现一次，没有额外冻结、外来张量或重复。它同时核对各组原始学习率、AdamW设置、cosine预算/下限及scheduler与实际optimizer的归属，不替换原优化器。

实际旧版完整输入为395,276训练、3,152验证、8,612测试次手样本消费；测试仅有4,306个唯一手样本。原缺图行为保留，完整评价需分开all/crop-present、重复消费/唯一人口及有效角分母。当前缺失的合并语料、旧标签、整episode滤波、公开归一化和单rank/workers0适配仍需披露，不能宣称PDF原表数值等价。

正式fit入口已接入；GPU资格、完整150轮、最终测试与报告仍按上述计划继续。完整兼容EMG150轮尚在运行，当前没有新的ViT-S融合最终MAE。

## 完整训练与恢复接口

`fusion_literal_train.fit_fusion`已接入原生完整轮次循环。full模式使用完整注册输入、真实完成的双分支交接和独立冻结cohort receipt，保留395,276训练/3,152验证次消费、原batch200/bf16/150轮；禁止工程截断或早停。`cpu_engineering`只训练前4个工程样本、batch2/两轮，`cuda_engineering`保留batch200/两轮、最多400个训练样本；两者均消费各自已准入队列的全部验证样本，不是论文成绩。

```python
from egoemg_repro.fusion_literal_train import fit_fusion, read_fusion_fit

result = fit_fusion(
    cohort, branches, output,
    scope='full', cohort_receipt=frozen_cohort_receipt,
)
verified = read_fusion_fit(output)
```

中断后重新准入同一数据和分支，再以`resume=True`调用。恢复必须先重推整个冻结验证集，并核对模型、优化器、scheduler、原生loop、callbacks和RNG六类状态；原生迭代器重建仍可能改变后续训练RNG轨迹，不承诺与不中断训练逐位相同。旧未完成结果原字节另存；未提交的日志尾部另存后回到最后完整检查点。若原生检查点已完成预算，只补做完整验证与选模验收、生成结果，不再更新参数；已有有效完成结果会拒绝重训。

私有fit身份同时记录数据/crop/upstream位置、完整config、分支文件及独立封存的producer位置。`read_fusion_fit`重新hash和扫描实际完整输入，并重新打开两个真实producer；不会只接纳一份自报完成的summary。该复核在结果重开时执行，不在每个训练batch重hash61.38GB。原native选中best、完整轮次记录、覆盖分母和checkpoint实际张量还需再次一致。上述私有位置应保留，公开报告只应包含安全摘要；最终测试/报告接口仍由计划Task3完成。

[CPU训练与恢复证据](../results/native-fusion-fit-cpu-20261028.json)已在匹配提交`aaadf52`的源码上独立重开：合成真实memmap/LMDB队列308训练/4验证/12测试个消费样本，明确只训练前4样本、batch2、两轮/4次更新；完整4样本验证、六状态恢复、298项原生状态严格重载和新进程完整验证预测摘要精确一致。另在已完成的任务自有CPU工程检查点上显式模拟结果文件丢失，重新完整验证并补写结果，新增优化更新0；原检查点、轮次与训练日志逐文件字节未变，旧结果原字节保留。该collector实际29.69秒exit0，无CUDA，没有测试集推理或新的融合精度。

[对应检查与发布验证](../results/native-fusion-fit-cpu-validation-20261028.json)记录43项原生检查、完整轻量1,460通过/1,020可选依赖跳过，均实际exit0、4个测试拥有进程独立确认退出、源码/测试摘要未变。保留一次长临时目录触发Unix socket路径上限的失败，缩短测试目录后整套通过，未改生产代码或测试以绕过；最初过期临时fixture的collector也保留失败，后续使用本轮保留的工程检查点完成。真实GPU资格、完整150轮、最终评价以及整个新融合实现的一次最终审查仍未完成。

## 已完成的交接工程证据

[真实视觉分支半链](../results/native-fusion-initializer-vision-half-20261028.json)使用冻结源码91a879a，在新服务器目录中实际100.05秒exit0。54个输入文件/61,379,659,366字节重新hash及完整数值扫描通过；原ViT-S124轮完整读取、实际last的SHA和52个共同payload精确核对。重新构造全部可能训练输入，与新融合**整个验证区间**及测试输入的交集都为0。四个拥有进程独立确认退出、84固定成员逐项未变；无模型构造、CUDA或优化更新。

新交接/原生构造与优化器76项、相关配置/数据及强化冻结读取器88项均通过；完整轻量1,460通过/1,000可选依赖跳过。两组原生检查含一个重复的冻结读取器用例，不把它算成两个独立用例。工程分支和真正的工程producer均无法升级为full。尚未完成兼容EMG150轮的正式接纳，因此该证据明确`full_two_branch_handoff_complete=false`，也不是新融合训练或精度。

## 冻结评价与独立复算接口（CPU已验证）

新增 `fusion_literal_eval` 使用融合训练自身的验证best，保持原生validation/test步骤、22角生成mask、缺图处理和完整repeat2消费。`evaluate_fusion(..., role='val')`先运行原生验证；在另一个进程调用 `verify_fusion_evaluation`执行新预测，再以 `freeze_fusion_validation`封存验证配对和事先声明的精确输入/预测一致规则。测试角色必须显式提供这个更早完成且属于同一fit的profile；测试原生pass完成后还需另一个进程重新推理，不能只重读同一份预测文件。

```python
from egoemg_repro.fusion_literal_eval import (
    evaluate_fusion, verify_fusion_evaluation, freeze_fusion_validation,
    publish_fusion_report,
)

# 每次独立复算在新的进程中执行；以下示意各阶段的参数。
evaluate_fusion(cohort, fit, val_native, role='val')
verify_fusion_evaluation(cohort, fit, val_native, val_replay)
freeze_fusion_validation(val_native, val_replay, profile)
evaluate_fusion(cohort, fit, test_native, role='test', profile=profile)
verify_fusion_evaluation(cohort, fit, test_native, test_replay, profile=profile)
publish_fusion_report(fit, test_native, test_replay, report)
```

固定原生batch200/bf16；CPU工程显式为batch2。保存实际样本索引、左右手、窗口、中心、原始目标/mask、图像有效性及EMG+RGB逐样本内容指纹。原repeat2在不同batch边界处也必须配对同一个输入；完整消费与唯一首遍分别统计，原样本加权批量弧度MAE与float64有效角度加权度数MAE分列。报告保留左右手、split、匿名用户、22角和指组统计，用户交集按实际metadata重新构造，不能声称严格未见自然人测试。

`publish_fusion_report`默认只接纳真实完整训练。仅在显式`require_full=False`（CLI `--engineering`）时允许工程报告，且不设置科学完成。公共目录同时含metrics.json/metrics.md、实际完整epoch journal派生的curve-data.json及PNG/SVG曲线；曲线中的训练复合loss和验证弧度MAE分轴显示。最终completion.json在重新核验原始证据后最后写入，已有目录禁止覆盖。重新核验还独立重开完整fit和原始预测，不能通过改summary、换成last或重封图片摘要代替真实曲线。

实际CPU检查已完成：31项评价检查、122项原生模型/数据及训练策略回归、完整轻量1,460通过/1,051可选依赖跳过均actual exit0；6个测试拥有者独立核对已退出，10个源码/测试摘要未变。包含真实两轮工程fit、验证与测试的新进程预测、完整4/12消费、6唯一测试样本、原缺图mask保留及错误best/未冻结测试/改日志/改图片拒绝。追加独立collector和真实完整数据/GPU检查继续；ViT-S融合完整150轮和最终科学误差仍欠。曲线核验要求匹配报告记录的Matplotlib/FreeType/字体内容环境。


[独立CPU评价证据](../results/native-fusion-eval-cpu-20261028.json)绑定提交c14e6f8：另一个真实进程重新预测全部12个工程测试消费（6个唯一、6个缺图），与原生测试的typed输入/预测精确一致；报告和最终verify也在各自的新进程完成，三个实际子进程均exit0并独立确认退出。新父进程再次重开完整fit、验证profile、原始预测和实际epoch曲线，未初始化CUDA；只支持两轮/4更新的合成文件工程范围，不增加科学完成数。

[检查记录](../results/native-fusion-eval-cpu-validation-20261028.json)另保留最初29项检查、追加两项curve绑定RED及最终31+122/完整轻量结果。最初collector在三个子进程成功后，用缺少Torch的system Python汇总时exit1；失败日志保留，之后使用匹配CPU解释器只完成独立最终核验，重启子阶段0、额外训练更新0。没有把第一次失败改记成成功。

[真实数据/GPU资格流程](../results/native-fusion-training-gpu-launch-20261028.json)已从c14e6f8冻结86成员单次启动，先在两个新进程重验完整输入，再检查原batch200/bf16的训练、恢复、完整验证和独立评价。启动时四个实际拥有者存活，当前启动快照只证明正在输入核验；不证明GPU阶段或150轮已完成。资格用明确未训练EMG工程分支，完整融合仍须等待兼容EMG150的真实best接纳及最终整体审查。


[当前训练源码的两次全量输入重验](../results/native-fusion-training-readmission-20261028.json)已完成：两个新进程分别96.61秒/85.49秒actualexit0，54文件/61,379,659,366字节和完整有限足迹的身份记录精确一致；四个对应worker/guardian独立确认退出，86固定成员未变，无CUDA或模型更新。资格流程随后进入prefix阶段，原拥有进程继续；10:29UTC实际观察到它在指定GPU上持有CUDA上下文，仍不能据此宣称完整训练/恢复/评价资格已通过。一次最终整体审查及兼容EMG150完成后的full交接/150轮继续保留。

### GPU阶段的实际终态与报告依赖修复

[11:04UTC前独立核验的终态](../results/native-fusion-gpu-report-failure-20261028.json)：原流程10:56:16UTC实际exit1，只失败在最终CPU报告导入Matplotlib。前9阶段均实际exit0：两次全输入准入、原batch200/bf16的一轮prefix与恢复到两轮/4更新、完整验证原生与新进程预测、事前profile、完整测试原生与新进程预测。恢复六状态和完整验证精确一致，验证/测试分别3,152/8,612消费，typed输入与预测摘要精确一致；prefix实际峰值分配16,886,108,672字节。所有20个worker/guardian与原recorder/controller实际退出，86冻结成员重新核验未变。该结果仍未完成整个GPU资格，也不是完整训练成绩。

根因是部署环境未安装项目已声明的`report`可选依赖，非模型或预测失败。新建隔离报告恢复目录，固定Matplotlib3.9.4及其依赖，只计划重做报告和新进程报告核验；旧失败日志、空报告目录、训练、预测与源文件均保留。不修改共享训练环境或已封存源码，不重启训练/预测，不改变一致门槛。恢复成功后再接纳完整工程资格及请求本计划唯一一次最终整体实现审查。

[报告恢复后的完整工程资格](../results/native-fusion-training-gpu-qualified-20261028.json)已实际完成：本机11个固定版本安装包共37,955,509字节，ZIP/本机SHA/完整远端SHA一致，隔离离线安装actualexit0、相同worker搜索路径下实际版本全部匹配。两个新CPU进程先生成报告，再独立重开完整原始fit/验证profile/测试链、重新生成图像并逐字节校验，均actualexit0/未初始化CUDA；整个恢复11:26:26UTC实际exit0，新增训练更新和预测重启均0。原报告失败与9个成功阶段不改写。24个阶段worker/guardian和4个流程拥有者独立确认退出，86原成员、3356恢复成员及11安装包再次全hash未变，最终报告六文件/completion摘要一致，PNG已目视检查。仍只是原batch200/bf16/两轮4更新/400训练样本工程范围；完整3,152验证与8,612测试消费不能替代完整训练。一次新融合最终整体审查、真实兼容EMG150双分支交接和新融合完整150轮仍欠。

[本计划唯一一次新实现最终审查](../results/native-fusion-training-final-review-20261028.json)现已完成：fresh-context只读审查74b83a2→7ab6e43的全部新交接/训练恢复/评价报告和八组优化器集成，0Critical/0Important/1Minor，可继续真实完整交接和训练。唯一Minor为`fusion_literal_train.py`末尾额外空行，仅影响range diff格式检查；为保留实际资格身份暂不改字节。8个涉及源码/测试与c14e6f8逐字节相同，无生产修复或新失败，不重复已通过套件。完整交接、完整150与科学结果仍须真实完成；此前模块与其他论文要求没有因审查scope被删除。当前Task4Steps1–3完成，Steps4–6继续。

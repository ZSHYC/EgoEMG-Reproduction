# 旧模型派生评价：当前工具范围

[tools/legacy_derived](../tools/legacy_derived) 已完成来源/协议元数据预检、纯NumPy归约及私有中心缓存接口。**四个旧模型已经各执行固定4手中心的实际GPU限定推理；完整派生评价、论文表格和定性图仍未完成。**

[实际GPU证据](../results/legacy-derived-bounded-gpu.json)对应2026-10-26T13:00:09UTC结束的独立运行：S、M、RN18、融合四阶段均exit0，每个模型只消费两个冻结窗口／4个手中心，A800／原bfloat16，均明确标记`engineering_partial_batch=true`和`this_derived_full_passes_per_model=0`。重新核对的13个服务器receipt与原私有归档逐字节一致。这里只验收限定推理接入；配置了原batch参数不代表实际执行过完整大小的batch，也没有由此完成2,792手全队列、Table12/14/15或Fig7/12。完整payload未由该限定pass重新hash，不能据此宣称整套数据准入。

预检CLI只开放 `python -m tools.legacy_derived.cli preflight --manifest FILE --output NEWDIR`，返回 `preflight_only/no_inference`、`benchmark_eligible=false`。它在读取模型tensor或数据payload之前核对四个已选旧模型、旧运行时源码、官方源码和原小型协议文件；当前新训练器、已预导入的同名运行时或被替换的来源会明确拒绝。真实旧源码与四份协议已在独立新Python进程中核验，未导入Torch/NumPy或打开checkpoint/datapayload。

纯归约器分别保留有效角数量加权MAE、逐样本MAE及用户间population SD（ddof0）；空集为null。12,000点窗口的物理中心，在原510点左上下文后的11,490点输出中是索引5,490。它保留未知手势、raw50、crop、active以及定义59类的不同统计人口，不把某个人口的分母套到另一个人口。私有非pickle缓存要求四模型共享同一typed target/mask/cohort及完整顺序，文件权限0600并校验外部整文件SHA；公开投影仅保留聚合统计。

37项定向测试实际通过，另有四项独立手算/来源替换/缓存负例及独立真实元数据检查通过。被测新工具独立于旧 `egoemg_repro` 包，不改变旧训练来源、模型、损失或初始化守卫。旧实验的0.01损失、12,000点输入和legacy数据限制继续保留。

独立[旧运行时worker](../tools/legacy_derived/worker.py)已实现来源保持、禁网加载、typed状态复核和私有中心cache。65项CPU定向检查及独立原故障回归通过；公开目录替换/最终source变更的两个真实RED已修复并独立复核。它仍只开放固定两测试窗口的工程模式，上述实际GPU检查不是全批量容量准入；下一步需要扩展受控完整pass，在冻结selected-best上每模型执行一轮完整派生推理、核对原基线并独立复算聚合。纯算术测试、元数据预检和4手限定推理都不能替代这些工作。Fig7的真实几何/渲染资产也需另行核验。

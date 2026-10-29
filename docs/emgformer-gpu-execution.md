# 原生 EMGFormer 显卡验证与完整执行

目前只完成执行流程的CPU预检，尚未取得显卡资格或新200轮结果。[实际证据](../results/native-emgformer-gpu-workflow-cpu-preflight-20261029.json)绑定core源码b09b58c及单独冻结的worker/controller字节：预检69.162秒、控制器89.217秒，16文件/16,540,457,757字节，0模型/0优化器/0更新/未初始化CUDA；4个拥有进程退出，来源和输入全SHA/九项属性不变。显卡准入辅助函数25项检查通过；优化模式两个拒绝负例、重复启动拒绝负例通过。独立流程审查待完成。

执行只调用已核验的[原生训练](emgformer-native-training.md)、[完整输入](emgformer-native-full-data.md)及[冻结评估](emgformer-native-evaluation.md)入口。显卡工程验证先使用原batch500/六worker/BF16、平衡的1,000训练手样本：一轮前缀，再由新进程恢复到两轮/四次总更新，保留全部验证1052与测试2792。完整验证、独立复算、冻结及报告验收后，才允许另一目录从官方初始化开始原生200轮/62470训练手样本/25000更新；不复用工程验证权重。

[worker](../scripts/run_native_emgformer_worker.py)与[controller](../scripts/run_native_emgformer_controller.py)是冻结运行目录内部入口，需要复制成private/runtime-worker.py与private/runtime-probe.py。它们从自身路径的上一级确定运行根目录；必须先准备完整的private/stage-receipt.json、request.json、registered-cohort.json，正式执行另需已解决的重要审查问题记录final-review.json。manifest绑定原生Python成员、执行脚本与请求/输入回执；使用已有耐久退出记录器启动控制器，不能直接把工作树当成配置齐全的运行目录。CPU预检来源与后续正式运行分别冻结，不会把先前证据重签到新源码。

request指定execution_mode=preflight_only时，只做只读CPU预检，不能启动任何模型。full_after_qualification模式包含20个独立拥有的阶段：preflight；qual-prefix/resume/accept；qual的val-native、val-replay、freeze、test-native、test-replay、report、verify；full-fit/accept；full的同样七阶段。工程范围始终不能生成完整科学资格。

本服务器执行将候选严格固定为GPU1/3的实际UUID；其他设备仍由既有任务保留。准入既要求空闲，也要求至少64GiB：任何compute application都会拒绝该设备，包括本用户的其他任务。控制器最多等待24小时，超时保留真实失败，不自动重试；实际GPU阶段同时检查CUDA输入/模型/输出、被动优化步数、NVML与拥有的PID/start/UUID绑定。资源降至host256GiB、输出15GiB或GPU8GiB保留量以下时，只关闭自己的阶段；其他用户进程不被终止。

公开脚本与CPU预检私有冻结脚本内容相同，可按报告中的SHA核对。更换服务器需要显式调整候选设备策略与请求并生成新的冻结证据，不能在活动运行上静默替换UUID或放宽资源条件。此处没有增加模型、改变损失/掩码/归一化，也不证明PDF人口或数值等价。完整轨迹分析、其他模型及全论文的其余工作继续单独登记。

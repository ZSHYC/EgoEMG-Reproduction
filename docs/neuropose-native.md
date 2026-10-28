# NeuroPose 原生连续训练

本路径执行官方 NeuroPose 的一次全新连续训练。原 `canonical_fp16_eval_copy_v1` 恢复后训练轨迹检查仍保持失败，相关源码/阈值/产物不修改。[执行裁决](../results/neuropose-native-execution-ruling-20261028.json)记录既有恢复与 fresh/fresh 诊断、推论边界及新路径的精确验收条件；这不是新的 GPU 资格或科学成绩。

新模块 `neuropose_native_train` 保留原厂模型、16槽/7790点、22角完整轨迹、生成2维mask、固定scalar归一化与原增强次序。使用原 Lightning loss、AdamW/Cosine、val_loss/min/patience50 的选模/早停，以及 native top-one/last checkpoint。完整模式为101,714训练/1,576验证/4,306测试手样本，B256/16-mixed/GradScaler、max100epochs、单rank/workers0。单rank的全局批量与BatchNorm统计不等于原六rank，旧版数据也未证明与PDF/缺失合并数据等价。

`fit_neuropose(..., scope='full')`必须先提供新的原始GPU资格文件，以及实际完整cohort、numeric与三阶capacity来源。完整入口仍等待本路径真实GPU验收完成；调用者布尔值、旧失败记录、CPU工程结果、改summary或调batch/precision均不能打开完整训练。没有`resume`或覆盖已有输出的接口；失败/中断保留为不完整，不能自动重启或续训。

```python
from egoemg_repro.neuropose_native_train import (
    fit_neuropose, read_neuropose_fit, reopen_neuropose_best,
)

# ordinary small memmap corpus; explicitly CPU engineering only
fit_neuropose(data, upstream=upstream, config=neuropose_recipe,
    partitions=frozen_partitions, output=output, scope='cpu_engineering')
record = read_neuropose_fit(output, require_full=False)
selected = reopen_neuropose_best(output, require_full=False)
```

CPU工程为B2/32-true/两轮，最多前4个训练样本及该小队列全部验证；CUDA工程为原B256/16-mixed/两轮、前512训练及完整1,576验证。两者从身份到reader都与full区分，默认reader拒绝工程结果。CPU RNG快照不读取或初始化可见CUDA设备。

每轮训练和验证均检查完整唯一样本覆盖及原native损失；实际optimizer post-step hook计数与尝试次数分列。epoch journal在完整验证后提交，native checkpoint携带身份、实际完整轮次/初始模型指纹/journal及RNG，保留普通模型/optimizer/scheduler/scaler/loop/callback状态。reader重开实际输入、配置和来源，核对journal、checkpoint与真正val_loss最优文件，检查状态有限性、实际AdamW计数、完整RNG和摘要；不以last替代best。

严格重开最佳权重后，原生完整验证必须重现选中轮次的typed预测/输入摘要和验证损失。后续完整测试还需提前冻结的完整验证native/独立预测配对、新进程完整测试复算、逐手/split/匿名用户/关节/指组的trajectory与center统计，以及真实曲线和公开报告。NeuroPose轨迹长度7790、窗口中心3895，不能套用VEMG的裁后6000点。

当前实施计划为[原生连续NeuroPose](superpowers/plans/2026-10-28-neuropose-native-uninterrupted.md)。完整数据GPU资格、一次新实现最终审查、完整训练和最终科学报告仍是必需后续；整个论文目标保持进行。

[CPU工程实证](../results/neuropose-native-fit-cpu-20261028.json)绑定源码a0e4f4f：原厂6,354,969参数，两轮/两次实际optimizer更新，小型普通memmap队列2训练/2验证/2测试手样本，只消费train/val。新进程严格重载真实best，再运行全部2个验证样本，与选中轮次的typed预测/输入摘要精确一致；父进程独立重开原始输入、checkpoint和journal，无CUDA，新增训练0，所有原fit文件字节未变。该过程实际10.84秒exit0，不是完整数据/GPU资格或最终科学误差。

[检查与执行身份](../results/neuropose-native-fit-cpu-validation-20261028.json)记录最终33项原生、60项相关及完整轻量1,474通过/1,070可选依赖跳过，均actualexit0；五个源码/测试摘要未变，八个检查拥有者和独立验证两个拥有者均确认退出。7项状态绑定/非法布尔值追加RED后修正，另补配置先行拒绝和CPU RNG不触碰CUDA的RED→GREEN。初次轻量检查误开Torch可选集成开关导致两项失败，原日志/状态保留；只修正该环境变量后全套通过，没有改生产代码或断言去绕过失败。资格/评价/报告仍为计划Task2，实际GPU/唯一一次新实现审查/full100仍为Task3；旧恢复轨迹失败结论保持。


评价模块 `neuropose_native_eval` 使用原生 Trainer 验证/测试及真正独立进程推理。保留六条有类型和批次顺序的 EMG、target、mask、sample、metadata、prediction 摘要链，以及统计量；不会保存整个多GB预测张量。原最佳权重、scaler、实际设备、进程身份、运行时、完整输入与来源均绑定，默认完整模式拒绝工程结果。训练 checkpoint 与结果记录实际 fitter 的 PID/start ticks，GPU资格需确认 fitter、native验证与独立复算进程均已退出。

```python
from egoemg_repro.neuropose_native_eval import (
    run_neuropose_pass, freeze_neuropose_validation,
    qualify_neuropose, publish_neuropose_report, verify_neuropose_report,
)

# Run native validation, then replay in ANOTHER process with the same fit.
run_neuropose_pass(fit, native_val, role='val', kind='native')
# In a separate process:
run_neuropose_pass(fit, replay_val, role='val', kind='replay', reference=native_val)
# Freeze only after both artifacts really exist and have been checked.
freeze_neuropose_validation(native_val, replay_val, profile)
# Full native test then another-process test replay both consume that profile.
run_neuropose_pass(fit, native_test, role='test', kind='native', profile=profile)
# In a separate process:
run_neuropose_pass(fit, replay_test, role='test', kind='replay', reference=native_test, profile=profile)
publish_neuropose_report(fit, native_test, replay_test, report)
verify_neuropose_report(replay_test, report)
```

显式CPU/CUDA工程推理须传 `require_full=False`，报告标为工程验证。CUDA资格另用 `qualify_neuropose`，只接受原B256/16-mixed、两轮512训练及全部1,576验证的真实CUDA工程fit/native/replay；CPU、旧失败gate或布尔值不能替代。资格reader重开numeric runtime、完整cohort与三阶capacity来源，并与当前源码核对。

测试统计保留全部22角/7790点/中心3895，按生成mask计算trajectory与center，含手、split、匿名用户、关节、手指和PD组；native批聚合指标单列。PNG/SVG曲线来自真实journal和所选checkpoint轮次，两个纵轴都是原生复合损失，val_loss不是角度MAE。独立report验证重新生成两种图并核对字节，completion最后写入；报告准备期间原输入或journal变化会拒绝发布。

[新评价CPU实证](../results/neuropose-native-evaluation-cpu-20261028.json)绑定源码4f9a3cb：原厂小队列native验证/测试及独立进程预测精确一致，额外新进程复算2测试手样本并生成报告，另一新进程重开原始证据、精确重绘PNG/SVG；原fit文件字节未变、无CUDA和新增训练。初次汇总父进程因项目根目录导入路径遗漏exit1，原状态/日志保留，两次子进程均exit0；修正收尾的独立父进程exit0且未重跑预测。[检查实证](../results/neuropose-native-evaluation-cpu-validation-20261028.json)为55原生、330相关与1475轻量通过/1091可选依赖跳过，均actualexit0，七个源码/测试摘要一致、六个检查拥有者和四个独立校验拥有者均已退出。[工程曲线](../results/neuropose-native-evaluation-cpu-curves-20261028.png)已目视检查。实际GPU资格、一次新实现最终审查、全数据100轮上限/50轮patience训练及完整科学报告仍未完成。


第一次实际GPU验收的两轮512训练/全部1576验证、4次实际optimizer更新已经完成；随后native最佳权重验证在设备检查处失败，未形成GPU资格。[原始失败实证](../results/neuropose-native-gpu-device-teardown-failure-20261028.json)保留全部失败日志与真实已完成fit。原因是绑定版本Lightning的Strategy.teardown会把模型移回CPU，旧检查取了收尾设备；当前评价器在每一次forward记录真实EMG与模型设备，要求全批次一致，并单独保留收尾设备。此修改不改变模型、数据、精度、批量、损失或精确预测标准。新源评价须再次完成全部验证native/独立复算和资格重开，才能开始full；旧完整工程fit的原训练源码/字节不变，作为冻结权重复用，不重新训练。

[设备记录修正的当前源码CPU实证](../results/neuropose-native-device-fixed-cpu-20261028.json)绑定cfbc24b：真实forward后移动模块的回归检查及全套56原生/330相关/1475轻量通过（1092项可选依赖跳过），均exit0。额外独立推理与另一进程重绘报告、父进程重开原始fit全部exit0，原fit字节不变、无CUDA/新增训练；PNG与已目视检查的工程曲线逐字节相同。实际GPU的全1576验证native/replay验收仍在进行，尚未取得资格或启动full。

[实际原B256/FP16 GPU资格](../results/neuropose-native-gpu-qualified-20261028.json)已完成：冻结两轮/4次更新的原工程权重，新源native与另一进程完整验证全部1576样本（6×256+40），有类型的输入/预测精确一致；7批均实际CUDA:0计算，native收尾CPU与replay收尾CUDA分别保留。最后CPU资格reader重开原始fit/来源/runtime/cohort/numeric/capacity全部通过，三阶段exit0、全部8个所属进程退出、107个stage成员及原fit全部字节未变，新增训练0。旧错误运行保持失败。仍须本新实现唯一一次最终代码审查后才能启动full；该资格没有最终科学成绩。

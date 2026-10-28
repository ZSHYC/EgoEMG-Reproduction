# 与融合划分兼容的官方 EMG 初始化

旧EMG-S已完成其原协议实验，但训练输入与新ViT-S融合的部分验证、测试输入重叠，[实际分区检查](../results/fusion-initializer-footprint-compatibility-20261028.json)已记录。因此新融合需要另训初始化模型，旧结果与权重保持原样。

## 当前已实现的部分

`emg_initializer_literal`原样合成固定官方`emgformer/regression_egoemg_incre_cotrain_wl12000_small`，只增加私有来源绑定；改动执行配置会被拒绝。原配方为8通道、12,000点完整轨迹、训练stride1,200、batch260、repeat1、150轮、AdamW学习率0.0005、bf16与原批量增强。原指尖损失别名归约后是0.01；按`val_mae/min`保存top3和last，没有早停。

归一化实际调用官方`_load_norm_stats`，从原配置声明的alias统计文件读取左右手各8维float32均值/标准差。不是融合用的16槽标量，也不是本项目重新估计的训练集统计。缺失Incre语料、旧版标签、整episode滤波与统计来源限制继续保留，不能宣称论文原表数值等价。

`build_initializer_partitions`保留官方episode上的1,200点训练网格及完整抖动范围，排除与**新融合全部验证区间及测试输入足迹**相交的训练锚点；所有可能训练输入行还必须是发布的train。验证使用现有融合验证区间内能完整放下的12,000点窗口，不建立初始化阶段的测试角色。[只读真实人口可行性](../results/fusion-compatible-emg-recipe-feasibility-20261028.json)为63,026训练、582验证手样本，已由下述真实完整准入确认。

`EmgInitializerDataset`使用真正的官方类实例执行未修改的完整`__getitem__`。适配只提供旧版物理字段别名、元数据、已选窗口索引和I/O；原归一化、变换、随机抖动、腕角换算、22维完整轨迹及生成有效性仍由官方方法处理。旧webcam元数据明示映射为head；缺少腕部视频路径时使用官方空字符串默认值。声明存在的手势或腕角有效性文件缺失会失败，不会静默删掉字段。

```python
from egoemg_repro.data import LegacyStore
from egoemg_repro.emg_initializer_literal import (
    compose_emg_initializer, emg_initializer_normalization,
)
from egoemg_repro.emg_initializer_data import (
    build_initializer_partitions, EmgInitializerDataset,
)

cfg = compose_emg_initializer(upstream)
norm = emg_initializer_normalization(upstream)
partitions = build_initializer_partitions(LegacyStore(data))
train = EmgInitializerDataset(data, partitions, upstream=upstream,
    config=cfg, normalization=norm, role="train")
```

样本按左手全部窗口、右手全部窗口排序，每手只消费一遍。`emg`为`8×12000`，`joint_angles`为`22×12000`，不是中心帧监督。读取器保留原数值；即使mask=false也不清理NaN，必须由后续完整数值准入明确拒绝。

## 实际证据与待完成项

[CPU证据](../results/emg-initializer-input-cpu-20261028.json)仅使用合成的真实memmap文件：与完整原生构造的读取器逐字节对照两手、训练/验证、可选标签、腕角换算、完整mask和RNG。初次35项缺模块检查失败后实现通过；增加边界检查并连同原融合配置/读取回归共87项通过。独立新进程读取4个完整轨迹样本、确认构造RNG及所有输入文件不变，无模型/CUDA。原生加载器实际执行了batch260配置下的最后3样本不足批次，不证明真实260样本GPU容量。

## 完整数值输入入口

`admit_emg_initializer_cohort`已实现完整数值文件hash，以及每个角色全部可能抖动窗口的双手EMG和22维完整目标轨迹扫描。它在任何数值映射建立前登记文件身份，绑定原生窗口索引、元数据、手、变换、归一化及只读映射布局；`expected`需要新一轮实际内容重算。声明且消费的手势、腕角有效性字段同样纳入文件身份，不要求未消费的图像数据库。

```python
from egoemg_repro.emg_initializer_cohort import (
    admit_emg_initializer_cohort, native_emg_initializer_loaders,
)

cohort = admit_emg_initializer_cohort(data, upstream=upstream,
    config=cfg, normalization=norm)
cohort.check()
loaders = native_emg_initializer_loaders(cohort)
```

两个角色均调用官方batch260加载器，单rank/workers0、repeat1、全局RNG、原生collate和最后不足批次。训练入口已在批次和完整轮次边界接入文件与来源守卫。

[独立CPU数值入口证据](../results/emg-initializer-cohort-cpu-20261028.json)为28项检查通过，以及两个真正新进程对合成真实文件的完整hash/轨迹扫描：16文件、328训练/4验证手样本，两次记录逐字节一致且实际退出0。最后可能抖动位置的EMG、非中心指角及腕角NaN（含mask=false）、文件替换时序、索引/步长/offset修改、伪造记录及有限值改动均有拒绝用例。未消费目标上的NaN保留。没有模型、GPU或优化更新。

[服务器真实完整准入](../results/emg-initializer-cohort-full-20261028.json)已完成：16个文件、16,540,457,757字节，63,026训练/582验证手样本；两次新进程完整hash和所有可能轨迹扫描逐字节一致，实际退出0，总计72.07秒。两角色共45,267,270个可能时间行、724,276,320个EMG元素及1,991,759,880个目标角元素均有限；全部拥有进程已独立确认退出。这一步未构造模型或初始化CUDA，不增加完成科学设置数量。

## 原生模型与完整轮次入口

`emg_initializer_runtime`在官方factory之前调用官方`setup_runtime`，用seed42随机初始化真正的`Emg2PoseFormer`，严格重载全部110个状态张量。官方完整22×11,490输出对齐、0.01有效指尖项、批量增强、所有参数的AdamW以及cosine150/eta_min5e-6保持原实现。20项原生检查通过，包含loss/backward/真实参数更新、原始schema与来源/版本拒绝。

`emg_initializer_train`的full模式要求已独立冻结并重新核验的真实完整数值记录，全部63,026训练手样本、全部582验证手样本、batch260、bf16、150轮、clip1、top3+last、val_mae/min，没有早停或测试角色。日志调用原TensorBoardLogger，仅将保存路径和version改为当前独立输出目录。engineering模式明确限制为两轮：CPU使用batch2/最多4训练样本，CUDA保持batch260/最多520训练样本；验证仍完整消费该输入队列。

每轮记录实际唯一样本、更新数、完整验证预测摘要和模型状态；只有完整轮次才能保存恢复凭据。新进程恢复前先重推整个冻结验证集，之后核对模型、优化器、scheduler、原生loop、callbacks、全部RNG六类状态并从下一完整轮继续。原生迭代器重建可能改变恢复后RNG轨迹，因此不承诺与不中断训练逐位一致。只排除Lightning原生不恢复的ModelCheckpoint.current_score暂态字段。

`export_emg_initializer_best`只导出原回调实际选中的best全部状态张量，并绑定来源检查点、数据划分、验证分数与逐张量字节/shape/dtype清单；默认拒绝工程或未完成150轮模型。不会用last替代best。[独立CPU整链证据](../results/emg-initializer-runtime-cpu-20261028.json)记录新进程重新核验原工程fit、全部数据与110张量导出，并重推最后一轮完整验证预测精确一致。14项训练/恢复/导出检查通过，包含两个真正独立进程的工程轮次和恢复、六状态及整套冻结验证一致、原失败和历史日志保留，以及被改动的最佳权重/历史/覆盖统计拒绝。

[真实GPU工程资格](../results/emg-initializer-runtime-gpu-20261028.json)已实际完成98.84秒exit0：两个GPU进程分别完整重验真实16.54GB数值队列，保持官方batch260/bf16，两轮各520训练手样本/582完整验证手样本，合计4次更新；新进程六状态及完整冻结验证均精确一致。第三个独立CPU进程重开实际fit与110张量best导出通过，所有8个拥有进程已独立确认退出、83个固定成员不变，A800峰值allocated/reserved为12.74/14.61GB。这是明确520样本的工程训练，不是全63,026训练人口或150轮。完整150轮与验证best最终验收仍未完成；整个初始化实现的一次最终整体审查也仍待完成。[后续完整计划](superpowers/plans/2026-10-28-fusion-compatible-emg-initializer.md)保持这些要求。以上工程检查没有产生新的论文测试精度。

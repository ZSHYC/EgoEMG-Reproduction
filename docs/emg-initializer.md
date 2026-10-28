# 与融合划分兼容的官方 EMG 初始化

旧EMG-S已完成其原协议实验，但训练输入与新ViT-S融合的部分验证、测试输入重叠，[实际分区检查](../results/fusion-initializer-footprint-compatibility-20261028.json)已记录。因此新融合需要另训初始化模型，旧结果与权重保持原样。

## 当前已实现的部分

`emg_initializer_literal`原样合成固定官方`emgformer/regression_egoemg_incre_cotrain_wl12000_small`，只增加私有来源绑定；改动执行配置会被拒绝。原配方为8通道、12,000点完整轨迹、训练stride1,200、batch260、repeat1、150轮、AdamW学习率0.0005、bf16与原批量增强。原指尖损失别名归约后是0.01；按`val_mae/min`保存top3和last，没有早停。

归一化实际调用官方`_load_norm_stats`，从原配置声明的alias统计文件读取左右手各8维float32均值/标准差。不是融合用的16槽标量，也不是本项目重新估计的训练集统计。缺失Incre语料、旧版标签、整episode滤波与统计来源限制继续保留，不能宣称论文原表数值等价。

`build_initializer_partitions`保留官方episode上的1,200点训练网格及完整抖动范围，排除与**新融合全部验证区间及测试输入足迹**相交的训练锚点；所有可能训练输入行还必须是发布的train。验证使用现有融合验证区间内能完整放下的12,000点窗口，不建立初始化阶段的测试角色。[只读真实人口可行性](../results/fusion-compatible-emg-recipe-feasibility-20261028.json)为63,026训练、582验证手样本，仍须由后续实际完整准入重新确认。

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

当前只有配置和完整轨迹读取接入。真实完整输入hash、全可能轨迹数值检查、原生模型/批量增强/优化/状态恢复、真实batch260 GPU资格、完整150轮与验证best验收仍未完成。先前融合的中心目标扫描不能覆盖这里的完整轨迹；本页和CPU证据均不增加已完成科学设置数量。[后续完整计划](superpowers/plans/2026-10-28-fusion-compatible-emg-initializer.md)保持这些要求。

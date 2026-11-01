# Table 11 第三方成绩来源核对

Table 11 标 † 的四行共12个 mean±SD 单元格，与原始基准文献逐值一致。SensingDynamics、NeuroPose 和 vemg2pose 的成绩来自 **Salter 2024 的 EMG2Pose 基准重评**，CLDM 来自 Verma 的直接 EMG→姿态评价。本项目没有重训这四项引用实验；本项目对 EMGFormer S/M/L 的全源250轮复现及最终 Table 11/13 仍待完成。

## 四行引用成绩

单位为度，±表示跨用户标准差；保留原 PDF 显示精度。User+Stage 对应来源中的 User, Stage。

| 方法 | User | Stage | User+Stage | 数值直接来源 |
| --- | ---: | ---: | ---: | --- |
| SensingDynamics† | 15.5±1.4 | 18.8±1.6 | 18.7±1.6 | [Salter Table 4，p.7](https://proceedings.nips.cc/paper_files/paper/2024/file/64f8884f6ba3d9ace5bb647c4f917896-Paper-Datasets_and_Benchmarks_Track.pdf) |
| NeuroPose† | 13.2±1.1 | 17.2±1.7 | 17.5±1.5 | [Salter Table 4，p.7](https://proceedings.nips.cc/paper_files/paper/2024/file/64f8884f6ba3d9ace5bb647c4f917896-Paper-Datasets_and_Benchmarks_Track.pdf) |
| vemg2pose† | 12.2±1.3 | 15.2±1.6 | 15.8±1.4 | [Salter Table 4，p.7](https://proceedings.nips.cc/paper_files/paper/2024/file/64f8884f6ba3d9ace5bb647c4f917896-Paper-Datasets_and_Benchmarks_Track.pdf) |
| CLDM† | 11.3±1.0 | 14.3±1.5 | 14.7±1.4 | [Verma v1 Table I，PDF p.8](https://arxiv.org/pdf/2508.08269v1) |

[固定官方 README](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/README.md)也给出这12格相同显示值。原 PDF 对 SensingDynamics、CLDM 的参数量为破折号，继续保留；没有把 README 的1.0M、7.0M反填进原表。README 另列的模型和原 PDF 不同精度的 S/M/L 值，也没有被添加或覆盖到原表。

## 方法出处与基准成绩出处

Salter §3.5 明确将 NeuroPose 和 SensingDynamics 称为重新实现的基线；其 NeuroPose 为 EMG2Pose 的采样／通道设置调整了降采样。该文 Table 4 是未知初始姿态的回归任务，Table 5 才是提供初始真实姿态的 tracking；两表不能互换。mean±SD 按用户汇总，不是多 seed 误差条。[原始基准 §3.3、§3.5、Table 4、C.7](https://proceedings.nips.cc/paper_files/paper/2024/file/64f8884f6ba3d9ace5bb647c4f917896-Paper-Datasets_and_Benchmarks_Track.pdf)

[NeuroPose 2021 作者版原文](https://www.cse.psu.edu/~mkg31/papers/Neuropose_www.pdf)是该架构的早期 Myo 研究，不是这组后来的 EMG2Pose 成绩来源。SensingDynamics 的原始方法文献仍有年份／刊名差异：EgoEMG 参考文献写2024／Science Robotics 9(93)，Salter 写2022a／bioRxiv。本次确立的是成绩的基准出处，未据此宣称两份文献元数据已闭合。[Salter 参考文献，p.14](https://proceedings.nips.cc/paper_files/paper/2024/file/64f8884f6ba3d9ace5bb647c4f917896-Paper-Datasets_and_Benchmarks_Track.pdf)

CLDM 对应 Verma Table I 的第一数值列 `sEMG-to-Pose`。同表的 tendon-to-pose、两步预测和物理仿真列，以及 Table II 的腱控制信号 MAE，均是另一任务，未用来填 Table 11。[Verma Table I–II](https://arxiv.org/pdf/2508.08269v1)

## VQ-MyoPose 子集脚注

EgoEMG 原脚注保留为“12-subject subset of EMG2Pose (12/27 stages)”。[VQ-MyoPose 原始 workshop 文稿 p.3 Training setup](https://hands-workshop.org/files/2025/VQ-MyoPose.pdf)明确写 **12名参与者、27种 stage**，采用70%／15%／15%划分；这里的12不能解释为 stage 数。它没有作为 Table 11 的一行，本次也未补入同全基准直接排名的成绩。未以该文引用的 vemg2pose 行证明双方在同一子集重新训练。

## 产物与验收范围

- [结构化来源与原表七行](paper-table11-reference-audit-20261102.json)：用户原 PDF、固定 README，以及四篇研究的五份远端文件身份；原值和来源判定分列。
- [12格逐项对照 CSV](paper-table11-reference-audit-20261102.csv)：两端 mean／SD、任务、来源页码及匹配结果。
- [独立读取核对](paper-table11-reference-readback-20261102.json)：实际源文件字节和另一种 PDF 文本排列下的数值复核。

本次核实文献中的显示值与协议说明，没有独立复现引用实验的原始样本、权重、选模和运行。参数量未重新测量，未新增科学模型设置，整张 Table 11 和全论文均未完成。

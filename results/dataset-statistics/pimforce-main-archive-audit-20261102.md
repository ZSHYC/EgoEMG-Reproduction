# PiMForce 主发布包：完整目录及原始 CSV 核验

这是 EgoEMG Table 1 的外部来源核验，不增加完成训练的模型数，也不把 Table 1 或全文标为完成。

2026-11-02（北京时间）从[作者项目页](https://pimforce.hcitech.org/)进入公开 Dropbox，实际读取[主数据 ZIP](https://www.dropbox.com/scl/fo/m7qye1l3ii6hlwgzlajyv/AEbamHZ9KDG81CeqnX0qYmw/Dataset_NeurIPS2024.zip?rlkey=eyt5gs66uimhiw3dismnylamw&dl=0)的完整 ZIP64 目录，再完整解压一个会话的三份 CSV。[2026-10-29 浏览器目录观察](pimforce-public-directory-observation-20261029.json)保留为历史证据。

## 完整主 ZIP 文件清单

[全部 4,242 条目录记录](pimforce-main-archive-members-20261102.csv)保存路径、压缩/解压字节数、CRC32、压缩方法、标志与实际文件头偏移；[观察报告](pimforce-main-archive-audit-20261102.json)保存来源、HTTP 范围和哈希。

| 实际目录项目 | 数量或范围 |
| --- | --- |
| 用户目录键 | User1–User21，共 21；不是独立核实的自然人身份 |
| 每个用户的会话目录 | Session1、Session2、Session3 |
| 会话目录合计 | 63 |
| 每会话文件 | `fsr_0..21.csv`、`join_0..21.csv`、`pps_0..21.csv`，各 22 |
| 非目录成员 | 4,158 个 CSV；三类各 1,386 个，无其他非目录成员 |
| 目录成员 | 84（21 个用户目录＋63 个会话目录） |
| 归档总字节 | 8,200,737,484 |
| 目录所列解压字节合计 | 39,997,239,158；不是已下载/解压的全量数据 |

全部路径唯一，63 个会话的三类动作编号集合均精确为 0–21。主 ZIP 没有图像或视频命名的成员；这一结论限定于该归档的完整文件清单，不扩张为整个公开文件夹不存在视觉数据。

实际 HTTP 返回 `206`、`Content-Range: bytes 8199688908-8200737483/8200737484`、`Content-Length: 1048576`，ETag 为 `1728974474612702d`。末尾 1 MiB 的 SHA256 为 `381378619d1dcfef903d012d275a6d2b8f89b1503505db3645f9f3488b55457a`。完整中央目录起点 8,200,376,813、长度 360,573 字节。ZIP64 读取使用完整文件的逻辑偏移，未将截取的末尾误作独立完整 ZIP。

## 三份真实 CSV 的内容检查

完整读取 `Dataset/User1/Session1/` 中的三个压缩成员，逐一对照实际本地文件头、原归档大小、文件名、ETag 与 CRC32，再检查所有 CSV 行的列数和数值。原信号文件只在本地保留，没有上传到公开仓库。

| 成员 | 数据行 | 列数 | 核实的结构 |
| --- | --- | --- | --- |
| `fsr_0.csv` | 2,975 | 6 | Timestamp＋5 个 FSR 通道 |
| `join_0.csv` | 57,720 | 30 | 索引＋timestamp＋8 个 sEMG 通道＋20 个手指角 |
| `pps_0.csv` | 3,557 | 66 | Time＋65 个 Elem 通道 |

三份共 64,252 行，所有单元均能解析为有限数值。20 个角列明确列在报告中，按五指各四列命名，没有显式以腕关节命名的列；这不替代角单位/校准的独立验证，也不证明其余 62 个会话的内容模式一致。

[独立复核报告](pimforce-main-archive-independent-readback-20261102.json)由另一进程读取真实末尾字节，用独立 `struct` 解析器重算全部 4,242 条中央目录，包括 ZIP64 的 64 位文件头偏移，逐字段匹配公开 CSV；没有调用首次使用的 `zipfile` 目录读取器。三份 CSV 另由 NumPy 数值读取器完整重读，确认行数、列数和有限性。全部输入的 SHA256 与九项文件属性在检查后保持一致，没有导入 Torch 或进行训练。

## 不合并的三种计数

- EgoEMG PDF Table 1 的 PiMForce `Gestures=63` 原样保留在 [as-reported CSV](table1-as-reported.csv)。
- PiMForce [原论文 §3.2](https://pimforce.hcitech.org/static/pdfs/Posture-Informed%20Muscular%20Force%20Learning%20for%20Robust%20Hand%20Pressure%20Estimation_compressed.pdf)写 22 类交互、每人 3 个会话；实际发布包每种文件的编号集合为 0–21。
- 作者公开[代码 README](https://www.dropbox.com/scl/fo/m7qye1l3ii6hlwgzlajyv/AO_3toLxUlsCwN2FvvBn6mA/Codes/README.md?rlkey=eyt5gs66uimhiw3dismnylamw&dl=0)写 21 个动作，与原论文和文件编号存在差异。报告另存 README 的实际字节数和 SHA256。

21×3 确实等于此次盘点的 63 个会话目录，但没有作者证据证明 EgoEMG 的原表把会话写成了手势，因此不把这种推测当作纠错结论。

## 边界与剩余工作

没有下载或校验主 ZIP 的全部 8.20 GB 字节，未验证全部 4,158 份 CSV 的内容或其总 SHA256。单独的 `Benchmark_PressureVision++_Dataset.egg` 未在本次读取；此前公开根目录显示的 58.96 GB 与主 ZIP 分列。代码目录的 MIT LICENSE 已读取并记录哈希，未据此认定所有数据文件均有相同许可。

Table 1 的 NinaPro 家族统计口径、EgoEMG 全模态实物与作者原表定义等仍欠核实。本项没有执行额外 PiMForce 训练、没有运行 EgoEMG 的 PiMForce 预处理，也不能替代 markers2mano 的 11 个视觉训练数据源或其缺失的训练包。

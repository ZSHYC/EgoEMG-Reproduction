# 双手全部 41 段：官方 FFT 重生成结果与旧发布 filtered 全量对账

2026-10-29 完成了之前尚欠的全量数值比较。使用已经由固定官方 `f7cbd048` 原入口生成的两份完整输出，逐项比较服务器保留的旧版 `emg_left_filtered` / `emg_right_filtered`。左右各 66,161,725 行 × 8 通道，共 **1,058,587,600 个 float32 数值**，覆盖全部 41 段；没有重跑 FFT 或训练。

| 全量指标 | 左手 | 右手 |
|---|---:|---:|
| 比较数值 | 529,293,800 | 529,293,800 |
| 数值不相等 | 2,595 | 524 |
| float32 位模式不相等 | 2,595 | 524 |
| 平均绝对差 | 2.124681376352926e-14 | 1.0121734295339103e-14 |
| 均方根差 | 4.376197711675012e-11 | 5.918089665877624e-11 |
| 最大绝对差 | 4.76837158203125e-7 | 9.5367431640625e-7 |
| 至少一项不同的录制段 | 40 / 41 | 38 / 41 |
| 比较 CPU 进程用时 | 116.207 秒 | 106.328 秒 |
| 进程峰 RSS | 55,377,920 bytes | 56,152,064 bytes |

合计 **99.99970536212591%** 的 float32 位模式一致。两份数组的完整 SHA 不同，因此不宣称逐字节等价。差值沿用数据原存储单位，没有额外缩放、归一化、舍入或阈值筛选，也未独立标定为 mV。差异原因未由本次对账单独确定。最大差出现在左手 episode 索引32/35、右手索引34，故首段最大差不能代表全量最大差。

## 输入与执行身份

| 手侧 | 原官方重生成 SHA256 | 旧发布 filtered SHA256 |
|---|---|---|
| 左 | `d4089674565753aa5c6cb6a6fc69cfd73f5dd7dddf5cf122ecab60788f8cc1ac` | `b948e1879bc6044a54758c865a137acb5b580ded18eca2f4a7d84d3f976ec3ea` |
| 右 | `90af062a9d2e1a8afd767b4b52cbdd2b17c456ba3f9c91226fcadccfd2a6ba18` | `2aeac3c7262fd42d7770051fe9df73660bafaf4627bc70626802fe13067e88d3` |

每个数组均 2,117,175,200 bytes。新输出绑定[完整官方重生成证据](emg-fft-bilateral-full.json)，旧数组绑定先前[完整融合输入准入](../native-fusion-cohort-fixed-full-identity-20261028.json)中的 `field:emg_*_filtered`。metadata SHA 为 `cb42886b5ca80a070d3b79f5e920cd01e364572e038ef11d5643e9ab73528a57`，41 段连续、无空缺、无重叠，覆盖起点0至66,161,725。

[公开比较程序](../../scripts/paper/compare_filtered_emg.py)以只读方式逐段、每块至多65,536行扫描，所有值均检查有限性。计算 float64 的绝对差和平方差，分别累计整手、每段、每通道；数值比较与 `uint32` 位模式比较分别计数，不把正负零默认为位相同。新数据从本机原数组直接流送到服务器 CPU 读者，没有在服务器或本机再保存完整数组副本。既有原 FFT 数组、released filtered 和训练输入都未改写。实际环境为 Python3.11.12 / NumPy1.26.4，无 Torch 导入或 GPU 分配。

复现单手比较的示例（两数组在同一主机时）：

```bash
python scripts/paper/compare_filtered_emg.py \
  --baseline /path/to/legacy/emg_left_filtered.dat \
  --generated /path/to/native-fft/emg_left_filtered_paper.dat \
  --metadata /path/to/legacy/metadata.npz --hand left \
  --baseline-sha256 b948e1879bc6044a54758c865a137acb5b580ded18eca2f4a7d84d3f976ec3ea \
  --generated-sha256 d4089674565753aa5c6cb6a6fc69cfd73f5dd7dddf5cf122ecab60788f8cc1ac \
  --metadata-sha256 cb42886b5ca80a070d3b79f5e920cd01e364572e038ef11d5643e9ab73528a57 \
  --output /path/to/new-output/left-comparison.json
```

输入经管道流送时，将 `--generated` 设为 `-`。程序拒绝截断、额外字节、非有限值、摘要不符、边界不完整、输入身份变化和覆盖已有结果；另外输出所属进程记录，便于退出后核验。

## 独立核验与边界

[手算数值检查](emg-fft-full-comparison-numeric-check.json)在全量运行前验证16值例子的绝对差和7、平方差和15、最大差3、4个数值不等/5个位模式不等，另检查正负零、NaN拒绝与流截断拒绝。

[全量退出后独立核验](emg-fft-full-comparison.json)确认左右比较实际 exit0，独立重开两个结果及源/所属进程记录；本机驱动和两个 SSH、服务器两个读者共五个进程均不再存活。完整新/旧 SHA 与先前证据一致，输入在读取前后及核验时的九项文件属性未变。41段及8通道的整数计数、最大差逐项精确对账；浮点正数和按 `n_terms × float64 epsilon` 相对界核对，两个首段指标也与先前独立比较一致。

逐段/通道底表：[左手 JSON](emg-fft-full-comparison-left.json)、[右手 JSON](emg-fft-full-comparison-right.json)。这项证据补齐了 **旧发布全部 filtered 数值对照**；[原官方 FFT 方法](emg-fft-bilateral-full.md)保留整 episode 的非局部依赖。独立硬件单位标定、按 split 独立滤波、当前发布/PDF 数据等价，以及其余标签/几何/训练项目，仍各自验收。本次不增加科学模型完成设置数。

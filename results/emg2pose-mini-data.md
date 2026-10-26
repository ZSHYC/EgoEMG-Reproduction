# EMG2Pose mini：数据工程验收

这是完整官方 mini 的转换与读取检查，**未训练模型，也不是 Table 11/13 性能结果**。全部30个原始文件仍为 `val`，未伪造 train 或独立验证划分。全文复现继续执行。

| 实测项目 | 结果 |
| --- | ---: |
| HDF5 文件 | 30 |
| 原始记录 | 4,218,756 |
| 四个输出字段字节数 | 645,469,668 |
| 原始有效行 | 3,559,584 |
| 连续有效 blocks | 1,015 |
| 按完整 session 生成的7790窗口 | 526 |
| 仅有效 blocks 生成的7790窗口 | 230 |

全部 EMG、20维弧度角度、float64时间戳与原始有效mask均已逐行位级复核。未修改的官方 reader 对780/3900/7790三种stride各检查120个窗口，覆盖全部30个session，共360窗；归一化、标签、mask与索引逐位一致。窗口总数与实际抽查数分别保存在[机器可读证据](emg2pose-mini-data.json)。

按官方代码，落盘 valid mask 仅结合 IK 成功和有限角度；原始 EMG/time 的 NaN、Inf 和位模式保留。官方 reader 在取样时会把整窗全False的mask改成全True，原行为保留并披露。原始 session 字符串可能跨多个文件/分组复用；后续正式评价须单独审计选择器，不能据本次重建推断作者未见 sidecar 或论文结果。

[转换实现](../egoemg_repro/emg2pose_data.py)补齐当前官方 loader 必需的 metadata.npz，保留 generalization、原 session/filename、split、side 与有效blocks；没有落第二份全量Zarr。旧工程转换来源与修复后来源各自保留，新验收的实际源码和私有完成receipt摘要见JSON。CSV、原始数据和参与者映射不随本报告再分发。

原始 mini 来自 [Meta 官方公开 S3](https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar)。完整tar接收长度647,198,208 bytes及本次传输SHA在JSON中；SHA标识本次收到的字节，不冒称作者提供的checksum。完整原始数据、S/M/L完整训练与最终测试仍未完成。

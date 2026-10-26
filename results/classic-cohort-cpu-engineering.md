# Classic 新分区：实际完整队列 CPU 数值检查

2026-10-26 已在真实 legacy 数据上分别完成 vemg2pose、NeuroPose 的完整队列准入，再各进行一次实际整文件重哈希与足迹数值重扫，所得 receipt 完全一致。[聚合JSON](classic-cohort-cpu-engineering.json)公开来源摘要和计数；没有构造模型、导入Torch、使用GPU或计算误差，**不构成正式训练或科研成绩**。

分区采用7,790点输入、train stride780、eval stride7,790、val fraction0.12，canonical SHA256为 `6c91278448705b2d32d38de63e98a931731ca527675f0d84c6097d3f73b26764`。

| 角色 | 窗口 | 单手样本 | 足迹唯一同步行 | 合并区间 |
| --- | ---: | ---: | ---: | ---: |
| train | 50,857 | 101,714 | 42,135,980 | 352 |
| val | 788 | 1,576 | 6,138,520 | 352 |
| test | 2,153 | 4,306 | 16,771,870 | 346 |

12数组的整文件SHA及manifest/metadata绑定66,161,725个源同步行；数值扫描覆盖冻结窗口足迹内65,046,370个唯一行。另有1,115,355个源行不在足迹内，不能据此声称它们全部finite。每个角色同时读取双手的原始浮点字段，验证固定布局/归一化后的EMG，以及20个手指角和腕部degree→radian后的22维角度，所扫描值均finite。所有窗口保留，不删除个别空标签样本；每个角色的实际generated监督非零。

VEMG数据/配方身份为 `1306202563fdbe62342506005055ed82cb5ccd3fcd135663ab727541d6a0625b`，NeuroPose为 `631e0939f19cb6e27804b019325ab787963f534071b2c9c66976e73c4bc77983`。二者raw数据及角度转换一致，输入布局不同；身份包含各自模型协议。固定官方scalar为mean0.000832、std8.6551、epsilon1e-6，来自已固定的公开norm资产，**没有从本项目独立留出的train行重新估计**；资产原采样与本分区的等价性未证明。

实际被测新准入源码SHA256为 `72267d660f55e9f5fbbd2e1e17edf07627ef01d2ba09e92fca1086c50a16dbd3`。独立复核重合并真实partition、重算receipt和data identity摘要，核对12字段与旧取得清单、原norm资产及全部角色/双手/元素/监督计数，均一致。私有执行脚本在导入前禁止Torch运行时。原始协议、路径、样本和模型权重不公开。

64MiB是每字段处理块上限，不是进程RSS上限。数值检查仅涵盖固定布局、scalar归一化和单位转换，随机增强仍需运行时逐batch检查。generated监督计数是唯一手行数，不是模型上下文后的有效角分母或MAE。

此队列的4,306个测试手样本既不是旧12,000点实验的2,792，也不是PDF的4,278。`cuda_qualified`、`gpu_capacity_qualified`、`production_ready`均为false；实际FP16/scaler、原batch600/256的train和validation容量、最终正式入口验收、完整训练/val-best/test及导出仍需完成。

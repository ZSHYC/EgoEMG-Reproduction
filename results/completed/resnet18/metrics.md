# ResNet-18：完整运行结果

已核对完整测试、验证选择检查点和协议身份；这不是论文表格等价性的声明。

| 采样 | 人口／分组 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | --- | ---: | ---: | ---: |
| center | all / pooled | 7.784887 | 59572 | 8094.161306 |
| center | hand/left | 7.964298 | 29762 | 4137.013704 |
| center | hand/right | 7.605765 | 29810 | 3957.147602 |
| center | split/both | 8.036029 | 4422 | 620.2083621 |
| center | split/gesture | 7.458456 | 31918 | 4154.913135 |
| center | split/user | 8.185562 | 23232 | 3319.039808 |
| center | component/fingers | 7.867119 | 54160 | 7436.553965 |
| center | component/wrist | 6.961960 | 5412 | 657.6073402 |
| center | left/both | 8.131132 | 2200 | 312.213053 |
| center | left/gesture | 7.642658 | 15968 | 2129.964246 |
| center | left/user | 8.375623 | 11594 | 1694.836405 |
| center | right/both | 7.941868 | 2222 | 307.9953091 |
| center | right/gesture | 7.274045 | 15950 | 2024.948889 |
| center | right/user | 7.996219 | 11638 | 1624.203404 |
| center | vision_valid / pooled | 7.690786 | 59132 | 7937.260582 |

缺失 crop：104 / 2792；全标签结果保留这些样本。

用户均值、总体标准差及匿名逐用户统计见 metrics.json。该标准差来自跨手、跨 split 合并的逐用户均值，并非每个条件的 σ；与论文条件内标准差的比较须等待独立审计。不同采样及 crop 人口不得混算。

图表未生成（显式 --no-plots）。

源代码、数据、配置、划分、归一化及所选检查点的 SHA256 身份见 metrics.json。

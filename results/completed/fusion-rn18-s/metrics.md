# ResNet-18 + EMGFormer-S：完整运行结果

已核对完整测试、验证选择检查点和协议身份；这不是论文表格等价性的声明。

| 采样 | 人口／分组 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | --- | ---: | ---: | ---: |
| center | all / pooled | 6.793533 | 59572 | 7063.423613 |
| center | hand/left | 6.994259 | 29762 | 3633.13206 |
| center | hand/right | 6.593131 | 29810 | 3430.291554 |
| center | split/both | 7.277857 | 4422 | 561.6937666 |
| center | split/gesture | 6.462222 | 31918 | 3599.937315 |
| center | split/user | 7.156528 | 23232 | 2901.792532 |
| center | component/fingers | 6.830018 | 54160 | 6456.212995 |
| center | component/wrist | 6.428419 | 5412 | 607.2106186 |
| center | left/both | 7.453927 | 2200 | 286.2102522 |
| center | left/gesture | 6.636947 | 15968 | 1849.678335 |
| center | left/user | 7.399149 | 11594 | 1497.243472 |
| center | right/both | 7.103530 | 2222 | 275.4835144 |
| center | right/gesture | 6.287301 | 15950 | 1750.25898 |
| center | right/user | 6.914825 | 11638 | 1404.549059 |
| center | vision_valid / pooled | 6.692349 | 59132 | 6906.825844 |

缺失 crop：104 / 2792；全标签结果保留这些样本。

用户均值、总体标准差及匿名逐用户统计见 metrics.json。该标准差来自跨手、跨 split 合并的逐用户均值，并非每个条件的 σ；与论文条件内标准差的比较须等待独立审计。不同采样及 crop 人口不得混算。

图表未生成（显式 --no-plots）。

源代码、数据、配置、划分、归一化及所选检查点的 SHA256 身份见 metrics.json。

# EMGFormer-M：完整运行结果

已核对完整测试、验证选择检查点和协议身份；这不是论文表格等价性的声明。

| 采样 | 人口／分组 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | --- | ---: | ---: | ---: |
| center | all / pooled | 16.490835 | 59572 | 17145.97553 |
| center | hand/left | 16.372327 | 29762 | 8504.521674 |
| center | hand/right | 16.609152 | 29810 | 8641.453857 |
| center | split/both | 17.580856 | 4422 | 1356.863376 |
| center | split/gesture | 16.356454 | 31918 | 9111.758444 |
| center | split/user | 16.467983 | 23232 | 6677.353711 |
| center | component/fingers | 16.694303 | 54160 | 15780.62919 |
| center | component/wrist | 14.454653 | 5412 | 1365.34634 |
| center | left/both | 17.326960 | 2200 | 665.3075156 |
| center | left/gesture | 16.415216 | 15968 | 4574.825125 |
| center | left/user | 16.132113 | 11594 | 3264.389033 |
| center | right/both | 17.832238 | 2222 | 691.5558601 |
| center | right/gesture | 16.297626 | 15950 | 4536.933318 |
| center | right/user | 16.802584 | 11638 | 3412.964679 |
| trajectory | all / pooled | 16.486140 | 684715272 | 197018208 |
| trajectory | hand/left | 16.374407 | 342609316 | 97913397.68 |
| trajectory | hand/right | 16.598037 | 342105956 | 99104810.3 |
| trajectory | split/both | 17.248946 | 51326308 | 15451831.24 |
| trajectory | split/gesture | 16.397417 | 366067240 | 104764386.6 |
| trajectory | split/user | 16.461176 | 267321724 | 76801990.18 |
| trajectory | component/fingers | 16.647407 | 622517680 | 180873800 |
| trajectory | component/wrist | 14.872062 | 62197592 | 16144407.97 |
| trajectory | left/both | 17.020081 | 25585582 | 7600362.36 |
| trajectory | left/gesture | 16.508087 | 183336994 | 52823140.58 |
| trajectory | left/user | 16.067508 | 133686740 | 37489894.74 |
| trajectory | right/both | 17.476431 | 25740726 | 7851468.879 |
| trajectory | right/gesture | 16.286380 | 182730246 | 51941245.98 |
| trajectory | right/user | 16.854996 | 133634984 | 39312095.44 |

用户均值、总体标准差及匿名逐用户统计见 metrics.json。该标准差来自跨手、跨 split 合并的逐用户均值，并非每个条件的 σ；与论文条件内标准差的比较须等待独立审计。不同采样及 crop 人口不得混算。

图表未生成（显式 --no-plots）。

源代码、数据、配置、划分、归一化及所选检查点的 SHA256 身份见 metrics.json。

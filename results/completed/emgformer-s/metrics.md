# EMGFormer-S：完整运行结果

已核对完整测试、验证选择检查点和协议身份；这不是论文表格等价性的声明。

| 采样 | 人口／分组 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | --- | ---: | ---: | ---: |
| center | all / pooled | 16.313757 | 59572 | 16961.86249 |
| center | hand/left | 16.276526 | 29762 | 8454.758098 |
| center | hand/right | 16.350928 | 29810 | 8507.104397 |
| center | split/both | 17.179378 | 4422 | 1325.877893 |
| center | split/gesture | 16.221521 | 31918 | 9036.590761 |
| center | split/user | 16.275715 | 23232 | 6599.39384 |
| center | component/fingers | 16.511034 | 54160 | 15607.39017 |
| center | component/wrist | 14.339532 | 5412 | 1354.472328 |
| center | left/both | 16.896425 | 2200 | 648.7761583 |
| center | left/gesture | 16.320975 | 15968 | 4548.560665 |
| center | left/user | 16.097679 | 11594 | 3257.421275 |
| center | right/both | 17.459528 | 2222 | 677.1017344 |
| center | right/gesture | 16.121955 | 15950 | 4488.030096 |
| center | right/user | 16.453078 | 11638 | 3341.972566 |
| trajectory | all / pooled | 16.289435 | 684715272 | 194667480.6 |
| trajectory | hand/left | 16.256548 | 342609316 | 97208642.32 |
| trajectory | hand/right | 16.322370 | 342105956 | 97458838.32 |
| trajectory | split/both | 16.979583 | 51326308 | 15210532.42 |
| trajectory | split/gesture | 16.259008 | 366067240 | 103880082 |
| trajectory | split/user | 16.198592 | 267321724 | 75576866.23 |
| trajectory | component/fingers | 16.438285 | 622517680 | 178601695.2 |
| trajectory | component/wrist | 14.799636 | 62197592 | 16065785.48 |
| trajectory | left/both | 16.762409 | 25585582 | 7485297.98 |
| trajectory | left/gesture | 16.395697 | 183336994 | 52463511.43 |
| trajectory | left/user | 15.968907 | 133686740 | 37259832.91 |
| trajectory | right/both | 17.195449 | 25740726 | 7725234.441 |
| trajectory | right/gesture | 16.121866 | 182730246 | 51416570.57 |
| trajectory | right/user | 16.428365 | 133634984 | 38317033.32 |

用户均值、总体标准差及匿名逐用户统计见 metrics.json。该标准差来自跨手、跨 split 合并的逐用户均值，并非每个条件的 σ；与论文条件内标准差的比较须等待独立审计。不同采样及 crop 人口不得混算。

图表未生成（显式 --no-plots）。

源代码、数据、配置、划分、归一化及所选检查点的 SHA256 身份见 metrics.json。

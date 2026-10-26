# ResNet-50：完整运行结果

已核对完整测试、验证选择检查点和协议身份；这不是论文表格等价性的声明。

| 采样 | 人口／分组 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | --- | ---: | ---: | ---: |
| center | all / pooled | 6.640058 | 59572 | 6903.851127 |
| center | hand/left | 6.848570 | 29762 | 3557.454836 |
| center | hand/right | 6.431881 | 29810 | 3346.396291 |
| center | split/both | 7.138400 | 4422 | 550.9307137 |
| center | split/gesture | 6.308939 | 31918 | 3514.547352 |
| center | split/user | 7.000120 | 23232 | 2838.37306 |
| center | component/fingers | 6.693153 | 54160 | 6326.838554 |
| center | component/wrist | 6.108719 | 5412 | 577.0125723 |
| center | left/both | 7.251464 | 2200 | 278.4362352 |
| center | left/gesture | 6.469925 | 15968 | 1803.130498 |
| center | left/user | 7.293614 | 11594 | 1475.888103 |
| center | right/both | 7.026455 | 2222 | 272.4944786 |
| center | right/gesture | 6.147772 | 15950 | 1711.416854 |
| center | right/user | 6.707737 | 11638 | 1362.484958 |
| center | vision_valid / pooled | 6.535572 | 59132 | 6745.024848 |

缺失 crop：104 / 2792；全标签结果保留这些样本。

用户均值、总体标准差及匿名逐用户统计见 metrics.json。该标准差来自跨手、跨 split 合并的逐用户均值，并非每个条件的 σ；与论文条件内标准差的比较须等待独立审计。不同采样及 crop 人口不得混算。

图表未生成（显式 --no-plots）。

源代码、数据、配置、划分、归一化及所选检查点的 SHA256 身份见 metrics.json。

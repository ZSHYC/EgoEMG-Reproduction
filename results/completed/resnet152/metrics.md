# ResNet-152：完整运行结果

已核对完整测试、验证选择检查点和协议身份；这不是论文表格等价性的声明。

| 采样 | 人口／分组 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | --- | ---: | ---: | ---: |
| center | all / pooled | 5.982738 | 59572 | 6220.417728 |
| center | hand/left | 6.189459 | 29762 | 3215.083025 |
| center | hand/right | 5.776350 | 29810 | 3005.334703 |
| center | split/both | 6.649401 | 4422 | 513.1905637 |
| center | split/gesture | 5.602483 | 31918 | 3120.998758 |
| center | split/user | 6.378270 | 23232 | 2586.228407 |
| center | component/fingers | 6.034570 | 54160 | 5704.300103 |
| center | component/wrist | 5.464036 | 5412 | 516.1176258 |
| center | left/both | 6.743313 | 2200 | 258.9246264 |
| center | left/gesture | 5.746373 | 15968 | 1601.480773 |
| center | left/user | 6.694610 | 11594 | 1354.677627 |
| center | right/both | 6.556420 | 2222 | 254.2659373 |
| center | right/gesture | 5.458431 | 15950 | 1519.517985 |
| center | right/user | 6.063126 | 11638 | 1231.55078 |
| center | vision_valid / pooled | 5.869706 | 59132 | 6057.81863 |

缺失 crop：104 / 2792；全标签结果保留这些样本。

用户均值、总体标准差及匿名逐用户统计见 metrics.json。该标准差来自跨手、跨 split 合并的逐用户均值，并非每个条件的 σ；与论文条件内标准差的比较须等待独立审计。不同采样及 crop 人口不得混算。

图表未生成（显式 --no-plots）。

源代码、数据、配置、划分、归一化及所选检查点的 SHA256 身份见 metrics.json。

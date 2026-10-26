# EMGFormer-L：完整运行结果

已核对完整测试、验证选择检查点和协议身份；这不是论文表格等价性的声明。

| 采样 | 人口／分组 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | --- | ---: | ---: | ---: |
| center | all / pooled | 16.612111 | 59572 | 17272.06979 |
| center | hand/left | 16.740644 | 29762 | 8695.841971 |
| center | hand/right | 16.483786 | 29810 | 8576.227818 |
| center | split/both | 17.813938 | 4422 | 1374.852319 |
| center | split/gesture | 16.436766 | 31918 | 9156.498246 |
| center | split/user | 16.624258 | 23232 | 6740.719224 |
| center | component/fingers | 16.801636 | 54160 | 15882.0876 |
| center | component/wrist | 14.715468 | 5412 | 1389.98219 |
| center | left/both | 18.019034 | 2200 | 691.8812291 |
| center | left/gesture | 16.687007 | 15968 | 4650.571739 |
| center | left/user | 16.571937 | 11594 | 3353.389002 |
| center | right/both | 17.610874 | 2222 | 682.97109 |
| center | right/gesture | 16.186243 | 15950 | 4505.926507 |
| center | right/user | 16.676381 | 11638 | 3387.330222 |
| trajectory | all / pooled | 16.591133 | 684715272 | 198272926.3 |
| trajectory | hand/left | 16.753646 | 342609316 | 100181114.5 |
| trajectory | hand/right | 16.428380 | 342105956 | 98091811.72 |
| trajectory | split/both | 17.412808 | 51326308 | 15598620.9 |
| trajectory | split/gesture | 16.532222 | 366067240 | 105625664 |
| trajectory | split/user | 16.514041 | 267321724 | 77048641.35 |
| trajectory | component/fingers | 16.739019 | 622517680 | 181869163.2 |
| trajectory | component/wrist | 15.110977 | 62197592 | 16403763.02 |
| trajectory | left/both | 17.543944 | 25585582 | 7834294.502 |
| trajectory | left/gesture | 16.844037 | 183336994 | 53898124.39 |
| trajectory | left/user | 16.478433 | 133686740 | 38448695.65 |
| trajectory | right/both | 17.282463 | 25740726 | 7764326.399 |
| trajectory | right/gesture | 16.219371 | 182730246 | 51727539.62 |
| trajectory | right/user | 16.549663 | 133634984 | 38599945.7 |

用户均值、总体标准差及匿名逐用户统计见 metrics.json。该标准差来自跨手、跨 split 合并的逐用户均值，并非每个条件的 σ；与论文条件内标准差的比较须等待独立审计。不同采样及 crop 人口不得混算。

图表未生成（显式 --no-plots）。

源代码、数据、配置、划分、归一化及所选检查点的 SHA256 身份见 metrics.json。

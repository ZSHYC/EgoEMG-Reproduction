# 完整密网格冻结预测与独立重放

本入口补齐固定官方版本的ViT-S密采样遮挡分析。索引、两遍配对输入和两遍原生GT已完成；新增完整预测流程须在集成审查、相应修正测试和真实batch64资格通过后执行正式预测。**代码接通不等于17,168点预测、Figure13或全文已经完成。** 结果状态以服务器实际终态和公开报告为准。

源代码为`f7cbd048b7092139852270491d3859cd472e0311`。可执行的原生中心前向替代同版本评价脚本中过期的配置和attention补丁；完整融合验证best、视觉LAST及其实际cotrain EMG初始化来源均重新核验，不以新Table2模型替换兼容初始化模型。独立EMG使用原8通道逐手归一化、相同7790窗口，融合使用原16槽unified归一化；独立EMG只传EMG，以原完整轨迹的`T//2`取中心。所有兼容处理保留在来源和限制说明中。

## 执行设置与产物

- 所有17,168唯一手请求，source stride1947，6个split/hand组；272批，尾批45/45/19/19/8/8。
- 官方支持的`--device cpu`路线，FP32、无autocast、模型eval、父预测进程2个Torch线程；batch64、4workers、pin_memory=True、shuffle=False、不丢尾。CPU上请求pin_memory与实际是否启用分别记录。
- 缺图695个、标签无效518个、零姿态GT518个全部保留；原始CSV误差按每样本22角未mask均值，不替代标准Table3的masked指标。
- 4个实际数据进程在正常完成及读取/预测异常时均退出；拥有进程的PID与启动时间一并记录。没有优化器构造或参数更新，不占用已分配GPU。
- 每批私有NPZ保留三分支中心预测、完整EMG轨迹、目标、原`B×1`标签掩码、图像有效性和三种弧度误差；日志记录精确键、逐样本输入指纹、实际批量输入指纹、typed数组及文件SHA。

资格先执行每组首批和尾批：528个真实请求、12批，再由新进程按相同设置重放。正式两遍各执行全部17,168个请求，重新载入实际训练来源、重新读取原始输入并前向；第二遍不能用保存的预测代替模型。每遍输入前后核对全部54文件/61,379,659,366字节，三个模型完整状态前后相同。最终连接实际密GT，重建原始CSV、bin/SEM/样条/Spearman并运行整个未修改的官方绘图main，另一个进程从原始包独立重建CSV、统计和PNG。

重放须由不同的实际进程产生，使用不同的规范化输出目录，并绑定第一遍完整结果文件和身份文件的精确摘要；同一遍自比较、复制进程身份或引用另一份前序结果均被拒绝。每遍记录实际CPU型号、主机、启动身份与Torch版本，报告另记Matplotlib版本。

## 服务器入口

须有已完成的实际来源目录及相应私有产物，不能仅下载仓库中的JSON充当输入。以下路径是当前复现服务器上的实际位置；新机器须先建立等价来源并保留其真实性，入口会核对固定的来源内容身份。运行环境除模型依赖外，还需原报告环境的numpy1.26.4、scipy1.13.1、pandas2.2.3、Matplotlib3.9.4；后者与官方environment的3.9.2差异明确记录。当前依赖位于独立目录，未修改在跑训练的共享环境。

```bash
python -B scripts/paper/run_dense_occlusion.py workflow \
  --evidence results/native-dense-occlusion \
  --upstream /home/shihao/projects/EgoEMG-Reproduction/upstream \
  --index-root /data/shihao/egoemg-native-dense-occlusion-complete-indices-20261101 \
  --paired-root /data/shihao/egoemg-native-dense-paired-inputs-v2-20261101 \
  --geometry-root /data/shihao/egoemg-native-dense-geometry-vits_s-full-20261101 \
  --data /home/shihao/projects/EgoEMG-Reproduction/data/legacy \
  --crops /home/shihao/projects/EgoEMG-Reproduction/data/crops \
  --fit /data/shihao/egoemg-native-fusion-full-59b5ed5f525f-20261028/private/full-fit \
  --output /absolute/new/dense-workflow
```

在服务器持久会话或专用后台recorder下启动，父控制器必须持续存活。当前会话的实际部署使用脱离SSH的recorder；不会依赖本机持续开机。入口拒绝复用已有输出目录，不自动重试或清理失败证据。7个拥有阶段依次执行资格A/B、资格验收、完整A/B、报告、独立报告重建；只有所有阶段及其数据进程退出后才写工作流终态。

`full-a/`和`full-b/`保存全部私有预测；`report/occlusion_fusion_results.csv`是原官方列，`raw_paired_errors.csv`补充原始弧度和窗口键，`bins.csv`与`statistics.json`保留完整底表，`occlusion_fusion_gain.png`由原官方脚本生成；`report-rebuilt/verification.json`记录独立重建。报告首先保持待验收状态；只有拥有阶段的独立重建进程及其子进程全部成功退出，控制器重新核验两份报告和五类数值/图片产物后，才写`report/completion.json`并原子更新报告完成标记。工作流控制器不导入Torch或构造模型，绘图/验收阶段禁止构造学习模块。

旧版数据、1920×1080官方视频回退、原X镜像/未用畸变以及训练输入窗口交集均保留并披露。即便此完整旧版流水线结束，也不直接证明PDF数值等价或全文完成。RN18路线仍需其正式兼容初始化、原生联合训练和真实训练来源覆盖；当前入口拒绝用legacy分支或未来数据分区冒充这些来源。

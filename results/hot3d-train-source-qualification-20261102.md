# HOT3D-Clips：公开训练清单与双设备完整片段核验

已固定发布版本，取得全部公开训练归档的文件清单，并完整接收 Aria、Quest3 各一个片段；服务器独立解析和数值复核通过。这完成了公开来源的格式与传输核验，**尚未取得全部训练片段、完整 HOT3D 或作者的实际混合训练集**。

## 清单范围与实际接收

对固定版本的两个训练目录逐页读取至无下一页，将所有路径与 `clip_splits.json`、完整 `clip_definitions.json` 交叉核对。公开训练清单共 **2,804 个归档／294,201,743,360 bytes**：Aria 1,516 个，Quest3 1,288 个。元数据对应420,600帧、257个源 sequence 和13个源 participant ID。这里的帧是片段元数据记录，participant ID计数也不是独立核验过的自然人数。[固定训练清单](../assets_hot3d_train_manifest.json)

实际接收使用每个设备目录按文件名排序的首个训练片段，没有根据姿态或结果挑选：

| 训练归档 | 完整归档字节数 | 帧数 | MANO记录 | UmeTrack记录 |
| --- | ---: | ---: | ---: | ---: |
| `train_aria/clip-001849.tar` | 104,069,120 | 150 | 300 | 300 |
| `train_quest3/clip-000000.tar` | 102,440,960 | 150 | 300 | 300 |
| 合计 | 206,510,080 | 300 | 600 | 600 |

两份归档均完整匹配发布方 LFS SHA256。每帧左右手均存在，当前两个片段没有缺失或空手记录，MANO注册缺失数与全零姿态数均为0；不据此推断其他片段也没有缺失。保留原始 `hands.json`、`info.json` 和共享 `__hand_shapes.json__`，共602个文件／2,055,653 bytes，其中全部79,642个数值有限。未重写任何原值，原始完整归档也保留在本机与服务器。

完整元数据还核对了1,028个测试片段及其两个子划分：pose 524个、shape 504个；两个测试子划分互不相交，合并恰好等于BOP测试划分。源 participant ID在训练与测试之间、两个测试子划分之间均无交集。官方说明测试手部GT不公开；本次没有请求测试归档或隐藏GT。[官方片段说明](https://github.com/facebookresearch/hot3d/blob/146b34afef8c1a32adeef7e981c070109f225c87/hot3d/clips/README.md)

## 实际格式以原始数据和官方加载器为准

- `mano_pose.thetas` 是 **15个PCA系数**。官方模型明确使用 `use_pca=True` 和 `num_pca_comps=15`；不能把它直接当成45／48维轴角，也不能补零当成已经适配的训练输入。[固定MANO加载器](https://github.com/facebookresearch/hand_tracking_toolkit/blob/9064a2fe7fda51bb1f1451a0d3d0da7f3ef5e64e/hand_tracking_toolkit/hand_models/mano_hand_model.py)
- `mano_pose.wrist_xform` 有6项；原函数将前三项传给 `global_orient`、后三项传给 `transl`，源码注释单位为米。`__hand_shapes.json__` 的10维MANO beta由同一片段的左右手共享；本次未展开PCA、加减均值、镜像或重建mesh。
- 实际600份 `umetrack_pose.joint_angles` 均为 **22项**，腕变换存为 `T_world_from_wrist`，含4项 `quaternion_wxyz` 和3项 `translation_xyz`。固定README仍描述20角及4×4 `wrist_xform`；官方解码器读取实际字典并转为矩阵。本次保留该差异，不按旧说明截断或改写数据。[固定官方解码器](https://github.com/facebookresearch/hand_tracking_toolkit/blob/9064a2fe7fda51bb1f1451a0d3d0da7f3ef5e64e/hand_tracking_toolkit/dataset.py)

这些原始22项也没有被认定为与 EgoEMG 的20指角加2腕角目标相同。上述核验只证明原始参数的来源、数值格式和传输一致性，不证明跨数据集姿态约定、模型兼容或几何精度。

## 独立复核与许可

本机使用 `tarfile` 和标准库逐项检查归档成员、重复JSON键、有限数值、计数和范围，并将300帧的设备、sequence、participant、逐相机时间戳与来源定义完整对照。服务器使用另一份512字节tar头解析程序，将所有文件正文及填充计入完整SHA256，独立提取相同原始JSON，再用NumPy复算；602个成员的字节摘要与全部数值计数／范围精确一致，没有导入 `tarfile` 或Torch。

首个独立解析器错误地拒绝了实际归档中的PAX时间元数据。定位后只增加对 `mtime` 扩展记录的校验，仍拒绝会改变路径的扩展；针对真实片段及畸形记录的5项检查通过。新的复核进程直接重用服务器上已经完整接收的文件，没有重新下载。接收和复核进程退出后，又重新哈希全部5个完整来源文件和602个保留成员，来源、报告与目录成员未变。[数值及原始来源证据](hot3d-train-source-qualification-20261102.json)

已完整阅读并逐页查看该版本附带的[许可协议](https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/30fe9674782f32e1e5edba98476b6ff4300132c5/hot3d_dataset_license_agreement.pdf)：手部标注为CC BY-NC-SA，序列与非手部标注为CC BY-SA。代码库的Apache-2.0许可不能代替数据许可。公开仓库只保存清单、来源身份和聚合核验记录，没有发布原始手部参数或MANO模型。

## 尚待完成

公开训练片段目前只接收2／2,804个；清单完整不等于数据已全部取得。HOT3D-Clips本身是原始HOT3D的精选子集，完整VRS发布使用另一个申请流程；当前片段不能替代论文所述约4M帧来源或作者确切采样版本。11源约712K混合的样本清单、抽样、去重、划分，以及缺失的markers2mano模型包、匹配权重和训练规格仍未恢复。[官方完整数据与片段的区别](https://github.com/facebookresearch/hot3d/blob/146b34afef8c1a32adeef7e981c070109f225c87/README.md)

本次没有安装来源SDK、执行来源README里的安装命令、训练新模型或替换EgoEMG标签。D07及整篇论文仍未完成。

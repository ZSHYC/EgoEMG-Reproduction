# FreiHAND v2：主包与完整官方链接来源验收

已完整取得[官方页面](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)链接的两个v2归档，共 **4,640,215,813 bytes**。两个包的全部文件CRC、根目录标注数值、共享路径及退出后完整SHA均已核验。这里的“完整”限这两份官方v2归档；EgoEMG作者实际选用的版本、样本、约712K混合及markers2mano资产仍欠，整篇复现尚未完成。

| 官方归档 | 完整字节数 | 文件成员 | 已验收内容 |
| --- | ---: | ---: | --- |
| [FreiHAND_pub_v2.zip](https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip) | 3,881,833,583 | 166,767 | 130,240张训练RGB、32,560张mask、3,960张评估RGB及7份根标注 |
| [FreiHAND_pub_v2_eval.zip](https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip) | 758,382,230 | 23,766 | 3,960个评估样本的RGB、附加逐帧文件及6份根标注 |

主包SHA256为 `0e1ce30e75d12d0874b0d39a93dee0dad8cd0121ebabf72d3c9745b0f68dca96`；评估包为 `7f1968695971479149e28c0fe9495fabee930c48d171cfba739cfd7eeca35712`。发布接口未提供整包密码学摘要，这两个值来自完整接收与独立重读。[完整清单](../assets_freihand_manifest.json)和[机器可读核验报告](freihand-full-source-intake-20261102.json)保留各字段与来源身份。

## 原始角色和跨包重复

训练标注各32,560行；按[官方map_id](https://github.com/lmb-freiburg/freihand/blob/5ea4ab9763fea0eec988a52bfa563333cb16523f/utils/fh_utils.py)映射，`gs/hom/sample/auto`四组RGB各32,560张，同一行标注供四种图像版本使用。评估标注各3,960行。训练RGB、mask和评估RGB的原始数字ID均从0连续，范围及分组已核对；这些数量不代表受试人数或经几何去重后的独立姿态数。

两个归档共有 **3,962个相同路径**：3,960张评估RGB及 `evaluation_K.json`、`evaluation_scale.json`。全部路径完整正文相同，两个根数组的有序数值也精确相同。因此190,533个物理文件成员对应186,571个不同文件路径，13份物理根JSON对应11个逻辑根路径。此处只合并报告计数，两个原始归档、全部原值和训练／评估角色均分别保留，没有自动将评估样本改作训练。

## 全部根标注数值

| 原始根目录标注 | 完整形状 | 数值数 |
| --- | --- | ---: |
| `training_K.json` | 32560×3×3 | 293,040 |
| `training_mano.json` | 32560×1×61 | 1,986,160 |
| `training_scale.json` | 32560 | 32,560 |
| `training_xyz.json` | 32560×21×3 | 2,051,280 |
| `training_verts.json` | 32560×778×3 | 75,995,040 |
| `evaluation_K.json` | 3960×3×3 | 35,640 |
| `evaluation_mano.json` | 3960×1×61 | 241,560 |
| `evaluation_scale.json` | 3960 | 3,960 |
| `evaluation_xyz.json` | 3960×21×3 | 249,480 |
| `evaluation_verts.json` | 3960×778×3 | 9,242,640 |
| `evaluation_errors.json` | 3960×2 | 7,920 |
| 按逻辑路径合计 | 11份，原角色分别保留 | **90,139,280** |

所有根标注数值有限，无整行全零记录。主包7份根JSON包含80,397,680个数值；评估包6份包含9,781,200个，按共享根路径扣除39,600个重复值后得到上表合计。`evaluation_errors.json`仍为来源字段，未当成本项目模型成绩或指定未经核实的单位。

固定[官方模型代码](https://github.com/lmb-freiburg/freihand/blob/5ea4ab9763fea0eec988a52bfa563333cb16523f/utils/model.py)的61维划分为 **pose48、shape10、UV根位置2、weak-perspective scale1**。最后三项不直接作为3D平移。官方先由UV、相机内参和scale恢复根位置，再减去已变形参考关节9的位置设置平移。训练向量中的scale为408.151929–1502.590092，评估为402.131103–1389.158943；单独的 `*_scale.json`参考骨尺度分别保存。

[官方可视化入口](https://github.com/lmb-freiburg/freihand/blob/5ea4ab9763fea0eec988a52bfa563333cb16523f/view_samples.py)对应 `use_pca=False, flat_hand_mean=False`。本次保留来源约定，没有安装或运行MANO／OpenDR／Chumpy、转换姿态或坐标、重建mesh或验证几何精度。MANO版本兼容仍按[固定README](https://github.com/lmb-freiburg/freihand/blob/5ea4ab9763fea0eec988a52bfa563333cb16523f/README.md)列为后续要求。

## 接收恢复和独立核验

主包首次GET在1,082,183,896 bytes处提前结束，原接收进程退出；保留原始前缀，原因仅确认到响应正文提前终止。后续同URL／ETag的下一MiB及边界重叠实际验证通过，新独立进程重用前缀和这段数据，接收167个连续严格HTTP206范围，于 **2026-11-01 21:35:09 UTC** 完整结束。每段限制16MiB，核对范围、长度、总长、编码和源版本，保留200GiB磁盘余量，失败即停。未重下已完成评估包，也未把原失败进程改写为成功。

服务器对主包所有166,767个文件完整读过并检查CRC，用NumPy检查全部7份根标注。原始JSON按字节复制到本机，使用隔离数据读取目录中的ijson 3.6.0／yajl2_c逐行解析，再以标准库递归标量逻辑复算形状、计数、有限性、范围、零行及MANO分块；全部统计精确一致。该读取峰值RSS为19,296 KiB，没有导入NumPy或Torch，也没有改动模型／测试运行环境。评估包已完成的双端独立核验沿用[单独资格记录](freihand-evaluation-source-intake-20261102.md)。

四个原接收／核验进程均已退出后，再完整重哈希两个ZIP及13份物理根JSON，核对报告、辅助源码、目录成员和文件代际；原前缀及下一MiB在最终主包中逐字节相同，167段正文重新哈希且覆盖连续。原始文件均未改变。CRC覆盖全部归档文件，深入数值检查覆盖根标注；附加逐帧JSON、图像和offset-map语义尚未重建。

数据限研究使用、禁止商用并要求引用。公开仓库仅保存来源、清单和聚合核验结果，没有发布原始标注、图像、MANO模型或下载凭据。本次新增模型运行与GPU任务均为0；来源验收不增加模型成绩。

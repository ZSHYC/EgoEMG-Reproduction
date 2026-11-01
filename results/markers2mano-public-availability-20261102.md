# markers2mano 官方公开资产核查

本轮在明确列出的官方公开来源中，仍未取得可运行的 markers2mano 模型包、匹配的 v51 权重或标注训练配方。EgoEMG 主分支仍为固定执行版本 `f7cbd048b7092139852270491d3859cd472e0311`；没有因为搜索结果而替换上游版本或修改实验。

## 已核查范围

官方账号公开仓库列表没有下一页；下列四棵默认分支文件树均完整返回，未截断。表内数量包含树节点，并非全部都是源码文件。

| 仓库 | 固定树版本 | 树条目数 | 返回被截断 |
| --- | --- | ---: | --- |
| [EgoEMG](https://github.com/zhenqis123/EgoEMG/tree/f7cbd048b7092139852270491d3859cd472e0311) | `f7cbd048b709` | 772 | 否 |
| [WristPP_code](https://github.com/zhenqis123/WristPP_code/tree/7953b3009574353e2bb02fcfbb681e55dbf8d593) | `7953b3009574` | 9 | 否 |
| [robot-data-collector](https://github.com/zhenqis123/robot-data-collector/tree/7ab84caad2199105673d20fce22cfdcfdd8d210f) | `7ab84caad219` | 381 | 否 |
| [ai-course-hw1](https://github.com/zhenqis123/ai-course-hw1/tree/4455a7a301a309677d0a7cff5a9a4925b30f4068) | `4455a7a301a3` | 16 | 否 |

EgoEMG 只有 `main` 公开分支，`v0.1.0` 标签指向 `289af31f007583587a9ef1bcceec037389331ecc`。该标签存在，但 [EgoEMG Release 列表](https://api.github.com/repos/zhenqis123/EgoEMG/releases?per_page=100)与 [WristPP_code Release 列表](https://api.github.com/repos/zhenqis123/WristPP_code/releases?per_page=100)均为空，没有从这两处取得模型包或权重资产。完整本地上游不是浅克隆，214个可达提交和该标签中也未找到对应包／graph_transformer路径。

[WristPP_code 的固定 README](https://github.com/zhenqis123/WristPP_code/blob/7953b3009574353e2bb02fcfbb681e55dbf8d593/README.md)明确仍为准备发布的占位仓库。robot-data-collector 是多设备采集代码，其公开子模块为语音／标记工具，未在这些来源找到标注模型。没有执行其安装、编译或网络服务示例。

## 调用入口提供了什么

[固定推理脚本](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/mano/infer_mano_for_egoemg.py#L100)依赖 geometry、graph_transformer 和 rigid_align。对于包含全局 `pose_head` 键的状态，它还明确加载 `markers2mano.archive.graph_transformer_4x.EfficientGraphTransformer4x`，以防误用另一种姿态头。默认权重指向 v51 的验证best epoch12。

这些调用点不是完整模型实现。脚本的1280隐藏维、12层、8头等fallback也不能当成已核验的训练配置；真实值仍需 checkpoint hyperparameters 与相同版本模型源码。当前尚缺模型／图谱与位置编码、匹配权重、训练入口／增强参数及11源抽样／去重／划分清单。[推理入口与匹配检查](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/scripts/mano/infer_mano_for_egoemg.py#L100)、[包的忽略项](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/.gitignore#L209)

## 检索与证据边界

仓库名／描述／README检索未返回 markers2mano 仓库；在 EgoEMG、WristPP_code、robot-data-collector 的代码检索返回的是导入／调用处，没有取得实现。仓库现有6个 issue 中，markers2mano 关键词检索为0。[预览集重投影抖动 issue6](https://github.com/zhenqis123/EgoEMG/issues/6)当次读取没有评论，它本身不能确立抖动原因或与本项目legacy监督等价。[下载 issue1](https://github.com/zhenqis123/EgoEMG/issues/1)维护者回复仍是已知百度发布入口，没有提供这次缺失的模型包。

[结构化来源与字节身份](markers2mano-public-availability-20261102.json)保存官方API响应哈希、完整树版本、查询范围和固定调用文件哈希。该核查不能证明全互联网不存在这些资产，也不代替取得、重跑或训练模型。本轮没有下载新模型／权重，没有构造未训练替身或新增科学成绩；数据标注与整篇论文仍未完成。

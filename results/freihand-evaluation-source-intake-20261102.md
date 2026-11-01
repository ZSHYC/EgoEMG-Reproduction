# FreiHAND v2：完整评估来源接收与参数约定

已完整取得[官方数据页面](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)发布的带GT评估包 `FreiHAND_pub_v2_eval.zip`，共 **758,382,230 bytes**。归档全部23,766个文件的CRC通过，六份根目录标注已在服务器和本机完整读取、独立数值复算一致。主归档仍在服务器单独续传；完整FreiHAND及作者实际训练混合尚未取得。

## 已核验的评估内容

| 原始根目录标注 | 实际完整形状 | 数值数 |
| --- | --- | ---: |
| `evaluation_K.json` | 3960×3×3 | 35,640 |
| `evaluation_mano.json` | 3960×1×61 | 241,560 |
| `evaluation_scale.json` | 3960 | 3,960 |
| `evaluation_xyz.json` | 3960×21×3 | 249,480 |
| `evaluation_verts.json` | 3960×778×3 | 9,242,640 |
| `evaluation_errors.json` | 3960×2 | 7,920 |
| 合计 | 六份文件各3,960行 | **9,781,200** |

全部根标注数值有限，无整行全零记录。原始JSON共214,371,948 bytes，保留文件、评估角色、样本顺序和全部原值，没有重划训练角色。`evaluation_errors.json` 只按原始字段做数值核验，没有把它认作本项目的模型成绩或指定未经核实的单位。

完整ZIP共23,773个成员，解压正文总计5,327,713,787 bytes：3,960张RGB、3,960份逐帧JSON、3组各3,960张PNG、3,960份NPY及六份根目录JSON。全部文件完整读过并校验CRC；逐帧附加文件、图像和offset-map的内容语义未单独重建。根目录参数／关节／顶点数组的全部值已另外完成数值检查。[完整统计及来源身份](freihand-evaluation-source-intake-20261102.json)

## 61维参数按官方代码解释

固定[官方加载与变换代码](https://github.com/lmb-freiburg/freihand/blob/5ea4ab9763fea0eec988a52bfa563333cb16523f/utils/model.py)的 `split_theta` 为：前48项pose、随后10项shape、2项 `uv_root`、最后1项weak-perspective scale。它先通过相机焦距／主点与该scale回投根位置，再在 `pose_by_root(root_id=9)` 中减去已变形参考关节的位置，确定全局平移。

这份61维向量的scale实际范围为402.131103–1389.158943，均为正；单独 `evaluation_scale.json` 保存的是发布方说明的参考骨尺度，实际范围为0.015407–0.035001。两个字段分别保留，后续适配遵循各自含义。

[官方可视化入口](https://github.com/lmb-freiburg/freihand/blob/5ea4ab9763fea0eec988a52bfa563333cb16523f/view_samples.py)使用 `use_mean_pca=False, use_mean_pose=True`；模型入口对应 `use_pca=False, flat_hand_mean=False`。本次没有安装或运行MANO／OpenDR／Chumpy，没有展开或改写参数、变换坐标、重建mesh或测试几何精度。官方v2与MANO版本的兼容说明仍按[固定README](https://github.com/lmb-freiburg/freihand/blob/5ea4ab9763fea0eec988a52bfa563333cb16523f/README.md)保留，来源兼容不能仅凭本次数值有限性宣布完成。

## 双端独立验证

服务器对完整归档重哈希并逐成员校验CRC，将六份根标注按原始字节提取，用NumPy对全部数组归约。复制到本机后，另一份只用标准库的程序递归检查每个维度与数值，复算计数、有限性、范围、零行及官方MANO分块统计；全部精确一致。原接收和评估核验进程退出后，又重新哈希完整ZIP与全部六份原始JSON，输入、报告和目录成员未变。

完整包SHA256为 `7f1968695971479149e28c0fe9495fabee930c48d171cfba739cfd7eeca35712`。发布方没有在此次接口提供整包密码学摘要；该值来自完整接收与独立重读，不冒充作者签名。[文件清单](../assets_freihand_manifest.json)

## 主归档的实际状态

本次两条官方链接在本机与服务器均返回HTTP200，较早记录的404已由本次实际观测更新。主包声明3,881,833,583 bytes，首次完整GET在1,082,183,896 bytes处提前结束，原进程已退出；保留前缀且未重新下载已完成的评估包。原因仅确认到响应正文提前结束。

同一官方URL、同一ETag的下一MiB实际返回严格HTTP206；新后台进程在保留原前缀的同时重用这段数据，随后以16MiB范围接收，逐次核验范围、总长、长度、编码和源版本，保留200GiB磁盘余量，失败即停止。**21:24:41 UTC的实际观测**确认原续传进程存活、无终端依赖，已提交2,291,192,024 bytes／72个新增完整范围；这仍是部分文件，不是完整主包或训练标注验收。

官方数据限研究使用、禁止商用并要求引用原论文。公开仓库只保存来源、文件清单、聚合统计和核验记录，没有发布原始标注或MANO模型。主包、作者精确11源约712K混合及标注器官方资产仍欠，整篇论文未完成。

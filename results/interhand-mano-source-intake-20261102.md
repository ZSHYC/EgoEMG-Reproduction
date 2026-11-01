# InterHand2.6M 的 MANO 参数来源接收

已完整取得[官方页面5fps v1标注链接](https://mks0601.github.io/InterHand2.6M/#annotations)下的 train、val、test 三份 `MANO_NeuralAnnot.json`，共 **50,044,309 bytes**。本机逐项检查与服务器独立 NumPy 复算一致，传输进程退出后又重哈希全部三份文件；原始值与空缺标记保持不变。

## 实际文件内容

| 原始角色 | capture 数 | 帧记录数 | 右手非空拟合 | 左手非空拟合 | 空手槽位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 27 | 18,718 | 12,077 | 12,064 | 13,295 |
| val | 1 | 2,736 | 1,839 | 1,796 | 1,837 |
| test | 8 | 6,890 | 4,533 | 4,541 | 4,706 |
| 合计 | 36（按角色分别计） | 28,344 | 18,449 | 18,401 | 19,838 |

每份非空拟合均含 pose 48、shape 10、trans 3 个有限数值，共36,850份手部拟合、2,247,850个数值。没有双手拟合均为空的帧，全零pose记录为0。`null` 没有补成零姿态，也没有丢弃原角色或记录。capture编号不是自然人计数；这些相机无关的拟合记录也不是2.6M张图像的完整图像标注数。

发布方文档将 pose 定义为减去 MANO mean pose 的轴角坐标，将 trans 定义为米。这里只接收和核验原值，没有加均值、镜像、单位转换、重建 mesh 或更改训练标签。[官方字段说明](https://mks0601.github.io/InterHand2.6M/#annotation-files)

## 来源和许可

[Moon 等的官方数据页面](https://mks0601.github.io/InterHand2.6M/)提供三份参数文件，并说明 MANO 拟合由 NeuralAnnot 得到。发布方代码库的[固定版本 LICENSE](https://github.com/facebookresearch/InterHand2.6M/blob/5d0e456f2345ef524bf71374141fbf3e11dd93f8/LICENSE)为 CC BY-NC 4.0；本项目公开文件清单和聚合记录，原始参数留在本机与服务器，不放入 Git 仓库。

接收使用已连接的 Google Drive：先核对官方目录、文件名、MIME和大小，再取得完整文件引用并落盘。清单SHA256来自本次完整接收及独立重读，发布方未在本次接口返回中提供可认证的整包SHA256；本项目哈希不冒充作者签名。

## 后续复用

[三文件清单](../assets_interhand_mano_manifest.json)保留公开Drive ID、确切字节数、完整SHA256和原始角色。可用项目现有 `download_one` 对照清单重新取得相同字节；本次实际接收走的是Google Drive连接，并未重复调用该下载器。

```python
import json
from pathlib import Path
from egoemg_repro.download import Asset, download_one
inventory = json.loads(Path("assets_interhand_mano_manifest.json").read_text())
for entry in inventory["files"]:
    asset = Asset(**{k: entry[k] for k in
        ("filename", "drive_id", "expected_bytes", "sha256", "group")})
    download_one(asset, Path("data/annotation-sources/interhand5fps"),
                 reserve_bytes=200 * 1024**3)
```

[数值统计及独立核验](interhand-mano-source-intake-20261102.json)列出三个文件的全部计数、范围、来源身份与验收边界。此次未取得完整 RGB、其他标注或30fps版本，未确立 EgoEMG 作者的抽样、去重、train/val/test混合清单，也没有恢复 markers2mano 模型、增强实现或训练权重。它完成的是一个公开参数来源的接收，D07及整篇论文仍未完成。

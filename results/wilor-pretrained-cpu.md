# WiLoR：真实预训练与禁网 CPU 接入

2026-10-26 已完成固定官方大权重获取、原 EgoEMG 22 角模型的合成 CPU 前向和严格离线重载。**这不是数据集性能、GPU 容量或完整训练结果。** 可核验的聚合身份见 [JSON](wilor-pretrained-cpu.json)。

实际使用 [EgoEMG f7cbd048 的 WiLoRViTPose](https://github.com/zhenqis123/EgoEMG/blob/f7cbd048b7092139852270491d3859cd472e0311/egoemg/models/modules/wilor_vit.py) 和 [WiLoR fcb91131](https://github.com/rolpotamias/WiLoR/tree/fcb911312a38fa8badd30d9656a167485d61b8f9) 的源码。26个所需源码、配置及许可文件与固定 Git blob 一致，没有使用 demo 的替代模型。

| 实测项目 | 结果 |
| --- | --- |
| 完整权重大小 | 2,564,989,533 bytes |
| 完整权重 SHA256 | `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2` |
| 原主干加载并逐 tensor 核对 | 404个，keys/shape/dtype/finite及数值全部匹配 |
| 总参数／可训练参数 | 631,631,657／631,579,158 |
| 冻结参数 | 52,499；仅原六个 MANO token/decoder 组件 |
| 原输入／实际裁剪后输入 | 1×3×256×256／1×3×192×256 |
| 原主干末层特征 | 1×1280×12×16 |
| 角度预测 | 1×22×1，float32且全部有限 |
| 完整模型保存后严格重载 | 模型及精确输入的预测逐位一致 |
| 禁网／CUDA | 网络尝试0次；CUDA未初始化 |

原主干加载仍使用官方的`strict=False`调用；检查入口先核完整 schema，加载后逐 tensor 比较，实际无 missing/unexpected key。两次新构造均使用真实权重，完整模型的第二次重载使用`strict=True`，没有接受随机主干 fallback。第一次角度头构造前seed42，第二次构造用seed43再载入保存状态，避免仅靠重复随机初始化得到相同预测。

保留原角度类的 **高度裁剪**：256→192，而不是改成 native demo 的宽度裁剪；head仍为1280→512/GELU/Dropout0.1→22。输入是一个合成RGB张量，目标和mask只用于核对原tuple接口，没有将合成误差当作benchmark MAE。两次前向前后所有参数及持久buffer均未变。

配置实际选择固定 GitHub 的2,228-byte原文件。获取的HF配置为2,233bytes，两者的MANO路径差异由原EgoEMG构造函数覆盖；mean参数经独立SHA核对后放入专用运行目录。该角度前向没有构造MANO几何模型或检测器。

最初整流下载因网络中断失败，原93,323,264-byte片段保留。显式后续attempt在新目录复制只读片段，按固定版本、精确Content-Range及storage ETag获取16MiB分段；完整文件再次核对官方SHA，原片段未变，无自动重试。随后CPU入口又独立重哈希完整权重。

运行实际为Torch2.2.2+cu118、NumPy1.26.4、timm1.0.15，额外依赖在隔离目录，未升级共享Torch/NumPy。模型资产适用[WiLoR许可](https://github.com/rolpotamias/WiLoR/blob/fcb911312a38fa8badd30d9656a167485d61b8f9/license.txt)；此处仅发布聚合身份，不发布实际权重或保存的模型状态。

下一步仍需明确缺失的22角独立训练配方，完成真实cohort优化、GPU容量与完整训练。尤其不能将MANO预训练配方当成该角度任务的配方，或由本次CPU通过推成论文原表已复现。

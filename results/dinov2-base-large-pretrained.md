# DINOv2-B/L：真实预训练与禁网CPU检查

2026-10-26，两项真实预训练权重均取得并校验，随后在冻结8dabdaf原视觉builder中完成CPU正确性检查。[聚合身份](dinov2-base-large-pretrained.json)保留精确HF版本、实际文件SHA、运行源码和加载后tensor身份。S的独立证据仍见[原报告](dinov2-small-pretrained.md)。

| 项目 | DINOv2-B | DINOv2-L |
| --- | ---: | ---: |
| 实际safetensors文件bytes | 346,334,872 | 1,217,502,758 |
| 逐字节核对的过滤后主干tensor | 174 | 342 |
| 含22角回归头总参数 | 86,182,166 | 303,833,622 |
| 原位置嵌入 | 1×1370×768 | 1×1370×1024 |
| 官方插值后位置嵌入 | 1×325×768 | 1×325×1024 |
| 合成输入/实际预测 | 1×3×256×256 / 1×22×1 | 1×3×256×256 / 1×22×1 |
| 禁网前向与严格重载 | 通过，0次网络调用、预测逐位一致 | 通过，0次网络调用、预测逐位一致 |

主干分别来自[timm的B预训练库](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m)和[L预训练库](https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m)。原生HF请求使用main，实际响应的不可变commit、size、LFS OID及下载后SHA必须一致；未手写refs、替换factory或用另一个模型冒充加载。实际CPU环境为Python3.11.12、Torch2.2.2+cu118（CUDA禁用）、timm1.0.15、HF0.28.1、safetensors0.5.2。

第一次使用真实pretrained=True，在活跃socket禁网守卫下加载现有私有cache；保存完整模型及精确合成输入/预测后释放模型，再以pretrained=False严格恢复，参数摘要与预测逐位一致。位置嵌入与官方插值结果逐字节一致。

**此处仅为预训练与CPU正确性证据。** 未做真实cohort优化、GPU容量检查、完整训练或测试准确率；B/L独立作者YAML仍未发布，本项目保留S派生配方的声明。权重、checkpoint及机器私有路径不随此报告分发。

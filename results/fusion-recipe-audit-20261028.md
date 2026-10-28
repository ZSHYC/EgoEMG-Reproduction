# ViT-S融合：固定官方配置与分支初始化审计

2026-10-28实际合成固定f7cbd048版本的五份Hydra配置，并重复核对关键值；[JSON](fusion-recipe-audit-20261028.json)保留配置、源码摘要及边界。这是来源审计，没有新模型构造、权重加载、训练或精度结果。

| 官方experiment简名 | 首卷积输入 | 窗口 | 训练stride | batch | 指尖有效系数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `egoemg_emgformer_small` | 8 | 12,000 | 1,200 | 500 | 0.02 |
| `regression_egoemg_incre_cotrain_wl12000_small` | 8 | 12,000 | 1,200 | 260 | **0.01** |
| `fusion_vits_s_center_8ch` | 8 | 12,000 | 1,200 | 200 | 0.02 |
| `fusion_vits_s_center_16ch_wl7790` | 16 | 7,790 | 400 | 200 | 0.02 |
| `fusion_vits_s_center_eval_released` | 16 | 7,790 | 400 | 200 | 0.02，评价时不作训练损失 |

系数按已核验的原Lightning别名累加规则计算：cotrain-S显式把`landmark/fingertip`置0，另保留`fingertip_distance=0.01`；不能将“EgoEMG统一都是0.02”套用到这条预训练配方。对应的独立S、cotrain-S和融合不是同一个训练实验。

官方README的已发布ViT-S融合评价入口为`fusion_vits_s_center_eval_released`，继承16通道/7,790点配方并清空两条初始化路径，由完整融合checkpoint提供所有权重。16槽位置为`[10,12,0,1,2,4,5,6]`；虽然layout名含`interpolate16`，实际配置`channel_interpolate=false`，原reader走**放置原8通道并补零**的分支，不能凭名字启用插值。

训练入口的`pretrained_emg_checkpoint`调用原`_load_pretrained_backbone`。它只保留指定前缀中**名称与shape都匹配**的张量，再以`strict=False`加载并打印缺失键。8通道首卷积权重不会被自动扩展成16通道；不匹配时跳过。后续实际实验应记录加载/跳过的键与权重来源，不能把部分初始化写成整分支严格恢复，也不能因此自行改变官方加载算法。该规则是源码核对，本次没有拿实际初始化checkpoint执行加载。

因此下一项融合工作须先固定“发布checkpoint评价”或具体“从头训练配方”的身份，再核验真实初始化与数据消费。当前已有旧版8通道S或ViT-S报告不能证明新的16通道融合已完成；也不能预设必须先训练一份0.02的S才能遵循官方cotrain初始化。合并训练来源、完整发布资产和论文原表等价仍未补齐。

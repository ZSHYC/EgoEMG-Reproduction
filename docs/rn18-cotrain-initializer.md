# RN18融合的原指定cotrain-S初始化

这是`fusion/fusion_rn18_s_center_8ch`的前置训练，使用固定官方版本`f7cbd048b7092139852270491d3859cd472e0311`中的`emgformer/regression_egoemg_incre_cotrain_wl12000_small`。它不新增一项论文模型成绩，也不表示RN18融合已经训练完成。

现有ViT融合的cotrain初始化用了另一组留出区，不能直接交接；在跑的原生EMGFormer-S使用200轮/指尖0.02，也不能代替本实验的150轮/0.01。新读取器保留原生模型、损失、优化器、完整轨迹及原始随机增强，只将可用旧版数据绑定到RN18融合的固定留出区。

| 项目 | 本路线执行设置 |
| --- | --- |
| 输入与监督 | 每手8通道×12,000点；训练和验证均为完整22角×12,000点目标及原generated mask |
| 窗口 | 训练stride1,200及原exclusive jitter；验证stride12,000；repeat1，无初始化测试角色 |
| 优化 | 原AdamW、lr0.0005；cosine150、eta_min0.000005；实际指尖系数0.01 |
| 正式预算 | 150轮，完整训练/验证；每轮241次更新，共36,150次 |
| 批量与加载 | 训练/验证均260；12workers，persistent=true、prefetch2、pin_memory=false；验证forkserver |
| 选模 | 原val_mae/min，top3＋last，无EarlyStopping；只导出实际验证BEST |
| 硬件适配 | 原声明四卡；本路线单rank及BF16，明确不宣称与四rank数值等价 |
| CPU工程范围 | 两轮，每轮4个训练样本、batch2/workers0；验证仍覆盖该工程数据的全部样本 |
| CUDA工程范围 | 两轮，每轮最多520训练样本；原batch260/12workers与完整验证 |

[两个独立进程的完整数据准入](../results/rn18-cotrain-full-input-admission-20261101.json)已实际完成：训练62,470、验证1,052个手部样本；16份消费payload、16,540,457,757字节。每进程在完整目标扫描前后分别整文件哈希，合计4遍；所有可能随机窗口的输入和完整轨迹目标均有限。将来融合的留出分区身份固定为`44e9ebc8a2f63a79699c5a976bf2717b0fe7a376427c7c46f538d0745c2b4bd0`，初始化训练与其验证/测试范围的重叠行数均为0。12个数据核验拥有者已关闭。数据核验没有构造模型、更新优化器或初始化CUDA。

实际可用的是41人旧版EgoEMG；原cotrain配置中的Incre右手数据未获得，未合成或重复填充。使用原统计资产中的每手、每通道float32归一化向量；`filtered_paper`仅通过IO别名读取实际`filtered`文件。保留完整目标和原mask，不改成中心监督，也不清理腕角或masked-false标签。

训练入口是`egoemg_repro.rn18_initializer_train`。完整训练必须重新准入实际数据，并提供独立冻结的完整数值receipt；工程数据、旧实验的相似配方或重新封装的JSON都不能获得完整权重交接资格。fit、checkpoint和export使用RN18专属身份，拒绝另一初始化实验族。恢复核对模型、AdamW、scheduler、callbacks、native loops和Python/NumPy/Torch随机状态，并在继续前完整重放冻结验证。实际worker/iterator需要重建，不保证与未中断persistent-worker训练逐步相同。

[独立CPU恢复与导出](../results/rn18-cotrain-native-cpu-recovery-20261101.json)已实际完成：6个拥有阶段全部退出0、14原拥有者关闭；真实一轮前缀接续到两轮/4次更新，模型、优化器、scheduler、callbacks、loops和RNG六域恢复一致，完整工程验证重放一致，110个原生张量与实际验证BEST一致，退出后新进程重新读取通过。部分训练不能导出，工程结果不能冒充完整初始化。80项完整运行时检查通过；明确禁用Torch时8项纯检查通过、72项运行时检查skip。

当前完整150轮、真实GPU资格及验证BEST权重尚未完成；RN18融合仍需其视觉LAST来源验收、两个分支交接、原生融合150轮和最终独立评价。完整论文目标继续保留。

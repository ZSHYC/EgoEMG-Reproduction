# 原生 EMGFormer 完整训练的实际用时

这份Table7补充统计读取已经完成的原生S/M/L各200轮、25,000次更新的真实fit结果，没有重新训练、加载模型或运行优化器。[JSON](emgformer-native-training-time.json)保留未舍入秒数、原始fit/identity文件SHA和已公开完成证明的SHA；[CSV](emgformer-native-training-time.csv)给出同一底表。

| 模型 | 实际参数 | 完整轮次／更新 | GPU | 精度／批量／worker | fit阶段墙钟 |
| --- | ---: | ---: | --- | --- | ---: |
| S | 3,460,134 | 200／25,000 | 1×A800-SXM4-80GB | BF16／500／6 | 7.055943小时 |
| M | 6,617,382 | 200／25,000 | 1×A800-SXM4-80GB | BF16／500／6 | 7.491039小时 |
| L | 16,320,806 | 200／25,000 | 1×A800-SXM4-80GB | BF16／500／6 | 8.303335小时 |

原始记录分别为25,401.3957263669、26,967.740244214423、29,892.004269579425秒，小时为原值除3600。单设备占用阶段小时与这些墙钟小时数值相同，**不是GPU活跃计算时间或论文RTX4090等价GPU-hours**，也不能用于推导纯训练吞吐或不同架构的受控速度排序。

计时字段为原执行源码`egoemg_repro/emgformer_train.py`的`elapsed_fit_seconds`。计时从调用原生`Trainer.fit`之前开始，涵盖训练、内部验证和sanity验证；在关闭读取进程、检查来源／训练记录及计算最后状态／检查点SHA后构造结果时结束。它不包含此前完整数据准入、模型构造、单独工程资格和最终val/test新进程评价。原记录没有将训练、验证、收尾的用时分别拆开，因此本表不能给出这三部分的独立工时。

2026-11-01的只读验收重新打开三组真实`fit-result.json`和`identity.json`，全部物理SHA与已公开的[S](../native-emgformer-small-complete-20261101.json)、[M](../native-emgformer-middle-complete-20261101.json)、[L](../native-emgformer-large-complete-20261101.json)退出后证明一致，内容seal与本项目最终报告的fit身份一致，六文件九项属性前后不变。原计时源码与执行提交`53272d738f571238b45ec3ab83540406c78eba63`逐字节相同；本次读取未导入Torch，新增训练更新为0。JSON／CSV重新读取后，全部原秒数和小时数值一致。

数据仍为可得旧版源、单rank运行；各模型的训练、验证／测试人口与论文的完全等价性未证明。原参数／结构审计见[架构报告](emgformer-architecture.md)，完整精度及选模身份见各原生结果目录。此处补齐实际本项目硬件与fit用时，全文目标仍未完成。

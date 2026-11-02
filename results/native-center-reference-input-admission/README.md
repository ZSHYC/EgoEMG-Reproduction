# 原生同帧评价的完整输入准入

**完整4282手请求的输入单元已验收；新模型评价与Table10尚未完成。** 本次完成[同帧计划](../../docs/superpowers/plans/2026-11-02-native-center-reference.md)的Task1，未新增训练或预测，也未把已完成的2792点报告改成同视觉人口。

来源为固定官方版本 `f7cbd048b7092139852270491d3859cd472e0311` 的完整 `collect_val_centers`；7790参考中心统一要求可容纳12000窗口，2141窗口／4282手请求，原官方边界规则排除24个手请求。这个事先按最大模型窗口确定的清单来自[实际网格与独立元数据oracle](../native-table10-source-audit/README.md)，不按测试误差选择人口，不证明作者使用了该共同人口。

| 完整准入项 | 实际结果 |
| --- | ---: |
| Gesture / User / Both手请求 | 2314 / 1646 / 322 |
| 全部请求 / 有效标签 / 无效标签 | 4282 / 4154 / **128保留** |
| 有效角度数 | 91388 |
| 有图 / 缺图请求 | 4112 / **170保留** |
| 每个请求的EMG / 目标 / mask | float32 `8×12000` / float32 `22×1` / bool `1` |
| 全部数据来源 | 17文件／17,069,751,557字节 |
| 两个独立完整CPU读取进程 | 69.928516秒 / 69.463702秒 |
| 退出后独立标准库读取 | 16.811596秒 |

两个真正独立的新进程逐请求使用既有官方纯EMG点读和冻结归一化，完整读取所有4282个输入；没有因为label mask、图像缺失或异常而减少请求。每个22角目标及bool掩码与已完成视觉第一遍的同一中心／手／split／私有用户记录逐字节一致。所有模型相关输入有限；新数据类对非法子清单、移位中心、非有限输入及源文件世代变化均报错。

两遍完整规范化EMG流、目标/mask流、私有身份流以及全部4282行typed输入摘要列表精确相同。实际完整源码确认S/M/L所消费的test modalities和transforms一致；三种配置的真实输入行为已由[新增17项检查](../native-center-reference-data-engineering-20261102.json)验证。类直接调用已有 `_getitem_center_supervised_emg`，不调用缺图像字段的融合取样分支；不修改旧数学、归一化或原官方源码。

两遍各自在消费前后核对17份完整来源SHA及守卫属性，315份冻结组件/测试胶囊文件和注册的完整官方源码均保持字节。另一个**未导入Torch、NumPy、Lightning**的CPU进程在三个原拥有者退出后，独立核对两份完整摘要列表、计数和身份，再对17.07GB来源重新做整文件SHA；独立读取者也已实际退出。模型和优化器构造、神经前向、训练更新、CUDA初始化均为0。上述时长属于输入验证，不是模型训练GPU-hours或推理延迟。

## 证据与继续工作

- [完整双遍及退出后读回](admission.json) · [实际源版本／拥有者／胶囊SHA](launch.json) · [产物清单](completion.json)
- 直接归档的实际服务器代码：[输入worker](worker-source.py)、[双进程控制器](controller-source.py)、[独立退出后读取者](reader-source.py)；[实际配置原字节](execution-config.json)与启动SHA一致。
- [数据类源码](../../egoemg_repro/center_reference_data.py) · [新增行为测试](../../tests/test_center_reference_data.py) · [17项检查与失败后修正证据](../native-center-reference-data-engineering-20261102.json)

私有每请求摘要文件保留在服务器准入根目录的 `private/first/input-digests.json` 与 `private/second/input-digests.json`，两者SHA均为 `5382d9afa6535abaae8b8826845bb787cd15ca4bf76b16c7de04701f3584f187`；后续预测必须绑定这个完整输入身份。源码和配置归档不包含原始数据或预测包，独立重放仍需要这些实际来源。

接下来使用既有200轮／25000更新及验证BEST169/124/142，新增FP32、原raw-token中点的完整S/M/L同帧预测、新进程独立模型重放与Table10逐用户mean±SD／总体样本MAE。旧BF16点6000与轨迹对齐6255的结果保持各自身份。新的数据类整合审查和完整模型入口尚未完成；本次没有新分数。旧版数据、标签来源、整episode预过滤、单rank、作者混合来源/人口以及其他全文缺口仍在。

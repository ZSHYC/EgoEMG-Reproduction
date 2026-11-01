# 原生EMGFormer全轨迹评价（进行中）

本工作补齐既定全文目标中的S/M/L全轨迹评价。现有原生200轮及中心评价报告保持原样；同名模型的legacy轨迹结果也不改写。完整窗口输入、同次原生前向捕获、原始包与统计接口已接通，并有实际训练模型的单窗口诊断；完整角色的前向/独立重放、报告和正式GPU执行仍属后续工作，当前没有新增全轨迹MAE。

[28项真实输入工程检查](../results/native-emgformer-trajectory/input-engineering-tests-20261101.json)已通过、无跳过；另一进程重新读取服务器完整测试日志及全部冻结源码字节并核对摘要。夹具为真实合成memmap、验证4/测试8手窗口，不是完整发布数据或已训练模型。原始只读NumPy数组经官方collate时保留1条Torch警告，输入文件字节及随机状态未变。一次完整新组件审查将在前向/报告/CLI接通后进行，当前不授予正式评价资格。

## 原生前向、原始包与统计

[`emgformer_trajectory_results.py`](../egoemg_repro/emgformer_trajectory_results.py)已接入原始Lightning完整目标前向，在同一次学习前向中观察head输出，并使用模型本身的原生对齐方法核对结果。[21项新增检查](../results/native-emgformer-trajectory/results-engineering-tests-20261101.json)通过：FP32/BF16张量无损读回、完整源标签与上下文目标/掩码、异常后的hook关闭、尾批/整批无效、完整样本顺序、两种MAE及逐角/指组统计均覆盖。工程统计使用明确的工程数组，没有伪造完整训练模型。

私有原始包保留低采样率预测、对齐预测、对齐目标/掩码、完整窗口目标/掩码六种张量；BF16保持原始dtype，不转成float32冒充原始字节。另存精确窗口键、逐样本/整批输入指纹和typed张量及文件SHA。统计流式读取完整角色，分别保留原生FP32按batch样本数加权MAE与FP64有效角度pooled MAE；轨迹和对齐轨迹中心单列。官方PD组仅proximal/mid/distal三组，Table12十组另包括五指和两个腕角，组之间存在重叠。

[真实训练S模型诊断](../results/native-emgformer-trajectory/trained-small-full-window-diagnostic-20261101.json)使用实际完成200轮、验证best第169轮的模型，重新核对其完整来源、载入全部3,460,134参数，在一个实际验证左手窗口执行直接原生前向与上述捕获前向。两次前向共享同一模型和生产进程，输出完全一致，模型状态前后未变，0优化更新/无CUDA；原始包由另一个禁止模型构造的进程读回，六个拥有进程全部关闭。全新进程重新载入模型的完整重放仍待后续完成。

| 已实际观察的S模型字段 | 数值 |
| --- | ---: |
| 原始目标长度 | 12000 |
| 左/右上下文 | 510 / 0 |
| head低采样率长度 | 230 |
| 原生对齐后长度 | 11490 |
| 对齐轨迹中心索引（零起算） | 5745 |
| 对应原窗口目标索引（零起算） | 6255 |
| 原中心点评价目标索引（零起算） | 6000 |

该诊断为CPU/FP32、一个验证窗口，完整点队列的16文件/16,540,457,757字节来源重新核对；**没有完成全体留出区全轨迹目标准入、batch500/6worker资格或完整val/test评价，也没有新全轨迹MAE。** 对齐中心与原中心目标不同，因此不能拿原中心结果填全轨迹派生中心。GPU BF16的插值不声称与CPU逐位相同；后续在相同正式设备上独立模型重放，并从保存的实际对齐输出重算误差。

固定官方`f7cbd048b7092139852270491d3859cd472e0311`的原生验证/测试默认`center_target_only=True`。全轨迹适配使用同一数据类支持的`False`设置，为验证/测试另建读取器，不修改原中心读取器、窗口、归一化、样本顺序或已选权重。后续模型执行须调用原始Lightning的完整目标路径：依照模型本身的左右上下文裁剪，再由原始`align_predictions`和`align_mask`插值，不手写新的对齐算法。

## 输入验收

[`admit_trajectory_inputs`](../egoemg_repro/emgformer_trajectory_data.py)先重新执行完整原生点队列输入准入，并与实际训练身份中的`cohort`精确匹配，再检查全部验证/测试窗口标签。原点队列仅数值检查了留出区中心，不能据此接受非中心标签。

- 原有16个输入文件完整流式SHA，源码、配置、元数据、分区及读取器身份均绑定。
- 每手输入8×12000，完整目标22×12000，生成有效性掩码12000；原左手窗口后右手窗口、单次消费顺序保留。
- 留出区全部窗口目标、左右腕角及原float32弧度转换均检查有限性；无效掩码下面的NaN也拒绝，不修补或删除样本。
- 原完整读取器的float32腕角转换不同于中心读取器的标量舍入；后续“对齐轨迹的中心”不得冒充原中心点评价。
- 数据进程序列化重开只读映射，并核验原文件代际和完整读取器指纹。输入验收不构造学习模型或优化器，不授予训练或科学结果资格。

以下示例需要真实私有数据与训练身份，公开JSON无法代替它们：

```python
import json
from pathlib import Path
from egoemg_repro.emgformer_literal import compose_emgformer
from egoemg_repro.emgformer_trajectory_data import admit_trajectory_inputs

upstream = Path('/absolute/pinned/upstream')
fit_identity = json.loads(Path('/absolute/completed/fit/identity.json').read_text())
config = compose_emgformer(upstream, 'small')  # 与实际来源规模一致
inputs = admit_trajectory_inputs(
    '/absolute/legacy/data', upstream=upstream, config=config,
    normalization=fit_identity['cohort']['normalization'],
    expected_point_identity=fit_identity['cohort'],
)
inputs.check()
```

实际旧版队列应为验证1052、测试2792手窗口。工程夹具的4/8等数量必须保持工程身份，不能重签名成完整发布数据。原生后续评价保留batch500、6workers、prefetch1、persistent=True、pin=False及BF16；显式CPU工程入口为batch2/0workers/FP32，不代替正式评价。

## 后续验收

仍需实现完整pass/profile/report及CLI生命周期，同一原生200轮验证best的完整val-native/新进程val-replay、测试前冻结轨迹评价规则、完整test-native/新进程test-replay，以及报告/原始包独立重算。输出应分别保留轨迹与对齐轨迹中心、左右手、Gesture/User/Both、逐用户、22角和指组的误差和分母，并披露此前中心测试已经执行。只能在这些实际完成后发布新全轨迹结果。

当前其他GPU任务和其后续队列继续，不能抢占或因观察超时重启。全论文、PDF数据人口等价、标注与其他清单仍按[全文状态](full-paper-status.md)逐项验收。[实施计划](superpowers/plans/2026-11-01-native-emgformer-trajectory.md)记录本组件全部后续步骤。

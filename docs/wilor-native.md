# WiLoR 原生22角实验

当前正在实现[完整WiLoR计划](superpowers/plans/2026-10-28-wilor-native.md)。Tasks1–3的配置、真实资产/模型、完整输入绑定、连续fit和严格最佳权重读取已完成。原生完整评价/报告、GPU资格、一次新实现审查、完整30轮和独立测试仍需逐项完成。全论文完成设置数不会因配置或工程检查增加。

Task3最终绑定源码8013690。[两轮真实CPU验收](../results/wilor-native-fit-cpu-20261028.json)使用原631,631,657参数模型、真实404张量初始化和普通合成memmap/LMDB：B1/FP32，每轮2训练/2验证、共4次实际更新；三个完整原生检查点共22,739,889,583字节，模型/优化器/调度器/全局RNG/覆盖及冻结状态均核验。另进程严格重载原生best后，全部两个验证样本的typed预测和原生MAE精确一致，不重开预训练权重、不增加更新；第三个独立读者重开原始链，11种实际检查点损坏均拒绝。三个阶段实际exit0、八个所属进程退出，原始文件和111份冻结成员再次hash未变。这个合成工程过程不是准确率结果。

[完整实际输入的两次独立重验](../results/wilor-native-data-full-20261028.json)各79.38/78.13秒actualexit0：52文件/57,145,308,966字节，全部可能训练抖动中心的目标均有限、RNG未变、没有模型或CUDA。注册legacy实际202,020/3,152/8,612消费及101,010/1,576/4,306唯一人口一致，六个所属进程退出，连续输入/源码generation guard通过；[完整内容身份](../results/wilor-native-data-full-identity-20261028.json)保持数据资格范围。先前30项数据边界检查已经包含在最终80项fit/data检查内。

[最终检查身份](../results/wilor-native-fit-cpu-validation-20261028.json)记录208项原生/相关检查和完整轻量1,552通过/1,200真实可选依赖跳过，均actualexit0、12份来源与实际拥有者核验通过。`fit_wilor`没有续训/覆盖或任意配方覆盖接口，完整模式在读取数据/模型前要求当前原始CUDA资格；`read_wilor_fit`重开原始epoch链及native best，`reopen_wilor_best`用自包含完整状态严格重载。完整GPU资格仍未执行，数据/CPU实证不能打开完整30轮入口；服务器检查点可用空间需要补足，相关目录信息已询问用户。

[配方来源审计](../results/wilor-angle-recipe-audit-20261028.md)核对了所附PDF第22页与固定官方源码。原仓库有WiLoR模块，独立视觉WiLoR训练YAML缺失；此处明确派生自官方视觉ViT-S配方，整份替换模块参数，再应用论文WiLoR的batch18、初始LR1e-5和30epochs。BF16、weight_decay0.01、梯度裁剪1、val_mae选模/top3+last/patience30等分别标为继承字段。

```python
from egoemg_repro.wilor_literal import compose_wilor, wilor_protocol
cfg = compose_wilor(upstream)
contract = wilor_protocol(upstream, cfg)
```

该API仅生成并验证配方，不构造模型、读取真实样本或准入权重。绑定的完整配置被更改时会拒绝；公开元数据排除数据目录和权重路径。`pretrained_path=None`是尚未准入资产的配置状态，实际原生模型入口必须另行核验真实完整权重，不能直接调用原类的随机初始化回退。

固定代码使用1280维特征/GELU/22角head，保留256×256输入的原高度裁剪192×256；原文1576维不覆盖代码。先前[真实CPU权重实证](../results/wilor-pretrained-cpu.md)已测631,631,657参数，其中52,499冻结、404个主干张量与真实预训练精确匹配；这些结果不证明本次新运行入口已训练或取得GPU资格。

新增入口`admit_wilor_assets`先完整核验固定目录源码、配置、mean与2.56GB原权重，再允许`build_wilor_runtime`调用原生构造器；捕获前已加载的错误`WILOR_PATH`会拒绝。模型契约校验实际结构、冻结状态和全部404个预训练主干张量，`wilor_optimizer_identity`校验原生head/backbone两组AdamW与恒定cosine行为。独立恢复要求完整初始化来源和严格模型状态，不重开原预训练权重，也不宣称训练轨迹续接。

[当前真实CPU工程实证](../results/wilor-native-model-cpu-20261028.json)绑定源码a52a0ba：真实模型完成一次原生AdamW更新，非零梯度有限、冻结参数未变；完整保存7,579,750,462字节私有模型/优化器/调度器包。另一个进程严格重载全部模型状态，两个普通合成memmap验证样本的原生预测精确一致，新增训练更新为零。第三个独立CPU读者重开原始包、来源与状态核验成功，两个阶段220份冻结成员未变，十个原进程与独立读者均已实际退出；没有CUDA或完整数据资格。

[最终检查](../results/wilor-native-model-cpu-validation-20261028.json)记录128项原生/相关检查通过（78模型入口、27WiLoR配置、23原视觉配置），完整轻量1,501通过/1,171真实可选依赖跳过，均实际exit0、十来源摘要未变。[两次工程驱动失败](../results/wilor-native-model-cpu-failures-20261028.json)保留：官方直接调用`model.forward`使hook漏记、Torch2.2的mmap要求字符串路径；只修正私有观测/加载驱动。第二次已成功的更新和原包保留，仅新进程恢复，未重跑训练、改生产源码或放宽比较。这里只证明工程入口，不报告准确率或full/GPU准入。

原base的指尖项与实验别名在原生Lightning中合计0.02。继承的cosine floor1e-5等于初始LR，实际学习率恒定；报告明确记录此行为。数据仍保留原7790点、stride780、评价stride7790、repeat2与生成中心mask、缺图的归一化黑输入语义。原生加载器在三个角色都使用batch18；配置中val_batch_size256未被make_data_module传入，不能据此声称验证批量256。

源码契约保留原六设备声明作为来源；未来实际单卡执行另有身份，不能称为相同的六卡优化轨迹。可用legacy数据与声明的合并训练数据、PDF原表的等价性仍未解决。完整数据预期消费202,020/3,152/8,612，唯一101,010/1,576/4,306，与新融合的395,276训练消费分开。所有实际原始资产和派生模型权重保持私有，公开仓库保存实现、配置和聚合证据。

[当前配置实证](../results/wilor-native-config-20261028.json)绑定源码028250c：独立新进程重新合成/校验成功并实际退出，Torch未导入，未构造模型/数据或读取权重。[检查身份](../results/wilor-native-config-validation-20261028.json)记录27项新配置及23项原视觉配置检查共50通过，以及完整轻量1,475通过/1,119可选依赖跳过，均actualexit0；七份来源摘要逐字节未变、四个检查拥有者和独立校验子进程均确认退出。原生数据模块用受控三行源验证三角色各repeat为六消费、批量18；不是实际完整队列。初次汇总对嵌套上游使用错误Git仓库而退出1，修正汇总器后重新校验成功，保留失败且未重跑模型/预测。

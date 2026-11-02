# Native Center Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定官方7790参考中心、共同12000边界清单上，完成原生S/M/L完整同帧评价与Table10，不改变原200轮选定权重或旧报告。

**Architecture:** 原官方完整中心函数选择清单；小型适配层调用已有纯EMG点读和严格已训练模型构造。新原始预测经独立重放、同视觉实际目标连接及退出后读取，复用既有point统计归约；产物独立保存。

**Tech Stack:** 现有Python3.11、NumPy1.26、Torch2.2.2、Lightning2.5；无新依赖。

**Spec:** [统一中心补充](../specs/2026-11-02-native-center-reference-addendum.md)，落实[全文设计](../specs/2026-10-26-full-paper-reproduction-design.md)要求4/7。

## Global Constraints

- 官方固定 `f7cbd048b7092139852270491d3859cd472e0311`；完整 `center_frame.py` SHA `9ed981e3652d94a25d4607ff3e0eb3802b0f279077eccafe933773026aeb085a`；不改官方源码。
- 共同窗口2141、手请求4282；Gesture/User/Both为2314/1646/322；参考4306减24仅源于最大12000窗口的原官方边界规则。
- S/M/L使用原200轮/25000更新、BEST169/124/142，严格完整加载；FP32、raw-token中点、22角原generated mask；原6000点、6255轨迹中心和BF16来源分别保留。
- 无优化器、训练更新、测试选模、镜像/标签/mask/归一化算法修改；缺crop不筛除，取样失败不得跳过，全部无效请求仍记录。
- 两个完整模型进程分离；完整来源SHA/文件世代、模型状态前后、输入与目标身份、物理拥有者退出及独立CPU读取；公开只含匿名用户聚合。
- CPU2/RAM128GiB/磁盘225GiB；GPU稳定空闲64GiB、运行8GiB；不影响别人的任务，观察超时不重启。
- 缺少正式预测、独立重放或目标连接时，资格/工程/脚本完成不代表Table10完成；整篇论文范围保持不变。

## Review Focus

- 7790与12000网格不可互换，官方最后行边界不可被自行修正：Task1用同一实际完整函数与独立元数据oracle检查。
- 单样本跨手、跨episode、尾部、非有限输入或全无效mask：Task1拒绝损坏字段，保留有效性分母和全部请求。
- 把辅助资格或旧2792点包冒充完整4282：Task2拒绝错误scope、权重/构造身份和人口，并检查typed完整覆盖。
- helper融合取样异常导致静默缩小人口、FP32被BF16替代：Task2实际纯EMG入口与dtype/全请求检查，异常使阶段失败。
- 私有用户泄露、独立进程未退出或源被改动：Task3匿名投影、源世代负例和仍活进程拒绝测试。

### Task 1: 原参考清单和纯EMG输入适配

**Files:** Create `egoemg_repro/center_reference_data.py`; Test `tests/test_center_reference_data.py`。

**Interfaces:** `reference_windows(data, upstream, *, required_window_length=12000)`返回完整有序episode/start/end/center/split清单和来源身份；`CenterReferenceDataset(data, upstream, config, windows)`返回左手后右手、repeat0的原生点样本和完整元数据。所有窗口必须来自固定原官方函数，不由调用方任意选择。

- [x] 编写并观察失败的真实数据类测试：重排split-label词表、episode首尾和最后行边界、跨手顺序、非官方任意子清单拒绝、非有限输入拒绝、无效mask仍返回请求。
- [x] 最小实现调用未修改的官方collect函数和原 `_getitem_center_supervised_emg`，复用已有view/归一化，不载入模型或图像。
- [x] 新组件测试通过；服务器完整2141窗口/4282手输入资格，source SHA/世代、所有输入有限和与已完成视觉目标逐请求精确连接；独立进程重读。四个旧探针只作既有工程证据，不重复冒充本步骤。
- [x] 提交Task1，ledger记录actual命令/结果/关闭身份，旧归约与数学源码保持原字节。

### Task 2: 选定原生S/M/L的完整FP32同帧评价入口

**Files:** Create `egoemg_repro/center_reference_eval.py`, `scripts/paper/run_native_center_reference.py`; Test `tests/test_center_reference_eval.py`。

**Interfaces:** `run_reference_pass(source_fit, admitted_inputs, output, *, kind, reference=None)`以冻结模型身份执行；`read_reference_pass(output)`拒绝来源/人口/typed包不一致。输出所有4282请求的原始point parts、输入指纹、前后模型状态、原生逐手float32旁证及既有point充分统计量，不产生训练状态。

- [x] 编写并观察失败的资格/全量混用、错误BEST/源模型、错误窗口中心/手/mask、FP32变BF16、取样异常、重复/缺失请求、输入变动和独立进程身份测试。
- [x] 最小实现严格现有原生构造/全state加载；保持原module.model forward及raw输出中点，复用 `emgformer_statistics`；不修改任何旧runtime/reducer。
- [ ] 新组件整套通过；真实三规模CPU/GPU短资格及逐字节严格重载；短资格不得写科学完成。与现有官方模型数学的关系通过调用边界和真实原生输出证明，不重新审查或测试旧模型内部数学。
- [x] 入口与来源/运行协议提交；一次新组件最终审查，实质发现一次修正并以失败再通过的测试核验，不二次审查已关闭数学。

### Task 3: 完整新进程重放、Table10与公开来源闭环

**Files:** Create `scripts/paper/collect_native_center_reference.py`, `results/native-paper-table10/`; Test `tests/test_center_reference_collection.py`; Modify `docs/full-paper-status.md`。

**Interfaces:** 独立CPU读取消费完整pass、模型/源/共同清单身份，输出9个split用户mean/SD、3个Avg以及每单元有效分母；现有视觉预测只按固定共同清单连接目标，不按误差选择。

- [x] 编写并观察失败的活进程、被替换来源、私有by_user键、旧点/轨迹报告替换、同N但不同目标、缺少无效请求和非有限聚合测试。
- [x] 实现model-free关闭后读取与匿名投影，复用原point统计和全量正确性凭据；保留逐手原float32与主要float64聚合的区别。
- [ ] 正式S/M/L各完整原生评价及独立新进程重放；完整模型状态、typed输入/预测/目标/mask与来源一致；GPU资格、短训练或子清单不替代此项。
- [ ] 实际拥有者全部退出后CPU独立读取，第三独立算术/身份核验；发布表格与PDF显示参照、全部限制；公开匿名不可变URL逐字节核验。
- [ ] 更新Table10实际状态与全文checkpoint，源/旧结果均未改变；只有所有交付实际验收才勾选完成，保持全论文goal active直到全部其余必需项也完成。

当前状态：**Task1已完成**：[17项新组件检查](../../../results/native-center-reference-data-engineering-20261102.json)及[完整4282输入的两个独立进程／退出后17.07GB来源重hash](../../../results/native-center-reference-input-admission/README.md)均通过，四原输入/读取拥有者已退出；源代码224f057保持。**Task2/3继续执行**：新增三部分组件共83/83检查通过；真实选定S/M/L的CPU四请求原生及独立重放共24次前向、12来源拥有者与独立读取者均已退出。最初S验收脚本把raw步数写成1915，已按官方与已验收轨迹的230更正；保留失败记录并复用成功S产物，无重复前向。一次新组件审查提出的资格摘要来源绑定问题已由7个RED→GREEN负例和真实6包/16文件重新读取修正；GPU资格、完整六次4282请求预测和独立Table10交付仍未完成；尚无新的全量模型分数。保留本计划workspace/ledger，不重跑已关闭的来源审计、旧数学或输入资格。

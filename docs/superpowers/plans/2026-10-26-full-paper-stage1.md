# Full-paper Reproduction Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 推进全论文目标，首先使 EMGFormer-L 能完整训练并严格导出，建立覆盖全部论文产物的证据清单，随后补充可直接获取的数据统计。

**Architecture:** 复用固定官方模型和当前训练/评估接口，先扩展 L 的配方与导出上下文白名单。新运行与此前四基线隔离。整个全论文设计保持有效，本阶段结束不等于总目标完成。

**Tech Stack:** Python、PyTorch 2.2.2、Hydra、NumPy、pytest、SSH、A800。

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`

## Global Constraints

- 当前官方版本固定 `f7cbd048b7092139852270491d3859cd472e0311`；有歧义优先官方代码。
- 既有四模型结果和运行身份不得覆盖。
- 配置存在、短训练和测试通过不能代替完整实验。
- 新实验不得根据测试结果调参或选模。
- 不发布数据、原始参与者身份、凭据及无授权资产。

## Review Focus

- 不支持的 EMG 架构仍应拒绝导出，不能用默认 510 上下文吞掉 vemg2pose/NeuroPose 的差异。
- L 的上下文必须由真实官方模型检查，轨迹长度为 12,000−510=11,490，不能照抄 S/M 结果。
- GPU 短检查与完整训练必须不同目录，观察失联不能自动重复启动。
- 新配置不得改变已完成实验的源码、配方、数据身份。
- 论文表中缺失项不能因本阶段仅处理 L 而消失。

### Task 1: L 配方、严格导出与真实训练

**Files:**
- Create: `configs/emgformer_l.json`
- Modify: `egoemg_repro/report.py`
- Test: `tests/test_report.py`, `tests/test_training.py`
- Modify: `docs/run-status.md`（仅增加当前全范围执行记录）

**Interfaces:**
- Consumes: `build_model(upstream, config)`, `build_export(run, checkpoint_loader=...)`, 当前 `egoemg_repro.train` CLI。
- Produces: 官方 `emgformer/egoemg_emgformer_large` 配方；L 的脱敏名称与严格 11,490 点轨迹校验；独立短检查及生产运行证据。

- [ ] 写导出测试：L 的真实上下文契约可导出，错误轨迹分母仍被拒绝；不支持的实验仍失败。
- [ ] 运行新增测试，确认因 L 未支持而失败。
- [ ] 添加 L 配方，官方模块宽 384、8 层、12 heads、FFN 1,536；使用官方 200 epochs、batch 500、lr 0.00005，保留可比较的 legacy 数据族声明。
- [ ] 最小修改导出名称和固定上下文支持；不更改已有运行时训练算法。
- [ ] 运行真实官方 L 构建及 CPU 输入形状检查，核验 22 维和 510/0 上下文；运行完整测试集。
- [ ] 部署独立代码目录与源身份，实际 GPU 有界优化、完整验证、严格重载通过后，启动独立 200 epoch 生产训练；记录进程身份、日志、退出与存储余量。
- [ ] 提交本任务并记录证据。最终导出与完整审计等待真实生产结束，不能预先计为完成。

### Task 2: 全论文产物清单与官方来源裁决

**Files:**
- Create: `docs/full-paper-status.md`
- Modify: `docs/reproduction-matrix.md`

**Interfaces:**
- Consumes: 原 PDF、固定官方源码、三份只读研究报告、原四模型证据。
- Produces: Tables 1–15 / Figures 1–13 / 算法和数据依赖的逐项状态及证据路径，后续模型和分析实施顺序。

- [ ] 逐条核验研究报告与论文和官方文件，记录真实代码存在/缺失，不能把 schema 字段当作已获取数据。
- [ ] 建立完整状态表，指出 L 当前真实阶段，以及未运行模型、EMG2Pose、标签训练、遮挡和手势分析。
- [ ] 记录 PDF 与代码裁决，例如输入、标签转换与统计口径；无来源时保留待核验状态。
- [ ] 检查文档链接和 git diff，提交。

### Task 3: 后续独立实施计划

**Files:**
- Create: `docs/superpowers/plans/2026-10-26-full-paper-models.md`
- Create: `docs/superpowers/plans/2026-10-26-full-paper-data-analysis.md`

**Interfaces:**
- Consumes: Task 2 的已核验官方 API、依赖、数据和资源。
- Produces: 可执行的模型适配、额外数据获取与统计/遮挡/标签流程计划，继续原全范围目标。

- [ ] 对每个真实接口写精确实现任务、输入输出和验证要求；对缺失官方模型/训练规格记录所需来源。
- [ ] 优先落实可获得的 beta 统计字段与官方分析所需字段，以及无账号要求的 EMG2Pose 资源；mini 仅用于工程验证。
- [ ] 以实际存储为约束设计获取顺序；不能删他人数据，不能用子集训练替代全量目标。
- [ ] 继续实现与运行；保持总目标 active，直到全部必需论文内容实际验收。

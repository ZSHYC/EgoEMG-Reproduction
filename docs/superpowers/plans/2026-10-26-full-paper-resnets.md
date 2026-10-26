# Larger ResNet Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为全论文目标补齐官方 ResNet-50/152 模型的实际构造、预训练来源、配方与后续完整训练入口。

**Architecture:** 继续组合官方 `fusion/vision_resnet18`，用经过白名单验证的显式 `module.backbone_type` 覆盖调用已有官方模型类；不在本项目重写 ResNet。独立 RN50/152 YAML 在固定版本缺失，清楚标注派生配方。

**Tech Stack:** Python、PyTorch/torchvision、Hydra、pytest。

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`

## Global Constraints

- 当前官方版本固定 `f7cbd048b7092139852270491d3859cd472e0311`；有歧义优先官方代码。
- 不修改 `train.py`、旧配方、已有结果或服务器冻结的 L 运行源码。
- 不把派生配置称为作者已发布的独立 YAML；没有实际训练不得称完整实验完成。
- 默认 RN18 和已有融合构造行为保持兼容；不关闭任何初始化来源/严格载入检查。
- 无随机初始化冒充 ImageNet 预训练；恢复不隐式联网。

## Review Focus

- 非法或错误 family 的 backbone 配置必须在实例化/联网前拒绝。
- resume 权重形状错误必须 strict load 失败，不能悄悄使用另一个 backbone。
- 不得把 RN50/152 报告成 RN18。
- torchvision 权重枚举、实际来源与已加载张量摘要均按真实 backbone 记录。
- batch 来自 PDF 而学习率来自官方 RN18 的派生依据须明确，不虚构原始训练身份。

### Task 1: RN50/152 构造、报告与配方

**Files:**
- Modify: `egoemg_repro/visual.py`, `egoemg_repro/report.py`
- Create: `configs/resnet50.json`, `configs/resnet152.json`
- Test: `tests/test_visual.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `build_visual_model(upstream, config, vision_checkpoint=None, emg_checkpoint=None, resume_state=None)`，现有 `pretrained_weight_identity(model)` 与报告接口。
- Produces: 新可选 JSON 字段 `vision_backbone_type`，仅 vision 下允许 `resnet18`、`resnet50`、`resnet152`；该字段存在时组合 `module.backbone_type=<value>`。未指定时保持原默认。report 状态/安全配置按实际 backbone 命名。

- [ ] 写真实官方 RN50/152 构造、中心预测反传/严格重载测试；在任何实现修改前运行，确认当前构造仍为RN18或缺配方而失败。预训练关闭，测试不得联网。
- [ ] 写非法类型/融合错误使用 override 的拒绝测试，以及 RN50/152 状态标签和安全配置保留测试。
- [ ] 实现受限 backbone override；复用现有 ImageNet 枚举和张量哈希，不复制官方网络。
- [ ] 新配方继承本项目RN18队列/150epochs/AdamW lr0.0001、WD0.01、cosine eta_min0.00001、bf16、clip1、seed42；RN50 batch400、RN152 batch180来自PDF Table8，说明独立YAML未发布、这是派生legacy对照。`official_batch_size` 与 `batch_size_note` 明确批量的论文依据，不能说已实测容量。
- [ ] 核验官方 RN50/152 是24,568,406 / 59,204,182总参数（含新角度头）；不将其等同论文仅backbone统计。执行完整pytest。
- [ ] 自检diff，写实际命令、RED/GREEN及完整测试结果到私有报告；不启动训练/不提交git，交主线程部署独立冻结目录。

### Task 2: 真实 GPU 验收与完整实验（主线程）

**Files:**
- Runtime: 独立服务器运行目录、配置/环境/进程/退出记录。
- Update: `docs/full-paper-status.md`, `docs/run-status.md`

**Interfaces:**
- Consumes: Task1 两份配方与经过核验的代码；当前真实数据、全部crops、官方ImageNet资源。
- Produces: 两项完整150epoch训练、验证选模、最终测试、严格导出和锁定审计的真实产物。

- [ ] 实际获取预训练权重并记录来源与张量摘要。
- [ ] 各自真实数据两步优化、全部验证重载和GPU容量核验通过后启动完整训练；与L使用不同GPU/目录。
- [ ] 验证完整预算、完整独立验证、best与最终测试，再导出和一次锁定审计；运行中不得算完成。

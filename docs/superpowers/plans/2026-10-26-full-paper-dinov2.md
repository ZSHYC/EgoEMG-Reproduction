# DINOv2 Vision Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接通未修改的官方 DINOv2 ViT-S/B/L 视觉单模型、真实初始化身份、离线严格恢复与安全报告。

**Architecture:** 组合官方 `fusion/vision_vit_small`，用 family 白名单选择 `vit_small/vit_base/vit_large`，继续调用官方 `VisionViTPose` 和 timm。围绕实际 timm 权重 transport 作有界观察，记录成功执行的 HF/URL 路由、resolved cfg 和 loaded tensor digest；不把候选 URL 当作实际来源。

**Tech Stack:** PyTorch 2.2.2、torchvision 0.17.2、timm 1.0.15、huggingface-hub 0.28.1、safetensors 0.5.2、Hydra、pytest。

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`；官方配置 `upstream/config/experiment/fusion/vision_vit_small.yaml`；重建调查 `/home/zshyc/EgoEMG/tmp/official-expansion-recon-20261026.md` §6。

## Global Constraints

- 官方固定 `f7cbd048b7092139852270491d3859cd472e0311`，不改官方网络、不改融合、不启动 GPU 生产训练。
- RN18/50/152 与已有融合兼容；所有跨 family、未知/非字符串 backbone override 在 Hydra/联网前拒绝。
- ViT 的官方输入是 256×256，不因 DINOv2 pretrained cfg 的 518 改变输入。
- 全模型完整实验未运行；工程验证和派生配方不能称完整论文复现。
- 本项目保持 WL12000/stride1200/eval12000、独立 val 和无 repeat 的 legacy 队列；披露官方 WL7790/stride780/eval7790/repeat2 差异。
- S 官方 batch360、150epochs、AdamW lr1e-4/WD.01；B batch160、L batch50 来自 PDF Table8，其他配方从 S 派生，独立 B/L YAML 未发布。
- 不升级 torch/numpy；无随机权重冒充预训练；恢复时 pretrained=false、strict load、不隐式联网。
- `validate_initializer` 运行代码身份 guard 保持严格；改 train.py 会使旧 clean-S 与新运行身份不兼容，融合初始化另行裁决。
- 不 commit/push；父任务已授权 inline TDD 实现，计划写后直接执行，最终交父任务审阅。

## Review Focus

- ResNet override 放进 ViT experiment、ViT override 放进 ResNet、vit_huge/未知/non-string/fusion override 都须早拒绝。
- HF 下载失败应原样失败，不假称已自动尝试 Meta URL；URL 成功与 HF 成功的 provenance 必须区别。
- 随机、clean checkpoint、self-contained resume 不产生新 pretrained identity；resume 保留原始身份。
- 公开 export 不泄漏 cfg 内本机路径/任意嵌套文本，也不能将 ViT-B/L 命名成 ViT-S 或 ImageNet。
- 大模型验证不得和其他模型常驻叠加；S 真 256px forward/backward/optimizer，B/L 至少真实构造、参数与 schema 核验，资源不足明确记录。

### Task 1: Family dispatch 与三份配方

**Files:** `egoemg_repro/visual.py`, `egoemg_repro/report.py`, `configs/vit_small.json`, `configs/vit_base.json`, `configs/vit_large.json`, `tests/test_visual.py`, `tests/test_report.py`。

**Interfaces:** 保持 `build_visual_model(upstream, config, vision_checkpoint=None, emg_checkpoint=None, resume_state=None)`；新增受限 ViT 值到现有 `vision_backbone_type`；report `_model_name/_safe_config` 使用相同 family 规则。

- [x] 写失败测试：三配方存在与精确数值；ViT 名称、安全 config、跨 family/未知拒绝。
- [x] 跑上述测试，确认因功能缺失失败；保存 RED 日志。
- [x] 最小扩展 family 验证与 label 映射，创建明确声明 legacy/S-derived 的三份配方。
- [x] 跑选择测试，确认 GREEN；RN 原覆盖保持通过。

### Task 2: DINOv2 实际初始化身份与安全导出

**Files:** `egoemg_repro/visual.py`, `egoemg_repro/train.py`, `egoemg_repro/report.py`, `pyproject.toml`, `tests/test_visual.py`, `tests/test_training.py`, `tests/test_report.py`。

**Interfaces:** `pretrained_weight_identity(model)` 保持 ResNet 行为；DINOv2 返回 family、timm model/version、成功 transport 的 source type/location、resolved pretrained cfg、与现有同格式 backbone tensor SHA256。`initialization_identity` 的 ViT policy 为 `DINOv2 backbone`。

- [x] 写失败测试：真实 timm loader + 仅外部 transport fixture 的 HF provenance、marker tensor、digest 变化；transport 失败不能随机回退；offline fresh request 早拒绝；random identity 不得冒充预训练。
- [x] 写失败测试：train policy/resume 原身份保留；safe export 保留 DINOv2 已知字段并剔除 cfg 私有路径，非法来源拒绝。
- [x] 运行 RED；最小实现 successful transport 观察（恢复原函数）、family-aware private/public identity、pinned 可选依赖。
- [x] 运行 GREEN；`validate_initializer` 原 strict guard 测试继续通过。

### Task 3: 真实 CPU schema、优化和 strict resume 验收

**Files:** `tests/test_visual.py`, `tests/test_training.py`；私有证据 `tmp/dinov2-implementation-report.md`。

**Interfaces:** 使用 Task1/2 的正式 builder；不 mock 网络结构，仅运输层 fixture 可代替联网权重。

- [x] 写/运行真实 ViT-S 256px 中心预测、反传、AdamW step、完整 strict resume/坏 schema 拒绝；现有官方模型行为无需新增实现，属于 Task1 dispatch GREEN 后的真实集成验证。
- [x] 先查 MemAvailable，B/L 分开真实 CPU 构造、embedding/head/schema/参数数目核验；不复制大 state，不做未授权 GPU 训练。
- [ ] 运行三修改测试文件和完整 `EGOEMG_UPSTREAM=$PWD/upstream /home/zshyc/EgoEMG/.venv/bin/python -m pytest -q`；保存所有失败或限制。
- [x] 自检 diff 和预训练路由源码，记录 RED→GREEN、实际参数/内存/依赖、未下载预训练和未做 B/L backward 等边界；交父任务 fresh review，不 commit/push。

## Execution rulings

- Ruling: 新增独立 `dinov2` optional extra，保留原轻量 `vision` extra；official-cpu 安装 dinov2，unit-light 不因此拉取未固定的 Torch。父任务另授权增加 `emg2pose=['h5py==3.11.0']` 及两个 CI install 点。
- Ruling: clean vision checkpoint 的 policy 明确为 `clean vision checkpoint`，不因请求 pretrained=true 而冒充新 ImageNet/DINOv2 下载；旧 policy 仍可导出，fusion 行为未改。
- Ruling: B/L 真实构造和 S 反传属于复用官方网络的集成验证，首次即通过，不伪造 RED；新增 dispatch/config/name/identity/export 行为均有实际 RED→GREEN 日志。
- Ruling: 按父任务协调，等待并行 EMG2Pose adapter GREEN 后统一跑全套，避免把其进行中的 TDD RED 当作本任务回归。
- Ruling: root 明确接手无 Torch light 环境验收、统一最终全套与独立 review；本实现者以 focused142通过和文件指纹冻结交接，不等待该环境或占用 review 槽位。
- CI follow-up: root 提供真实 unit-light 45项 shape_statistics 因无 upstream 而失败；新增 pinned-source bootstrap 和 report extra，保留无 Torch/无全局 integration opt-in 的轻量层级。YAML/脚本语法检查通过，修复后真实 light 全套由 root 执行。
- Independent review fix: effective ViT recipe 限制真实初始化family，拒绝改policy藏RN/DINO包；resolved cfg要求8个真实核心字段，已知矛盾拒绝、未知私有字段仍脱敏。35 failed/6 passed RED → report整文件121 passed GREEN；S/B/L合法fixture与实际timm1.0.15 cfg一致。仅改report.py/test_report.py，原reviewer复审与root组合回归待接续。

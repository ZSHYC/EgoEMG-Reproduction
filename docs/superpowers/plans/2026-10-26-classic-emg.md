# EgoEMG vemg2pose / NeuroPose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax. This document authorizes no new training run by itself.

**Goal:** 接入未改动官方 vemg2pose 与 NeuroPose，完成独立训练、完整验证选模、严格恢复和可核验导出，明确区分新官方配方适配与已存在 legacy 协议。

**Architecture:** 推荐新增独立 `classic_emg_*` 路径：保留实际官方网络和 Lightning `_step`、优化器、AMP 与回调；外围只负责受限合成、旧数据的明确布局/固定归一化、权威角度 mask、独立分区、身份和产物核验。不修改现有 `train.py/data.py/visual.py/report.py`，不扩大它们已批准的 guard，不 monkeypatch 生产代码。

**Tech Stack:** 固定 upstream `f7cbd048b7092139852270491d3859cd472e0311`；Torch 2.2.2、Hydra 1.3.2、PyTorch Lightning 2.5.0.post0（新运行明确记录版本）、原官方 NumPy/Torch transforms 与 metrics。GPU 16-mixed 由真正 Lightning GradScaler 提供；CPU 工程验收明确为 float32。

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`；已有 `<original-project>/tmp/official-expansion-recon-20261026.md` §1/3/4/9；`tmp/emg2pose-adapter-recon.md` 仅作为分离协议的先例。本文件是只读调查和计划，尚未实现、未完整训练。

## Global Constraints

- 论文歧义优先固定官方实际代码及 resolved config；保留有冲突的原字段和具体裁决，不以注释覆盖执行行为。
- 原四完整实验、当前 L/RN50/RN152 的 legacy `.01` 损失协议及现跑 job 保持原身份；它们不等价于当前官方 Hydra 合成的 `.02` 损失。当前 DINO JSON 同样仍是 legacy `.01`，不能因官方 DINO 实验有 `.02` 而重命名已有产物。
- 新模型/窗口/归一化形成独立实验族。不能合并为旧 12,000 点、8 通道、训练区间 per-hand/per-channel 归一化的同一控制比较。
- 完整独立 validation 决定 best；冻结 test 只在终止后以该 best 执行。原官方 classic `allowed_splits=[val]`/`test:[]` 不能替代这个要求。
- 不弱化 `validate_upstream`、`validate_initializer`、runtime SHA、配置/分区/数据/归一化/selected-best 检查；旧结果和旧运行根目录不可改写。
- 本次仅写本报告；无生产修改、共享环境安装、大权重下载、GPU/服务器工作或完整训练。

## Review Focus

1. YAML 声称16槽但 VEMG 首层8ch：真实错误必须先 RED；不改卷积权重形状。
2. tuple 已裁剪/插值：第二次裁剪或采用20维元数据会悄悄改变标签人口。
3. 16槽空位先零填、再 scalar 归一化：空槽并非最终零；增强顺序必须与数据 reader 一致。
4. `16-mixed`/clip0/空标签批次/早停：禁止以 JSON 名称代替真实 Lightning 语义。
5. best-loss 与 best-MAE 不同、恢复后旧 best/回调/scaler 或身份丢失：必须用故意相反的验证排序和负例证明。

## 1. 已有证据与新的重要核实

固定 upstream 本次 `rev-parse HEAD` 为规定 commit，`git status --porcelain -- egoemg config` 空。未改官方树。

| 项目 | vemg2pose | NeuroPose |
|---|---|---|
| 官方 experiment | `emg2pose/regression_vemg2pose_egoemg` | `emg2pose/regression_neuropose_egoemg` |
| 实际 class | `VEMG2PoseWithInitialState` | `PoseModule(featurizer=NeuroPose)` |
| 参数（含头） | 5,963,948 | 6,354,969 |
| 实际输入 | `(B,8,T)` | `(B,16,T)` |
| context | 1790/0 | 0/0 |
| 类 metadata out_channels | 20（基类默认、并非真实输出） | 20（同左） |
| 实际输出 at WL7790 | tuple：`(B,22,6000), (B,22,6000), (B,6000)` | tuple：`(B,22,7790), (B,22,7790), (B,7790)` |
| 官方 batch | 600 | 256 |
| 默认增强 | ExtractToTensor → RotationAugmentation | 同左 |

两项真实 CPU forward 已成功，`target == original_target[...,left_context:]` 逐位相同、预测 finite。VEMG 输入16ch真实失败：`weight [256,8,11] expected 8 channels, got 16`；NeuroPose 输入8ch真实失败：最后 spatial pooling 输出宽度0。不得将这两个负例当“可用配置”。

VEMG 不使用真实初始位置：`provide_initial_pos=False`、22维零初态；64特征+22状态 → SequentialLSTM 86→44，hidden512、2层、scale.01。每 trajectory reset，rollout50Hz，`round(500*50/2000)=12` 个 rollout step 用位置、随后积分速度（Python round 的实际结果，不写成推测13步）；WL7790裁后6000点对应150 rollout step，再由官方插值回6000。NeuroPose 16槽经时间总池化320，原 featurizer 输出7680，再由 BaseModule 插值回7790。

**新增 loss 证据：不能只读 experiment 内的两行。** `config/base.yaml` 已有 `landmark/fingertip:.01`，上述 EgoEMG experiments 追加 `fingertip_distance:.01`；Hydra 合并后两者都存在。`lightning.py:644` 起兼容映射逐 key 相加，两个键都读同一 `*_landmark/fingertip`，有效系数 **.02**。本次执行原 AST 中的原始 assignment+for loop，以 MAE=2/FK=3 得 2.06。没有改 loss，也没有去重别名。附录给出每个模型的 resolved 原字典、实际 reducer 输出与源码 hash。

| 官方合成族（逐个实测） | raw loss_weights（另见完整JSON） | 有效 fingertip | monitor / budget / EarlyStopping |
|---|---|---:|---|
| EgoEMG EMGFormer S/M/L | mae1 + hierarchical.01 + legacy-alias.01 | .02 | val_mae / 200 / patience200 |
| RN18；同experiment显式50/152 backbone派生 | 同上 | .02 | val_mae / 150 / patience30 |
| RN18+S 8ch | 同上 + delta_reg0 | .02 | val_mae / 150 / 无 EarlyStopping |
| DINOv2-S | 同上 | .02 | val_mae / 150 / 见附录实际callback |
| vemg2pose / NeuroPose（含两项 with_aug） | 同上 | .02 | val_loss / max100 / patience50 |
| EMG2Pose S/M/L | 仅 mae1 + hierarchical.01 | .01 | val_mae / 250 / 见附录 |

当前 `train.OfficialLoss` 和现有 JSON 只实现 `.01`，这说明其与当前官方合成 loss 不等价，**不说明已完成数字失效，也不授权改写旧结果**。新 classic 使用原 raw dict 和原 Lightning reduction，不能只写 `fingertip_weight=.02` 然后丢弃双key来源；汇总有效系数作为审计字段。

**官方 Lightning 的行为需要保留：** batch masked-L1 + FK 的 `val_loss` 经 `self.log(..., batch_size=B)` 按 batch-size 加权的 epoch mean；不是全局有效角度数加权，也不是 degree trajectory-MAE。两者均可报告，只有前者选模。全空 label 的官方 graph-connected zero 仍会经过自动优化（AdamW 可能因 weight decay 改权重），现有 legacy runner 会 skip，该差异不能照抄成新 classic。官方 LandmarkDistances 接受22通道，本次零输入22/20都成功；随机正例仍应在实现测试中比对，不能沿用旧调查“22一定报错”的未验证断言。新 classic 直接用官方 `_step`，不替换 FK 实现。

## 2. 最小、可审阅的官方优先裁决

### 2.1 模型与布局

- **推荐 VEMG：保留未改8ch网络，把数据布局显式裁决为 `target_hand`。** 这是修复官方16槽数据/8ch模型内部矛盾的最小数据适配，必须记录 `declared_layout=emg2pose_interpolate16`, `effective_layout=target_hand`, `reason=official model Conv1d input=8`。不得声称原 YAML 原样可训练。另一条16输入卷积改造会改变官方 state schema，作为另一个架构变体，本计划不选。
- NeuroPose 保留16槽 `[10,12,0,1,2,4,5,6]`，单目标手8通道放槽、空槽原始零填、`channel_interpolate=False`；不是左右手拼接。不要把视觉 sparse16 的“已经归一化8ch后放槽”直接照搬。
- wrapper 校验实际类型、Conv/Linear schema、context、参数数目和一次真实22D输出；不改 `out_channels` 属性，不给两个不支持它的构造器传 out_channels。
- 两个 with_aug 已发布且 compose 同样双损失key。首轮先接论文基础classic（默认rotation），额外增强作为可明确命名的两份配置；不把未跑增强实验填成结果。

### 2.2 归一化和增强

实际 reader `_load_norm_stats` 优先 `egoemg__filtered`，固定资产 SHA 见附录：**mean .000832、std8.6551、epsilon1e-6**；asset note 为8,000,000样本 post-layout pooled stats，未证明这些样本正好来自当前独立训练区间。此处不能宣称 train-only 估计。当前 asset 的 per-hand条目只有 scalar，没有 `per_channel_mean/std`，所以即使 VEMG 改成8ch target_hand，当前官方 reader 也 fallback 到同一 scalar。不要擅自改 dataset_name 为 `egoemg_8ch` 或换 std12.275496，亦不能把 EMG2Pose 的18.49拿来用。

顺序固定：读取 filtered float32 `(T,8)` →（NeuroPose）原始零填16槽 → `(x-.000832)/(8.6551+1e-6)` → 原官方 resolved train transform → 转 `(C,T)`。默认 Rotation 在归一化之后对完整8/16轴 roll ±1或0；NeuroPose 不能只旋转活槽再复位。16槽空位归一化后约 **-9.6128286e-5**，训练旋转还会移动槽位；val/test只 ToTensor。不要强制最后空槽全零以“修复”这个值。

`with_aug` 的原顺序：ExtractField → Rotation → ToFloatTensor → channel mask(p=.25) → time mask(num_masks=0) → frequency mask(3) → Gaussian noise(SNR25–35,p=.5)。其标题“No normalization / only rotation”与实际字段不符，以 resolved data+transform 为准。

### 2.3 窗口、独立验证与终止

新族采用官方 WL7790、train_stride780、eval_stride7790、jitterFalse、无padding/重采样/数据repeat；重用现有 temporal holdout 算法生成**新**分区。它仍按完整row footprint排除train/val/test交叠，val_fraction=.12，左右手成对。wrapper 必须改写 manifest 中“train-only normalization”的描述为固定官方 asset、重算分区 hash；不能留下与实际行为相反的 provenance。保留 legacy 预滤波跨时间依赖、测试用户交集、生成标签来源等既有限制。

官方 dataset 原 allowed_splits 按窗口中心过滤，val不是本项目独立训练holdout、test为空；独立holdout属于明确适配。WL7790新test中心和标签人口不同于旧12k队列，不能直接拼旧排行榜称完全同人口；如果另需旧队列控制比较，再写单独12k派生配置，VEMG轨迹应为10210、NeuroPose12000，不能冒称官方WL。

经典实际 monitor **val_loss/min**、max_epochs100、每epoch完整validation、EarlyStopping patience50、min_delta默认0、check_on_train_epoch_endFalse、check_finiteTrue。保留早停机制，100是上限。首个val建立best后连续50次不改善，在第51个validation check触发；sanity checks不计入正式检查。记录实际 completed_epochs/stopped_epoch/stop_reason，不能为凑满100关掉回调或把早停epoch写100。非有限val触发停止不能算科学完成。cosine T_max仍为100，不因早停事后重排。S/M/L官方patience200在正常首轮best+200不改善的规则下通常超出200预算；不能据此把classic的50也删除。

### 2.4 FP16、clip0、身份

推荐直接用官方 Lightning 自动优化和真实 `precision='16-mixed'`，而非扩展现有只支持bf16的 `train.precision_context`。AMP原逻辑：scale backward → unscale optimizer → hooks/clipping → scaler.step（overflow可skip）→ scaler.update；scaler state应随checkpoint恢复。clip0在 Lightning Precision 中 `<=0:return`，不会把梯度归零；本次实际原 Precision clip0 保持grad2，而当前 `torch.clip_grad_norm_(...,0)`把grad2变0。生产FP16溢出由scaler处理并记录skipped update/scale；CPU float32工程分支做有限梯度检查；不把CPU通过称FP16/GPU通过。

新运行拒绝未知experiment、layout/normalization不匹配、任意module override、checkpoint/private分支非空和未声明resume。采用新runtime身份scope：哈希全部新classic源、实际调用的旧pure helpers、选定recipe、完整resolved module/loss/transforms、固定norm asset、数据/分区/标签mask policy、Torch/PL/Hydra版本。保留现有 strict guard；不导入旧clean S当classic初始权重。后续融合若受共享源码变化影响，必须另作兼容裁决/新clean训练。

## 3. 文件边界与接口（推荐路径）

本任务推荐不改现有runtime文件；允许的后续受控公共修改仅 pyproject/CI/docs 注册，新模块独立评审。与进行中的 `emg2pose_runtime.py` 不共享编辑。

| 新文件 | 责任与接口 |
|---|---|
| `egoemg_repro/classic_emg.py` | frozen `ClassicContract`；`compose_classic(upstream:Path, variant:str)->DictConfig`（4项白名单）；`build_classic_model(upstream, cfg)->torch.nn.Module`；`align_classic_output(model,batch,contract)->(prediction,target,angle_mask)`；`resolved_classic_identity(cfg,contract)->dict`；`ClassicModelAdapter(official_model,contract)` 仅适配tuple mask并代理context/align_mask/featurizer/decoder属性，不增加参数或改官方class。先validate pin/clean，再按实际官方factory使用 `_convert_='all'`。 |
| `egoemg_repro/classic_emg_data.py` | `build_classic_partitions(store, *, val_fraction=.12)->dict`；`classic_normalization(upstream)->dict`；`ClassicDataset(root,windows,*,contract,norm,transform)`，包装 `LegacyDataset(norm=None)`，仅改emg、保留每角mask与手/用户/分区；`build_classic_datasets(...)->dict[role,Dataset]`。不碰视觉数据。 |
| `egoemg_repro/classic_emg_train.py` | `build_classic_lightning(upstream,cfg,identity)->official.EmgPredictionModule`：调用真实 `egoemg.train.make_lightning_module(cfg)`，之后仅把 `module.model` 包为 `ClassicModelAdapter`；原Lightning class不改，身份/RNG由独立callback保存恢复；`ClassicProtocolCallback` 记录真实完整val/stop/overflow；`ClassicCheckpointIO` 复用既有15GiB reserve+原子安全落盘原则；`run_classic(args)->dict`/CLI。不替换官方 `_step`、configure_optimizers、precision plugin。 |
| `egoemg_repro/classic_emg_report.py` | `build_classic_completed(run:Path)->dict`、`build_classic_status(run)->dict`、严格独立复算CLI；独立schema验full或真实earlystop、best val_loss、完整全test、每角coverage与协议差异。复用 `metrics.AngleAccumulator`，不将classic送旧 `_trajectory_points` 白名单。 |
| `configs/vemg2pose.json`, `configs/neuropose.json` | 明确variant、WL7790、780/7790、batch600/256、seed42/max100/val_fraction.12、official scalar norm、16-mixed/clip0、val_loss/min、patience50、raw及effective损失；与实际Hydra字段逐项核验。两份with_aug配置放在基础闭环之后。 |
| `tests/test_classic_emg.py`, `test_classic_emg_data.py`, `test_classic_emg_training.py`, `test_classic_emg_report.py` | 分别承担以下RED→GREEN。纯配置/AST/metadata测试无需Torch；真实模型/PL测试单个importorskip，不能整个模块skip。 |
| `pyproject.toml` / `.github/workflows/tests.yml` | 新optional `classic-emg-runtime` 与现有EMG2Pose runtime保持同一已核验Torch/PL/Hydra/Scipy/Pandas/joblib版本，不升级torch/numpy；轻量job执行纯测试，official-cpu job明确装runtime并执行真实模型/PL测试。保留当前两层CI，不造第二套。 |

`ClassicModelAdapter.forward` 调用未改官方model，再调用 `align_classic_output` 验官方返回tuple3项、22D、context及target长度相符，保留官方已经插值的预测/target，**只**从batch.angle_valid_mask裁同一span（wrist独立），不第二次interpolate或以tuple的统一label mask替换。它是有意延续本项目per-angle有效性协议；官方reader丢弃wrist_valid、只用generated_label_valid，这一点要公开，不能声称所有mask行为逐位官方相同。

## 4. 可独立评审的 TDD 任务

### Task 1：官方合成与模型 contract

Files：Create `classic_emg.py`、`tests/test_classic_emg.py`。

- [ ] 先RED：`test_rejects_unknown_variant_or_module_override_before_factory`；`test_resolved_loss_keeps_both_raw_keys_effective_dot02`（实际原AST、不是自写去重）；`test_vemg_contract_records_declared16_effective8_without_network_edit`；真实 `test_official_models_forward_7790_and_bad_layouts_fail` 断言参数5963948/6354969、context1790/0与0/0、metadata20但输出22、长度6000/7790。`test_neuropose_factory_convert_all` 固定 Hydra list→Python值转化（默认instantiate会把ListConfig传pool导致TypeError，官方make_lightning_module实际用_convert_='all'）。
- [ ] 运行 `pytest -q tests/test_classic_emg.py`，确认因缺接口/错误contract真实RED。
- [ ] 最小实现受限compose/build及contract，无上游编辑；将声明/实际布局两项都写身份。
- [ ] 同命令GREEN，冻结本任务diff与测试证据给root；不自行commit/push。

### Task 2：固定归一化、真实布局与新分区

Files：Create `classic_emg_data.py`、`tests/test_classic_emg_data.py`，基础两份JSON。

- [ ] 先RED：8ch取目标手；16ch准确anchors、先zero-fill再scalar（空槽约-9.6128286e-5）；左右手不拼接；两手都不取当前train-row norm；asset SHA/mean/std/epsilon不符拒绝；default rotation在完整layout后并与原Compose同seed逐位相同；with_aug若启用，与原resolved整链同seed相同。验证train/val/test手/用户/mask不被改写。
- [ ] `test_classic_partitions_preserve_disjoint_roles_and_new_grid` 验7790/780/7790、双手paired、中心分区、同row无交叠、val来自官方train、test未用于norm/selection、新hash和正确norm描述；故意放入窗口重叠必须失败。
- [ ] `pytest -q tests/test_classic_emg_data.py` 真实RED后实现包装器，复用旧 `LegacyDataset(norm=None)` 和 `build_partitions`，不改旧文件；GREEN。保留label单位rad与独立wrist mask。

### Task 3：官方 Lightning 训练/精度/早停/恢复

Files：Create `classic_emg_train.py`、`tests/test_classic_emg_training.py`；受控依赖注册。

- [ ] 先RED：tuple不得二次裁剪；不同finger/wrist validity仍保持；错误channel/tuple/target/context拒绝。`test_selection_is_full_batch_weighted_val_loss_not_mae` 构造两候选（低MAE但高FK），验证选择loss最小；不等batch/valid数量证明batch-size weighting而非global valid-count均值；max_batches工程标记永不eligible。
- [ ] 真实CPU一步证明clip0梯度未归零、参数更新；空label批次须按官方zero loss自动优化，不能套旧skip；对非有限loss失败，FP16梯度overflow采用scaler真实skip策略并记证据。
- [ ] 原 EarlyStopping 实测：第一check建立best；后49次未改善不停止、第50次停止；恢复wait_count/best/stopped_epoch；故意MAE改善但loss不改善仍按loss计数。budget最多100，提前停止写真实epoch；不把NaN earlystop标complete。
- [ ] `test_epoch_checkpoint_resume_restores_optimizer_scheduler_callbacks_rng_and_best`：小真实CPU网络/两份官方class短序列分别跑uninterrupted vs epoch边界严格resume，比较state schema/下一步预测、optimizer、scheduler、RNG及callback；不接受只model_state。原始/effective recipe、norm/data/partition/runtime任一变化在构造或读数据前拒绝。
- [ ] 以实际 `make_lightning_module`（不手抄其构造字段）+ Lightning automatic optimization完成GREEN，沿用 EMG2Pose Task3 的固定pin/严格resolved协议身份方法，但不耦合其20D数据入口。seed在model构造**之前**（披露对官方train先构造后setup_runtime的确定性顺序修正）。原 `loss_weights` 原封传入，不复写为单scalar。adapter引入的新state key前缀必须记录并严格重载，不冒称可直接接旧checkpoint；不通过strict=False掩盖。不得把PL的model/target通道自动截断行为当schema验收，外围先拒绝。
- [ ] checkpoint保留完整Lightning model/optimizer/scheduler/precision-plugin/scaler/loop/ModelCheckpoint/EarlyStopping +本项目identity/RNG；输出具有随last携带的selected-best snapshot（去递归）、能搬迁离线重建best，best/last混改失败。ModelCheckpoint IO必须15GiB余量+保守临时payload预算+atomic/fsync，禁止直接无保护写满盘。构造恢复不联网、不取任何预训练资产。
- [ ] CUDA专项后续在独立新GPU可用时做真实16-mixed forward/back/optimizer/scaler +严格重载（包括overflow skip与scaler state）；本次未做、CPU不能证明。生产run在通过前不得开。多worker/持久worker中断后augmentation RNG逐位continuation尚需专门证据：默认可审计支持epoch边界checkpoint；若不能证明worker RNG跨restart相同，明确“不保证位级训练续序”，不能用CPU workers0测试冒充workers12逐位续训。任何未来mid-epoch支持另开测试/保存sampler与worker state，禁止静默重放已完成更新。

### Task 4：严格导出与独立复算

Files：Create `classic_emg_report.py`、`tests/test_classic_emg_report.py`。

- [ ] 先RED：report不能把context固定510；WL7790轨迹VEMG6000、Neuro7790；中心是原窗口3895减leftcontext（2105/3895），不得取裁后轨迹一半。每关节/左右手/Gesture-User-Both/用户/trajectory-center覆盖sum/count可复算。
- [ ] 完整测试与每epoch val event覆盖对齐；选择 `best_val_loss`，MAEdeg是独立字段；最佳步骤必须确有完整val。正常max100或有限、真实、patience50触发earlystop均可完成，记录`configured_max_epochs`/`completed_epochs`/`stop_reason`，不能强制epoch==100。外部kill、max_steps、NaN/Inf/缺best或缺full test都不可eligible。
- [ ] 脱敏导出保留family、声明/实际布局、norm数值/asset SHA、raw/effective loss、precision实际使用、stop/selection口径和已有数据限制；拒绝谎称legacy.01与official.02等价，拒绝强行套原S/视觉身份。
- [ ] 只用冻结selected-best、原数据/分区/norm/模型source、严格schema做独立CPU/GPU复算，差异有阈值和实际dtype记录；test结果必须来自同一checkpoint digest。完整JSON与MD同证据；完成矩阵状态由root据实际scientific产物更新。
- [ ] 定向GREEN后才与root安排一次全套，避免与EMG2Pose Task3并行反复消耗。旧initializer mismatch仍必须拒绝的回归测试保持GREEN。

## 5. 后续真实运行与验收边界

拟议入口（目前不存在、未执行）：`python -m egoemg_repro.classic_emg_train --data data/legacy --upstream upstream --config configs/vemg2pose.json --output runs/classic-vemg-official-scalar-seed42 --device cuda:N`；Neuro对应另一新目录。GPU N由当时实际空闲资源审计决定，本次不预占任何GPU。

先有完整数据/独立分区审计与微型CPU/真实GPU工程验收，再按原batch600/256测容量；不够时保留原batch并记录实际batch调整，不能以梯度累积伪称相同BatchNorm批次。NeuroPose含BatchNorm，DDP多卡与单GPU的batch-stat语义不等价；官方devices[0..5]和其batch_size可能是每rank参数，单卡配方适配必须写world_size/per_rank/global batch，不能默认600是官方全局总量。训练预算是max100加原EarlyStopping，不是保证跑满100。

本次未取得新增数据，没有真实cohort性能；未做GPU FP16/GradScaler、backward/optimizer、官方Lightning `_step`全链、完整val/test或恢复验收。已有current legacy dataset/data receipts可复用，但不可消除legacy标签/预滤波和分区局限。

## 6. 本次实际 CPU 证据与限制

环境 `<original-project>/.venv/bin/python`，Torch2.2.2+cpu，CUDA_VISIBLE_DEVICES=''，OMP/MKL/torch线程均2；开始MemAvailable 3,251,924KiB。运行 `PYTHONDONTWRITEBYTECODE=1 ... python -u -B`，无缓存源码写入。真实模型循环：

```python
with initialize_config_dir(config_dir=str(upstream/'config'), version_base='1.1'):
    cfg = compose(config_name='base', overrides=['experiment=emg2pose/'+name])
model = instantiate(cfg.module, _convert_='all').eval()
batch = dict(emg=torch.randn(1,channels,7790),
             joint_angles=torch.randn(1,22,7790),
             label_valid_mask=torch.ones(1,7790,dtype=torch.bool))
with torch.no_grad(): pred,target,mask = model(batch)
assert torch.isfinite(pred).all()
assert torch.equal(target, batch['joint_angles'][...,model.left_context:])
```

真实成功命令session74413 exit0：VEMG .4304s、maxRSS274132KiB；Neuro .0410s、进程累计maxRSS279352KiB，输出/负例见§1。一次更早诊断因将配置子树从完整Hydra根复制后丢失`${to_tensor}`而exit1；另一次用默认instantiate NeuroPose遇ListConfig池化TypeError。修为真实官方factory的 `_convert_='all'` 后成功，没有改模型。

真实 callback/clip 命令session98053 exit0：仅在进程PYTHONPATH前置已有 `tmp/emg2pose-runtime-deps`，PL2.5.0.post0；EarlyStopping actual `_evaluate_stopping_criteria` 返回checks0/49/50分别(wait0,false)/(49,false)/(50,true)，Precision.clip0 grad2保持2。原torch clip0 grad2→0。EarlyStopping源码SHA `c348fb5f7a2c3255b1db8f0fc42f3c8cf8be4b989da115e48a1cb338486bc554`；Precision源码SHA `9651372ff90383370474c5d0a8c3d0cbff7c19c7fb50608126e71da2cb731c70`。PL混合精度源码确认16-mixed使用GradScaler，state_dict/load_state_dict保存恢复scale；GPU行为未实测。

尝试真实 `egoemg.train.make_lightning_module` 的隔离CPU全链命令session11388实际exit1，缺`joblib`（upstream utils顶层还依赖pandas）。未安装/伪造依赖或声称完整官方step已通过；Task3必须以正式隔离runtime补全后实跑。AST loss reducer只是原控制流的局部真实执行，不能冒称完整训练链。用本机已有PL并未改共享env。


## 执行安排与mask裁决（2026-10-26）

根代理按已批准全文目标先执行Task1，仅新增官方compose/model contract及其测试，保留原始模型tuple和2D mask，不改变官方network或旧运行。Task2–4仍需独立实现与审查；Task1不引入角度mask wrapper、不生成训练成绩或进入GPU。

后续训练必须显式区分`official_literal_loader`与`legacy_reproduction_angle_mask`：官方loader仅generated validity与本项目独立20+2 mask并不等价。原计划中的per-angle adapter是明确的数据协议适配提议，不可默认标成官方字面行为；正式准入前须决定并记录实际mask协议，保持用户“歧义优先官方代码”的要求。

11项实际损失合成与别名执行证据已另作[独立公开审计](../../../results/official-loss-audit.md)，不在本计划重复转储。早期完整Lightning factory导入缺joblib的诊断是历史限制；现有EMG2Pose隔离runtime依赖已补齐，新的Task1必须实际调用正确factory再验。所有私有机器路径在此使用逻辑位置表示。

# Table 4 / Table 6 Semantic Tables and Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This document is a private proposal, not authorization to modify a server or evidence that its tasks have run.

**Goal:** 从固定官方来源生成可追溯的60类词表、22维角度语义表，并在明确队列和分母下统计真实legacy gesture/active覆盖，不把定义表当作实测覆盖或模型误差结果。

**Architecture:** 两个小模块：`semantic_tables.py`只编译定义、来源和差异；`semantic_coverage.py`先投影根代理已完成的真实全帧审计收据，不重复扫描或下载，再允许独立追加冻结窗口的中心读取。公开JSON/Markdown用固定白名单投影，原路径、episode/subject映射和窗口清单仅留私有receipt。保留现有训练、标签和`results/completed`。

**Tech Stack:** Python标准库AST/JSON/hashlib、NumPy、pytest；既有Poppler只用于PDF参考核对。定义/统计实现不依赖Torch/GPU、不安装新包。

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`；当前状态`docs/full-paper-status.md`的Table4/6/14/15条目；复用已有私有`<original-project>/tmp/full-paper-inventory-20261026.md`，尤其§11，不重做全论文清单。

## 本轮已经核对的事实

- 只读调查于2026-10-26完成；未写生产代码/公共状态，未连接服务器、下载数据、改环境或运行训练。本轮只写此私有计划。
- 实际从PDF印刷p.15和p.20抽取完整页面并渲染目视核对Table4/6；另读p.19腕角约定。PDF SHA256为`124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725`。
- 官方checkout实际HEAD为`f7cbd048b7092139852270491d3859cd472e0311`，Git状态干净。
- 小型AST/JSON检查确认官方词表ID完整为0–59，三个族为30/18/12，成员总数和去重数均为60；50是`raw`，属于symmetric。此检查**没有读取实际gesture数组，不能证明实际60类均出现**。
- 实际legacy metadata的41个episode半开区间连续，`end-start==length`，合计66,161,725行，首start=0、末end=N；NPZ中无gesture/active数组。未输出任何原始参与者值。
- 调查后段根代理提供了本轮03:57:25 UTC的真实服务器全三数组重新hash/全同步行扫描收据`tmp/legacy-gesture-frame-audit.json`。本代理只读该本地收据并独立核平衡157条聚合记录，没有再次连接服务器或扫描数组。现在已有**实际60类覆盖**证据，来源是这份真实全帧审计，而不是官方词典。

## 字段到底有没有取得

结论：**已有完整服务端实物，且根代理本轮已在线重hash并全帧扫描；本机没有完整payload。不得再写“gesture/active尚未获取”或“本轮没有当前帧覆盖证据”。本代理复核的是当前收据，不把它说成本代理亲自重扫。**

| 字段 | manifest契约 | 本机实物 | 已有完整获取证据与位置 |
| --- | --- | --- | --- |
| `label_gesture_class.dat` | int32 `[66161725]`，264,646,900 bytes | 在`<original-project>`及本worktree的完整文件名检索均未找到；两个本地legacy目录也无此实体 | 服务器`<server-legacy-data>/`；根代理本轮当前审计重hash与既有transfer SHA `55a07d9d639483e54eef963a527d50554902d632eea4e42a29651f40f83b4212`精确相符 |
| `label_gesture_active.dat` | bool `[66161725]`，66,161,725 bytes | 同上，无完整实体 | 同一服务器目录；根代理当前重hash为`27bfef6dc99d98110613acc8685dad1ee5321f119098d9bac772c27d4f77d7cb`，与ledger精确相符 |
| `frame_split_id.dat` | int8 `[N]` | `<original-project>/data/legacy/`存在，字节数N | metadata映射0=train、1=user、2=gesture、3=both |
| 指角、generated validity、两手腕角/validity | 见下方Table6映射 | 本地legacy仅manifest、summary、NPZ、split及部分EMG下载；并无完整这些payload | 同两份旧protocol列有完整文件大小和transfer记录；服务端既有盘点确认腕角、generated labels已经拥有 |

证据层级与边界：

1. **当前主证据**`tmp/legacy-gesture-frame-audit.json`：`observed_utc=2026-10-26T03:57:25.322850+00:00`，SHA256 `c590c1a82831c297a75c3cd67b7de47ac8abb47a7e685faf2936cca79c7f08eb`。三数组实际shape/dtype/bytes/hash均记录并与ledger匹配。split文件SHA为`9ecef5c0057c639aa90e9574f9cf30f9de070b833f3a38bd2cd0f679c31b31ef`。每行共享class/active/split，`no_hand_duplication=true`，无窗口/用户数量声明。
2. **历史交叉证据**`<original-project>/tmp/extra-data-recon-20261026.md` §1为服务端实物盘点；`.../completed-20261025/rn18-protocol/provenance.json`及S protocol的`data.verified_transfer_records`保存相同bytes/SHA。旧protocol的`data.scope`仅metadata/大小/ledger，不能将它单独称当前重hash；当前根代理审计已补足这一步。
3. 私有`.../s-audit-success/posttrain.config.json`的`data_stats`记录这些实体的五元stat快照，生成代码`posttrain_audits.py:193–198`逐个校验实际stat后保存；五元组为device/inode/size/mtime/ctime，不能误把第一个数当大小。
4. 本地`assets_manifest.json`列出它们，只证明获取清单和schema，不单独证明已下载。本地legacy ledger只有3个metadata文件；extras ledger只有calibration及两个beta，不能冒称包含gesture。
5. 下一执行者直接消费当前主证据及其内容hash，无需再扫描、下载或拷回整套标签。只有后来明确新增窗口/中心分析时才需要获准的服务端point-read；其来源/统计范围单独保存。

当前审计可直接形成的真实帧统计（本代理由收据聚合行复核，而非再次读取payload）：

| 同步时间行口径 | 行数 |
| --- | ---: |
| 所有同步行 | 66,161,725 |
| active=true | 65,600,616 |
| active=false / class=-1 unassigned | 561,109 |
| 实际出现的定义类 / active内定义类 | 60 / 60 |
| raw50同步行 | 86,952 |
| train / user / gesture / both | 49,409,330 / 6,469,595 / 9,020,820 / 1,261,980 |
| single / symmetric / asymmetric的已定义类行数 | 37,306,077 / 16,856,826 / 11,437,713 |

所有family行数合计65,600,616，加unassigned等于N。这些是时间帧数，不能乘二后改称独立动作/窗口/用户；raw排除后的已定义时间帧65,513,664也不是Table14/15误差样本N。

## 精确来源映射

下表路径相对本worktree；定义来自官方代码，PDF参考保留自己的页码/SHA。当前文件hash见文末。

| 产物字段 | 确切来源 | 输出时的解释 |
| --- | --- | --- |
| gesture ID/name/description | `upstream/scripts/paper/generate_per_gesture_appendix.py:25–86`，`GESTURE_NAMES` | PDF Table4没有逐类数字ID；ID由固定代码赋予，描述采用官方原文，并另留PDF描述/措辞差异栏 |
| family与30/18/12定义数量 | 同文件`:90–94`，`FAMILIES` | 显式集合，不能把30–47视为全部symmetric；33/34等属asymmetric，51/52属symmetric |
| raw排除边界 | 同文件`:76/:92`；PDF Table4 p.15，Table14/15后续59类范围 | 完整词表保留ID50；后续error59视图显式排50，定义族数变成30/17/12，实际出现类别和样本数另算 |
| 官方样本gesture标签位置 | `upstream/egoemg/datasets/egoemg_memmap_dataset.py:1683–1689` | `get_gesture_class(idx)`取解析后的窗口中心；缺字段时返回-1是宽松fallback，统计模块不能把它伪装真实全量标签 |
| 官方分析选择/误差 | `upstream/scripts/eval/analyze_per_gesture.py:62–123,128–145,226–263,323+` | 默认仅gesture split；分手采样；EMG取输出token中点；读取mask后逐样本MAE仍直接mean。与本项目masked/aligned中心指标不同，不能复用这个脚本生成本轮“定义/覆盖”冒称误差结果 |
| activity筛选的一个实际官方用途 | `upstream/scripts/paper/plot_paper_figures.py:199–213` | t-SNE使用label_valid & active & finite-all20，再丢<20帧类别；这是特定图的筛选，不是Table4覆盖的默认分母 |
| 手指0–19名称/顺序/组 | `upstream/egoemg/constants.py:11–47`，`JOINTS` | `NUM_JOINTS=20`只指手指，不应修改成22；指组与proximal/mid/distal都是源映射，非推测 |
| 22维完整名称 | `upstream/scripts/paper/plot_paper_figures.py:39–64` | 前20与constants一致，20=`WRIST_FE`、21=`WRIST_RU`；PDF Table6 p.20逐行一致 |
| 腕FE/RU的位置 | `upstream/egoemg/metrics.py:252–283` | 20=flexion，21=deviation；该metric若拿3D mask会把通道20 mask用于两腕维，不能据此断言21的独立有限性已处理 |
| 磁盘单位与转换 | `upstream/scripts/paper/plot_paper_figures.py:97–113`；dataset`:1864–1878`；`egoemg_repro/data.py:170–177` | 指角磁盘rad，腕pitch/yaw磁盘deg；模型统一rad；报告deg。不能把PDF“reported in degrees”当成磁盘22列全是deg |
| 手侧与raw mask | dataset各路径`hand_idx=0 if left else 1`；`egoemg_repro/data.py:173–177` | generated validity列0左/列1右；前20与各手generated有效性关联，末2与对应wrist validity关联；再逐维finite |
| canonical左手与角度生成版本 | 实物`manifest.left_hand_strategy=flip_local_z`、`generated_joint_angles_semantics`；`metadata_summary.generated_joint_angles_note` | 两手保持同一维度顺序。不要凭“左手”给角度整体乘-1；几何反射/正负轴须保留来源。legacy summary明确axis-angle projection，与PDF B.3 landmark IK不同 |
| 腕角符号 | PDF B.2 p.19 | extension/radial为正，左手法线L×F；这是论文定义，尚不能只读现有腕角证明原始marker算法确按此实现 |
| 名义指角限位 | `upstream/egoemg/UmeTrack/dataset/generic_hand_model.json`的`joint_limits[:20]`；IK脚本`:348–354` | 作为模型名义限位而非实测极值/legacy已约束证明。asset有22对limit，但不能把其最后两对±π/4自动当作独立mocap FE/RU的范围；这两维输出null并解释 |

## 论文与源码/实物的差异，必须同行展示

1. Table4把ASL1–ASL9合并印作一行，代码分成9个ID；需展开定义但保留该PDF来源行，不能声称PDF直接给出0–8。
2. 少量描述仅措辞不同：`nocontact_grab`在代码多出“object”；Kiss/MiddleOppo大小写、单复数不同；Checky、PinchWring代码语法更简略。官方描述优先，PDF描述另列，不擅自润色后声称逐字相同。
3. raw50明确在symmetric族中；名称“unconstrained”不足以推翻显式族群映射。`generate_per_gesture_appendix.py`的family/scatter函数只排负ID，不自行排raw；后续59类分析必须输入/配置明确排50，记录排除前后N。
4. Table6的“reported in degrees”指报告口径；实际磁盘混合单位。不存在一个现成的磁盘`N×22`数组，需从20D指角和两个腕字段拼接。
5. 当前官方memmap loader的普通路径`:1884–1887`读到wrist validity后丢弃，并说明未传loss；`:1905–1906`只surfaced generated hand mask。其center快路径同样只返回generated label validity。官方Lightning`:519–541`的额外腕遮罩针对缺腕数据集名称，不是本次legacy逐手腕有效性。**本项目已完成legacy**使用独立20+2、逐角finite mask，须分别标`official_literal_loader`与`legacy_reproduction_angle_mask`，不能悄悄统一后追溯改旧成绩。
6. legacy metadata明确角度来自axis-angle projection；PDF B.3是landmark-level IK。维度语义对齐不证明标注生成算法等同，也不证明2.8mm拟合误差在当前数据成立。

## Global Constraints

- 固定官方`f7cbd048b7092139852270491d3859cd472e0311`，验证clean tree与所读常量内容hash，不静默采用移动main。
- 不改变旧训练data/manifest/acquisition ledger、模型、mask实现、分区或`results/completed`；只新增定义/统计入口和独立产物。
- 原始参与者标识、episode/path/subject表、逐人结果不得进入公开JSON/Markdown；公开sources仅固定相对源码名和内容hash。
- 完整词表恒为60定义，raw50始终保留；实际覆盖数由实际数组决定，缺数组不得输出60/0填充式实测结论。
- 每项结果明确单位、source identity、选择集合、分母、排除及状态；先实现fixture RED/GREEN，再进行真实CPU验收。
- 本轮只写此计划；下列执行任务尚未实现。当前不接触服务器/大下载/共享环境。

## Review Focus

- family不是连续ID分段：33/34、50/51/52必须按集合分族；重复/漏分/未知ID明确失败。
- 词典数量、实际帧覆盖、窗口中心类别、手展开样本和有效角计数是五种不同量；全程避免用一个“N”覆盖它们。
- 左/右mask列不可交换，腕deg不可漏转/二次转，单个NaN不能把其他21维误作全有效或全无效。
- missing/partial字段与readonly查表fallback必须显式拒绝统计；定义产物仍可独立成功，但coverage标未计算而非零。
- 公共投影不能递归转储manifest/metadata/partition/source cfg；字段hash和私有映射分离，公开输出不得出现测试身份sentinel或机器路径。

## 分母契约（在实现前固定）

1. **timeline frames**：每个全局行只计一次，N=66,161,725。按raw split、class、active=true/false分别累积；-1允许作为显式unassigned桶。先报全部帧，再报active帧；active不是label-valid替代物。意外的其他负数/≥60类别须报schema错误，不能静默过滤。
2. **hand-frame opportunities**：同一时间行左/右各一个机会，合计2N；gesture/active是共享的1D时间标签，不是每手独立动作记录。可报告各手原generated/wrist validity真值数，须名为raw mask frame counts，不能称逐角finite后计数。
3. **windows**：只使用一个显式冻结的`partitions.json`，绑定其内容及内部partition SHA；每个半开区间一次，检验不跨episode、center=`start+(end-start)//2`、无重复区间。不同窗口重叠不应把timeline行统计重复加权。记录WL/stride/role/grid/jitter和拒绝数。`role=train/val/test`与原split=train/user/gesture/both分栏。
4. **window centers**：每个上述窗口在center取一个class/active，沿用官方取中心规则，不能换成majority。左/右展开前为window-centers，展开后为hand-centers，分别给出分母，后者通常是前者两倍但因明确过滤可减少。
5. **center angle validity**：对声明的单手中心，0–19为`generated_label_valid[row,h] & finite(finger_j)`；20/21为该手`wrist_angles_valid[row] & finite(pitch/yaw)`，变成22个独立valid counts。清理NaN后再判finite会造假；原数组不改。可另列“20指全有效”“两腕全有效”“22全有效”帧数，名称不可混用。
6. **masked error denominator**：本轮不产生模型误差。以后是每模型同配对人口的`sum(abs_error_rad)/valid_angle_count ×180/pi`；trajectory含时间/角维，不能与每中心22角分母或window个数混用。官方literal无mask逐样本mean须单独标协议。
7. **59类后续视图**：对已明确的实际population再排raw50，保存`before_count/raw_excluded_count/after_count`，未观测类保留零覆盖定义行而非编造误差0。不能将旧legacy 2,792手中心/2,688有图人口声称为PDF Table14的4,278样本。

## Task 1: 带版本的词表和22维语义表（本机可立即实现）

**Files:**
- Create `egoemg_repro/semantic_tables.py`。
- Test `tests/test_semantic_tables.py`。
- 真实运行后Create `results/dataset-semantics/gesture-vocabulary.json`、`gesture-vocabulary.md`、`joint-angle-semantics.json`、`joint-angle-semantics.md`。

**Interfaces:**
- `build_semantic_tables(upstream: Path, metadata_root: Path) -> dict`：只读固定常量、模型名义limits，以及实物manifest/summary的白名单语义字段。AST读取常量而不运行paper main；FAMILIES中的`list(range(0,30))`用仅此语法的安全解析，禁止对任意AST `eval`。
- `export_semantic_tables(upstream: Path, metadata_root: Path, output: Path) -> dict`；CLI `python -m egoemg_repro.semantic_tables --upstream PATH --metadata-root PATH --output PATH`。
- 每gesture行含`id/name/description_official/family/paper_reference/analysis59_excluded`；metadata含`definition_count=60`和family definition counts。本定义导出自身的`observed_class_count=null`、`coverage_status=not_computed`，实际帧覆盖链接Task2产物；不能因读取不到payload而置0。
- 每angle行含`index/name/finger/joint/motion/groups/disk_field/disk_component/disk_unit/model_unit/report_unit/hand_index/mask_policy/nominal_limit/source_reference`；wrist名义model limit为null。PDF引用明确是固定人工核验的表引用，非CLI每次重新打开PDF。

- [ ] **先写RED：** `test_complete_60_definitions_keep_raw_and_noncontiguous_families`断言0..59全覆盖、30/18/12无重叠、33 asymmetric而50/51/52 symmetric、raw不丢。
- [ ] **先写RED：** `test_22_semantics_preserve_finger_order_units_and_mask_sources`用手写索引序列核验0–19，20pitch=FE、21yaw=RU，hand列0/1，rad/deg/rad/deg四层明确；不得仅拿生产列表与自身比较。
- [ ] **先写RED：** `test_nominal_finger_limits_do_not_become_mocap_wrist_limits`保留前20 source limits、末2 null；`test_pdf_descriptions_are_reference_not_unverified_equal`覆盖ASL合并行和Checky措辞差异。
- [ ] **先写RED：** 错commit/dirty官方文件、重复family/缺ID/错schema、公共输出PRIVATE身份及私有路径sentinel、未知manifest字段不可泄露。
- [ ] 运行针对测试，确认接口缺失/行为缺失导致失败，保存私有RED日志。
- [ ] 实现最小AST读取、严格schema/来源检查、显式公共白名单与JSON/Markdown导出；不改现有dataset或metrics。
- [ ] 运行针对测试到GREEN；用本机真实metadata和固定源码实际导出；人工逐行对照本轮已核验PDF页（60定义/22索引/族归属/单位差异），检查JSON/Markdown来源hash、无路径/身份泄露。
- [ ] 交独立review；根代理决定集成/commit。此任务只可称“Table4/6定义产物已核验”，不能将实际覆盖或Table14/15误差置为完成。

## Task 2: 直接发布已验证帧覆盖的聚合产物（本机可立即实现）

**Files:**
- Create `egoemg_repro/semantic_coverage.py`。
- Test `tests/test_semantic_coverage.py`。
- 从已有真实全帧审计成功编译后Create `results/dataset-semantics/coverage.json`、`coverage.md`；不重扫数据。
- 私有run receipt用`tmp/semantic-tables-run.json`；路径/原映射/partition原件不公开。

**Interfaces:**
- `build_frame_coverage(audit_path: Path, *, upstream: Path, metadata_root: Path) -> dict`：验证审计schema、三个字段的dtype/shape/bytes、布尔值类型、split/class域、无重复聚合键、N和每组计数平衡，以及审计明示的hash/ledger匹配状态；再与Task1定义/实物manifest N和split映射核对。不能仅相信`actual_class_ids`，要从计数重新推导并检查一致。
- `export_frame_coverage(audit_path: Path, *, upstream: Path, metadata_root: Path, output: Path) -> dict`；CLI `python -m egoemg_repro.semantic_coverage --frame-audit PATH --metadata-root PATH --upstream PATH --output PATH`。明确`computation_input=verified_frame_audit`、`source_array_rehash_performed_by=recorded_audit`和审计UTC/hash，本模块不冒充重新读取过原数组。
- 返回`timeline_frames`、`full60_view`、`exclude_raw59_frame_view`和平衡表；`window_centers`、`hand_centers`、`raw_mask_hand_frames`、`center_valid_angles`均为`status=not_computed`，计数null。可以解释双手机会为2N的派生量，但不得伪装实测有效手/角数量。无预测/MAE字段。

**独立可手算RED fixture：** class=`[0,0,1,50,-1,33,33,59]`，active=`[T,F,T,T,F,T,F,T]`，split=`[0,0,1,2,3,2,2,3]`，在测试内写出对应聚合记录和工程来源标签。应得8 timeline帧、7 known/1 unassigned、5 active；class0=2、class1=1、raw=1、class33=2、class59=1；59帧视图known=6、active known=4。实际覆盖仅5个定义类，必须与60个定义分开；fixture不能冒称真实release验收。

- [ ] 写上述fixture计数、聚合记录顺序不变性、全60定义与实际5类区分、frame不能自动变window/hand-center的RED测试。
- [ ] 写缺字段描述、错shape/dtype/bytes、`matches_acquisition_receipt=false`、非法split/意外class、重复聚合键、负/非整计数、count总和不平衡、缺identity/dirty源码拒绝的RED测试；缺审计不返回全0/60实测覆盖。
- [ ] 写隐私RED：向manifest/审计插入身份/路径sentinel，公开所有aggregate不转储未知字段；私有receipt路径明确留私有。
- [ ] 运行针对测试确认RED，实现最小只读聚合投影，GREEN后独立review。
- [ ] **真实本机验收：** 读取当前`tmp/legacy-gesture-frame-audit.json`，核其文件SHA与本计划记录一致；重新累加157条聚合记录，而非重新扫描/下载原数组。输出必须由计数推导出60 observed/60 active-observed、active 65,600,616/561,109及上述split/family总和。
- [ ] 检查`sum(all buckets)=N`，已定义类行+unassigned=N，排raw前后守恒；保留raw50=86,952。所有窗口/单手中心/有效角计数仍null，不以N或2N填充。
- [ ] source/audit读取前后hash稳定、JSON/Markdown可读、隐私检查通过后发布aggregate。由根代理统一项目light/full检查并决定commit，不重复训练、清理数据或修改旧结果。

## Task 3: 后续追加冻结窗口/中心及mask审计（独立范围，非本轮帧覆盖的前置）

**Files:** 后续Modify `egoemg_repro/semantic_coverage.py`与其测试，新增私有`tmp/semantic-center-audit.json`。Task1/2已可独立验收，不应等待此项或再扫全帧。

**Interface proposal:** `collect_center_coverage(data_root: Path, partition: Path, *, upstream: Path) -> dict`，在后来允许的服务端只读CPU环境执行；只点读选定冻结`runs/.../partitions.json`的class/active/split/20+2角和对应validity，不重新生成分区、不再全行扫描class数组。原大数组使用现有transfer identity+读前后stat保护，并记录实际中心字节摘要与point-read scope，不声称本次全文件重hash。

- [ ] 先RED：两episode `[0,4)`、`[4,8)`的两个4长窗口中心2/6，对上方fixture应为2 window-centers（1、33各1）、4 hand-centers，只有第一个center active；跨episode、duplicate或错center窗口失败。
- [ ] 先RED：手指π/2原样，腕90/-45度转成π/2/-π/4一次；左指valid/腕invalid、右指invalid/腕valid不串；插入单维NaN只影响该角finite mask，清理后不能重判为有效。
- [ ] 实现后对首/中/末及边界中心与`LegacyDataset`相同样本的单位和mask独立比较；窗/中心/有效角三套分母分栏。选择旧冻结测试人口时对照2,792手中心；其他grid报实际，不能调参迎合4,278。
- [ ] 新审计验收后再补coverage中对应`not_computed`栏，保留此前帧审计身份与时刻。模型误差、trajectory对齐和59类三模型配对评价仍是之后的派生评价，不因中心标签已读取就标为完成。

## 当前真正缺的内容与下一步裁决

- **并不缺词表/维度定义或当前帧覆盖证据，也不需要先取大数据。** Task1/2均可在本机直接实现；Task2只消费根代理当前真实全帧审计，不再扫描或下载。
- **窗口/中心/手mask统计仍缺。** 当前三字段审计不包含冻结window/hand-center/有效角分母；Task3独立追加。不能把实际60类帧覆盖推广为每个split/每手/选定测试中心均有60类。
- **三字段当前重hash已有，其他大角数组/腕有效性本轮未重验。** 仅后续中心point-read时检查其现存大小/身份/稳定性；不阻断当前定义和帧覆盖产物。
- **缺严格对应原v1 Table14/15的样本清单、逐角/逐样本三模型预测与未舍入误差。** Task1/2不解决它，未来从锁定best一次统一派生评价，不能从已有总体均值还原逐类误差。
- **缺legacy与PDF landmark IK/腕marker算法等价证明。** 定义表保留正面的版本差异；不因22维顺序吻合而宣布转换/标注质量复现。
- **腕实际物理限位没有对应证据。** 不能从generic hand model最后两行机械填入mocap腕表；本轮输出null及原因即可完成真实语义表。

## 本轮所读核心来源SHA256

| 来源 | SHA256 |
| --- | --- |
| 实物manifest | `ebcef7877444253869845b9337b7c0df2b16b0b6f55d83c58d6c3c03bae8ac09` |
| 实物metadata NPZ | `cb42886b5ca80a070d3b79f5e920cd01e364572e038ef11d5643e9ab73528a57` |
| 实物metadata summary | `34736a9ef3c311a0ddc4a060d6a996c6c850c26f90d1cb32856c9231fe87be62` |
| 官方constants.py | `01f4ad9d6be5b587d46d28b01c2325d584a593a63bd14c79dd0f8e2a1e561099` |
| 官方generate_per_gesture_appendix.py | `590b85ee7f1675e82ca7cf02fa72464b6c3c335b068467ff8d35484fd91ecf1d` |
| 官方analyze_per_gesture.py | `448e83f7a3698334d1813fda8ce0b7303672f9d95740442157ded4aa244d28a7` |
| 官方plot_paper_figures.py | `0da286e14f434d8b66a3fa7cb297b4175a3b77f899ef1c7642409936cd755289` |
| 官方egoemg_memmap_dataset.py | `9f8999ccc6a2d964713ff85886781ecc4d41ddb18f2986e0fbe5c6cc11236060` |
| 官方generic_hand_model.json | `2eb8029a8e1cedc2cf5b40c14c0955e2004277024df3c6304ce0bb74106a7c31` |
| 现有egoemg_repro/data.py | `de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e` |
| 现有partitions.py | `ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678` |
| RN18旧protocol provenance | `ae260b37ea588f188d26619c54ca2123a5c7d01de652b56b4d5057c02e4bd871` |
| S旧protocol provenance | `dedc2f4fe764d011dc61e769ede70ab6b510568b563aa818318ad9d93c23c381` |
| extra-data-recon旧实物盘点 | `0292fb425ba67e7d177e96b7101ab607a5eaec1e78ba3c9ee664eb86e1ef0926` |
| 当前legacy-gesture-frame-audit.json | `c590c1a82831c297a75c3cd67b7de47ac8abb47a7e685faf2936cca79c7f08eb` |

以上是当前源码/元数据身份、根代理当前全帧审计身份和历史收据身份，各自作用域已注明；未将历史ledger当作本代理读过的完整数组。此计划只交根代理评估和执行安排，不更新公共状态。

## 执行安排（2026-10-26）

根代理按已获批的全文目标推进Tasks1–2；只新增独立定义/覆盖模块、测试和聚合产物，先RED后GREEN并独立审阅。Task3的窗口/中心追加仍未执行，不把当前全帧覆盖当成Table14/15模型误差。私有审计、原始映射与运行路径不发布；本计划用逻辑位置代替机器绝对路径。

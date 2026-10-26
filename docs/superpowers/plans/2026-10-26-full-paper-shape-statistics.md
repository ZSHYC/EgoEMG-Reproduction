# Legacy Shape Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从实际获取的左右 MANO beta 字段重算 Table5 所需统计，并生成真实手形分布图和来源记录。

**Architecture:** 严格按manifest读取每episode左右各10维beta，一手仅计一次，复用固定官方脚本的concatenate/mean/std/min/max/range算法，并额外明确输出真正的P95−P5。所有公开产物仅包含聚合统计与内容摘要。

**Tech Stack:** Python、NumPy、Matplotlib、pytest。

**Spec:** `docs/superpowers/specs/2026-10-26-full-paper-reproduction-design.md`

## Global Constraints

- 当前官方版本固定 `f7cbd048b7092139852270491d3859cd472e0311`；有歧义优先官方代码。
- 官方 `scripts/paper/plot_beta_diversity.py` 用ddof0与max−min，未算P95/P5；论文标题差异必须披露，不改公式迎合参考数。
- 不改旧训练数据manifest、acquisition ledger或四模型结果。
- 原始参与者标签、左右逐人beta、机器路径不得写入公开聚合结果。
- 41episode/82hand是数据表计数，不能据此补造人口年龄/性别信息。

## Review Focus

- 使用metadata `episode_beta_idx` 查表，防止episode排序或重复引用错把行数当独立participant。
- 缺文件、字节数错、非有限beta、dtype/shape错均须明确失败，不能填0或静默丢人。
- 双手和并后同一维度的std为ddof0；range与P95−P5是两个不同字段。
- 公开结果不得泄露episode_id/subject及逐人参数。
- plot图注应说明官方bar图与PDFboxplot差异，不给PCA维度编造固定语义。

### Task 1: beta统计读取、导出与作图

**Files:**
- Create: `egoemg_repro/dataset_statistics.py`
- Create: `tests/test_dataset_statistics.py`
- Create: `results/dataset-statistics/{shape-statistics.json,shape-statistics.md,shape-distribution.png,shape-distribution.svg}`（只有实际计算后生成）

**Interfaces:**
- Consumes: `--metadata-root`内manifest.json/metadata.npz；`--beta-root`内两个beta.dat和已有acquisition_ledger.json。
- Produces: `python -m egoemg_repro.dataset_statistics --metadata-root PATH --beta-root PATH --output PATH`。JSON记录源manifest/metadata/beta/官方统计脚本SHA、shape、样本数、ddof=0、percentile_method=linear及每维mean/std/min/max/range/p05/p95/p95_minus_p05；不含原始身份与原始beta表。绘图按十维中性beta1..10显示合并82手的箱线分布。

- [ ] 先写独立可手算beta fixture测试，核验每维左右拼接后的mean/std/range与分位差，重排episode_beta_idx可正确读取，不把未引用行计入统计。
- [ ] 写损坏字节长度/NaN/无效beta索引及公开输出去身份测试；运行确认功能缺失导致失败。
- [ ] 实现严格字段读数和来源校验，只保留被metadata引用的不同beta索引，对每索引左右各一次；结果计数明确是beta row/episode并不声称自然人验证。
- [ ] CLI额外接受 `--upstream PATH` 获取已固定官方脚本的内容摘要；拒绝未知官方版本或脏源。生成JSON/Markdown与PNG/SVG，不生成PDF。
- [ ] 使用原项目 `data/legacy` 的 metadata，以及当前工作树 `data/legacy-extras` 的两个已下载 beta 及 ledger；运行生成真实结果并核验图像。
- [ ] 明确记录代码使用range、PDF误标P95−P5的差异；实际数值与PDF若不同如实报告。运行完整pytest，写私有实现报告，不提交git。

# Native Dense Occlusion and Paired Figure13 Plan

> Execute inline with superpowers:executing-plans. Existing approved full-paper design and explicit official-code priority authorize this continuation. One review of this new integration after implementation; no review or rerun of closed MANO/zbuffer/model mathematics.

**Goal:** Execute the original quarter-stride evaluation and Figure13 analysis on all available legacy samples, pairing genuine frozen vision/EMG/fusion predictions and actual original GT geometry. Preserve numerical/source differences and unavailable actual-video/full-corpus limits.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements2/4/7 and D19/D20 in [full-paper status](../../full-paper-status.md). No new scientific preference is needed: pinned executable source controls ambiguities.

**Architecture:** Separate source-bound dense selection, numeric geometry extraction, frozen-model predictions and aggregate analysis. Reuse original `EgoEmgMemmapDataset._build_window_index` and `_resolve_index_to_center`; reuse accepted native mesh/occlusion operators unchanged. Existing standard-role runner/receipts stay immutable. Dense roles remain a separately named evaluation population, never silently substituted into standard model reports.

**Global constraints**

- Upstream f7cbd048b7092139852270491d3859cd472e0311; selection exactly `scripts/eval/evaluate_occlusion_fusion.py`: evaluation WL from exact fusion config, stride=`val_test_stride//4`, jitterfalse, eval_center_stride=None, all native center-filtered indices, order split(user/gesture/both)→hand(left/right)→native index, repeat1, no max-samples subsampling. Support explicit ViT-S+S7790/1947 and RN18+S12000/3000 recipes, never mix their grids or infer count17168.
- Keep native center-only split filtering even where full windows cross split boundaries; report full-window split crossings and intersections with actual fixed model training/validation footprints before inference. No claim of independent unseen inputs merely from a heldout center, no test-driven sample deletion or retuning.
- Sources/data must be nonsymlink physical paths, complete byte-bound and guarded before import/read and afterward. Reopen actual expected identity; an arbitrary resealed token cannot grant an inference/training permit. Grid admission is explicitly data-only, no models/updates/CUDA or geometry execution.
- Native geometry receives actual episode beta and per-frame pose/world/camera transforms, right MANO/X-left mirror and source original calibrated1920x1080 fallback. Video sizes/distortion/mirror physical equivalence remain unproven. Preserve zero poses and all tracked/stale/marker flags. No synthetic/zero beta or invented finite score.
- Genuine completed frozen branch/fusion lineage and exact native weights only. Official standalone-head/center-forward dispatch must be independently checked; absent branches cannot be reported as trained zero predictions. Preserve original lossless radians and source conversion57.3/round4 and round6 occlusion as separate fields. Source evaluation MAE ignores mask, so report literal metric alongside explicit valid/crop/zero populations rather than silently imposing a new mask.
- Plot filtering is source-exact: occlusion notNaN and visionMAE>0.01; base0.05 bins merged to default100samples, final<10 discarded; weighted cubic UnivariateSpline, filtered per-sample Spearman, panel(a)SEM only. Preserve bottom tables, source native outputs and differences from PDF LOESS/confidence/textρ; never copy paperρ/p.
- Protect all14 closed scientific settings, actual server jobs/reserved devices and licensed data/models. CPU engineering/sampling is not scientific full-paper completion. No hidden retries, model-batch reductions, truncated population or qualification-weight reuse.

## Review Focus

1. Quarter stride must use integer floor1947 for7790, not1947.5/1948 or shared eval_center_stride; exact episode-start grid and native center split boundaries must agree, including odd WL.
2. Center-only filtering admits input overlaps. Require visible measured diagnostics against fixed producers, not a blanket leakage-free claim or hidden safety filter.
3. A resealed grid receipt, reordered hand/split indices, wrong typed arrays or changed source/input cannot silently authorize downstream geometry/predictions.
4. Native standalone model and fusion center outputs have different dispatch/target conventions; pair exact hand/episode/center keys and record all unavailable-mask/crop/zero populations before aggregation.
5. Native rounding, filters, adaptive bins and spline weights materially affectρ/curves. Test actual source functions on adversarial boundary/empty/small bins and preserve unfiltered data privately.

### Task 1: source-bound native dense selection

**Files:** create `egoemg_repro/occlusion_dense_data.py`, `tests/test_occlusion_dense_data.py`.
**Interfaces:** `dense_occlusion_protocol(upstream,config,*,recipe)->dict`; `native_dense_points(data,*,upstream,config,recipe,expected=None)->dict(identity,points)` with readonly typed columns sequence/episode/start/end/center/split_id/hand_index/native_index/group_index. Receipt is data-only, not a model/geometry permit.

- [x] RED: both exact recipes/1947 and3000, unknown recipe/wrong config/changed stride reject; source-exact order and centers across tiny multi-episode and mixed-split fixtures, including first/last fit, oddWL, empty split, no repeat/subsampling and windows crossing center split retained.
- [x] RED: malformed shape/dtype/metadata/split labels, symlink input/source and forged expected receipt reject. Real native method parity, no model constructors and unchanged RNG with CUDAhidden.
- [x] GREEN: minimum wrapper calls unchanged native methods on explicit IO-only instances; byte-hash/guard required providers and three full numeric index inputs, strictly typed unique keys. No raw target/EMG/model/geometry reads.
- [x] Run new tests in pinned rich CPU runtime with0skips; commit sources. Expected: all new cases pass and no constructor/CUDA/full-science permit.

### Task 2: actual full dense populations and geometric inputs

**Files:** `egoemg_repro/occlusion_dense_footprints.py`, `tests/test_occlusion_dense_footprints.py`, new collector/CLI and `results/native-dense-occlusion/` evidence; reuse closed finite-scan/source/kernel helpers without modifying them.
**Interfaces:** exact fresh grid rebuild→typed point payload→numeric geometry admission, separate complete-mode identity; `summarize_dense_footprints(points,partitions)` checks paired typed keys and uses closed union/intersection primitives to count unique temporal rows and separate overlapping hand requests. Include independently reconstructed training/validation footprint intersections by model producer.

- [x] Two detached fresh processes per recipe reopen all real legacy index inputs; all keys/counts/order/typed bytes agree. Actual ViTS17168/RN18=11114, all10ownersclosed; no preset count.
- [x] Two fresh complete audits reconstruct actual completed ViTS fusion/vision-initializer/EMG-initializer partitions and the admitted future RN18 dataset, verifying24 full boolean row masks. ViTS1280handwindows intersect fusion training inputs while all heldout centers remain outside; retain/disclose all original keys.
- [x] After RN18 fusion/initializer training closes, bind its actual producer partitions and weights before RN18 dense prediction; prospective dataset diagnostics alone are insufficient.
Task2 geometry-input interface refinement: add `egoemg_repro/occlusion_dense_geometry_data.py` and `tests/test_occlusion_dense_geometry_data.py`. `extract_dense_point_arrays(points,fields,betas,metadata)` is a pure typed IO transform with no provenance permit, returning source-order selection with explicit pool_index plus native-helper-compatible shared temporal arrays. `admit_dense_geometry_inputs(data,geometry,index_root,*,geometry_reference,index_reference)` returns a registered live `DenseGeometryInputs` exposing identity/data_by_recipe/selection_by_recipe/check. Fixed immutable public geometry/index reference hashes authenticate all actual point/identity/bulk bytes; actual beta/calibration hashes are pinned from the accepted small-asset transfer. Both recipes share one complete raw byte admission, with final raw rehash after extraction; no repeat full finite scan of already byte-identical accepted arrays. `export_dense_geometry_inputs` / `verify_dense_geometry_export` write/rebuild both private typed point packets and source-order maps, exact identity/completion. Check source/inputs before IO/import, read-only snapshots and unforgeable process registry. Test actual beta lookup, zero/flags, odd/mixed ordering, malformed schemas/metadata, corrupt/resealed references/points, mutation/forged objects and fresh export reconstruction. No Torch/native geometry/models or asset math in this component; native wrapper is separate and frozen standard runner remains unchanged.

- [x] Extract every selected real pose/world/camera/tracked/stale/valid flag and actual episodebeta from fully admitted eight files. Verify full bytes/numerics/source/guards and fresh reconstruction; no learned model or native geometry permit from the index alone.
Native-wrapper interface: add `egoemg_repro/occlusion_dense_geometry.py` and `tests/test_occlusion_dense_geometry.py`. First implement `load_dense_geometry_inputs(export,evidence,*,recipe,metadata,calibration)` and `select_dense_requests(loaded,mode)`: fixed actual full extraction proof authenticates every exported byte/typed array, actual metadata/calibration and source-order pool map; a process registry prevents resealed/forged live inputs. This consumes the actual authenticated derived packet, as the closed standard runner does, and does not redundantly reread unconsumed37.8GB bulk files for each mesh. Explicit `complete_dense` vs `bilateral_probe` selects all native keys or actual first nonzero bilateral keys, never nearest standard-grid data. Keep metadata/context flags and every zero request. Only the later native run action can construct the unchanged original MANO layer/operator; loading/selection alone is0operator/0model/0CUDA. Run/independent raw verifier then integrate the same frozen source geometry_frame/_native_runtime and unchanged kernel functions, with separate complete-dense identities and strict scope/populations.

- [x] Stage immutable committed source; execute full original geometry plus independent fresh native replay/verification for dense grids, retaining private arrays. Same accepted source helper/operator policy, no changed rasterizer or new licensed-math review. Close all exact owners and export actual populations.
Both actual workflows closed: RN18proof83012647... and ViTproofacc901cf...; 28 original native owners +4 postexit owners closed, every typed array equal. This closes only the geometric-execution substep; Task2 actual RN18 trained-producer footprint binding and Task3 full frozen predictions are still pending.

### Task 3: frozen paired predictions and native analysis

Paired input IO implemented: `occlusion_dense_inputs.DensePredictionDataset` uses the original coupled and EMG-only center helpers, exact native7790/12000 windows, separate original normalizations, all keys/masks/crop flags. Actual13passed/0skip including independent genuine full trajectory reader parity. This pure IO class provides no full-content or trained producer/model permit; complete actual paired IO and trained-state/dispatch integration remain pending.
Actual pairedIO sourcea1a41d4 completed both full passes: all28282 hand requests perpass, four54file raw SHA passes and all6ownersclosed. Native heldout-center selection consumes54 physical files versus prior standard full-footprint53: crop:000004 is independently joined to source catalog/episode metadata/2026-10-25 transfer SHA; extra file is retained. Initial operational53-file equality failed before point reads, all4ownersclosed; fixed separate v2 root preserves every old fileSHA and adds the explicit extra. Full input/model permit remainsfalse until complete IO closes and trained inference integration is separately admitted.


Statistics subcomponent implemented and independently source-main checked: `egoemg_repro/occlusion_dense_analysis.py`, `tests/test_occlusion_dense_analysis.py`, actual16passed/0skip; evidence `results/native-dense-occlusion/source-analysis-tests-20261101.json`. Source eval compatibility diagnostic proves missing experiment/dataset and stale attention patch; use same-pinned native callable center dispatch in later integration, not a changed model. This does not close Task3: actual trained producer/state/data admission, complete-key frozen predictions, replay and real-data Figure13 remain pending.
Ordered join subcomponent also implemented: `occlusion_dense_predictions.join_dense_predictions`, actual26passed/0skip, proof `results/native-dense-occlusion/prediction-join-tests-20261101.json`. This validates ordered complete keys/geometry/real error transport only; it neither constructs a model nor admits trained/full inference. Standalone8/per-hand-vector and fusion16/unified-scalar must have separate producer-correct inputs on identical7790 windows; preserve native standalone outputT//2.

**Files:** new dense inference/replay and source-analysis adapters/tests, bottom tables/plots.
**Interfaces:** source-bound completed producer admission and exact typed point join→native head predictions/raw radians→native degree/rounding columns→original plotting functions and independent aggregate verification.

- [x] RED→GREEN exact standalone/fusion center dispatch, full trained state/producer/config/footprint binding, genuine missing crop/mask handling and per-sample MAE conventions. Existing core model families stay closed.
- [x] RED→GREEN no missing/duplicate/reordered joins; source rounding/filter/bin/SEM/spline/Spearman boundary cases and raw-report reconstruction. Preserve original arrays and separate masked/pooled diagnostic metrics.
- [x] One new-unit review and one material fix pass, then relevant CPU acceptance. No second review or unrelated core suites.
- [x] Complete ViT-S native full frozen predictions on the official-supported CPU FP32 device after genuine B64 qualification: two actual17,168point/272batchpasses, new-process learned replay, original report and independent reconstruction allclosed. Postexit raw/input54file61.380GB/SHA/source/numericCSV/PNGverification passed,112explicitowning/dataworkeridentities+collectorclosed. No model updates or original GPU rescheduling.
- [ ] Complete the RN18 route after its actual compatible producer training closes; preserve its11,114sourcekeys and independently bind its weights/inputs/replay/statistics. ViT-S completion cannot substitute for this route.

### Task 4: publication and full-paper accounting

- [x] Publish ViT-S full resultCSV/raw-radianCSV/binCSV/statisticsJSON/originalPNG/report/completion and postexit proof under results/native-dense-occlusion/vits-full; include actual16,650filter/16,409bin/241right-boundary populations, meanΔ0.119740877°,ρ0.023640393/p0.002283599 and source/PDF differences. Keep rawNPZ/model/dataset private.
- [ ] Publish corresponding RN18 result artifacts only after actual complete source-bound execution; do not reuse ViT-S predictions.
- [x] Mark only actual D20/Figure13 scope proved. ViT-S available-legacy pipeline is complete and scientific within its stated scope; PDF population/numerical/physical camera equivalence and RN18 remain unproven. Keep figure13_reproduced/full_paper_complete false and all other obligations active.

## Preflight rulings

Task1 selection feeds Task2 numeric extraction: array key order/dtypes/window family must agree; role is native heldout-center, not standard strict full-footprint test. Task2 real geometry feeds Task3: original score and invalid row keys must survive, no missing-key drop. Task3 full frozen predictions feed Task4: all native/raw/rounded/filter populations must reconcile and proof must refer to actual closed processes. Existing full design selects original code and explicit fallback, so no renewed brainstorming preference/approval round. This is reversible authorized server/repository work. If source native selection overlaps trained inputs, retain and disclose it; do not silently redefine original selection as a safer subset.

Task3 current checkpoint2026-11-01: complete pairedIO proof387735569301d6714b205c381c82ca1b77d3007b1e745dc0bc5dc91cfe4f1456 now closes the full-input-read substep, not inference. Actual complete trained triplet reload10passed; native forward/center/error/group-tail adapter18passed including independent whole official eval module collate parity. `_collate_fusion` is in the source evaluation script, not datamodule; preserve its field omission literally. New runner/raw export/fresh full replay/real-data analysis integration still pending; final new-unit review only after that integration is implemented. Native EMGFormer-M200/fullvaltest/replay is separately complete; not part of dense prediction permit.

Execution route refinement (source device choice, not a scientific scope change): the pinned evaluator explicitly accepts --device cpu and performs FP32/no-autocast forwarding. All GPUs remain assigned to existing own/protected work. The full ViTS dense inference runner may use the server CPU with this native supported option after the single final integration review and real B64/4worker/pin-memoryTrue/group-tail acceptance. State the actual hardware/threads in every result; do not shrink batches/population, change normalization, steal a GPU or treat small interface diagnostics as B64/full qualification. RN18 remains pending its actual compatible producer and native joint fit. Real full batch schedule is272 with six tail sizes45/45/19/19/8/8; source defaults remain intact.

2026-11-02 RN18 Task2接续：原生150轮／46950更新及BEST145已完成并公开。先执行[只读实际生产者审计入口](../../../results/native-dense-occlusion/rn18-source-audit-20261102/README.md)：绑定原退出后凭据和三种真实生产者，重建完整11114网格与实际数据范围，独立布尔行掩码核对，整SHA重读选定和导出权重。此操作不调用模型、不提供推理资格；Task2该绑定仅在实际双遍成功和退出后读取完成后勾选。后续RN18独立三模型native class／全state／FP32／规范化／dispatch整合仍须测试先失败再通过和一次新整合最终审查，不重开已关闭ViT数学与数据几何。

2026-11-02 Task2实际RN18三生产者范围／权重绑定完成：[双完整CPU审计与独立退出后读取](../../../results/native-dense-occlusion/rn18-source-audit-20261102/README.md)。6原拥有者和独立读取者关闭；11114请求、18完整布尔行掩码，5权重字节再次核验，0Torch／模型／前向／更新。融合与EMG训练输入交集928778行／756请求，视觉961800行／778请求，训练内中心均0；保留全部原始请求并披露输入重叠。此项不等于实际native triplet构造和Task3完整预测／Figure13；既有ViT全流程和数学保持关闭。


2026-11-02 Task3 RN18接口进展：[真实已训练三模型和原生左右手输入接口](../../native-rn18-dense.md)通过23项／0跳过新组件检查；源guard／prepared引用替换负例实际先失败再修正通过。14测试拥有者与独立读取者关闭，155冻结文件及当前source/test SHA一致。尚未进行新运行器的完整最终审查和B64／全量预测。

RN18子路线的剩余交付（延续本Task3／4和原定性计划）：
- [x] 真实完整三模型重新构造、全张量／BN／FP32／原生方法身份及双手单样本输入检查；来源引用完整绑定。
- [ ] 单独RN18证据／全输入／批输出／完整新进程重放及退出后收集运行器，新接口测试先失败再通过；复用原有数值／几何操作，保留ViT身份与结果。
- [ ] 一次完整新RN18整合审查及必要实质修正；已关闭原数学和ViT流程不重审。
- [ ] 真实原B64／4worker及分组尾批资格、完整11114原生预测与新进程重放；不能用单样本／资格替代。
- [ ] 完整原遮挡分析／Figure13、精确普通2208子网格的Figure7、独立退出后重读及公开不可变字节验收。

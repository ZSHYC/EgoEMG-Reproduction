# Reproduction completion and publication design

The user authorized completing the project, improving documentation, and polishing the public repository. This extends the approved reproduction design; it does not relax its scientific acceptance criteria or change the identity of any running experiment.

## Completion requirements

1. Finish EMGFormer-S, ResNet-18, and their residual fusion on the verified legacy cohort, including complete validation-selected test results and recoverable checkpoints. Add the already proposed EMGFormer-M comparison after its own real-data GPU acceptance. Record unavailable paper-equivalence evidence explicitly instead of claiming numerical equivalence.
2. Publish sanitized, provenance-bound result tables and training curves. Incomplete or bounded runs must remain visibly incomplete and cannot populate final benchmark tables. Pooled and per-user summaries, center/trajectory sampling, counts, and missing-crop populations remain explicit.
3. Explain the paper's dataset, annotation pipeline, architectures, losses, and evaluation, separating PDF statements, pinned source behavior, and reproduction adaptations with page/code references.
4. Give the repository a readable academic presentation: original SVG banner and architecture diagram, clear navigation and quick start, accurate status and result links. Preserve the Chinese-first audience and external licenses. No paper figures, datasets, participant names, private paths, or weights are redistributed.
5. Verify published files, rendered visuals, commands, complete test suite, hosted CI, remote visibility, and actual experiment artifacts before marking the goal complete.

## Implementation choices

The selected approach builds on the existing GitHub repository and pinned official models. A documentation-only restyle would leave the final reporting gap; a separate website would add maintenance without helping command-line reproduction. We therefore improve the repository itself and add an independent reporting module.

The publication layer may add modules, tests, documentation, assets, and optional reporting dependencies. Existing executed model/data/training modules and the three selected recipe files remain byte-identical while their jobs run. New reporting files are outside the trainer's explicit runtime identity list. Changes that would alter an experiment's runtime identity require a separate decision and fresh run, never a silent in-place edit.

For visual identity use a restrained navy/teal/amber academic palette, readable labels, and vector diagrams. These are builder choices under the user's request for a polished repository. The user explicitly authorized all requests and execution; no extra approval round is required.

## Locked-checkpoint metric audit

The frozen training accumulator stores all-split per-user summaries, which cannot reconstruct per-condition user dispersion. A separate evaluator will recompute predictions from each completed, validation-selected checkpoint, verify the already reported center MAE, and add hand/split/user intersections plus identical crop-present center populations for every modality. The training driver still performs its final test pass once. This explicitly declared additional locked-checkpoint audit is verification and metric disaggregation; no checkpoint selection, model update, hyperparameter choice, or retry based on its test accuracy is permitted. Final result metadata records both passes and their agreement/tolerance. This clarifies the once-after-selection policy rather than treating repeated test-driven selection as acceptable.

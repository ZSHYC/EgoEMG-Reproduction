# EMGFormer architecture audit and figure implementation plan

> **For agentic workers:** Continue the approved inline full-paper analysis work; preserve every running/qualified runtime.

**Goal:** Supply Figure4 and Table7 with measured original-factory S/M/L parameter counts and forward shapes, preserving code-over-PDF differences.

**Architecture:** A standalone analysis script composes each pinned official experiment without overrides, seeds before the original factory and performs CPU eval/no-gradient synthetic forwards. Hooks record real frontend/decoder/head shapes, and the original Lightning forward supplies aligned predictions/targets/masks. A renderer consumes the fixed public audit JSON. No dataset, checkpoint, optimizer or scientific training is involved.

**Tech Stack:** Existing pinned Hydra/OmegaConf/PyTorch runtime and Matplotlib; JSON/Markdown/SVG/PNG.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements1/2/5/7; Figure4/Table7 and existing paper-guide§3.

## Constraints and review focus

- Keep upstreamf7cbd048b7092139852270491d3859cd472e0311 clean and all staged model code unchanged.
- All three official EgoEMG recipes use8target-hand channels,12,000train/eval points and22outputs. Do not falsely unify them with7,790point visual/classic/EMG2Pose recipes.
- Record actual `tds_slim`,2stages/globalSE/noncausalRoPE/pre-LN/ReLU head, rather than inferring topology from Figure4's simplified four blocks/GELU.
- Assert real parameter counts and original tuple alignment for both12,000and a separately marked diagnostic7,790length. Neither synthetic input nor a successful forward is a benchmark.
- Record declared normalization asset availability and unified-data dependency without substituting an unverified alias or loading datasets.
- Publish only normalized source filenames/digests and model facts; no private paths or raw input/output tensors.

## Task1: Measure original factories

- [x] Add `scripts/paper/audit_emgformer_architecture.py probe --upstream ... --output ...`.
- [x] Verify original factory configurations, six actual CPU forwards, typed target/mask equality and finite output; bind clean source/config and model digests. Compare exact counts with prior independently completed model records where available.

## Task2: Render and reconcile evidence

- [x] Add the script's `render` command; draw actual source topology, dimensions, recipe table and explicit PDF/recipe differences.
- [x] Inspect PNG, repeat render for byte identity, publish JSON/Markdown/SVG/PNG, update Figure4/Table7/paper-guide evidence and correct the earlier overly broad status statement about all12krecipes differing from official inputs.

Verification uses actual original models and independent existing parameter records plus visual checks, not implementation-mirroring tests for a reversible analysis renderer. Full training and all other paper requirements remain separate.

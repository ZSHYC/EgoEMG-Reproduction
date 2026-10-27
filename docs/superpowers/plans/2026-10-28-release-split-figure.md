# Released participant-gesture split figure implementation plan

> **For agentic workers:** Use the existing inline execution workflow; this is a bounded analysis/plot task under the approved full-paper design.

**Goal:** Reconstruct Figure3(b)'s protocol schematic and separately expose actual old-release participant/gesture label membership without concealing overlaps.

**Architecture:** One standalone script scans only three shared per-frame label arrays and metadata boundaries, hashes their actual bytes, and checks every aggregate against the existing independently recorded frame audit. A fixed anonymous aggregate JSON is the input to a deterministic Matplotlib renderer. Its left panel is explicitly the paper's conceptual6-user/10-gesture holdout; its right panel is actual released membership, with multicolor cells retaining multiple labels. No model/training/source pipeline is changed.

**Tech Stack:** NumPy, hashlib, Matplotlib; JSON/SVG/PNG. Read-only PDF inspection used Poppler.

**Spec:** [Approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md), requirements1/4/5/7; Figure3(b) evidence row in full-paper-status.

## Constraints and review focus

- Preserve fixed official revisionf7cbd048b7092139852270491d3859cd472e0311 and all running/frozen model code.
- Keep all60classes, including raw50. Shared synchronized frames are counted once, not per hand; unassigned rows stay separate.
- Verify field schemas, all396,970,350actual bytes and metadata boundary stability; every class/split/activity count must equal the existing frame audit.
- Keep alias secret, raw IDs and machine paths private; published aliases apply only to this plot. A published aggregate is sufficient for exact rendering.
- Multiple labels in a participant/gesture cell must remain visible; never silently choose a single label. Full-frame user overlap may differ from a window-center cohort.
- The paper panel is a schematic, not a measurement. The released-data panel is an old-release audit, not proof of PDF cohort equivalence or independent train/validation/test roles.

## Task1: Complete read-only label scan and anonymous matrix

- [x] Add `scripts/paper/render_release_split_matrix.py scan --data ... --frame-audit ... --vocabulary ... --private-output ... --output ...`; stream bounded chunks and freeze source identities, save alias mapping privately, emit only anonymous counts publicly.
- [x] Run once on actual server data, independently match all existing class/split/activity counts and transferred field hashes; inspect totals, overlap counts and source/code digests. Preserve failures; do not touch running jobs.

## Task2: Render and publish Figure3(b) comparison

- [x] Implement `render --input ... --output ...`; left explicit schematic, right actual41x60 membership; save SVG/PNG and a compact readable provenance/limitations report.
- [x] Re-render from the public aggregate, inspect PNG and validate all row/column/split totals. Update the full-paper evidence row and commit only actual verified outputs.

This reversible plot task uses real full-array balance/hash checks and visual verification rather than implementation-mirroring unit tests. Mathematical training and GPU qualification remain separate and untouched.

Actual validation: owned real-server scan1.688242215s exit0;396,970,350labelbytes match recorded hashes, all class/split/activity records match prior audit, public coverage perclass/persplit also exact.41metadata keys/40defined keys/60gestures/65,600,616defined+561,109unassigned frames.556multilabel cells; full-frame Train∩User and Train∩Both each5keys, deliberately distinct from center-cohort1/1. Private aliases remain local. SVG/PNG inspected and independently repeated render byteidentical; owner/worker/guardian independentlynonlive. Frozen model sources and active jobs unchanged.

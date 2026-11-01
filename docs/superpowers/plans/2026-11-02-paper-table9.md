# Paper Table 9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline. This is a read-only derived artifact under the existing approved whole-paper scope; it introduces no model, training, data-selection or scientific-method change. One fresh review of this new small component at the end; no closed model/math reviews.

**Goal:** Export the five completed EMG methods' actual bilateral trajectory errors, exact denominators and explicit PDF comparisons for Table9.

**Architecture:** A pure CPU module reads five immutable public completed reports, independently recomputes30hand×split cells and conserves hand/split/overall sums/counts. A deterministic CLI emits JSON, long-form CSV, hand-gap CSV and Markdown, then a fresh process rebuilds and verifies every artifact from the same pinned source reports.

**Tech Stack:** Python standard library plus the unchanged source-generation guard; no Torch, optimizer, model loading, dependency installation, new inference or training.

**Spec:** [Approved whole-paper requirements1/4/7](../specs/2026-10-26-full-paper-reproduction-design.md), Table9/D.1 as actually extracted and visually inspected from local EgoEMG.pdf printedp23.

## Global Constraints

- Preserve all14scientific settings, current10detached server workflows, all model/checkpoint/report bytes and original f7cbd048 upstream.
- Table9 is a10row×3split MAE table, not a standard-deviation table. The caption does not specify per-cell user weighting. Report actual pooled valid-angle-time MAE and leave PDF aggregation equivalence unestablished.
- Use retained complete S/M/L trajectory reports at2792handwindows/12000input/11490output and completed original vemg2pose/NeuroPose reports at4306handwindows/7790input/6000or7790output. Never relabel these as one controlled population or replace S/M/L with center metrics.
- PDF cells in gesture/user/both order: S left12.4/14.7/16.5,right13.2/16.6/18.4; M left11.5/15.5/17.0,right12.1/16.6/17.8; L left11.4/15.3/17.6,right12.0/16.0/17.7; vemg2pose left14.7/15.7/17.2,right15.2/17.0/17.5; NeuroPose left15.9/15.4/15.7,right15.7/16.0/16.9. Preserve raw precision in computed metrics and deltas; no fitting/rounding to match the paper.
- Pin each complete source report's actual byteSHA, require its real completion/selection identity and reject changes even under resealed summaries. Read-generation guards precede hashes; readers/public artifacts contain no private subject keys or machine paths.
- New analysis completion is distinct from paper numerical equivalence, current-native S/M/L trajectory completion and whole-paper completion. Full IK execution remains gated on its actual qualification; do not create that GPU workflow while it is pending.

## Review Focus

1. Unweighted cell means cannot replace valid-angle weighting; hand/split/overall margins and count/error conservation must match all source reports.
2. Center fields, missing/duplicate hand or split keys, boolean/negative counts, NaN and zero-count nonzero errors must reject.
3. Public modern/legacy source protocols and different2792/4306populations must stay visible; no causal/model ranking claim from mixed protocols.
4. Same-size source replacement during reading, altered pinned source/report identity and output generation changes must reject.
5. Empty cells remain null; no fabricated SD, user identity, natural-person equivalence, model run or scientific accuracy claim.

### Task 1: pure exact trajectory cell reader

**Files:** create egoemg_repro/paper_table9.py and tests/test_paper_table9.py.
**Interfaces:** trajectory_cells(statistics)->dict with30cell primitives per five reports via build_table9(repo)->sealed aggregate; the arithmetic helper itself grants no scientific permit. SOURCE_REPORTS pins the five existing metrics files/byteSHA; PDF_REFERENCE preserves the30visuallychecked cells.

- [x] Write handwritten unequal-count fixtures and rejection tests for center mode, key coverage, invalid counts/errors, incorrect reported means and violated margins; run missing-module RED.
- [x] Implement independent math.fsum/count×180/pi reconstruction, source pins and complete model/selection/trajectory checks, all30PDF deltas and15absolute left/right gaps. Expose different source populations/recipe identities and global completion=False.
- [x] Run only the new tests to GREEN, with guards denying ML imports. Commit source/tests.

### Task 2: actual export, independent verify and status

**Files:** same module/tests; create results/paper-table9/{metrics.json,cells.csv,hand-gaps.csv,README.md,completion.json}; update docs/full-paper-status.md Table9 row only plus current checkpoint.
**Interfaces:** export_table9(repo,output)->report; verify_table9(repo,output)->report; python -m egoemg_repro.paper_table9 export|verify --repo PATH --output PATH.

- [x] Write tests for altered raw source bytes, reader races, deterministic exports, exact independent CLI roundtrip, changed/missing output files and readonly source bytes; watch each missing-feature RED.
- [x] Implement fresh immutable output, stable CSV/Markdown and final completion marker after independently rebuilding source reductions. No plots are needed for this tabular result.
- [x] Run relevant tests without skips; execute real five-report export and separate fresh-process verify with Torch imports forbidden. Inspect all30actual cells/counts/conservation and preserve source/report/PDF hashes.
- [x] One fresh final review of this new component, one focused material fix if required, then publish the exact tested artifact/source bytes. Update Table9 at its true derived-analysis scope; all remaining full-paper requirements remain open.

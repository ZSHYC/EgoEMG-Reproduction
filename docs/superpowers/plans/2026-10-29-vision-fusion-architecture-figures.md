# Vision and fusion architecture figures

**Scope:** Fill Figures 5 and 6 under the [approved full-paper design](../specs/2026-10-26-full-paper-reproduction-design.md). This is descriptive analysis, with no model, data, training, queue or server changes.

**Method:** Audit the clean pinned upstream source, extract seven variant dimensions and actual head activations, and reconcile fusion dimensions with the retained native construction receipt. Render source-based PNG/SVG diagrams using the existing Figure 4 style. Record source regions, hashes, retained evidence and the distinction between static derivation and actual earlier model execution.

- [x] Bind the actual source/config files and existing native evidence. Use the center-supervised ViT-S + EMGFormer-S training recipe, not an unrelated fusion mode.
- [x] Render separate ResNet, DINOv2 and WiLoR paths, including WiLoR's vertical crop and different activation. Draw the native fusion validity gate, attention pool, convolutional projection and additive residual.
- [x] Inspect both images, reproduce the artifacts in a separate process and compare bytes; update the figure inventory with limitations and reproducible commands.

Preserve ReLU versus GELU, DINOv2's direct timm vector output, 16-slot/7,790-point fusion input, and normal(std=1e-5)/zero-bias residual initialization. Do not label the earlier 8-channel diagnostic trace as a new 16-channel forward or the earlier random-initializer construction as completed trained-branch initialization. Use source/fact reconciliation and visual checks rather than new implementation-mirroring tests for this reversible renderer. No new benchmark or full-paper completion claim.

# Paired synthesis audit: a usable cheap resource, not a selected method

Work date: 2026-10-03 PDT / 2026-10-04 UTC. Zero GPU hours.
Main PanTS grid and protected test evaluation were not queried or changed.

## What changed

We moved beyond catalogue metadata and inspected the actual small public
LeFusion Normal/Demo arrays. This creates a reproducible resource for future
paired diagnostics without generating new scans, retraining a generator, or
reusing the protected PanTS test. It does NOT establish a new paper idea.

Public data revision: `b0516d354c4473ea4f8539cad395dfddb5944215`.
Normal.tar: 6,344,704 bytes; Demo.tar: 41,293,312 bytes. Both match the SHA256
values pinned in the previous source audit. Raw archives stay OUTSIDE Git at
`../lefusion-public-demo-audit/`; no extraction or checkpoint execution.

Dataset card declares MIT. The TCIA collection page was unavailable through the
web tool, so upstream rights and all source provenance are NOT certified here.
No raw images, masks, identifying headers or archive payloads are published.
Report rows use hashed public case keys. Scientific reuse/redistribution claims
need a separate complete provenance/terms check; do not assume MIT metadata
automatically describes every upstream asset.

## Observed contents, not README assumptions

- **30** source ROIs, **90** generated images, three variants per source.
- Source filenames represent **22 distinct public patient IDs**, not 90
  independent samples. Patient-level clustering is mandatory for statistics.
- 30 normal masks and 90 variant masks; no missing pairing groups.
- All 90 variants share their source array shape, affine/grid and binary mask.
- Shape is 32 x 64 x 64. Positive mask counts range from **153 to 8,578** voxels.
- Foreground mean intensity increases from variant 1 to 2 to 3 in **30/30** ROIs.
  That is descriptive consistency with intensity controls, not clinical quality
  or a proof of subtype correctness. Generation seeds/exact settings unknown.

The [author README](https://github.com/HINTLab/LeFusion) advertises 20 normal
ROIs, but the pinned archive contains 30. Use actual membership, not the README
number. Masks specify intended synthesis regions; they are NOT independent
radiologist judgments of the generated nodules. Healthy-region labels likewise
remain source assertions, not our clinical adjudication.

## Critical unit contract verified

The source images have CT-unit intensities. Generated NIfTIs contain normalized
values. The pinned [author inference loader](https://github.com/HINTLab/LeFusion/blob/03dc67bd8169ced5f8bb6a8707d73377f110ebff/LeFusion/dataset/lidc_hist_in.py)
clamps to [-1000,400] and rescales to [-1,1]. The [output writer](https://github.com/HINTLab/LeFusion/blob/03dc67bd8169ced5f8bb6a8707d73377f110ebff/LeFusion/inference/inference.py)
saves that normalized output without inverse HU conversion.

Correct source comparison: `normalized = (clip(HU,-1000,400)+1000)/700 - 1`.
Do not send normalized files directly into a detector expecting HU. Do not infer
physical HU from the mere presence of a NIfTI header/CT-like appearance.

Initial geometry JSON (`lefusion_public_asset_audit_20261004.json`) deliberately
preserves the first diagnostic pass: its raw/clipped background differences
compare INCOMPATIBLE units and are **invalid image-error measurements**. They
were not used as scientific evidence. Use
`lefusion_public_asset_normalized_audit_20261004.json` for matched-unit values.
The current script emits matched-unit values only; the first artifact is retained
to make the correction explicit rather than silently replacing results.

After matched-unit conversion, average background MAE across all 90 variants is
0.0115942 normalized units; maximum per-variant background MAE is 0.0128479.
Thus the saved variants do not preserve background bit-identically. This is
not a finding of clinical harm. Author source replaces known context with
forward-noised source during denoising and returns the final model output,
without final exact source pasting for LIDC. That is consistent with residual
background differences; it is not proof of the precise archived run's cause.

The author segmentation README tells users to adapt preprocessing; a complete
end-to-end LIDC segmentation adapter is not supplied there. Therefore the unit
contract is a **reuse pitfall**, not evidence the published segmentation
experiments used wrong units or that their reported results are invalid.

## Next candidate: separate target change from context change in a verifier

A discriminating *diagnostic* is now possible without fresh synthesis:

1. Keep all 30 source ROIs and all three variants; do not pick detector-friendly
   examples. Freeze membership before any task-model outputs.
2. Compare each saved generated image with a counterfactual using exactly the
   same generated foreground but the matched-unit source background.
3. Compare with source-only and a background-only perturbation control. These
   isolate a verifier's response to foreground versus background differences.
4. Before executing a pretrained detector, audit its required CT units,
   orientation, resampling, output coordinates and permitted patch shape. Show
   native competence on separate real annotated development nodules. A competent
   synthetic-condition judge does not automatically certify clinical realism.
5. Count original patient clusters and source/model overlap. The 22-ID demo
   collection alone cannot establish independent test generalization or a tiny
   clinical false-negative bound. No PanTS test data borrowed for this step.

**Not yet a method or approved GPU test.** Generic counterfactual sensitivity,
foreground compositing, local quality scores and background-shortcut detection
are existing ideas. A viable contribution would need a substantial, reproducible
failure in strong verifiers, a remedy beyond those baselines, independent tasks
appropriate to the claim, and evidence its total cost/utility is worthwhile.
If the observed differences are just explained by units, ordinary residual
noise, or an inadequate detector, keep the engineering fix and kill the paper
proposal. Do not label intensity-control order as lesion detectability.

The small resource removes a data-access/generation-cost blocker. It does not
remove the adequate-native-model, novelty, downstream-utility or provenance gates.

## Additional model lead audited and not trusted blindly

[MedCondDiff](https://github.com/ruiruihuangannie/MedCondDiff) README advertises
weights in releases. Public releases API returned `[]` on this pass; pinned
revision `c4bd17d17b3381a8fb90a4634d580e9b392c0c5c` contains code/config but no
verified pretrained segmentation weights in the inspected tree. This is not
proof no weights exist elsewhere; it is not currently a reusable checkpoint.

Sampling code already extracts static conditioning features once, so caching
that extractor is not a new speedup. The [paper](https://arxiv.org/html/2512.00350v1)
defines timing per forward pass and describes multi-step sampling; do not treat
its per-pass table as certified complete multi-sample inference cost. Its
four-mask best-selection statement is about visualization, not an established
claim that quantitative results are oracle-selected. No code installed/executed.

## Resource/decision boundary

This pass used CPU parsing plus 47,638,016 downloaded bytes. No new model weights,
GPU allocations, downstream training, production actions or patient uploads.
The 2 charged GPU-hour total screen cap remains unchanged; historical accounting
is not a newly verified posted debit. No selected S-tier idea yet.

# Next-step preflight after Claude review731dcad

Preparation only; no new GPU job authorized. October1,2026.

## Immediate sequence

1. CPU source-mask feasibility on existing005/120/165: build one fixed
   parenchymal ROI from label1, excluding tumor(label2)+5mm and organ-edge2mm
   by voxel-center physical distances. Report retained size; fail on empty
   regions. Margins are engineering choices, NOT validated medical thresholds.
   Never retune them to obtain a larger feature change. Mask source is fixed
   across conditions, not resegmented on reconstructed images.
2. Lock a small texture implementation/config before extracting real-image
   features. Prefer validated PyRadiomics over hand-codedGLCM/GLRLM. Base
   pants_venv currently hasSimpleITK/nibabel, NOT PyRadiomics (read-only check).
   No install into the frozen shared runtime. An isolated CPU dependency
   target needs version/ABI checks and toy/phantom contract tests first.
3. Extract paired original/preprocessing-control/reconstruction features
   using identical masks, HU units and one pinned binning/grid policy. Record
   first-order mean/variance separately from textures. Report feature changes
   descriptively; three positive cases cannot estimate cancer discrimination.
4. In parallel, CPU audit PANORAMA checkpoint identity, exact patient fold
   exclusions, metadata/license and per-case download feasibility. Keep
   checkpoint-selection exposure explicit. Use weights_only loading where
   possible; do not execute arbitrary pickled checkpoint code. ZIP directory
   names alone do not prove checkpoint training identity.
5. Prepare ONE native-spacing insertion host with matched unmodified and
   modified volumes. Positions/masks/contrast selected before any VAE outputs.
   Start with ONE insert, not20 interacting inserts in a small pancreas.
   Confirm fit in parenchyma and preprocessing doesn't clip/distort it.
6. Before GPU use request a NEW explicit capped calibration budget. Include
   both whole-volume reconstructions, setup, metrics and interactive charge
   multiplier. Fixed-spacing expansion is separate: calculate shape/memory
   first. Do not rely on the withdrawn0.3-0.5hour multi-host estimate.

## Corrections to the new sub-visual direction

- MSD0/1/2 masks contain NO duct annotation. Do not compute duct diameter
  from them or invent a duct mask. Even with a fixed source duct contour,
  geometric diameter is unchanged by definition on a same-grid reconstruction;
  test re-estimated duct visibility/geometry separately if suitable masks and
  a verified estimator become available.
- Parenchyma distant from a tumor is not proven healthy tissue. Current
  scans contain visible tumors, NOT longitudinal prediagnostic scans. Feature
  change does not establish loss of early-detection information.
- "Sub-visual" is not established merely because a feature is numerical.
  The duct study itself describes visible dilation/narrowing. Do not state
  that no human could see any contributing signal.
- No frozenREDMOD checkpoint or full extraction pipeline has been acquired.
  Mayo's primary announcement linksGut paper337266; web retrieval returned403.
  Do not claim that a genericGLCM set replicatesREDMOD's predictive signal.
- The duct study's original data are available on reasonable request, NOT
  verified as an immediately downloadable public prediagnostic cohort. Do not
  contact authors without user authorization.
- PyRadiomics docs explicitly note deviations fromIBSI binning/resampling.
  Pin settings/version;25HU is an example default, not a proven choice for
  subtle pancreatic texture. No automatic per-image normalization or changing
  bin width separately between original/reconstruction. Native anisotropic
  texture is an initial within-case diagnostic, not cross-case comparability.
- Shape features from an identical ROI are expected identical and cannot
  establish image fidelity. Check this as a contract, not a success result.
- Diagnostic-caseAUC is not prediagnosticAUC. Fix feature direction, selection
  and any classifier on independent original training data before paired
  evaluation. Re-fitting separate classifiers on each condition answers a
  different question. Include negatives and patient-level uncertainty;
  near-chance baselines cannot support an "information was lost" claim.
- Four-host signal thresholds and20+20 featureAUC rules are exploratory
  prioritization heuristics, NOT established clinical or statistical safety
  thresholds. Negative tiny pilots are inconclusive, not evidence of safety.
- A synthetic-insert negative result must NOT block real-lesion testing.
  It checks signal transport in that construction, not realisticPDAC texture.

## Controls to lock before any larger run

Keep rawHU and bias-corrected summaries separate. Prespecify Gaussian sigmas
in physicalmm for the cheap baseline; if adding noise-power-matched blur,
define the matching rule and estimate it on separate background calibration
regions/cases. Label this outcome-calibrated matching as such, not fully
prespecified. Smoothing alters noise correlations: simpleROI variance is NOT
an observer detectability/SNR model. Observer templates/noise estimation require
their own definition/validation; do not relabel CNR as matched-filterSNR.

Matched host reconstructions isolate insertion response of a nonlinear VAE.
They do not make many inserts in one host independent samples. Patient is the
statistical unit. Compare actual paired lesion transitions and false positives
when a detector is ready; preserve all cases, including original-image misses.

## Protected items and spending gate

Do not change PanTS training/evaluation job47320183, its environment, or
test-case selection. Do not publish CT images or reader key. No new GPU hours
spent by preparation; prior campaign totals remain in RESEARCH_HANDOFF.
Remaining.08889allocation-hours is NOT assumed enough for a newGPU experiment.

## Sources checked for this preflight

- https://github.com/AIM-Harvard/pyradiomics/blob/master/docs/faq.rst
- https://pyradiomics.readthedocs.io/en/stable/features.html
- https://github.com/AIM-Harvard/pyradiomics/blob/master/examples/exampleSettings/exampleCT.yaml
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12153928/ (duct study/data availability)
- https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-ai-detects-pancreatic-cancer-up-to-3-years-before-diagnosis-in-landmark-validation-study/
- https://gut.bmj.com/content/early/2026/04/22/gutjnl-2025-337266 (linked paper;403 here)

Prepared tooling: prepare_texture_roi.py and test_texture_roi.py. These check
ROI feasibility only; no radiomics, detector, model observer or cancer result.

## Completed preparation checks

All12 local metric/ROI tests pass. Bounded one-thread CPU audit on cached,
hash-verified labels120/165 completed with the fixed5mm/2mm margins:

| Case | Retained parenchymal voxels | Physical ROI volume |
|---|---:|---:|
|120|16138|30.165mL|
|165|25526|60.859mL|

Both nonempty, neither has a duct mask. This verifies available measurement
regions, NOT feature extraction or diagnostic signal. Raw JSONs stay remote
as texture_roi_v1_120.json/texture_roi_v1_165.json. Initial launch accidentally
redirected Python's here-document stdin to/dev/null, producing an empty log
and no audit; corrected launch preserved that log and explicitly verified
CPU_ROI_AUDIT_COMPLETE in texture_roi_v1_attempt2.log. No GPU use or data loss.

# Evaluation geometry blocker: no automatic repair

CPU audit job 47450997 failed after 82 seconds with exit code 1:0, first at
`unetpp_ds/PanTS_00009232`. No prediction, GT, score or source manifest changed.
This is NOT a passing audit and GPU continuation has not been submitted.

## Header-only follow-up on actual saved files

Both UNet++ cells have 570 masks. Comparing all 1,140 saved mask headers to
their saved GT found the **same six mismatched case IDs in both cells**:

| Case | Max affine-element difference | Max voxel-corner displacement (mm) |
|---|---:|---:|
| PanTS_00009232 | 221.067291 | 467.497028 |
| PanTS_00009357 | 225 | 474.369054 |
| PanTS_00009362 | 303.5 | 573.743221 |
| PanTS_00009452 | 217.5 | 451.973001 |
| PanTS_00009480 | 278 | 522.747455 |
| PanTS_00009515 | 2 | 1283.222506 |

All six have matching array shapes. The large last-row physical displacement
despite small matrix-element difference demonstrates why both matrix entries
and physical extent matter. Physical positions use the NIfTI header convention;
this report does not validate the source headers' clinical truth.

For 9232, GT shape is (491,330,89), with diagonal affine
(-0.865234017372,-0.865234017372,5) and zero origin, axis codes LPS.
Both predictions have diagonal (-0.865234017372,+0.865234017372,5), origin
(221.067291259766,-221.067291259766,-220), axis codes LAS. GT qform and
sform numerically agree (codes 0,2), and prediction qform/sform agree (1,1).
Therefore selecting a different valid qform/sform or loosening tolerance
cannot explain away the discrepancy. These are not rounding-scale differences.
The other 564 saved pairs per cell pass the header-only shape/affine comparison;
they have NOT all passed the full decoded-voxel audit, which stopped at 9232.

The diagnostic `evaluation/diagnose_header_geometry.py` reads headers only and
was run read-only via the existing authenticated connection. No test outcomes
were compared to choose a correction or cohort. Python compilation passed.

## What the source code establishes, and what it does not

`convert_pants_to_nnunet.merge_labels` combines organ mask arrays in index space
and copies the header/affine of the first available organ into the combined GT.
It does NOT validate every organ's geometry against the CT or against the other
organs. This is a real missing conversion guard. It does not by itself prove
whether a particular original organ header is wrong or a voxel transform is needed.

The original CT's bytes were copied into imagesTs; predictor provenance records
their SHA256, but not a separate source-header inventory. The old node-local CT
staging is no longer persistent. Saved prediction headers are not an independent
replacement for checking the original CT and individual organ masks.

The existing orthonormality correction only nudges tiny direction nonorthogonality;
it does not fix axis-sign changes or large origin differences. Neither the new
audit nor the final scorer should be bypassed or set to a huge affine tolerance.

## Required resolution before scoring/resume

1. Recover original CTs and original individual masks for the six cases from
   pinned/hashed sources, preserving the existing GT and predictions as evidence.
   Compare CT SHA256 against saved predictor provenance. The public test source
   is the [pinned 901-case archive](https://huggingface.co/datasets/BodyMaps/PanTSMini/tree/3b1cd61108116b58ea5c1ddb3512c1847d965f96),
   about 28 GB; this is a CPU/IO task, not a new GPU experiment. Check available
   persistent space before staging and use a verified reusable CT cache.
2. Inspect every source organ's affine/shape and establish the intended CT voxel
   mapping independently of model predictions. A manual/author geometry check
   may be required where source metadata conflicts; do not pick alignment by Dice.
3. Only after evidence establishes the mapping, create versioned derived GT if
   necessary, preserving originals plus exact voxel/geometry transformation and
   hashes. A header-only change requires proof the voxel arrays already align;
   flipping/resampling requires its own validated mapping. Do not silently
   overwrite readable GT or alter the existing evaluation manifest.
4. Audit all 901 source pairs, not just the first six flagged saved predictions.
   Check implications for conversion guards and the completed study's provenance.
   These test-header findings alone do not prove training was invalid.
5. Resolve any required scoring/provenance protocol revision explicitly, then
   rerun the integrity audit and resume unchanged inference if appropriate.

Do not drop the six cases, score by array shape alone, realign to whichever
prediction overlaps best, rerun training reflexively, or claim the table complete.

## Resolved source/tumor mapping for the six flagged cases

Recovery 47451639 completed in 30m19s (0:0), verifying both original archive
checksums and all six CT fingerprints. Source header inventory shows the
pancreas and pancreatic-lesion masks have exactly the CT affine and shape
in all six cases. Other organ headers conflict (including adrenal-left, the
converter's first reference); the author combined-label file also conflicts.
Do not claim all source organs are correctly mapped.

Independent CPU job **47452439** completed in **63 seconds**, exit **0:0**, with
`ALL_RECOVERED_TUMOR_ARRAY_AND_SOURCE_GEOMETRY_CHECKS_PASSED`. It used the original
frozen conversion code and checked recovered mask hashes before decoding.
In **all six cases**:

- Reconstructing all classes with original merge order reproduces every saved
  combined-GT voxel exactly. Its inherited affine also matches the original
  first-organ reference, explaining the header lineage.
- Extracting `saved_GT == 28` is exactly equal, voxel for voxel, to the original
  `pancreatic_lesion > 0` array. The original target has CT-aligned geometry.
- Thus the target arrays are index-aligned to the source CT according to the
  original tumor annotation's physical mapping. The saved combined-GT header
  incorrectly describes those target voxels. This evidence does not use model
  predictions to select alignment, compare Dice, or choose easier cases.

**These six tumor targets need no voxel flip or resampling.** A future separate,
versioned tumor-reference derivative may use the verified target/CT geometry
while preserving target voxels and original artifacts. That repair has NOT
been applied. The original partial audit still fails on original GT headers;
this narrower source proof must not be mislabeled a full-grid integrity pass.

Before completing evaluation, audit the full 901-case source target mapping,
document the versioned reference/provenance convention, and test the final
auditor/scorer against it. Resume original prediction without changing its
checkpoint, inputs, TTA or outputs. Do not bypass existing source manifests or
change predictions to compensate for GT metadata. No retraining is warranted
solely by the verified tumor-reference header issue; broader training/organ
geometry is not certified by this six-case check.

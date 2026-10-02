# PANORAMA judge: source audit before GPU calibration

Status: preparation only. No weights downloaded, detector executed, or GPU job
submitted by this audit. Protected PanTS evaluation remains untouched.

## Sources actually read

Pinned baseline revision: `d08f2356fa70d9460881fec0aedba4ebd1566c7e`.
Read complete `src/process.py`, `src/data_utils.py`, `requirements.txt`,
`Dockerfile`, and `README.md` directly from the publisher's repository.

- [process.py](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/d08f2356fa70d9460881fec0aedba4ebd1566c7e/src/process.py)
- [data_utils.py](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/d08f2356fa70d9460881fec0aedba4ebd1566c7e/src/data_utils.py)
- [requirements](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/d08f2356fa70d9460881fec0aedba4ebd1566c7e/requirements.txt)
- [Dockerfile](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/d08f2356fa70d9460881fec0aedba4ebd1566c7e/Dockerfile)
- [README](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/d08f2356fa70d9460881fec0aedba4ebd1566c7e/README.md)

## Actual stages and failure attribution

1. Read CT as SimpleITK float32; clinical fields are printed, not supplied to
   the networks in this wrapper. That observation is not proof that every
   external challenge implementation ignores clinical information.
2. Resample stage-1 input to XYZ spacing `(4.5,4.5,9.0)` mm using B-spline,
   rounded dimensions, original origin/direction, identity transform.
3. Predict low-resolution pancreas. Crop the original-resolution CT using
   predicted pancreas bounds mapped through physical coordinates.
4. Margins `[100,50,15]` are divided by spacing: operationally millimeters,
   not README's centimeters. Do not multiply by ten. SimpleITK indexes XYZ;
   its NumPy arrays are ZYX. Crop finishes use exclusive Python slicing.
5. Predict stage-2 probabilities and segmentation using the CE trainer and
   `checkpoint_best_panorama.pth`; stage 1 uses `checkpoint_final.pth`.
6. Build a pancreas-region mask from predicted segmentation labels 1,4,5;
   dilate with a 5x5x5 voxel cube. Zero PDAC probability channel 1 outside it.
   This is voxel-based dilation, not a fixed millimeter radius.
7. Call external `extract_lesion_candidates`; expand candidates to original
   image size and take the maximum candidate-map value as patient score.

There are at least three distinct potential causes of a lower final score:
pancreas localization/crop exclusion, predicted-mask suppression, and PDAC
probability change. Candidate extraction can add a fourth. Reporting only the
final score would not distinguish them.

## Fail-closed execution requirements

- Choose a case's recorded held-out fold for BOTH stages. Wrapper defaults to
  `0,1,2,3,4`; that is unsuitable for out-of-fold tests on its own training
  cohort. Confirm patient-disjoint membership, checkpoint header, and actual
  dataset/plan label mappings before inference. Validation-based checkpoint
  selection still means these are not untouched external-test patients.
- Use unique per-case/per-arm input AND output directories. Wrapper uses a
  fixed `scan` basename and `--continue_prediction`; reusing output directories
  could silently serve an earlier arm's predictions. Never rely on a successful
  process exit alone: require fresh provenance and complete expected outputs.
- `CropPancreasROI` asserts exactly two unique labels before extracting bounds.
  Empty/full masks or unexpected labels are pipeline failures, not tumor-negative
  results. Log and stop that case; do not invent a GT-centered rescue crop.
- Preserve the published B-spline resampling for the primary arm. Its default
  outside value is the image pixel-type enum (`GetPixelIDValue()`), NOT a chosen
  CT background HU. Verify its actual effect; do not silently repair it in one
  arm. Any deliberate correction requires a separately labeled protocol.
- Verify original, reconstructed, and control physical grids: size, spacing,
  origin, direction, finite intensities. Restore the native geometry after VAE
  padding/orientation. Assert crop and probability-map shape agreement before
  expanding to native space; GT must never select a crop or tune predictions.
- Set actual `nnUNet_results` explicitly. Wrapper sets legacy `RESULTS_FOLDER`,
  while Dockerfile supplies `nnUNet_results` separately. Do not assume the
  legacy variable directs an nnU-Net v2 installation to the intended models.
- Do not reproduce this Dockerfile blindly: it clones moving nnU-Net without
  checking out a revision, and leaves SimpleITK and report-guided annotation
  unpinned. Pin/audit a compatible runtime in isolation; never alter the live
  PanTS trainer environment. Candidate-extraction defaults remain UNVERIFIED
  until the actual package version/source is selected and read.
- Test geometry and output reuse protections on CPU toy inputs first. Then
  price one-case original/reconstruction inference before adding cases. Use
  explicit wall-time/memory limits, bounded downloads, no automatic retries.

## Prespecified outputs needed for the tiny calibration

Save locally/private, with hashes and arm provenance:

- stage-1 pancreas mask and actual native crop bounds;
- fraction of GT tumor retained inside crop, plus pancreas overlap (diagnostic
  scoring only; GT never feeds inference);
- unfiltered PDAC probability channel, predicted pancreas/dilated mask, and
  filtered probabilities; fraction of GT tumor excluded by that mask;
- final candidates and patient score, native-grid lesion metrics using locked
  matching/threshold conventions, runtime and peak memory.

Primary comparison keeps the released pipeline identical across arms, including
arm-specific automatic localization. Before/after filtering is failure
attribution, not a replacement primary endpoint. A shared original-derived crop
could be a prespecified secondary diagnostic, never an undisclosed substitute.

If the original lesion is already missed, the paired reconstruction miss is
uninformative about compression-induced detection loss. One positive case
cannot establish sensitivity, false-positive rate, AUROC, or clinical safety.
No added HU-shift correction arm based on the observed outcomes. No test-case
selection by whichever images make the reconstruction look worst.

## Remaining gates

Actual plans/label schema; pinned candidate-extraction code; isolated runtime;
safe bounded checkpoint acquisition; CPU geometry contracts; then one-case
cost/runtime calibration. Detector weights and real inference remain untested.
This audit does not claim that the judge is ready or the idea is proven.

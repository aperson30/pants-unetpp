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

## CPU guard implementation (2026-10-02)

Added `judge_contracts.py` and eight passing tests in `test_judge_contracts.py`.
Local isolated Python3.12.14/NumPy2.5.3/SciPy1.18.1; test runtime 0.026s.
No new dependencies or GPU hours. Tests cover XYZ bounds versus ZYX arrays,
exclusive crop endings, shape/bounds rejection, affine geometry agreement,
nonfinite/singular grids, exact audited label1/4/5 plus 5-voxel-cube dilation,
GT-only crop-exclusion scoring, invalid probabilities, and atomic refusal of
an existing per-arm output directory without overwriting its sentinel.

These are helper unit tests, NOT execution parity with the official wrapper.
The local audit environment has no SimpleITK. Actual image reader/orientation,
B-spline resampling, physical-coordinate crop behavior and NIfTI export must
still be tested in the isolated inference runtime. Candidate extraction and
released model-plan label validation remain open. Helpers are not yet wired
into a real detector runner, so they cannot be claimed to protect a GPU job.

## Actual publisher-code CPU execution (2026-10-02 follow-up)

Installed ONLY task-local CPU wheels under `work/judge_cpu_deps_v1`:
SimpleITK2.5.3, report-guided-annotation0.3.4, tqdm4.67.1, no dependencies
auto-upgraded. Initial package import exposed missing tqdm; explicitly adding
that small dependency fixed the import without touching any trainer environment.
Existing Python3.12.14/NumPy2.5.3/SciPy1.18.1 were reused. This is a selected
research runtime, NOT proof of the historic baseline container's package versions.

Executed independently fetched, unmodified pinned `src/data_utils.py` on toy
images. Source SHA256:
`5f09cb980619e1f459a0163e6808a1bc20188a466c000218719eff36cd1ad68a`.
PyPI0.3.4 extractor source SHA256:
`15119b3bf469d8505c949c6c901321a0941dae340f070270f533dd15cb546990`.
Read the complete installed extractor source, not just its README. PyPI code
differs from current upstream main; do not substitute main and claim same runtime.

Nine upstream-execution tests plus eight guard tests PASS, 17 total in 0.050s.
Tests exercise NIfTI write/read geometry, actual B-spline resampling, actual
physical crop with anisotropic spacing AND rotated direction, empty localization
assertion, exact mask helper parity, native expansion geometry, and extractor
edge cases. Tests require explicit source path and verify source/version hashes;
without it the optional upstream suite skips, rather than implying execution.
SimpleITK emits a NumPy2.5 deprecation warning; no test failure. This tiny-input
execution does not validate large-volume memory, nnU-Net preprocessing/network
load, Linux/HPC runtime compatibility, historic container parity or detection.

Verified extractor0.3.4 default is `dynamic-fast`: threshold is scan-wide
maximum/2.5; 26-connected components with <=10 voxels are discarded, retained
component values become their maximum probability. Default-fast does NOT apply
the exposed five-candidate limit or adjacency removal used by iterative mode.
Toy tests retained seven components, removed a ten-voxel component but retained
eleven, and erased an unchanged twelve-voxel .15-probability component when a
remote .8 component raised threshold to .32. These are algorithm contracts,
NOT patient evidence. Preserve this primary pipeline; don't relax minimum size
or thresholds after seeing reconstructed lesions. Save raw maps for attribution.

### Repository model plans and label schema read

Read complete plans/dataset JSONs under pinned `src/nnUNet_results/` for both
stages, plus customTrainerCEcheckpoints.py (CE trainer subclass, no replacement
network). Stage1 is background0/pancreas1; stage2 background0/tumor1/veins2/
arteries3/pancreas4/pancreaticduct5/commonbileduct6. Both specify CT and
SimpleITKIO, numTraining2238, identity transpose. Stage1 3d spacing ZYX
9/4.5/4.5 mm, patch48x96x96, CT clips[-152,253], mean42.803375/std93.941116.
Stage2 spacing1.5/.747/.747 mm, patch48x128x288, clips[-55,430],
mean109.620674/std77.439339. These fixed intensity statistics must come from
the released plans, not be estimated separately per reconstruction arm.

Stage1 plans have LEGACY UNet_class_name/base-feature fields; stage2 plans have
new architecture metadata. Selected nnU-Net runtime must explicitly support
both. Repository JSONs are evidence of intended configurations, not verification
that the weight archive contains identical JSONs or loads in that runtime.

### Gates still open

Bounded checkpoint acquisition and matching archive metadata; compatible pinned
nnU-Net runtime supporting both plan schemas; actual model-load/preprocessor
checks; runner integration of guards and provenance; priced one-case inference.
No weights downloaded, GPU job submitted, or protected evaluation modified.

## Archive correspondence and runtime parsing verified; paused before weights

Read only plans/dataset JSONs directly from both official Zenodo11160381
archives using bounded HTTP ranges: 3567 bytes stage1, 4124 stage2. Parsed
JSONs match pinned repository exactly. A separate stage1 directory-only check
used2163 bytes. Thus correspondence now verified, not just presumed.

Live authenticated DeltaAI connection worked. Existing environment is
PyTorch2.10.0+cu129/nnUNet2.8.1. Read its ConfigurationManager conversion code;
ran actual PlansManager/label-manager parsing on both published plans in memory.
Legacy stage1 converts successfully to PlainConvUNet, five stages with features
32/64/128/256/320 and two heads. Modern stage2 resolves six stages with features
32/64/128/256/320/320 and seven heads. Required architecture/import symbols
resolve. NO network instantiated, checkpoint loaded, CT preprocessed or model
inference performed. Shared training environment unchanged.

Prepared fetch_judge_fold4.py, fixed official URLs and fold4 member allowlist.
No pickle execution; streams only selected members, verifies ZIP CRC and SHA256,
limits size/bytes/requests/time, refuses overwrite and retains failed partials.
Three transfer tests and five existing range tests pass. RangeReader default
32-request limit preserved; explicitly bounded up-to512 option supports large
member transport. Transfer helper has NOT been launched; weights NOT downloaded.
Expected total two checkpoints378,590,434 uncompressed bytes (~361MiB).

User requested pause here. Next: review/launch bounded transfer, then isolated
actual network-load/preprocessor tests; GPU inference only after those pass.
No new GPU spending or protected evaluation modifications.

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

## Bounded transfer launched; compute connection needs reopening

After user resumed, reviewed downloader and re-ran three transfer/six range
tests successfully. Streaming uses4MiB blocks to reduce HTTP overhead; default
metadata timeout remains180s, transfer opts into300s per archive and660s total
watchdog. Existing range-call default32 preserved; bounds remain explicit.

Launched a frozen local snapshot detached/hidden, PID22644 at launch. Source
SHA25615b5db7fa92cdcda97aad84af03845b6b07ef0c8651d754485ca622969ec9483.
Outputs `work/judge_weights_fold4_v1`, logs `work/judge_transfer_v1.*.log`.
Last inspection: stage1 partial exists, zero bytes, no success/error output yet;
NOT completed. No retry or duplicate transfer. Partial is not usable weights.

DeltaAI compression socket and Bridges2 progress socket subsequently absent;
requested user reopen DeltaAI. No model-load attempt or GPU job submitted.
User explicitly requested allocation resources: actual model load/inference must
run on allocated compute, not HPC login nodes. Do not reserve a GPU simply for
this local download. Remaining screening cap applies, including failed jobs.
Restore authenticated access, verify transfer completion/hash and staging space,
then prepare one bounded compute check. Protected evaluation untouched.

## Transfer completed directly on DeltaAI (2026-10-02)

Local detached transfer remained zero-byte and was stopped only after checking
its exact executable/command line matched our own helper (PID22644). Zero-byte
partial retained, no retry running locally. Resumed authenticated DeltaAI master
works. Actual `quota -s` reports home6180M used of102400M soft/103G hard;
this is user quota evidence, unlike raw filesystem free space. Shared projects
still near quota; no model files staged there.

Copied frozen helper+RangeReader via authenticated scp to
`/u/asanjeev/compression_probe_smalloutputs/judge_stage_fold4_v1`, mode700.
Source hashes matched. Detached remote transfer PID2688525 at launch, timeout670s
outside helper's660s watchdog, no retry. Completed marker and both CRC/hash
results verified. Public model members only, no clinical image upload.

| Stage | Uncompressed bytes | Downloaded archive bytes | Requests | SHA256 |
|---|---:|---:|---:|---|
|Pancreas|132172006|122696071|44|81fba70fc62e7d0aaf8f5a3c3e5ddc9bd88fc326b35736444e970812b1cac6b1|
|PDAC|246418428|228900174|71|c5a201f0f05030f4e367332f5f64caad66138fe56ea1e1fe1c2f2d9b6c635d1c|

Total351596245 archive bytes rather than all five folds. Archives lack ETag;
ZIP CRC, TLS/public release URL, and local SHA provenance are checked, NOT
claimed publisher-signature authenticity. `models/completion.json` true;
weights_loaded false. No GPU hours used. Protected evaluation untouched.

Prepared `judge_load_smoke.py` after reading installed network-construction
API signature. Requires scheduled allocation, one visible CUDA GPU, completed
transfer, matching hashes/fold/archive metadata, strict state load, finite tiny
FP32 synthetic output. Uses `torch.load(weights_only=True)` with NO unrestricted
fallback or blanket globals allowlist. Unsupported metadata fails cleanly for
review; no repeated allocations until cause understood. No full CT inference.
AST syntax check only passed locally (local runtime lacks torch); smoke script
NOT executed, NOT submitted and NOT certified runtime-ready. Next review and
one short capped allocated-compute smoke, then actual preprocessing/pair setup.

## Allocated load gate submitted, not completed (2026-10-02)

Static metadata audit using torch.serialization.get_unsafe_globals_in_checkpoint
found ONLY numpy.core.multiarray.scalar/numpy.dtype in both files; no tensor
weights loaded. Bounded pickletools inspection (no pickle execution) foundf4/f8
dtype descriptors. Narrow restricted loader supports that scalar's NumPy1/2
module paths, np.dtype and explicit float32/64dtype classes, never arbitrary
globals or weights_only=False. First synthetic test exposed NumPy2_core alias;
fixed and repeated successfully before any allocation. This is serialization
compatibility work on official public model files, not security probing.

Frozen actual smoke source SHA256:
`d5fc186b3f47ddbbac9ccad35a6fdcb6f5829ea256763a4c5f22964c18891dcf`.
Remote home `judge_load_gate_v1`, source.sha256 verified, bash syntax checked,
synthetic float32/64+tensor roundtrip passed. No real weights loaded yet.

Submitted HELD job3294898, accountbdyo-dtai-gh, oneGPU/twoCPU/8GB/5min,
no requeue. Regular ghx4 test-only estimate was a week later versus near-term
interactive estimate (both uncertain). Updated SAME held job partition to
ghx4-interactive; no duplicate actual job. HeldReqTRES reportedbilling1000;
conservatively reserve2x interactive charge <=.166667GPUh, verify actual running
AllocTRES/accounting later. Prior screening estimate.448611, worst-case total
with this reservation.615278of2. Posted allocation debit still unverified.

Initial ordered-string release guard rejected field-order mismatch; no release
occurred. Replaced with independent exact-field checks, verified hold/partition/
time/resources, released job. Lastlive statePENDING Priority, elapsed0, no log/
result. NO successful model load or synthetic forward claim yet. Slurm test-only
IDs3294897/3294901 are not submitted jobs. Frozen batch script uses regular
directive but actual held-job partition override documented above; no post-submit
code mutation. Protected PanTS evaluation untouched; no automatic retry.

## Actual model-load gate PASSED (job3294898)

Slurm COMPLETED exit0:0,14 allocation seconds,AllocTRESbilling2000/cpu2/GPU1/
mem8G. Charge-equivalent14/3600*2=.00777778GPUh; updated campaign estimate
.45638878of2 (prior.448611 approximate). Actual posted debit remains unverified.
No retry/continuation submitted. Sourcehash check passed in allocation.

Both real official fold4 checkpoints passed restricted loading, exact archive/
checkpoint plans+dataset correspondence, strict state_dict load and finite
synthetic outputs under PyTorch2.10+cu129. Pancreas shape1x2x16x64x64,
PDAC1x7x16x64x64. GPUpeak127192064/184667648bytes. Synthetic forward timing
.327832/.013786s includes different warmup states and is NOT a comparative
benchmark, full-CT inference estimate or accuracy result. Legacy plans warning
is expected conversion; strict realstate load succeeded.

Scalar result preserved as judge_load_result_3294898.json. Still no CT passed
through this detector, no tumor-recall result, no final candidate map, no claim
of historic container parity. Next real preprocessing/paired-inference harness
with guard integration, raw/masked/candidate outputs and separately capped cost.
Protected PanTS evaluation untouched.

### Three-arm diagnostic pilot released (results pending)

DeltaAI3295187: one GPU, 32GB, 20min hard cap, interactive 2x charge factor,
no retry. Held resource fields/source hashes verified before release.
Campaign worst-case estimate1.123056/2 charged GPUh; posted debit unverified.
13 CPU tests and exact real CT orientation roundtrip passed. Compare native,
clipped and whole-volume posterior-mean MAISI reconstruction with frozen
fold4 detectors. GT never sets crops; preserve raw/masked/GT-matched candidate
scores and crop/mask inclusion. No tuned clinical threshold or recall claim.
Transport-selected validation case is not a cohort or external test; frozen
judge response loss alone cannot prove irreversible information loss.
Private inputs/maps stay under home paired_judge_pilot_v1; last snapshot
PENDING. Next check completion JSON and accounting, not just allocation state.

Correction:3295187 failed before detector outputs (parameters=None API bug),
105s/0.058333 charge-equivalenth. Fixed single-fold parameter list; real-state
CPU regression passes both checkpoints and installed prediction weight loop
(neural sliding-window calculation mocked), plus four geometry tests. No
clinical results yet. Replacement3295395 released PENDING with identical
20min/no-requeue cap, v2 private folder; campaign worst-case1.181389/2 chargedh.

Completion update:3295395 COMPLETED0:0,126s,0.070charge-equivalenth. Native and
clipped scores identical; mean GT PDAC0.171565 to0.023676 after reconstruction,
all GT retained by crop/mask. No recall or irreversible-loss claim. Scalar
JSON preserved. Locked replication plan in JUDGE_REPLICATION_PLAN_20261002.md;
3295549 released PENDING, same3arms on two additional histopathology patients.
Resource cap explicit56M including padding and >=110GBfreeGPU, no geometry
changes. Selection is size-biased and before predictions. Worst-case campaign
1.251389/2chargedh. Completion/accounting still required for replication.

Resource amendment:3295549 FAILED at110GB free-memory guard in38s, no outputs.
Official architecture lists96GBGPU/120GBCPU despite misleading120gb Slurm label.
Large selected patient remains blocked, no replacement. Smaller locked patient
3295577 released PENDING,33M padded cap and65GB actual-free gate, same pixels/
model/three arms.10min hard cap/no requeue. Campaign worst-case0.939167/2;
see plan amendment and handoff39. No clinical result or automatic retry.

3295577 subsequently failed Slurm step OOM after114s, no arm results,0.063333h.
CUDA free/total101.50/102.09GB passed gate; host32G limit and installed MONAI
GroupNorm CPU-concat path identified. Sampled RSS is not actual kill peak.
3296011 released PENDING with host96G, sameGPU/billing2000/10min and unchanged
scientific settings; stage memory logs added. Five CPU tests pass1.408s.
Campaign max1.002500/2chargedh. Larger case stays blocked; no auto-retry/evalchange.

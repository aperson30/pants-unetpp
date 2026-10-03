# Independent judge: a concrete lead, not launch-ready

Verified primary-source lead: DiffTumor authors release pancreatic tumor
segmentation checkpoints for U-Net, nnU-Net and Swin UNETR, trained on real and
synthetic tumors. U-Net weight URL is explicit in their README:
https://github.com/MrGiovanni/DiffTumor
https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/unet_synt_pancreas_tumors.pt

Their actual validation code defines a MONAI3D residual U-Net with channels
(16,32,64,128,256),3outputs, strides2x4,num_res_units2. It applies RAS,
1mmresampling, HU[-175,250]->[0,1], minimum96cubepadding,96cube slidingwindows,
batch1,75%overlap,Gaussian weighting. It then replaces the organ channel with
an external organ_pseudo mask and suppresses tumor outside that mask.
https://github.com/MrGiovanni/DiffTumor/blob/main/STEP3.SegmentationModel/validation.py

Why this matters: a published checkpoint alone is NOT a verified end-to-end
judge. Need identify/pin external organ-pseudo producer and label map, validate
preprocessing/native-grid restoration and strictstate loading. Do not treat
their label2tumor as PANORAMAlabel1 or PanTSlabel28. Avoid their optional
strict=False load path for our compatibility preflight. Legacy transforms
AddChanneld/metadatainversion require checked MONAI1.5.1 equivalents, not a
blind modernport. Runtime needs real-shape calibration before GPUhours claim.

The checkpoint is a separatelytrained model, not necessarily patient-independent
or clinicallyindependent. MSD and PANORAMA datasetnames differ but patient
disjointness, checkpointfold and healthy/synthetic-source overlap are not
verified. Another CNNjudge can still share the same domainshift; agreeing
frozenjudges would strengthen task-response evidence, not prove irreversible
loss or replace a readerstudy. No known clinicalthreshold from our probe.

Cheapest safe nextpreflight: pin source/weights, verify license/provenance,
restrictedweights-onlyCPUload, exactarchitecture/keyshape match, tinyCPUforward
and transform/nativegrid regression. No trainable adaptation or new GPUjob
authorized by this document. Only then propose a bounded full-context inference
with both native/reconstructed controls and frozen tumor-specific metrics.
Do not silently substitute GT organs, a reducedcrop or unverified mask to make
the pipeline easier. OneGT-assisted score would not be end-to-end detection.

## CPU preflight completed — 2026-10-02 16:26 PDT

Pinned GitHub source: `ad45fb4e8bbec94105938e69deaca62fd6812c0c`.
Pinned Hugging Face revision: `089ba0f3f94a7858603a55106791d7d977d7bc0b`.
The public model API reports the pancreatic U-Net file as 19,260,008 bytes,
SHA256 `c3d7ca09aa57ce20e3537c5dfae611b4759bbd000f626fcb2d4bb180b1090d5d`.
Downloaded only that file into private home staging (not project quota),
verified exact bytes/hash before loading. No patient images downloaded here.

`difftumor_cpu_preflight.py` passed on PyTorch 2.10.0+cu129 / MONAI 1.5.1:
63 state tensors, 4,807,482 parameters, exact key match and strict=True load
into the published architecture. One-thread CPU-only 32-cube forward produced
finite `[1,3,32,32,32]` outputs. CUDA hidden; execution timeout 55 seconds.
This is architecture compatibility, NOT accuracy or full-volume runtime.
Result: `difftumor_cpu_preflight_result_20261002.json`.

First restricted load correctly stopped on legacy NumPy scalar/dtype metadata.
After static inspection, allowed only those specific types and float32/float64
dtype classes; retained weights_only=True. No arbitrary-global or unsafe-pickle
fallback. Checkpoint fields are only best_acc, epoch, state_dict: no fold or
patient-training manifest, so checkpoint-fold identity remains unverified.

Important source discrepancy: GitHub LICENSE identifies CC BY-NC-ND 4.0;
Hugging Face card metadata says apache-2.0. Do not assume one blanket permissive
license for both code and weights, or claim redistribution rights resolved.
Normal academic investigation here does not establish permission to distribute
derived artifacts. The original weights remain private, not committed.
Sources:
https://github.com/MrGiovanni/DiffTumor/blob/ad45fb4e8bbec94105938e69deaca62fd6812c0c/LICENSE
https://huggingface.co/api/models/MrGiovanni/DiffTumor?blobs=true

The pinned GitHub tree contains 120 precomputed pancreas NIfTI masks in
`STEP3.SegmentationModel/organ_pseudo_swin_new/pancreas/`, with MSD-style
`pancreas_*.nii.gz` names. These are NOT a mask producer for our PANORAMA cases.
Targeted inspection of README, validation.py, main.py, monai_trainer.py,
hg.sh, postprocess.py, TumorGeneration/utils.py and INSTALL/FAQ did not identify
a pinned mask-generating checkpoint or procedure. This is a scoped search,
not proof no producer exists anywhere. No mask downloaded or substituted.

Remaining gates: identify organ-mask producer (or explicitly define a different
raw-head diagnostic), verify full transforms and native-grid inversion, and
document training overlap uncertainty. A raw full-volume tumor probability
comparison could test whether a separately trained network also responds to
reconstruction, without relying on the external mask. But that would NOT be
the official postprocessed DiffTumor detector, clinical detection, or its
published benchmark result. Keep this distinction explicit before launch.
Prefer matching the same native/mean/sample controls over training a new judge;
do not fit or choose a judge based on which supports our hypothesis.

No new GPU job, no GPU hours, no independent tumor score obtained here.
Campaign estimate remains 1.109722/2 charged GPU-hours (posted debit unknown).
PANORAMA fold4 remains the ONLY actually executed tumorjudge. Protected
evaluation job 47320183 untouched. Full second-judge inference is not ready.

## Follow-up: producer family identified; geometry gate passed

The primary paper appendixE.2 names reference55 (CLIP-Driven Universal Model)
as the source of organ pseudo labels. Its official repository publishes Swin
weights and pred_pseudo.py. Exact historical mask weights/runtime and
checkpoint fold overlap remain unverified; no extra weights downloaded.
https://arxiv.org/html/2402.19470v2
https://github.com/ljwztc/CLIP-Driven-Universal-Model

Three synthetic CPU regressions now pass for the modern image-transform port:
RAS/1mm/padding and inverse shape/affine; flipped/permuted anisotropic grids;
linear-coordinate landmark restoration; invalid shear/channel rejection.
Tests caught and fixed Invertd skipping restoration for plain Tensor predictions:
wrap prediction in MetaTensor so modern MONAI uses the original trace.
Explicit orientation labels prevent changing-default behavior.

Resampling is NOT an exact intensity inverse. Phantom2.5mm slice spacing
produces a predicted0.2native-voxel edge clamp (0.5mm) due to integer1mm
extent rounding; interior landmarks restore within1e-4voxel. Tests assert
that analytically expected edge value, not a silently relaxed accuracy gate.
These are synthetic modern-runtime contracts, not legacy-version numerical
parity or clinical accuracy. Real-case grid assertions still required.

See NOVELTY_AND_DECISION_GATE_20261002.md for the explicitly different
raw-head protocol and its predeclared investment gate. No GPU submitted.

## Final follow-up: raw-head diagnostic completed, negative investment gate

The preceding no-GPU statements describe the earlier preflight only.
Job3297419 subsequently completed0:0 in209 allocation seconds. Strict frozen
checkpoint loading, real-case geometry contracts, all six arms and exact
three-channel repeats passed. A separate CPU audit verified saved-map hashes
and scored all1756 annotated voxels; GT was not an inference input.

Native tumor mean .00329347 fails the predeclared .01 adequacy floor.
Mean/seed0/seed1/native ratios were .300/.404/.272, but seed2 was1.679.
All arms, including native, had zero tumor argmax overlap with annotation.
Thus this is NOT confirmation of clinically introduced misses or uniform
tumor suppression. See difftumor_audit_3297419.json and the final decision report.

The independent-network runtime check is resolved for this raw-head question.
The official external-organ-mask pipeline, exact historical producer weights
and patient-independent training provenance remain unverified. These limits
are retained, not bypassed; this weak-native result does not justify spending
more GPU hours to reconstruct a clinical validation pipeline or select a
favorable replacement judge. Campaign estimate1.225833/2 charged-equivalent
GPU-hours; posted debit unknown. Main evaluation47320183 untouched.

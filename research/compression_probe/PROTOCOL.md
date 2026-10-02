# Compression-only lesion-preservation screen

Status: first attempt 3289813 hit host-memory OOM; retry 3289833 COMPLETED.
Measured total cost and limited fidelity observations: results_20261001/SUMMARY.md.
Separate from segmentation experiment.

## Verified inventory (October 1, 2026)

- Bridges-2 connected via bridges2-compression.sock; 47320183 was PENDING.
- Existing nine-case calibration labels all have ZERO class-28 voxels.
  Metadata-level `tumor?` selection did not guarantee pancreatic-lesion voxels.
  Do not use this subset as positive tumor evidence or retroactively claim
  previous timing/parity checks covered positive class-28 targets.
- Existing full development validation summary has 174 tumor-positive cases
  among 1800. Candidate IDs come ONLY from reference counts, never model Dice.
- MONAI is absent from the existing venv. Do not install into that environment;
  it serves the frozen segmentation evaluation.
- Official tutorial migrated to NVIDIA-Medtech/NV-Generate-CTMR.
  Source revision examined: 5cb04e82fed71f2fe64a2617ab695be1f5a37fea.
- Public weight nvidia/NV-Generate-CT, revision
  430f2c82a96dce44b455de5876d43a5a9f753cb2,
  models/autoencoder_v1.pt, 83,831,868 bytes, SHA256
  1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3.
  Download ONLY this file, not the full generator/mask/data bundle.

## First screen, not publication-grade evidence

1. Obtain original CT and explicit pancreatic-lesion masks for a few positive
   DEVELOPMENT cases; confirm actual mask positivity and geometry. Small
   candidates by native voxel count: 3548 (36), 6854 (66), 133 (126), 5782 (126).
   Larger controls: 8132 (3729), 3700 (3750), 605 (3948).
   Counts are not mm3; use physical volume before size-stratified selection.
   Prefer existing caches or individual downloads; do not download the full
   1.1TB corpus for a handful of cases. Include a cached negative control.
2. Pin case selection before inspecting reconstruction/model outputs. Check
   overlap with public generator training data where identifiable; no claim
   that all development cases are unseen by MAISI. Reserve 901 test cases.
3. Use isolated dependencies and scratch/output directory, explicit input hashes,
   model hash, source/version/config manifest; no trainer modifications.
4. Follow pinned official CT transform: RAS, HU [-1000,1000] mapped to [0,1],
   original spacing, divisible-by-four padding, no random augmentation. Recheck
   installed API and exact weight/config compatibility with strict state loading.
5. Keep preprocessing-only control. Compare native CT vs control separately,
   then control vs reconstruction on identical grids. Never conceal resizing,
   clipping, orientation or affine repair under a compression result.
6. Begin with deterministic posterior-mean reconstruction (not diffusion).
   Confirm encode/decode API and record this departure from sampled posterior
   forward; a preservation result does not prove sampled reconstruction safe.
   If necessary follow with fixed-seed posterior samples, cost included.
7. Avoid small decoder tiles in the primary diagnostic. If whole-volume memory
   is too large, stop and choose a documented alternative; do not silently
   attribute stitching artifacts to compression. A contextual ROI can be a
   preliminary diagnostic only, never a whole-volume detection conclusion.
8. Time one case first on scheduled GPU, one process, inference_mode/eval,
   conservative precision; capture peak allocated/reserved memory and runtime.
   No compile/autotune search needed for a handful of forward passes. Bound
   walltime and disk consumption; do not run GPU work on login node or consume
   the queued two-GPU evaluation allocation. Establish numeric budget before
   submitting or expanding the screen.
9. CPU metrics: per-component physical size, lesion/boundary HU MAE, signed
   lesion-to-local-ring contrast and CNR. Undefined contrast/CNR stays null;
   do not hide it in an aggregate. Negative controls cannot demonstrate tumor
   preservation. Image quality scores or detector disagreement are not clinical
   realism. Same-grid physical metrics reject shear and unspecified units.
10. Visual overlays at fixed HU window, original/control/reconstruction, matched
    slices; source-mask overlays do not prove synthesized lesion existence.
    Only afterward consider paired segmenter inference with unchanged settings.
    Original-image detector misses are evaluator limitations, not VAE erasure.

## Decision

Damage visible after preprocessing alone -> address that confound first.
Damage after VAE alone -> examine compression fidelity before depth routing.
Preserved in this small screen -> proceed to noise reconstruction, not a claim
of clinical equivalence. No absolute quality cutoff invented from this pilot.

## Sources

https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/tree/5cb04e82fed71f2fe64a2617ab695be1f5a37fea
https://huggingface.co/nvidia/NV-Generate-CT/tree/430f2c82a96dce44b455de5876d43a5a9f753cb2
https://github.com/Project-MONAI/tutorials/blob/main/generation/maisi/README.md

## Authorized MSD pivot (October 1)

The user chose to avoid PanTS for this quick screen and approved the proposed
0.25-hour initial cap. Target: one DeltaAI GH200, ghx4-interactive, 7min maximum.
The partition charges 2x, so this is <=0.117 physical GPU-hours and <=0.234
allocation-equivalent hours, provided the verified request reserves one GPU.
Request only eight CPUs/32G CPU RAM; do not accidentally reserve a second
GPU-equivalent through excessive CPU/RAM, or request an exclusive node.
Original submitted header requested 7m30s, rounded by Slurm to 8min. Caught
while ON HOLD; enforced limit corrected to 7min via scontrol before release.
The repository template now requests 7min directly; the Slurm-stored first
driver retains its original header. Predictor/metric code is unchanged.

First attempt ran 1m43s, one GPU, billing=2000 (interactive 2x): 0.02861
physical GPU-hours, ~0.05722 charge-equivalent hours. Step MaxRSS=33839296K
and OUT_OF_MEMORY, consistent with exceeding requested 32G host memory.
No reconstruction completed; exact allocating operation unconfirmed.
No CUDA-out-of-memory exception was recorded. This is not a quality result.

Retry 3289833: eight CPUs, 96G RAM, STILL one GPU and billing=2000, verified
while held. Limit FIVE minutes; process timeout 260s; max-cases=1; no automatic
retry/requeue. Both attempts together <=0.11195 physical GPU-hours and <=0.2239
charge-equivalent hours. Stored retry script byte-matches the repository
template at submission. Template now describes this retry, not attempt 1.
Only memory request/case count/time cap and diagnostic stage logging changed;
same selected case, weight, full-volume geometry and FP32 reconstruction.

Data mirror: Angelou0516/msd-pancreas at
327dd551a9e51295c34e14f311d8330ef44b6cac. This is a third-party redistribution,
not independently byte-matched to the original official archive; explicitly
retain that provenance limitation. Tumor label is TWO, not PanTS label 28.
Four prespecified masks: pancreas_001,004,005,006. Select up to two positive
cases by physical tumor burden before reconstruction. Downloaded positives:
005 (7602.96 mm3), 006 (13615.86 mm3). These are not voxel-tiny lesions; the
first run is a pipeline/fidelity screen, not evidence covering tiny tumors.
CTs + VAE total ~137MiB; full dataset/generator never downloaded.

GPU reference precision is FP32, including normalization: norm_float16=False.
This is deliberate and logged; default norm_float16=True without autocast
produced a half-input/float-weight error on CPU, caught BEFORE allocation.
CPU gate includes strict checkpoint loading plus an 8-cubed encode/decode.
MONAI 1.5.1 is imported from its pinned pure-Python wheel; incomplete target
install is retained as failed setup evidence, never used or added to the main
environment. Verify code, wheel and input-manifest hashes before submission.

Published runtime guidance, not a measurement of this probe:
https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/running-jobs.html
https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/job-accounting.html

## Follow-up: small-lesion candidate search (CPU only)

`inventory_small_msd.py` defaults to the first 24 lexicographically sorted masks
at the SAME pinned dataset revision, sequentially and with download/volume
bounds. It downloads no CTs, starts no GPU job, and writes into an exclusive
new directory. Remote execution is detached with a ten-minute timeout and
one-thread CPU-library limits. Original pilot artifacts are not overwritten.
An optional bounded window (`--start`, `--count`, maximum 48 masks per run)
supports a separate follow-up without silently changing the original inventory.

Rank positive cases by total physical GT tumor burden before inspecting any
new reconstruction. <=1000mm3 (1mL) is an exploratory screening cutoff, not
a clinical-stage definition. This bounded subset is not a representative
benchmark and the smallest case here is not necessarily the dataset's smallest.
Use whole-volume reconstruction with the same preprocessing-only control;
do not substitute cropped/tiling results for native whole-volume fidelity.

Original cap accounting: 0.17056 allocation-equivalent hours spent, leaving
0.07944 of 0.25. Another allocation comparable to the completed pilot would
exceed that remainder. Additional GPU work requires explicit budget approval;
the CPU mask inventory is not itself a GPU submission authorization.

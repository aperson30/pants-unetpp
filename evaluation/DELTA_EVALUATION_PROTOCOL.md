# Delta post-training evaluation protocol

This is the reviewed plan for the four 1,000-epoch grid cells. It is prepared,
not submitted. `delta_evaluate_grid.sbatch` is the launch artifact.

## Readiness gate

Do not submit merely because all four `checkpoint_final.pth` files exist.
nnU-Net writes that checkpoint before its final validation finishes. First wait
for the grid job (including any bounded successor jobs) to exit successfully,
then require each fold's validation directory to contain exactly 1,800 readable
segmentations. The evaluation job enforces both the four checkpoints and the
four validation counts. It refuses to fabricate a successful cross-validation
stage from partial outputs.

## Storage and execution

- Request one H200 node with two GPUs, matching the validated PyTorch
  2.10.0+cu129 stack used for training.
- Download and convert only the 901-case PanTS test split and answer key into
  job-local `/tmp`. Never restage or preprocess the 9,000 training cases.
- Run two model predictions concurrently, then the other two. Every model uses
  a separate local output directory and probability CSV.
- Extract the maximum class-28 softmax probability immediately, delete the
  large probability archive, and rewrite the segmentation to labels `{0, 28}`.
  This preserves all five tumor metrics while greatly reducing persistent data.
- Persist the tumor-only masks, probability CSV, per-case audit CSV, metric
  JSON, and logs under `/work/hdd`. The script copies each model only after its
  exact 901-case audit passes.

## Metric definitions and unresolved publication detail

The scorer preserves the definitions used by the project's earlier 901-case
run:

- DSC: class-28 Dice, mean over ground-truth-positive cases.
- P-Sen: a positive patient is detected if any tumor voxel is predicted
  anywhere.
- T-Sen: a 6-connected ground-truth tumor component is detected if any
  predicted tumor voxel overlaps it.
- Spe: a negative patient is correct only when no tumor voxel is predicted.
- AUC: ROC AUC using the maximum class-28 softmax probability in each volume.

The public PanTS paper and README name the metrics and describe patient- and
tumor-level detection, but do not publish enough evaluator detail to establish
the component connectivity, DSC averaging set, or continuous AUC aggregation.
R-Super's released evaluation instructions describe a different confidence- and
volume-threshold sweep. Therefore the JSON records all choices and calls this
the **project protocol**, not a proven bit-for-bit reproduction of an
unpublished PanTS leaderboard evaluator. Confirm these choices with the PI
before presenting direct leaderboard claims; do not change them between grid
cells.

## Failure behavior

Any failed nnU-Net batch, missing/corrupt NIfTI, missing probability, duplicate
CSV row, non-finite value, shape/affine mismatch, or case-ID/count mismatch is a
hard failure. A failed job cannot print the completion marker. Re-running is
safe: complete case/score pairs are reused, while partial pairs are deleted and
recomputed.

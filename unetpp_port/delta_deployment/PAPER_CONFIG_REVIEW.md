# Fifth configuration: launch review

Status: **prepared, not submitted**. The launch artifact is `paper_config.sbatch`.

## Scientific definition

This configuration changes the standard UNet++ DS-on cell in the two coupled
ways specified by the UNet++ paper:

1. every full-resolution decoder branch receives the same relative loss weight;
2. inference ensembles every decoder branch after the output nonlinearity.

The loss weights are the paper's literal, unnormalized `eta_i = 1` for every
branch. The branch losses are summed, not averaged. This is a resolved design
decision based on the paper's explicit text; it intentionally does not copy the
official nnU-Net integration's fallback to nnU-Net's default decaying weights.

This choice introduces a real interpretation limitation. With N branches, the
total loss and gradient can be roughly N times the scale of a normalized
equal-weight objective. Because the learning rate is otherwise unchanged, the
effective SGD step scale differs from the four main grid cells. The fifth
configuration is therefore a reproduction of the paper's coupled supervision
and ensemble rules, not a clean isolated ablation of those rules at matched
optimization scale. Any performance difference may be partly attributable to
the loss/LR-scale shift and must be reported that way.

The previous implementation averaged raw logits. That is not the paper's
ensemble: softmax(mean(logits)) is a normalized geometric mean, not the
arithmetic mean of branch predictions. The reviewed implementation now returns
`log(mean(softmax(branch_logits)))`. nnU-Net's required downstream softmax
therefore recovers the arithmetic probability average exactly for PanTS's
exclusive 29-class output. Training logits and losses are unchanged.

The implementation still generalizes UNet++ to the encoder depth selected by
nnU-Net v2 rather than forcing the paper's original four decoder branches. That
is intentional: forcing another depth would change the planned architecture and
make the fifth cell cease to be a controlled comparison with the UNet++ grid
cell. Describe it as "paper supervision and ensemble rules" rather than a
bit-for-bit reproduction of the original 2D architecture.

## Launch design

- One H200, physical batch size 4, 1,000 epochs, same PyTorch
  2.10.0+cu129/cuDNN stack and quality-neutral trainer plumbing as the grid.
- Sparse epoch validation remains enabled, but final 1,800-case validation is
  mandatory.
- The job will not start scientifically meaningful work until the reference
  UNet++ grid cell has a saved plan, final checkpoint, and exactly 1,800
  readable final-validation outputs.
  This gate runs before any shared trainer-package file is written, so an early
  or accidental submission cannot alter the environment of an active grid job.
- It regenerates the full-data plan, changes only batch size to four, and
  requires exact JSON equality with the reference grid plan.
- It constructs fold 0 from the reference cell's exact 1,800 validation case
  IDs, eliminating split drift across independent node-local preprocessing.
- Only 9,000 training images are downloaded. The 901 test images are omitted;
  final test inference belongs to the separate `delta_evaluate_paper.sbatch`
  job. This is not a reduced-data experiment: the exact installed nnU-Net v2
  training entry point and trainer package were searched for `imagesTs` and
  contain no training-time consumer of that directory. All 9,000 training
  images, all training labels, preprocessing, augmentation, 1,000 epochs, fold
  validation, and physical batch size four remain unchanged.
- Source files are copied explicitly into the installed trainer package and
  hash-checked. Smoke, trainer wiring, compile-target, target-reuse, confusion
  count, and sparse-validation contracts run before staging/training.
- A bounded after-any successor provides recovery across the 48-hour limit.
  A final checkpoint without completed validation triggers `--val`, not a false
  success. The successor is cancelled after a verified completion marker is
  atomically written.
- Completion requires a final checkpoint, 1,800 readable validation outputs,
  exact validation-ID equality with the reference grid cell, and a marker that
  records commit, checkpoint SHA-256, trainer, epoch count, and batch size.

## Final pre-submission checklist

1. The four-cell grid and any successor jobs have exited successfully.
2. The reference UNet++ DS-on folder contains `plans.json`, a final checkpoint,
   and exactly 1,800 validation NIfTIs.
3. Pull the reviewed commit onto Delta; require a clean checkout.
4. Run `bash -n paper_config.sbatch` and `sbatch --test-only` on Delta.
5. Confirm the isolated PyTorch 2.10 overlay and trainer package are still
   readable.
6. Confirm at least 300 GB free under `/tmp` and adequate checkpoint space under
   `/work/hdd`.
7. Confirm the launch commit retains literal unnormalized `eta_i = 1` for every
   branch and that the report identifies the resulting loss/gradient-scale
   confound. This decision is resolved; do not renormalize the weights.
8. Submit only after an explicit final go-ahead. Record the job ID and frozen
   commit; do not submit a PSC duplicate simultaneously.
9. After verified training completion, submit `delta_evaluate_paper.sbatch` to
   predict all 901 held-out cases and compute the same class-28 metrics as the
   four-cell grid. Never infer test completion from the training marker alone.

## Bridges-2 option

The new Bridges-2 allocation is a legitimate fallback. PSC currently documents
10 nodes with 8x H100-80GB, partial-node jobs through `GPU-shared`, a 48-hour
maximum, and a charge of 2 allocation units per H100 GPU-hour. The measured
42.3-GiB H200 UNet++ footprint fits 80 GB, and BF16 is supported. This does not
establish performance or queue advantage. Before porting the Paper run, use a
short real-data calibration with the same physical batch, patch, PyTorch stack,
compile settings, and trainer; compare both seconds/update and charged units.
Do not move the already-queued Delta grid merely because PSC access exists.

## Primary references

- UNet++ paper (equal branch supervision and branch-result averaging):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7357299/
- nnU-Net v2 predictor used by the evaluation path:
  https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py
- PSC Bridges-2 user guide (current GPU types, limits, and charging):
  https://www.psc.edu/resources/bridges-2/user-guide/

# 2x2 training progress and validation recovery — 2026-09-29

All four cells completed 1,000 epochs. Completion was verified from epoch 999 /
`Training done.` log entries and saved final checkpoints, not from Slurm's status.

| Cell | Training | Fold-0 validation | Saved predictions |
| --- | --- | --- | --- |
| UNet++ DS-on | 1,000 epochs complete | Complete | 1,800 |
| UNet++ DS-off | 1,000 epochs complete | Complete | 1,800 |
| Plain U-Net DS-on | 1,000 epochs complete | Complete | 1,800 |
| Plain U-Net DS-off | 1,000 epochs complete | Interrupted | 428 |

Bridges-2 job `47233546` started 2026-09-28 01:22:04 EDT and ended 2026-09-29
12:12:31 EDT (09:12:31 PDT), after 34h50m27s. Its overall Slurm status is FAILED
because task 1 of step 2 hit the cgroup RAM limit during validation. The other
plain U-Net task continued and completed validation. The failure did not occur
during training and does not require retraining either model.

## What is preserved

The four final checkpoints and model metadata have separate, hash-verified Ocean
backups under:

`/ocean/projects/cis260296p/asanjeev/pants_unetpp/checkpoint_safety/47233546_all_four_final/`

`checkpoint_manifest.json` records source/backup paths, checkpoint sizes, SHA-256
hashes, training completion evidence and the three available class-28 validation
summaries. This manifest was collected after the job exited. The source weights
and all existing predictions were left unchanged. The earlier UNet++ backups
under `checkpoint_safety/47104601_e35d3a7/` remain intact.

This GitHub directory preserves the progress evidence and reproducible recovery
code. Model weights and image data are excluded from Git by the repository's
policy; they are stored in the verified Ocean backups, not in this repository.
Both backups and original weights are on Ocean, so this protects against a bad
resume or accidental overwrite, not loss of the Ocean filesystem itself.

## Interrupted validation

The job reported one cgroup OOM-kill in step `47233546.2`; the DS-off validator
then raised `Some background workers are no longer alive`. The allocation had
220 GiB requested; the paired trainer step requested 200 GiB. The installed
nnU-Net default was eight export workers per model. Export resamples 29-class
logits into the original CT geometry, which can require much more CPU RAM than
patch inference. Concurrency is a plausible contributor; the exact allocation
that triggered the OOM has not been profiled, and Slurm MaxRSS is not the summed
peak memory of the entire cgroup.

Prepared recovery uses one H100 and one export worker with the same final
checkpoint, model classes, BF16 setup, sliding-window overlap, mirroring, label
mapping and nnU-Net inference/export/scoring functions. It checks the checkpoint
against the saved SHA-256 and confirms epoch 1,000. The new preprocessing plans
and dataset metadata must equal those saved with training before inference can
start. Completed cells cannot be trained again by the recovery branch.

The resume script reads existing predictions fully, checks geometry and label
values, and reuses only readable cases from the original 1,800-case fold. Missing
or corrupt predictions are recomputed; they are never excluded from scoring.
With all 428 existing predictions readable, 1,372 cases remain. nnU-Net's own
validator scores all 1,800 after prediction finishes. Checkpoint hashes must
still match after completion. Python stdout is unbuffered for visible progress.

The current conservative launcher still stages/preprocesses the full training
dataset on the new node. This is additional staging cost, not additional model
training. It requests one GPU, avoiding an idle second H100. Recovery has not
been submitted as of this record; review/Slurm preflight precede submission.

Four CPU tests passed on Bridges-2's Python environment: backups refuse a
different existing destination; stale checkpoint hashes abort before model
loading; invalid labels, shifted geometry and truncated predictions are rejected;
and the mocked validation workflow filters completed inputs while retaining the
full fold for scoring. Both Bash scripts passed syntax checks. These tests do
not constitute a GPU inference parity measurement or prove the full resume has
already run; production inference remains nnU-Net's installed validator.

## Results are not complete yet

The manifest's class-28 Dice is nnU-Net's native fold-validation summary. It is
not the project's positive-case-only Dice and is not the 901-case test result.
Aggregate organ Dice does not establish tumor detection validity. After the
remaining validation finishes, the next stage is the audited 901-case test
prediction and the five tumor metrics. No scientific quality acceptance is
claimed from the current three summaries.

## Subsequent recovery submission

With explicit user approval, the prepared recovery was submitted at 2026-09-29
14:19:24 PDT (21:19:24 UTC) as Bridges-2 **47272328**, pinned to
`eafd5741eb5bdfa0f68dc3898981b24664bc8e2e`, run ID
`20260929T211924Z_eafd5741eb5b`. Slurm confirmed one H100-80, 12 CPUs, 220 GiB,
48h and no dependency. Initial state was PENDING with no start estimate yet.
Its stored batch script matched the reviewed source SHA-256:
`43defea9f05da6441e8abf218479de34756c2b32d63d62eba68afa2350f21627`.
Follow `grid_logs/grid_47272328.log` for staging, prediction auditing and recovery.

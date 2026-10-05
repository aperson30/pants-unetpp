# Finish the protected 2x2 evaluation, not a new idea screen

User paused idea discovery on October 5. Focus is the original four-cell study.

## Live observation

Bridges-2 evaluation 47320183 reached its 48-hour limit. Scheduler start/end:
2026-10-03T03:48:20 / 2026-10-05T03:48:37 (raw cluster timezone, not converted).
Both UNet++ cells have 570 compact masks and 570 score rows. No plain-U-Net
cell directories exist yet. All 901 converted GT files remain on Ocean.
The queue was empty at the check. No final grid metrics file was found.
File counts alone do not certify readable/valid predictions.

## Preserve the experiment

Use training revision `61be5315e8cc5f4ec35637c9b748e5ebc33da1b7` and original
evaluation revision `5e99901263f4a9f2092d6674c7c566ba4d9d43cc`. Do not change
the predictor, metrics, trainer stack, checkpoint, TTA, patch overlap, case
membership, or original evaluation source manifest. The original predictor
already resumes only mask+score pairs passing its input geometry/provenance
checks. Its tests cover crashes, partial writes and skip-on-resume behavior.

New `audit_partial_resume.py` is a separate CPU integrity check, not inference
or a scorer. It checks original frozen source fingerprints, saved checkpoint
provenance, exact input membership and cross-cell identity, compact labels,
decoded mask/GT geometry, and mask/score pairing. Unpaired artifacts are listed
as pending, not counted as finished. It computes hashes without modifying
outputs or examining comparative test accuracy. The audit does not independently
re-hash current model binaries or certify newly staged CTs; the original
evaluation readiness/provenance guards do that on continuation.

CPU launcher `audit_partial_evaluation.sbatch` uses a shared read lock and a
frozen audit snapshot with the original evaluation utilities. No GPU requested.
Local checks: 12 existing guard/artifact/publication tests passed; the new
partial-audit invocation passed 5 collected tests (3 new + 2 imported fixture
tests). Shell syntax and Python compilation passed. These are logic checks,
not a GPU runtime guarantee.

CPU audit submitted as job **47450997**, pinned audit revision
`ce67418bd55307ac654e4d148e9c30e3d3bb44f1`, original evaluation utilities.
Submission overrides: 6 cores / 12000M and `--qos=low`; current template records
these defaults. Initial default-QOS and memory/core submissions were rejected
without allocations. Live association permits low on RM-shared but lacks its
normal rm QOS. Test-only accepted the corrected request; no GPU continuation
submitted yet. If low-priority audit is preempted, it changes no prediction data.

## Ordered remaining steps

1. Complete CPU audit of saved real artifacts; inspect its explicit success
   marker and exit status. Stop on mismatched source, corrupt paired masks,
   wrong checkpoint provenance or geometry. Never bypass these gates.
2. If clean, submit the original frozen evaluation launcher with the original
   TRAIN_COMMIT/EVAL_COMMIT. It restages CTs because the old node-local data
   expires with the allocation. It also compares saved GT and CT fingerprints;
   differences must fail, not be silently overwritten.
3. Reuse 570 verified pairs per UNet++ cell, compute remaining 331 each, then
   run both plain-U-Net cells. No training is repeated. Existing resume outputs
   are persistent; an interrupted local batch may be recomputed.
4. Audit all four complete 901-case outputs, then run the CPU scorer and verify
   the final report. Do not call the table complete until these succeed.
5. Write methods/fidelity/limitations while prediction runs. Keep the fifth
   paper-design config separate. No method tuning from protected test scores.

No promise that another 48-hour allocation necessarily finishes: volume sizes,
export cost, queue delay and filesystem load vary. Further allocations can
resume with the same scientific protocol. Do not add unreviewed auto-retry or
relax provenance merely to meet a deadline.

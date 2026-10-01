# Remaining evaluation: layered safety and throughput audit

## Scope and decision

All four 1000-epoch training runs and their 1800-case validations are complete.
The remaining work is 901-case test prediction per cell, CPU scoring, and the
final table. Bridges-2 job **47319378** was independently checked this audit:
PENDING (Priority), runtime zero. Its frozen evaluation revision
`28d2733ee7525c14874c4eca27c9fe18f7c7ab8b` was not edited or replaced.
No extra GPU job was submitted. No experiment settings were changed.

This is a risk-reduction audit, not a guarantee against hardware, scheduler,
filesystem, or network failure.

## Layers

| Layer | Existing protection / audit result | Remaining limitation |
| --- | --- | --- |
| Scientific contract | 1000 epochs, physical BS4 training, full inference TTA, Gaussian weighting, step size 0.5, final checkpoints, class-28 metrics unchanged | The project metric conventions still need PI confirmation before leaderboard claims |
| Source and environment | Frozen revisions, source hashes, exact version assertions, strict model loading, checkpoint hashes | Matching imports alone cannot prove numerical quality |
| Allocation | Two-task GPU UUID probe and explicit Slurm binding | Queue estimates are not reservations |
| Data | Exact 901 test IDs; input hashes equal across cells; local staging avoids the exhausted project quota | Label source URL is not revision-pinned; retain archive hashes and GT snapshot |
| Conversion | Training-compatible class mapping and affine repair; no experimental GT rewrite | Missing organ masks and per-organ geometry need source-convention verification before calling them corruption |
| Prediction | Official nnU-Net path; finite probability and mask geometry checks; bounded subprocess timeout | Five-case groups reload the model; a failed group can require recomputation of unfinished cases |
| Persistence | Compact masks copied through temporary files; score CSV replaced atomically; mask without score is recomputed | Atomic rename is not a promise of survival through every storage/power failure |
| Restart | Input/checkpoint/source provenance must match; already complete cases skipped | Never silently modify frozen source to bypass provenance |
| Scoring | Expected checkpoints and identical inputs checked; original five metrics retained; tested component-count optimization | A passing small fixture does not establish real tumor recall |
| Delivery | New independent full-volume artifact auditor and portable SHA-256 snapshot | A snapshot detects subsequent changes, not valid-looking scientific errors already present |

## New tests and artifact gate

`evaluation/test_prediction_guards.py` now injects five failures: CLI crash,
timeout, missing NPZ, NaN class-28 probability, and score-write failure after
mask persistence. Each must raise, leave no successful score, and recover on
retry. Existing provenance and geometry tests remain in place.

`evaluation/audit_test_artifacts.py` is separate from the frozen job. It decodes
every GT/mask, checks exact case membership, compact labels 0/28, finite integer
GT labels 0..28, valid geometry, complete finite score CSVs, expected checkpoint
identity, and identical CT input hashes across cells. It hashes all masks, GT,
score CSVs, and prediction provenance. Both positive and negative GT cases are
required. It keeps only one GT and one prediction volume decoded at a time.

Run from an audited checkout, with an appropriate CPU Python environment:

```bash
python -m evaluation.audit_test_artifacts \
  --evaluation-dir "$EVAL_DIR" --out-manifest "$EVAL_DIR/artifact_audit.json"
```

Copy the complete evaluation directory and the manifest to the scoring machine,
then verify the copy before scoring:

```bash
python -m evaluation.audit_test_artifacts \
  --evaluation-dir "$LOCAL_EVAL_DIR" \
  --reference-manifest "$LOCAL_EVAL_DIR/artifact_audit.json" \
  --out-manifest "$LOCAL_EVAL_DIR/artifact_audit_verified.json"
python -m evaluation.score_grid_cpu --evaluation-dir "$LOCAL_EVAL_DIR"
```

Do not interpret an old success manifest as a fresh pass if the current audit
exits nonzero. Keep the completed checkpoint backups unchanged.

Local suite: **11 tests, 29.894 seconds, OK (one installed-stack test skipped)**.
That installed-stack test was separately run on Bridges-2 CPU: **2 tests,
45.246 seconds, OK**. CPU-only export parity is not GPU forward-pass parity.

## Best remaining speed opportunity, not deployed

The queued reference path starts nnU-Net every five cases: 181 starts per cell,
724 across the grid. `evaluation/predict_persistent.py` could load once per cell
and avoid full-probability NPZ disk round trips. The actual installed-stack CPU
test now confirms exact compact masks and class-28 maximum probabilities against
the official exporter for the tested crop/transpose/resampling/logit fixtures.

Before deployment it still needs same-input/same-checkpoint **GPU** prediction
parity, actual wall-clock measurement including export, worst-volume memory
measurement, and worker/interruption cleanup checks. No measured speedup or
saved GPU-hours is claimed. Do not hold or replace the queued reference job for
an unmeasured candidate. Larger groups and parallel conversion also need bounded
memory and throughput measurements rather than assumption-based deployment.

## Failure/recovery runbook

1. Check Slurm state and the last actual trainer/predictor log line. Allocation
   is not proof of useful GPU work; a final checkpoint is not proof of scoring.
2. Preserve logs, manifests, completed masks/scores and final checkpoints.
3. If timeout, I/O, or prediction failure occurs, identify the failed layer first.
   Verify persistent provenance and valid completed-case count before restarting.
4. Resume with the same frozen contract. Partial cases are recomputed; completed
   matching mask/score pairs are retained. If source changes are necessary,
   document migration explicitly rather than bypassing hashes.
5. After all four prediction cells complete, run the independent artifact audit,
   verify transfer, then CPU scoring. GPU prediction success alone is not the
   completed 2x2 table.
6. Check the final table against all four scorer outputs and per-case tumor
   failures. Report tumor recall/whole-lesion misses, not organ-average Dice.

Potential storage/startup improvements remain optional follow-ups; protecting
completed scientific work takes precedence over speculative throughput gains.

## Follow-up: independent reviewer findings and fixes (21:43 UTC)

The independent Sol review found a missing regression: an invalid mask with an
existing score could be replaced, followed by a score-write failure; a retry
then accepted the replacement mask with the old score. Both predictor paths now
atomically invalidate pending scores **before** publishing or deleting masks.
The exact old-score 0.1 / new-score 0.875 interruption test passes. Invalidation
failure aborts rather than proceeding. Scientific probabilities are unchanged.

Additional repository fixes, NOT installed into the already-frozen queued job:

- CLI runs in a dedicated POSIX process group; timeout, nonzero exit, or Python
  interruption kills the group and reaps its leader before temporary cleanup.
  A real WSL process/grandchild test passed (2.054 seconds). This is not a test
  of every Slurm cancellation or uncatchable SIGKILL scenario.
- GT copies use a checksum-verified temporary file and atomic rename. Unreadable
  partial destinations can recover; readable but changed GT is rejected, not
  silently overwritten. Matching completed copies are retained.
- Old scoring/audit success markers are moved to timestamped history before a
  rerun. Failure leaves no current success marker. CPU scoring now invokes the
  independent artifact auditor itself and embeds its snapshot in grid output.
  Submission checks the report against current artifacts rather than merely
  treating nonempty JSON as scientific completion.
- Follow-up review confirmed the stale-score fix. Its direct-file scorer CLI
  compatibility finding was fixed and `--help` verified. Scoring also repeats
  the input audit before publication to detect concurrent changes. Completion
  verification checks five finite [0,1] metrics, case count and tumor/connectivity
  protocol; empty metric dictionaries and NaN cannot pass.
- Candidate predictor remains undeployed. Its pending-score bug is fixed via
  the shared tested helper, but its full GPU/runtime gates still remain open.

Validation: full local suite before the last additional stale-marker test:
16 tests, 56.443 seconds, OK (one installed-stack test skipped). Additional safety
suite including failed scoring/audit publication: 6 tests, 2.889 seconds, OK.
Both shell scripts pass bash syntax checks. No extra GPU-hours were spent.

**Deployment boundary:** 47319378 is still PENDING (Priority) with its original
frozen script. A repository push cannot patch that job. Do not silently edit its
snapshot. Applying these job-side fixes needs an explicit replacement/migration
decision, a new pinned revision and appropriate provenance handling; cancelling
and resubmitting may lose accumulated queue position. No such action was taken.

**Open:** affine correction has no evidence-backed upper limit yet. Do not pick
an arbitrary threshold or change GT construction only for the test set. Measure
the source deviations and review dataset mask/geometry conventions first.
Mutable label-source provenance and persistent-storage capacity gates remain
follow-up risks; do not claim all possible failure modes have been eliminated.

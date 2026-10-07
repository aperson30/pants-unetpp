# Protected 2x2 paper completion packet

Prepared October 5, 2026. This is an execution/reporting checklist, not final
results and not a new experiment. Idea discovery remains paused.

## What can be prepared before the source audit passes

- Methods/provenance inventory below, with missing facts explicitly marked.
- Resume and scoring review using frozen code; no new inference settings.
- Table structure and integrity/readiness gates; no provisional accuracy claims.
- Geometry incident disclosure: original files remain preserved, no target
  orientation selected using model predictions or Dice, no case exclusions.

## Methods facts and their boundaries

The four cells are plain U-Net and custom nnU-Net-v2 UNet++, each with deep
supervision on/off. All four trained for 1,000 epochs and completed the
1,800-case fold validation, as recorded in EVALUATION_PROTOCOL.md. This is not
the same as completed test evaluation. Report the actual fold/seeds/plans,
optimizer, learning-rate schedule, patch size and loss from the saved plans,
trainer/checkpoint provenance and logs, not a calibration plan or guessed
nnU-Net defaults. Do not imply multiple training seeds or full cross-validation.

Training continuation source:
`61be5315e8cc5f4ec35637c9b748e5ebc33da1b7`.
Original test-evaluation source:
`5e99901263f4a9f2092d6674c7c566ba4d9d43cc`.
Earlier epochs used equivalent custom trainer files; the continuation commit
must not be described as the revision used for every epoch without file-level
evidence. The fifth paper-design configuration is separate, not a fifth result
to fill in from this grid.

Original test inference retains all 29 output channels (background plus 28
semantic classes), default FP16 autocast, full mirroring and tile step 0.5,
checkpoint_final.pth and the original CLI. Stored masks retain only 0/28;
compaction occurs after inference and is not two-class model training. The
saved probability score is the maximum class-28 softmax probability per case.
BF16 training/validation and FP16 CLI test inference are different paths and
must be reported explicitly. Physical training batch size remains four;
ordinary gradient accumulation is not claimed equivalent to batch Dice.

## Test source and geometry disclosure

The frozen image archive and label archive are identified by SHA256 in the
original evaluation outputs. Test membership is exactly PanTS_00009001 through
PanTS_00009901 (901 cases). No test outcomes may determine tuning, selection,
alignment or exclusion.

Six initially flagged combined-GT headers inherited conflicting metadata from
the first organ mask. Six-case original-source checks proved exact tumor voxel
equality and CT-aligned original tumor annotations; this is not an all-organ
certificate. The complete strict source audit remains pending. Original-CPU
replay job 47461462 reproduced CT 9812's exact historical inference hash
02eca9599c5266c09e88d00238b73fe23b537633828d60e07211459b8bf3813f
in 12 seconds on an H100 node's Intel CPU, without changing source voxel values.
CT 9871 also reproduced its original inference hash in diagnostic job 47458101.
The audit's optional exact-replay handoff independently rechecks raw/code/file
hashes, shape, affine, voxel equality and unchanged evidence; it does not accept
approximate identity or promote a replay diagnostic into a cohort certificate.
Full CPU audit 47497014 COMPLETED in 23m41s using snapshot ebd86ea and this
verified replay: all 901 CT identities and tumor masks passed, no CT mismatch
or source error. It records 151 tumor-positive cases and nine combined-GT header
mismatches; the earlier six flagged cases were not the complete mismatch count.
Reference builder 47497024 started after this pass. Its derivative and the saved
partial predictions must still be independently checked before GPU continuation.
No weakened provenance gate is acceptable; no final model metrics are claimed.

The separate tumor-reference derivative, if certified, must be disclosed as
unchanged class-28 index masks on the verified prediction-CT grid, with no
resampling, flip or case exclusion. Keep its source-audit, original-GT and
derived-reference hashes in the final scoring provenance. Do not silently
replace test_ground_truth in the original evaluation directory. Other organs
are not certified by this tumor-only reference.

## Resume review: avoid an easy moving-HEAD failure

The current submit_evaluation.sh derives EVAL_COMMIT from live checkout HEAD.
That is not the right resume entry point for the existing protected run:
the original source-hash gate would reject a changed launch/scoring revision.
For the continuation, use the ORIGINAL frozen evaluate_grid.sbatch with the
original TRAIN_COMMIT and EVAL_COMMIT explicitly supplied. Keep original GT
publication and all inference settings untouched. This is a preparation note,
not authorization to bypass the pending source/partial-artifact gates.

Before submitting: full-source pass, certified derivative manifest, audit of
saved pairs against verified CT geometry, current checkpoint/readiness checks,
Ocean headroom, syntax/Slurm test-only, and no concurrent evaluation writer.
The existing GPU script ends with a predictions-only marker; CPU scoring is
separate. A 48-hour timeout can resume valid persistent pairs with unchanged
fingerprints; another allocation is not guaranteed to finish in 48 hours.

## CPU scoring preparation

The original score_grid_cpu.py assumes original test_ground_truth geometry.
Do NOT run it against silently substituted labels or suppress its geometry
audit. Prepare an explicitly versioned scoring adapter/output directory which
binds the original inference manifest and the separately certified tumor
reference manifest. Preserve the original metric implementation/math, exact
901 IDs, checkpoint/input identity, mask/score pairing and pre/post artifact
snapshots. Publish a final report only if all four complete cells pass.

The separate adapter is implemented as evaluation/score_versioned_grid_cpu.py,
with read-only certificate and four-cell prediction-audit helpers. Synthetic
fixtures exercise the ORIGINAL checksum-bound compute_tumor_metrics.py, known
metrics and refusal gates. Real 901-case use is still blocked by source
certification and incomplete predictions; fixture success is not a cohort pass.
Outputs must be in a fresh directory outside the original evaluation/reference
trees. Diagnostic source reports are refused. No original GT substitution or
metric-math change is performed. A worker failure, changed inputs or a failed
denominator/protocol check prevents publication of grid_metrics.json.

Source-audit numeric check: preserved int8 lesion masks carry slope
0.003921568859368563 and intercept 0.501960813999176, producing decoded foreground
1.0000000591389835. Exact equality to 1 was an audit bug. The updated binary
validation allows near-one decoding (absolute tolerance 1e-6), requires EXACT
zero background, then uses the frozen converter's source_data > 0 membership
and exact voxel-for-voxel equality with saved class 28. No rounding, annotation
edits, resampling, inference changes or new tumor definition. Negative values,
tiny positive background, fractional masks and extra label classes are rejected.
CT 9812's exact input hash has now been reproduced and the complete strict audit
passed. These checks do not certify the other organs.

Read-only partial resume checker: evaluation/audit_versioned_predictions.py
with --partial, snapshot d11910f, job 47497134 dependent on successful reference
builder 47497024. It independently verifies the full reference, then checks all
existing mask/score pairs against certified CT geometry and original checkpoint,
input, predictor, protocol and plans identities. Unstarted cells are explicitly
recorded as remaining work. Orphans, extra IDs and unprovenanced outputs fail.
Its separate report always declares final_scoring_ready=false; the default full
audit/scorer still require all 901 cases in all four cells. It never retires or
rewrites predictions. Six prediction and ten certificate fixture tests passed.

## Final table structure (no accuracy values until certified scoring)

| Model | Deep supervision | Tumor Dice, GT-positive cases | Patient sensitivity | Lesion sensitivity | Specificity | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Plain U-Net | Off | Pending | Pending | Pending | Pending | Pending |
| Plain U-Net | On | Pending | Pending | Pending | Pending | Pending |
| UNet++ | Off | Pending | Pending | Pending | Pending | Pending |
| UNet++ | On | Pending | Pending | Pending | Pending | Pending |

Use the recorded project protocol: class-28 Dice on GT-positive cases;
patient detection sensitivity; six-connected GT-component sensitivity;
specificity on GT-negative cases; AUC from maximum class-28 probability.
Report positive/negative denominators and lesion counts from certified scoring,
not approximately 10% prevalence. Public PanTS materials do not fully specify
the scorer conventions: obtain PI confirmation before claiming exact official
leaderboard equivalence. Aggregate organ Dice is not the tumor success signal.

## Required delivery artifacts

1. Four complete 901-case mask/score pairs with checkpoint/input fingerprints.
2. Full source-audit pass, derivative reference manifest and explicit scoring
   provenance; preserve original evidence and historical failed audit logs.
3. Four per-case metric tables and metric JSONs; independently verified final
   grid report. Do not label a partial prediction collection as the final table.
4. Methods/config inventory tied to actual saved training artifacts; fidelity
   audit and limitations, including one-fold/seed coverage actually performed.
5. PI-reviewed interpretation. No claim that a depth/supervision variant helps
   tumor detection before the actual class-28 results exist.

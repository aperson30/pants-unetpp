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

### Saved hyperparameter inventory (read October 6, Pacific time)

Read from each model's actual plans.json and fold_0/debug.json, not a calibration
sample. All four plans files share SHA256
`a9fd3a3963ad25e0070feb938b7faa5a8102201bdb8fbeb40c19624436389f1a`.

| Setting | Recorded value in all four cells |
| --- | --- |
| Configuration / fold | 3d_fullres / 0 |
| Epoch budget / training iterations per epoch | 1,000 / 250 |
| Physical batch / patch | 4 / [64, 160, 224] |
| Target spacing, transposed axis order (mm) | [1.25, 0.7929689884185791, 0.8046875] |
| Normalization / batch Dice | CTNormalization / true |
| Optimizer | SGD, momentum 0.99, Nesterov true, dampening 0 |
| Initial learning rate / weight decay | 0.01 / 3e-5 |
| Scheduler | PolyLRScheduler (exponent not extracted in this inventory) |
| Foreground oversampling | 0.33; foreground is not synonymous with tumor |
| Recorded validation cadence / iterations | every 5 epochs / 50 iterations |
| DDP | false |
| Loss | DC_and_CE_loss; DS-on additionally uses DeepSupervisionWrapper |

The 250 training updates per epoch imply 250,000 planned optimizer updates per
cell; an nnU-Net epoch is NOT necessarily one exhaustive pass over every case.
Startup debug.json records settings, not proof of completion: current_epoch=0
there must not be interpreted as the final epoch. Completion is established by
the independently checked final checkpoints and validation artifacts.

debug.json SHA256s, in model order:

- UNet++ DS-on: 2f5963793b908d1bdbaa7a38ec53ed22bc0979667198e7ea55debbe5a6cb9505.
- UNet++ DS-off: 665549b664d6a7199a98281a22a708a63ebf57f02c466aaef615a87e06223726.
- Plain U-Net DS-on: 447bb871cbe0e9e65c90b2f7b0e17b78397f14770f725c13c057e7d5fc9dfc54.
- Plain U-Net DS-off: bc7d185cdc187418566ed95c7c4fa87607ef5bc4570741eac776bdb64b73c692.

The plans architecture field is a PlainConvUNet template even in the UNet++
model folder. Describe the UNet++ architecture from its actual custom trainer
and network source, not that template. Do not claim these hyperparameters were
chosen by test-set tuning; recorded equality supports a controlled comparison,
but the historical selection rationale requires the original protocol/source.

### Tumor-specific training curve, not final test accuracy

Read-only extraction from the original training logs found exactly 1,000
recorded epochs (0 through 999), with no duplicate epoch records, for each cell.
The logged Pseudo dice vector has 28 foreground entries; entry 27 is class 28.
Sparse validation repeats the preceding value on skipped epochs. Summaries below
therefore use only epoch % 5 == 0 plus epoch 999: 201 finite validation points
per cell, not 1,000 independent validation measurements.

| Cell | First nonzero tumor proxy epoch | Median proxy, epochs 400-499 | Median proxy, epochs 900-999 | Epoch-999 proxy |
| --- | --- | --- | --- | --- |
| UNet++ DS-on | 330 | 0.0081 | 0.4538 | 0.3181 |
| UNet++ DS-off | 285 | 0.0000 | 0.3137 | 0.2875 |
| Plain U-Net DS-on | 490 | 0.0000 | 0.3988 | 0.4814 |
| Plain U-Net DS-off | 625 | 0.0000 | 0.4128 | 0.2243 |

All four had a zero median in epochs 0-99. First nonzero does NOT imply sustained
tumor detection: e.g. UNet++ DS-off has a first nonzero at 285 but a zero median
at 400-499. These are sampled-patch pseudo-Dice values, not full-volume tumor
Dice, patient/lesion sensitivity, percentages of patients detected or final test
metrics. End-point variability is another reason not to rank models by a single
training log value. No test-driven tuning, early stopping or architecture claim
is made. Plain DS-off's signal first appearing at 625 supports the decision not
to truncate this experiment at 500 epochs.

Training log SHA256s (chronological within each cell):

- UNet++ DS-on: e61df8d87b784b3b81072080b522b378e08f057c6b2ca9977232e6c545a8ee87;
  3f544476a0239b255284db9ce4f1416fd20bc344b6ac9e7f723586840843b36d.
- UNet++ DS-off: dbf19933c54c99567001009fd5eb1d466f149b03406b3492b89d340de365af8c;
  1678423a3cf44067573aa314aade71e0164dd0e190e3e331b44779ad2920f570.
- Plain DS-on: d5a54bbed2f108cd08e0fccc89717d85a1abf4c02a74c61a537480f424eb9959.
- Plain DS-off: 98b7db11982d69079a73d7bea8efa30a6ea72da8b2245f5112e1a31c7aff6a97;
  624cd2523e840302f2293482a950a8011523a2b19e4aa8fbc365fb2588622c98.

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

Continuation 47497144 is now submitted using the ORIGINAL frozen launcher,
afterok:47497134. All original source hashes, four final checkpoint hashes and
validation-summary identities passed a read-only preflight. Version metadata:
torch 2.10.0+cu126, torchvision 0.25.0, nnunetv2 2.8.1. No final grid report exists.
This is a queued continuation, not resumed prediction yet. Dependent CPU scorer
47497147 uses pinned snapshot 2185542 and the original metric file, afterok of
successful inference. Timeout/failure prevents scoring; no automatic GPU retry.

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

Independent final report verifier: evaluation/verify_versioned_grid_report.py.
It rechecks the complete input/reference snapshot, exact metric source identity,
all twelve per-cell output fingerprints, unique per-case IDs, original probability
scores, source tumor voxel counts and denominators. It independently aggregates
Dice/detection/specificity and uses pairwise probability ranking (ties count 0.5)
to verify AUC without relying on sklearn's implementation. Comparison tolerance
1e-12 only accommodates floating-point summation; original score math is unchanged.
Toy validation passes with unchanged inputs/outputs, and rejects consistently
edited metric JSON/hashes, duplicate CSV rows and a partial-report schema.
Read-only CPU verifier job 47497968, pinned snapshot 9c64275, is queued afterok
of scorer 47497147. A successful scoring job alone is not the final delivery gate;
the independent verification marker and matching report SHA256 are required.

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

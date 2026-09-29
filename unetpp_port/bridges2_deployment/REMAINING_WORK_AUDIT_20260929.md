# Remaining 2x2 work: throughput, cost and failure audit

This is a preparation/audit record, not permission to skip evaluation or change
the experiment. Four final 1000-epoch checkpoints exist. Three full validations
are complete; plain U-Net DS-off validation is being recovered by job 47272328.
No retraining, new GPU benchmark or test-evaluation job was submitted in this audit.
The recovery's pinned code, environment and allocation were not modified.

## Priority order across the whole remaining workflow

| Stage | Finding / potential gain | Status and safety gate |
|---|---|---|
| Protect finished work | Ocean is persistent but is not a backup. A second copy prevents a storage/account failure from forcing retraining. | All four final checkpoints copied to the user's Windows machine; SHA-256 matches the saved manifest. Ocean backups are retained. |
| Interrupted validation | Reuse only fully readable, geometrically valid completed cases; one exporter bounds CPU-memory concurrency. | Already in the pinned recovery. It still scores all 1800 cases. Leave this running job alone. |
| Validation restaging | Current conservative recovery still restages/preprocesses 9000 cases even though it does not train. A future validator could stage only the 1800 fold-validation GTs and preprocess only pending inference cases using the *original full-training plans*. | Large potential CPU/IO saving, not deployed. Require saved full split/case IDs, original plans/metadata and per-case preprocessing parity against the full-dataset path. Never replan on the smaller validation subset. Do not interrupt this allocated recovery to rewrite its data path. |
| Test-data staging | Only 901 test cases are needed, not another 9000-case training download/preprocessing pass. The test-image archive's measured HTTP content length is 27,994,666,473 bytes (~26.1 GiB). | Existing evaluation already uses the test split. Next improvement: cache converted test CT/GT once on Ocean, with a lock, complete case-ID/content/geometry audit and atomic READY manifest. Size the *finished cache plus temporary downloads* before starting; archive size alone is not total staging space. |
| Repeat staging after interruption | A verified persistent test cache avoids redownload, conversion and changing source data on later nodes. | Proposed, not implemented/deployed. Pin image revision `3b1cd61108116b58ea5c1ddb3512c1847d965f96`; record label-archive hash too because its JHU URL is unversioned. Never trust file existence as cache readiness. |
| Test conversion | The existing converter already has safe per-case process parallelism, but the test-only CLI did not expose it. | Added `--workers`, default still 1. Serial versus two-worker synthetic image/label/affine parity passed, including overlapping pancreas/tumor masks with tumor taking precedence. Choose a bounded worker count from measured memory/IO, not all cores blindly. |
| Prediction process startup | Current launcher uses batches of five: 181 CLI/model starts per model, 724 across four models. | Prepared `evaluation/predict_persistent.py`: one model load per cell, four total. Not wired into the launch script. Needs real-case GPU parity and total pipeline timing before deployment; fewer startups is not a measured hours-saved claim. |
| Probability export | Existing CLI saves full 29-channel compressed NPZs to Ocean, reloads them for one maximum probability, then deletes them. | Candidate uses the exact installed nnU-Net probability-space exporter in memory and stores the same tumor mask/maximum probability. Still computes all 29 channels, full softmax, resampling, crop restoration and transpose. Does NOT replace this with logits argmax or patchwise maxima. Actual installed-stack CPU export test timed out after 120s without a result: NOT cleared for deployment. |
| Export overlap / RAM | Serial in-process export bounds memory but can leave the GPU idle. More export workers can reverse that gain or reproduce the CPU cgroup OOM. | Candidate initially has one preprocessor and synchronous exporter. Measure end-to-end GPU-idle time and host-memory peak before considering a bounded single-worker exporter pipeline. Model startup alone is insufficient evidence of a net win. |
| Existing fallback predictor memory | Closing `NpzFile` leaves `probabilities` and its `tumor` view alive until the next assignment, overlapping consecutive volumes. | Added explicit deletion after score extraction in `predict_and_shrink.py`. No probability, mask, precision or metric math changed. This changes only future source snapshots, not the active recovery. |
| GPU concurrency | Prepared evaluation has two independent exclusive `srun` calls without the training launcher's distinct-UUID preflight. | Must carry over the verified one-two-task-step/explicit binding pattern before submission. Confirm distinct physical UUIDs and actual overlapping inference, not two ranks both saying CUDA device 0. This script has NOT been submitted or declared ready. |
| CPU scoring with GPUs held | Current evaluation retains both H100s while computing per-case metrics. | Move scoring to a legitimate CPU-only resource or the user's local CPU after preserving test GT and compact predictions. The local audit environment is ready. RM-shared `--test-only` rejected this GPU account's QoS; do not try undocumented QoS to bypass resource eligibility. |
| Lesion scoring | Original code rescans the full volume for every GT tumor component. | Exact vectorized any-overlap rule implemented and tested. Connectivity and the five metric definitions are unchanged. See measured CPU results below; no end-to-end GPU-hour saving claimed. |
| Multi-model scoring | Four separate scorer invocations reload the same GT and relabel its connected components four times. | Possible combined per-case CPU scorer, sharing only GT computations. Lower priority than removing GPU-paid setup/scoring. Must compare all four output tables exactly against independent scoring. |
| Checkpoint/provenance guards | Old evaluation wrapper hardcodes `b2ca075...` as a training commit. Actual training/continuation provenance differs. | Compare the exact 13 custom trainer/architecture files across frozen sources, record per-cell source revisions and expected checkpoint SHA-256. Local diff from `b2ca075...` to `61be531...` showed no trainer/architecture changes, but equivalence is not the same as correct provenance. Fix the label/guards before launch; do not overwrite installed shared trainer code while recovery runs. |
| Precision consistency | BF16 training-validation comes from the trainer mixin; ordinary nnU-Net predictor initialization builds the architecture without calling that mixin's constructor. Test CLI therefore defaults to FP16. | Candidate defaults explicitly to FP16 to preserve the currently prepared CLI. BF16 is an explicit protocol decision, not an automatically quality-neutral speed tweak. Record it and use the same choice for all four cells. |
| Resume correctness | A file count or `checkpoint_final.pth` does not establish complete validation. Likewise, existing test masks without a corresponding valid score/provenance cannot be reused. | Require full-fold validation summaries and case identities; prediction resume needs checkpoint/source/input hashes, valid score, complete readable mask and matching affine. Candidate hashes inputs/checkpoint/sources and audits masks. Full launch guard still needs updating. |
| Final table / scientific review | Aggregate organ Dice can obscure missed pancreatic tumors. | Report all five declared tumor metrics, GT-positive DSC, per-case results, whole-lesion misses and denominators. Keep the agreed patient-sensitivity definition distinct from lesion overlap. No test-informed retraining or checkpoint cherry-picking. |

## Exact scoring verification and realistic limits

- Exhaustive 2x2x2 GT/prediction masks: 256 x 256 x 3 connectivities =
  **196,608 comparisons**, all equal to the original implementation.
- Another 60 larger random-volume comparisons passed.
- Existing end-to-end synthetic five-metric test passed unchanged: DSC 0.5,
  patient sensitivity 1, tumor sensitivity 0.5, specificity 1, AUC 1.
- Local CPU: Intel Core i7-1185G7, Windows, isolated CPU audit venv (not training
  stack), synthetic 96x160x224 volumes, five-repeat medians:

| Disconnected GT components | Original seconds | New seconds | Component-scoring speedup |
|---:|---:|---:|---:|
| 1 | 0.01996 | 0.01821 | 1.10x |
| 5 | 0.03216 | 0.02214 | 1.45x |
| 20 | 0.06873 | 0.01555 | 4.42x |
| 900 (stress case, not representative PanTS) | 2.91009 | 0.01645 | 176.86x |

These are CPU function timings, not actual PanTS scoring or H100 inference
speedups. Most GT-negative cases already skip component labeling. Do not apply
the stress-case ratio to the full evaluation or claim hours saved from it.

## Candidate inference acceptance gate (still open)

1. CPU export: compare original `export_prediction_from_logits(...,
   save_probabilities=True)` plus shrink against the candidate, including crop,
   nontrivial transpose, resampling, tiny class-28 signals, ties/near-ties and
   geometry. Exact masks and maximum scores are required at the same precision.
   `test_compact_export.py` provides this harness. Actual installed-stack test
   did not complete in its two-minute timeout; the local geometry test passed.
2. Runtime: same real inputs/checkpoint, same FP16 or agreed BF16, same full
   TTA, tile step 0.5, Gaussian blending, resampling and fold 0. Compare old/new
   class-28 masks, voxel probabilities/maxima and all five derived metrics.
3. Budget: time the whole pipeline (preprocessing + GPU + CPU export + IO),
   monitor CPU/cgroup RAM and GPU allocation peaks. Check worst-sized CTs,
   restart after a deliberately incomplete case, worker crash/timeout and
   corrupt output. Bound the probe and preserve original outputs separately.
4. Freeze the validated sources/environment and only then switch the production
   launch script. Retain the reference path as fallback. No such GPU probe was
   submitted in this audit.

The candidate uses an upstream private preprocessing iterator; its child-worker
exception cleanup is imperfect. It must run as a dedicated Slurm step, not in a
long-lived process. Better explicit worker cleanup remains a pre-production task.

## What is deliberately not being optimized away

No smaller test set, missing classes, cropped-context shortcut, reduced TTA,
different sliding-window overlap, reduced precision without a decision, early
training termination, ordinary batch-1 accumulation, or test-selected checkpoint.
The four 1000-epoch models are already finished; tuning their training kernels
now cannot save their already-spent hours. The fifth Paper config is separate
and must not delay the primary grid's completion.

## Sources checked against actual installed code

- [PSC Bridges-2 guide](https://www.psc.edu/resources/bridges-2/user-guide/):
  Ocean is not backed up; LOCAL is allocation-lifetime storage. CPU RM and GPU
  allocations have separate eligibility; respect account/QoS limits.
- [Official nnU-Net predictor examples](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/examples.py):
  persistent predictor API. The installed 2.8.1 implementation, not moving master,
  was inspected for exporter math, preprocessing queues and autocast behavior.
- [Slurm srun](https://slurm.schedmd.com/srun.html): step resources and GPU binding.

Bottom line: the highest-value remaining changes are avoiding repeated setup
and keeping CPU work out of GPU-paid time. Safe scoring/plumbing fixes landed;
cache/scheduler/provenance changes and the persistent inference path are still
gated. There is no defensible new end-to-end hours-saved number yet.

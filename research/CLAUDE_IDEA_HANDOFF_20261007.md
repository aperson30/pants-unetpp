# Claude handoff: find a genuinely strong, resource-light research direction

Consolidated October 7, 2026 by Codex. This is the entry point; linked artifacts
contain exact protocols, source revisions, contracts and every retained result.
**No S-tier idea is established.** Please challenge the conclusions and search
outside the exhausted branches. A strong new explanation is welcome; a forced
positive result or a renamed published method is not.

## 1. What the user wants and what you should deliver

The user wants the strongest plausible top-conference idea for the least total
work and compute, not a fashionable combination or an expensive generator
trained from scratch. The PI mentioned UNet++ as a diffusion backbone, pruning,
parameter quantization and faster training. These are interests, not permission
to change the completed segmentation comparison or claim a new method already
exists. No requirement that the final idea use every one of those ingredients.

An excellent response would provide:

1. One strongest direction and two serious alternatives, with the actual
   scientific mechanism, importance, closest primary-source competitors and
   exactly what might differ. Do not give acceptance probabilities or S-tier
   labels unsupported by evidence.
2. A decisive falsifiable test, cheap enough to reject the direction early;
   strong ordinary baselines, confounds, fixed endpoints and stop criteria.
3. The cheapest credible COMPLETE paper, not just a cheap pilot: assets,
   replication, tuning, downstream training, compute accounting, independent
   data and any expert evidence actually required.
4. What you disagree with in this handoff and the evidence that changes the
   decision. A previous NO-GO is scoped to its tested claim, not an instruction
   that all nearby work is impossible.

The user does not have a clinical expert available on demand. Use public
annotations/resources for an honest technical initial screen. Do not replace
clinical evidence with an LLM, detector confidence, an annotation contour or
visual similarity; instead narrow the claim to what can actually be measured.

## 2. Protected project and latest known status

Repo: https://github.com/aperson30/pants-unetpp, branch main. Local checkout:
`C:\Users\adity\Documents\Codex\2026-09-17\we-re-training-unet-custom-ported\work\pants-unetpp-opt`.

The original paper experiment is plain U-Net versus custom nnU-Net-v2 UNet++,
each with deep supervision on/off. **All four finished 1000 epochs and the
1800-case fold validation. The final 901-case test table is NOT finished.**
One fold/seed coverage only; do not imply five-fold or multiseed replication.
The fifth paper-design configuration is separate, not a completed fifth cell.
Its requested design uses literal summed equal branch losses, eta_i=1, and
branch-averaged inference; summed loss changes effective SGD update scale and
therefore is not an isolated supervision/ensemble-only ablation.

Latest LIVE observation during the preceding paper-work turn, 07:30 UTC October7:

- Inference continuation **47497144** RUNNING on Bridges-2 w009, two H100s;
  start 07:13:07 UTC = October7 00:13:07 Pacific.
- Startup software/source/four complete validations/distinct-GPU checks passed.
  Images downloaded; labels were downloading. New inference was not yet confirmed.
- Reusable predictions: 570 mask/score pairs in each UNet++ cell, zero in each
  plain-U-Net cell. Remaining 2464 pairs across four cells, not 2464 patients.
- Source audit **47497014**, derivative reference **47497024** and independent
  reuse audit **47497134** all completed and passed.
- CPU scoring **47497147** depends on successful inference; independent final
  verification **47497968** depends on successful scoring.
- Original inference launcher/predictor are frozen. No automatic retry;
  Requeue=0. Wall cap October9 00:13:07 Pacific is NOT a completion forecast.

This status can age. **Do not touch, cancel, resubmit, repurpose, tune on or change
this chain for idea research.** Older notes referring to job47320183 describe
the previous timed-out evaluation; that is not the current continuation.
Paper-status source: `unetpp_port/bridges2_deployment/PAPER_COMPLETION_PACKET_20261005.md`.

Quality constraints: pancreatic tumor is **class28**, background plus28 semantic
classes (29 outputs), rare and voxel-small. Training BS4 is physical, patch
[64,160,224], batch_dice=True. Four batch1 accumulation steps are NOT equivalent.
No shortening to500 epochs, context-starving crops, changed class exposure,
changed inference/TTA/precision or quantization in the protected experiment.
Organ-average Dice does not certify tumor performance. Stored 0/28 masks are
post-inference compaction, not two-class model training.

Strict full test source audit independently verified 901 CT identities/tumor
annotations, 151 positive/750 negative cases, and found nine combined-GT header
mismatches. Separate tumor reference corrects metadata only, no resampling,
voxel editing or case exclusions; originals remain preserved. That certificate
is NOT an all-organ label audit. Final five metrics use an explicitly recorded
PROJECT protocol, not a verified unpublished official PanTS scorer. No protected
test fitting for a new idea; use development data/public independent assets.

### Engineering lessons relevant to any new proposal

These are historical measured context, not universal hardware predictions:

- On GB10, `channels_last_3d` was about 3.5x slower on the related lab model;
  true FP16 weights, batching tricks and cuDNN benchmarking did not help that
  workload. Do not extrapolate to H100/GH200 or assume all CNNs behave the same.
- Early compile failures/stalls had sm_121 target/toolchain and node-noise
  issues. Later validated compiled stacks existed. Old 'compile always broken'
  notes are historical, not the complete final state. Current protected test
  CLI intentionally uses original eager FP16 inference; do not change it.
- Real nine-case GH200 calibration using actual trainers/data: UNet++ .452s,
  plain U-Net .113s per train step versus GB10 3.51s/.83s. Full-loss compile
  added roughly 4.2%/4.0% in that calibration. These are STEP timings, not full
  dataset completion/validation/setup costs or a new diffusion-training benchmark.
- Dead-head skipping, BF16, loss/network compile, sparse validation and
  quality-neutral plumbing were engineering improvements, not novel research
  contributions. Snapshot/config fidelity matters more than remembering an
  approximate aggregate hours-saved headline.
- PyTorch 2.12.1 Delta parity failed the agreed tumor-gradient tolerance
  (~.22526% versus ~.1% bar); isolated 2.10 was used. Library mixing later caused
  a real torchvision startup failure. Verify imports FROM THE ACTUAL LAUNCH
  environment, not a different login environment or successful CPU smoke test.
- Sparse validation must be consistent across cells. A final checkpoint does
  not mean full validation/test scoring finished. A hard-coded `.npz` count can
  fail on a different preprocessed storage format; validate actual contracts.
- Physical batch four, context, 1000 epochs and tumor endpoints are scientific
  requirements. Gradient accumulation, patch reduction, early stopping and
  altered sampling are not automatically quality-neutral optimizations.
- INT8 PTQ previously degraded tiny rare tumor logits despite acceptable
  organ-average Dice. INT8 training acceleration is not available merely by
  using an inference quantizer. FP8/QAT/pruning remain new scientific experiments
  requiring rare-class validation; no measured payoff here justifies a claim.

The underlying loss was SGD + Dice/CE, initial LR .01, momentum .99,
Nesterov, weight decay3e-5, PolyLR; 250 updates per epoch (not exhaustive data
passes), foreground oversampling .33 (NOT tumor-specific). The documented
class-28 sampled-patch pseudo-Dice onset was UNet++ on330/off285, plain on490/
off625; first nonzero is not sustained recall. Patch proxies do not rank final
patient/lesion performance. See the paper packet for exact curves and hashes.

## 3. What we already tried: evidence and decisions

### A. Medical VAE/compression: real phenomenon, unproven clinical meaning

Initial rationale: a pretrained autoencoder might be the bottleneck before
diffusion depth/pruning matters. If small-lesion evidence does not survive
encoding, a better denoiser cannot simply recover that evidence from nothing.
But reconstruction changes are NOT by themselves irreversible information loss.

Pinned MAISI VAE, FP32, deterministic posterior mean, native spacing,
orientation-only RAS and padding, whole-volume encode/decode. This is an
intentional diagnostic, NOT established parity with production MAISI synthesis
preprocessing. No diffusion training/generation was performed in these screens.

Three MSD feasibility cases, selected using annotation burden before their
reconstruction outcomes (not independent clinical or model-holdout certification):

| Case | Annotated burden | Observed reconstruction result |
| --- | --- | --- |
| pancreas_005 | 7.60mL, not voxel-tiny | Mean local contrast 43.67→43.32HU (~99.19% retained), tumor HU MAE36.21 |
| case120 | .849mL,454voxels | Contrast -65.55→-58.58HU (~89.37% magnitude retained), tumor MAE18.27; surrounding tissue also shifts |
| case165 | 1.061mL,445voxels | Contrast11.79→11.20HU (~94.98% retained), tumor MAE42.12; CNR INCREASES under smoothing |

Ring/slice controls show strong measurement dependence and heterogeneous
within-lesion effects. CNR can improve because background variance shrinks.
HU MAE, mean contrast, CNR and GT contours are not detection/clinical endpoints.
Differences between patients do not identify a causal lesion-size effect.

Measured whole-volume forward/copy ~96–108s on DeltaAI GH200120GB; GPU peaks
~33–39GiB for these shapes; host high-water ~31–36GiB. Allocation/setup can be
larger. First32G host request failed,96G corrected it. Slurm sampled MaxRSS can
miss true peaks. No raw CTs/weights/reconstructions committed to Git.

Further controlled-signal probes found contrast/phase response worth recording,
but not a clinical-erasure mechanism. See the contrast result/code/phase controls
in `research/compression_probe/RESEARCH_HANDOFF_20261001.md`, not a new amplitude
sweep. Equivariance and task-aware compression already have substantial prior art.

### B. Frozen judges/posterior controls: why the strong interpretation stopped

Two PANORAMA cases were probed with the published task network. Original versus
posterior-mean tumor-region probability decreased86.20%/74.46%. Candidate overlap
was358→249voxels for100226, and0→0 for100430. Neither establishes a newly missed
lesion. High patient-wide maxima can occur elsewhere, outside the known lesion.

Historical native replay for100430 FAILED the original1e-4 numerical gate.
Changing backend policy did not retrospectively fix it. A separate same-process
repeatability experiment had exact repeats, but historical max probability drift
was .001598/.001856. Autotuning is a plausible confound, not a proven cause.

New same-process fixed-window factorial controls: probability weakening persists
in both fixed publisher windows (~80.53%/76.94% decrease), so window change ALONE
does not explain this patient's effect. All four arms still have ZERO
GT-overlapping candidate voxels. Do not label it clinical loss or claim crop
placement is irrelevant generally. Preserve original failed replay separately.

On100226, posterior mean/seeds0,1,2 retain13.80/10.80/47.53/35.29% of the stronger
PANORAMA judge's native tumor mean. All have some candidate overlap. Posterior
seed variation matters; no best-seed selection or independent-subject statistics.

Second network: public DiffTumor U-Net, strict exact weights, whole-image FP32
96cube sliding windows, official RAS/1mm/HU preprocessing, CPU accumulation and
native-grid restoration. Executed a RAW-HEAD diagnostic, NOT official organ-masked
postprocessing; never supplied GT organs. All repeated arms matched exactly.

DiffTumor native tumor mean **.00329347**, below predeclared adequacy floor .01;
ALL native/reconstructed arms have0 tumor-argmax overlap. Seed2 tumor mean is
.00552814,1.679×native, contradicting universal suppression. **NO-GO for this
current lesion-erasure/judge route.** No clinical recall loss, irreversible
information loss, independent-patient validation or novel remedy demonstrated.

The public DiffTumor weights are independently TRAINED but not proven patient-
independent: released headers lack fold/training-patient manifest. Official organ
mask producer family was identified, exact archived pipeline remains unverified.
GitHub CC BY-NC-ND versus HF apache metadata discrepancy remains unresolved.

Additional engineering negative: internal MAISI convolution splits4 versus1
gave exact output parity on one patient, unchangedGPUpeak24,250,846,720bytes and
~14.2% slower one-shot runtime. Not a demonstrated memory/speed optimization;
do not revive it to unlock the resource-blocked large case100259 or replace
that case after observing difficulty.

### C. Adaptive UNet++ depth: existing native adequacy gate failed

Four smallest annotation-positive development cases were frozen BEFORE outcome
inspection; two negative controls also frozen. Free saved-summary inspection:

| PanTS case | Reference tumor voxels | Overlapping predictions | Off-annotation predictions |
| --- | ---: | ---: | ---: |
| 00003548 |36|0|936|
| 00006854 |66|0|0|
| 00000133 |126|0|0|
| 00005782 |126|0|535|

Deepest baseline0/4 on this EXTREME stress set. This is not overall validation
accuracy, nor a proof all174 development positives are missed. A cheap exit can
match zero recall while preserving no lesion. The predeclared relative-safety
probe stopped: no controller/oracle GPU experiment and no easier replacement
cohort. Shallow rescue is UNMEASURED; overthinking/exit ensembles have prior art.

UNet++ pruning, MESS, class-aware thresholds, calibrated exits and safety/risk
control are required competitors. A truncated-network timing is an oracle lower
bound; deployment must pay preceding heads, router and any recomputation. Block
input/output similarity alone does NOT prove a block is safely skippable.

Statistical feasibility: even174 untouched independent positives with ZERO added
misses give1.707% one-sided95% binomial upper bound for ONE fixed policy, not1%.
299zero-failure positives needed for1%. Splitting/stratifying makes bounds weaker.
These are design math, not clinical tolerances or conformal-risk claims.

### D. Diffusion multi-depth/unbiased-gradient training

Explored nested-depth/control-variate/multifidelity ideas on CPU. Deep-only
parameters, residual variance, gradient clipping and optimizer state undermine
a naive 'unbiased gradient implies same training' argument. Unbiased raw
gradients do not give identical Adam/momentum/clipped updates or final quality.
Control variates, multi-fidelity estimation and timestep allocation are crowded.
No training speedup or promising neural cost/variance result was demonstrated.
Do not count skipped FLOPs as wall-time, or call a small toy a matched-quality win.

### E. Synthetic-example utility and validity

Real question: filters may reject useful difficult samples while retaining
generator errors. But 'hard mining', 'mask agreement', gradient alignment and
background-shortcut diagnosis already have direct competitors.

Pinned NAM code audit: HAT retries within the SAME condition (does not simply
delete rare class counts); LSRS uses global squared prediction differences;
QSF overlays a2D contour for MedGemma Yes scoring. Contour shortcutting and
lesion-region dilution remain hypotheses, not observed model failures. Extended
repo features are not all claims of the original conference paper.

Unsigned influence toy: flipping a label can preserve squared class-gap score
while changing beneficial versus harmful direction. Signed multiclass alternatives
already exist; not a novel fix and not an audit of actual published C2I execution.

Batch-Dice selection toy: companion-conditioned replacement differs algebraically
from isolated-example scoring. Binary256seed screen10reversals; more relevant
29channel aggregate1/256, exploratory rare-target0/256. Exact sufficient-statistic
reconstruction contract passes (max error1.1991e-8), but calculus alone is not
novelty, real-backbone cost is unmeasured, rare-class benefit absent. **PARK.**
VIF/JEST/ACID/FROST/class-influence are required baselines; no GPU investment yet.

### F. October7 new exact-score rare-mode solver screen

216fixed CPU comparisons,9.207s excluding imports/setup; ZERO GPU hours.
1D VE two-Gaussian mixture with analytic score, exact finite-noise initial
quantiles and exact quantile-preserving terminal reference.4096quadraturepoints,
not patient samples. Euler/Heun/RK4 and linear/geometric/rho7 grids;16/32/64/128
TOTAL score calls, intermediate stages counted. All configurations retained.
Score finite-difference/single-Gaussian/quantile/finiteness/call-count contracts pass.

Atweight.01/separation4/32calls/rho7: Euler globalMSE.00681972 versus rare-positive
regionMSE.646583,12.195% exact-positive trajectories cross outside positive region.
Ordinary Heun/RK4 remove those misses at SAME budget; nonzero shape/mass errors
can remain. Heun/rho7/32 loses zero positives on all six mixtures but can add
negative-to-positive crossings. No claim of perfect distribution recovery.

**Decision:** a weak Euler comparison can manufacture a compelling 'rare-mode'
story already fixed by a conventional baseline. Do not build a new method on it.
This does not test learned scores, 3D localized lesions or strong neural samplers.
Prevalence of a rare mixture mode and spatial size of a lesion are DIFFERENT.
INDIS/AYS/Gaussian Mixture Solvers already occupy generic schedule/multimodality
space. All code/results in `research/IDEA_REOPENING_20261007.md` and linked JSON.

## 4. What still has potential: questions, not selected contributions

These are judgments about information value, not novelty/acceptance scores.

**First lead: condition-specific fidelity under acceleration.** Can a strong
sampler preserve average image quality while violating a localized annotated
condition, in a way optimized-global/instance-aware/spatial baselines do not
already fix? An inexpensive observable that predicts AND repairs that failure
could matter. Need adequate baseline, independent condition endpoint, comparable
cost, and distinguish annotation compliance from clinical realism. Tiny toy and
Euler failure are insufficient. A broad 'global scores miss small objects' claim
already collides with CompLift/ASOB; localized controller alone is likely crowded.

**Second lead: controlled synthesis utility versus verifier shortcuts.** We have
small public paired assets enabling factorial foreground/background interventions
without new generation. A reproducible failure in a competent verifier, with a
remedy that outperforms published mitigation and simple compositing controls,
could be useful. Generic filtering is not new. Downstream augmentation-UTILITY
claims eventually require actual matched-budget downstream training/held-out
replication; inferential score alignment alone cannot certify benefit.

**Third lead: delayed tumor representation versus delayed hard decisions.** The
recorded patch proxy stays zero for hundreds of epochs. Did features/probability
ranking improve before argmax detections emerged, or was rare signal absent?
Different explanations suggest different compute investments. Long-tail
decoupling/logit adjustment/RankSEG already exist. Need historical checkpoints
or probabilities and localization at fixed FP burden, not lower-threshold(anywhere).
Standard training folders presently contain ONLY best/final for each cell;
best epochs/other backups not exhaustively audited. No temporal sequence proven.
Do not propose costly full retraining to create this evidence without a distinct
claim/cheap surrogate. No changes to original evaluation decoding.

**Conditional fallback: posterior/judge-robust codec audit.** Real score variation
is documented and might motivate a carefully scoped model-behavior study. Weak
judge/native adequacy, domain shift, existing compression work, few selected
patients and absent clinical evidence prevent the strong version. Could become
meaningful with genuinely distinguishing independent evidence, not more seeds
or more detectors until one agrees. Broad clinical-erasure story is NO-GO now.

Please propose entirely different routes if these remain crowded. The goal is
not to defend Codex's favorite idea or mechanically join two buzzwords.

## 5. Usable assets and critical reuse traps

### Small public LeFusion assets — inspected, not merely advertised

Author code revision03dc67bd8169ced5f8bb6a8707d73377f110ebff;
HFdataset revisionb0516d354c4473ea4f8539cad395dfddb5944215.
https://github.com/HINTLab/LeFusion

- Normal.tar6,344,704bytes + Demo.tar41,293,312bytes, hashes matched.
- Actual30sourceROIs/22patientIDs/90variants, NOT90independent patients or
  advertised20sources. Three variants per ROI; shape32x64x64; intended binary
  mask153–8578voxels; paired grids/shapes/masks verified.
- Source is HU; generated files are NORMALIZED, not HU. Official relation:
  `(clip(HU,-1000,400)+1000)/700 - 1`. Do not feed[-1,1] as CT HU to a detector.
- Source-normalized background differs from generated background: meanMAE
  .0115942, max.0128479normalizedunits. Not automatically clinical damage.
- All90 foreground-only/background-only intervention contracts pass; median
  background squared-change fraction4.875%, max69.212%, min.241%. This is
  change versus source, not generation-error or task-harm measurement.
- Hard pasting can create seams. Fixed masks are intended conditions, not
  independent radiologist confirmations. Three intensity variants are not
  independent subjects. Original unit-mismatch diagnostic retained and invalid
  image-error interpretation explicitly corrected.
- MIT card metadata does not settle all upstream TCIA patient-data rights;
  complete provenance/reuse terms still need checking. Raw payloads OUTSIDEGit.

Private local archives: sibling `work/lefusion-public-demo-audit`.
Contracts/data audit: `research/PAIRED_SYNTHESIS_ASSET_AUDIT_20261004.md`,
`paired_context_contract.py`, `paired_context_contract_20261004.json`.

### Public task-model / native-data leads

- MONAI lung-nodule RetinaNet bundle, HFrevision9d6622fda0e52be0a4155fed2e0afb9c17bcd80b:
  https://huggingface.co/MONAI/lung_nodule_ct_detection
  Advertised83.7MB public weights, source/config inspected, weights NOT verified
  downloaded/executed here. Research box detector, NOT segmentation judge.
  RAS/.703125,.703125,1.25mm/HU[-1024,300]→[0,1], defaultROI512,512,192;
  metadata variable patches divisible16,16,8. Coordinate/NMS/native adequacy/
  LUNA16patient-overlap gates remain. Region patches not automatically adequate.
- Lung-DDPM author three realCT/SEG demo pairs:
  https://github.com/Manem-Lab/Lung-DDPM
  label0background/1lung/2nodule, not whole-lung lesion labels. Public file payload
  access/headers/rights not established in this handoff. Their scaler/transpose/
  resized output with inherited affine is not automatic detector-grid parity.
- DiffTumor: https://github.com/MrGiovanni/DiffTumor
  Public19.26MB pancreatic U-Net downloaded/hash/strict architecture verified;
  finiteCPUforward and geometry tests passed.4.807Mparams/63state tensors.
  See failed adequacy result above. No proven training overlap, official mask
  postprocessing integration or blanket redistribution permission.
- PANORAMA selectively accessed via boundedHTTP ranges/ZIPdirectory/CRC, not
  full48GB download. Original manual annotations and matched CT geometry
  verified on selected cases. Automatic duct labels are NOT manual duct truth.
  https://zenodo.org/records/10998332
  https://github.com/DIAGNijmegen/panorama_labels
- Public UNet++v1 liver/tumor checkpoint link reached anHTMLpage; actual usable
  checkpoint retrieval/all-head training/fold provenance not verified. Its code
  has documented branch/loss mismatches; don't impute source bugs to unseenweights.
- MedCondDiff advertises weights; scoped public releases API wasempty on audit.
  Code/config lead, not a verified accessible checkpoint. Static-conditioning
  extractor caching already implemented, not our new optimization.

No established custom diffusion training pipeline or trained diffusion-UNet++
checkpoint for our paper. Existing work primarily reused pretrained codecs/task
models. Do not infer available diffusion assets from the segmentation port.

## 6. Prior art map and reading order

These links are investigated competitors, not claims we reproduced their results.
Details of inspected abstract versus source/paper are in the linked local notes.

- Adaptive depth/pruning: originalUNet++, MESS, calibrated/class-aware exits,
  Fast yet Safe, CRISP. Read `RESOURCE_MINIMAL_IDEA_SEARCH_20261002.md`.
- Medical compression: [Foundation VAE](https://arxiv.org/html/2605.30893v1),
  [MedVAE](https://arxiv.org/abs/2502.14753),
  [EQ-VAE](https://arxiv.org/abs/2502.09509),
  [Medical Latent Learnability Gap](https://papers.miccai.org/miccai-2026/1053-Paper3049.html),
  MAISI-v2/tumor objectives/pathology compression/region-separated latents.
  FoundationVAE already includes reconstructed-image nnU-Net tumor evaluation
  on MSDpancreas; generic VAE→segmentation comparison is not an empty niche.
- Synthetic validity/utility: [NAM](https://github.com/JackCD99/Native-Adversariality-Mining),
  [CompLift](https://arxiv.org/html/2505.13740v1),
  [ASOB](https://arxiv.org/html/2607.03831v1),
  [C2I](https://arxiv.org/html/2607.12464v1),
  [FROST](https://arxiv.org/html/2609.29988v1),
  [Grad-Mimic](https://arxiv.org/abs/2501.06708),
  [DiffAug](https://arxiv.org/abs/2508.17844), PRISM,
  [sample-selection/model collapse](https://arxiv.org/html/2606.13732v2).
- Attribution/batchcoupling: [VIF](https://arxiv.org/html/2412.01335v1),
  [JEST](https://arxiv.org/abs/2406.17711),
  [ACID](https://arxiv.org/html/2411.18674v2), influence functions/classinfluence.
- Background/counterfactual: [MU-Diff](https://www.nature.com/articles/s44387-025-00016-8),
  [RoentMod](https://www.nature.com/articles/s41746-026-02497-6).
- Diffusion efficiency: Min-SNR, adaptive nonuniform timestep training, AYS,
  INDIS, GaussianMixtureSolvers, simplediffusion, improvednoiseschedules.
  Latest sources/collision scopes in `IDEA_REOPENING_20261007.md`.
- Decision rules: [RankSEG](https://www.jmlr.org/beta/papers/v24/22-0712.html),
  [long-tail decoupling](https://iclr.cc/virtual_2020/poster_r1gRTCVFvB.html).

Read FIRST:

1. This handoff.
2. `research/IDEA_SEARCH_DECISION_20261002.md` (decision/evidence/costs).
3. `research/IDEA_REOPENING_20261007.md` (new solver screen).
4. `research/PAIRED_SYNTHESIS_ASSET_AUDIT_20261004.md` (actual cheapassets).
5. `research/compression_probe/GO_NO_GO_RECOMMENDATION_20261002.md`.

Then relevant branches, not every historical log:

- `research/IDEA_REOPENING_20261003.md`
- `research/SYNTHETIC_UTILITY_SOURCE_AUDIT_20261003.md`
- `research/BATCH_CONDITIONED_UTILITY_20261004.md`
- `research/compression_probe/NOVELTY_AND_DECISION_GATE_20261002.md`
- `research/compression_probe/POSTERIOR_CONTROL_20261002.md`
- `research/compression_probe/SAME_PROCESS_WINDOW_20261002.md`
- `research/compression_probe/RESEARCH_HANDOFF_20261001.md` (longchronology;
  useful for exact early design changes, not the current entrypoint).
- Original outside-Git brief: sibling `work/diffusion_idea_screen_20261001.md`.
  Its early promising hypotheses were subsequently screened; don't treat its
  preliminary 'no medical version found' search claims as current novelty proof.

## 7. Compute/storage/access/accounting limits

Approved screen cap: **2 TOTAL charged GPU-hours**, including failures,
calibration and evaluation, not two per job/allocation. Last verified historical
reconstructed DeltaAI ledger: 4413 weighted seconds = **1.225833/2**, remainder
.774167. This includes 17 allocations, not just successful neural time. Posted
debit, rounding and current other spending remain unverified; refresh before
any proposed GPU submission. Subsequent noted CPU-only screens used zero GPU-hours.

Available resources in historical context (verify live, don't assume queues):

- JHU GB10/DGX Spark: one GPU per physical node, 128GB CPU/GPU unified RAM;
  activation bandwidth bottleneck, caching allocator can starve the OS. bdmap1
  also serves LIVE BodyMaps with real users: no exploratory jobs/changes there.
  Other nodes are not yours just because SSH works; coordinate before access.
- DeltaAI: GH200 120GB, four GPUs/node, aarch64; account `bdyo-dtai-gh`,
  home `/u/asanjeev`; shared `/projects/bdyo/asanjeev`. Slurm modules/isolated env.
- Delta: H200/A100, x86. **Delta and DeltaAI share taiga bdyo project quota**;
  filesystem PB free != group quota available. Earlier separate-storage claim
  was wrong.
- Bridges-2: H100s/Ocean; account `cis260296p`. Protected job uses two H100s
  at the recorded snapshot. Ocean is allocation-backed storage, not unlimited disk.

MFA is personally approved by the user. Agents use an existing WSL Unix
ControlMaster, not native Windows OpenSSH multiplexing. Never ask for/publish
passwords or bypass MFA. Connection names age; don't assume an old socket works.
The paper turn used `/home/adity/.ssh/controlmasters/bridges2-paper-oct6b.sock`
(last verified live). Remote root:
`/ocean/projects/cis260296p/asanjeev/pants_unetpp`.
Do not open fresh allocations for read-only diagnostics achievable on CPU/login.

Inference-only full-paper cost is NOT automatically small: an archived
hypothetical 2000-volume-pass design at 10/30/60 seconds costs 5.6/16.7/33.3
physical GPU-hours before router profiling/repeats/staging/billing. Those times
were assumptions, not measured 3D exit runtimes. The earlier 36R repeated tiny
training design is a cost sensitivity, not a forecast. Count generation,
scoring, rejected attempts, selector training and downstream training. DDP
does not automatically lower total charged GPU-hours.

## 8. Working discipline and what NOT to claim

- Research first: source/asset/contract checks, then the smallest distinguishing
  test. Check simple strong controls before training a new clever controller.
- Frozen cohorts/gates are not rewritten after outcomes. Exploratory follow-ups
  are allowed if explicitly NEW hypotheses; never retrospectively pass a failed gate.
- Native model adequacy is necessary for interpreting retained clinical/task
  ability, not a requirement that every interesting toy be clinically adequate.
- Full context, proper units/affines/native grid, strict weights and provenance
  checks. GT is for scoring, not a silent inference crop/mask/helper.
- No raw patient data/weights/private images in Git. No license/provenance
  assumptions from a dataset card alone. No unsafe pickle fallback.
- No model confidence → clinical visibility; no detector failure → information
  erasure; no posterior seeds → independent patients; no aggregate Dice →
  tiny-tumor safety.
- No official-scorer parity, complete-grid or multi-seed claims until the
  actual evidence exists. No lab website/other users' directory changes.
- Repo coordination: pull before reading/writing status; update your own
  `coordination/CLAUDE_STATUS.md`, not Codex's entries; scoped commit/push.
  Preserve unrelated untracked deployment drafts and active frozen snapshots.
- Ask only when authority, material choice, access, new spending or PI alignment
  is genuinely missing. General enthusiasm is not an unbounded compute budget.
  The user requested this handoff, not new jobs.

## 9. My candid judgment for Claude

The best information-per-resource work so far has been killing unsupported
stories with controls: exact CPU algebra/solver baselines, source audits, native
adequacy and same-input/window/posterior repeats. We have engineering assets and
interesting observations, NOT a selected high-novelty contribution.

The largest conceptual mistake to avoid is optimizing an assumed problem before
showing that a competent model actually has it and that simple baselines don't
fix it. The largest resource mistake is calling a two-hour pilot an inexpensive
paper when the claim ultimately needs expensive counterfactual training or
independent cohorts.

Claude's source-audit and criticism inputs have been useful, especially when
they found collisions/confounds and narrowed claims. Agreement or eloquence
alone is not validation. Please bring a fresh mechanism or clear new evidence,
not just praise/repackage this work. A justified 'none of these merits investment'
with a better search strategy is more useful than confidently naming an S-tier
idea that collapses under the first strong baseline.

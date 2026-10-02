# Compression research: evidence and next-step handoff

Updated October 1, 2026. For Claude/PI critical review, not a launch instruction.

## 1. Research question and current verdict

Goal: find a useful, PI-aligned efficiency/quality research direction cheaply,
before committing to diffusion training. Generic noise-adaptive depth is crowded:
the original idea screen discusses ASE, multi-decoder diffusion, Early-Bird,
TDC and other competitors. Those historical literature claims are not newly
reverified by this handoff; Claude should critically check them before relying
on novelty or published speedup comparisons.

We pivoted to the earlier bottleneck: does pretrained 3D CT autoencoder
compression preserve small pancreatic lesions? This separates reconstruction
damage from diffusion/noise/depth damage. It is NOT itself a faster-training
result, a new method, or an established top-conference contribution.

Current verdict: three reconstructions show case-dependent intensity changes.
In two small lesions, whole-lesion contrast summaries can obscure heterogeneous
slice-level changes. We have NOT demonstrated lesion disappearance, downstream
recall loss, human-reader harm, a causal size effect, or VAE-specific pathology.
Do not start expensive lesion-preserving VAE training on this evidence alone.

## 2. Boundary with the main PanTS paper

This exploratory work is separate from the frozen 1000-epoch segmentation grid.
No training/evaluation code, checkpoint or shared runtime was changed for it.
The last verified historical status was four trained cells and completed full
development validations, with Bridges-2 test evaluation job47320183 pending.
That is NOT a live queue check and NOT evidence that the final2x2 table is ready.

The nine cached PanTS calibration labels inspected for this screen had ZERO
class28 voxels. Prior timing measurements remain timing evidence; they must
not be represented as positive-tumor quality validation. Full development
validation had174 positive cases among1800. Root cause of calibration selection
mismatch remains unresolved. User chose MSD rather than PanTS for this screen.
MSD tumor label2 must not be confused with PanTS tumor label28.

## 3. Reproducible setup and protections

- One scheduled DeltaAI GH200120GB; no GPU work on login nodes, no DDP.
- Public pretrained MAISI autoencoder_v1.pt only; no diffusion checkpoint,
  training, quantization, segmenter inference or depth-routing experiment.
- Whole-volume FP32 encode/decode, including normalization; deterministic
  posterior mean, native spacing, RAS orientation, HU[-1000,1000]->[0,1],
  center padding divisible by4. No spatial crops or decoder tiles.
- Isolated MONAI1.5.1 wheel import; base pants_venv left untouched. Official
  default config differs: norm_float16=true and num_splits=4, versus false/1.
- Source revision5cb04e82fed71f2fe64a2617ab695be1f5a37fea,
  weights revision430f2c82a96dce44b455de5876d43a5a9f753cb2.
  Weight SHA2561f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3.
- MSD mirror Angelou0516/msd-pancreas revision
  327dd551a9e51295c34e14f311d8330ef44b6cac. Input hashes checked. Mirror
  bytes NOT independently matched against original official MSD archive.
- Finite values, mask positivity, shape/affine and physical spacing validated;
  preprocessing-only control retained. No checkpoint/data overwriting.
- Hard job/process time limits, one GPU verified before release, bounded
  memory, no automatic retries/requeue, actual allocation accounting captured.
- First job hit32G host-memory limit; retry requested96G, still exactly one
  GPU. Actual Python RSS was30.58-35.89GiB on successful cases, explaining
  why GPU memory alone was insufficient for safe resource planning.
- Case165's first download was truncated and rejected by hash check BEFORE
  GPU use; retained as incomplete evidence, second CPU download passed.

## 4. What actually ran

| Case | Selection and tumor burden | Tumor HU MAE | Boundary-band HU MAE | Local contrast control -> reconstruction | CNR control -> reconstruction |
|---|---|---:|---:|---|---|
|005|Initial pipeline case;7.60mL,3609voxels|36.21|36.09|43.67->43.32HU|.56165->.54443|
|120|Smallest total GT tumor burden among120 inspected masks;.848618mL,454voxels|18.27|20.08|-65.55->-58.58HU|-1.1691->-.9785|
|165|Second-smallest among same120;1.060962mL,445voxels|42.12|45.04|11.79->11.20HU|.22660->.26608|

Each has one6-connected GT tumor component. Cases120/165 each span three axial
tumor slices, at spacings(.683594,.683594,4mm) and(.976562,.976562,2.5mm).
Selections made before inspecting that case's reconstruction. This bounded
subset is NOT representative or the global dataset minimum.1mL is an
exploratory cutoff, not a clinical staging definition;165 is above it.

Preprocessing tumor-region/boundary-band MAE was0 on all three cases. This
does not mean whole-image clipping was identity. Boundary-band HU MAE measures
intensity error near the original contour, NOT predicted boundary displacement.
CNR is a local image statistic, NOT detection accuracy.

### Post-hoc sensitivity checks

Case120 all-tissue contrast retention across1-3/2-5/3-7mm rings:
.8725/.8937/.9210; GT-pancreas-only:.8635/.8524/.8611. Its axial ratios:
.8915/.8583/1.0148. Mean tumor shift-1.96HU versus standard-ring-8.93HU:
contrast attenuation includes surrounding-tissue changes, not just tumor fading.

Case165 all-tissue ratios:.6307/.9498/1.0898; pancreas-only:
.6307/.9443/1.1606. Axial absolute contrast pairs25.86->16.27,
22.97->7.24,.85->13.41HU. The last baseline is near zero: a large ratio is
misleading. Tumor/ring mean shifts-32.52/-31.93HU nearly cancel in the
whole-lesion contrast average. CNR increased17.42%, potentially from reduced
surrounding variance, despite substantial reconstruction differences.

These ring/slice analyses are POST-HOC exploratory descriptions, not independent
replications. Three slices from one lesion are not three independent cases.
Local CPU metric halos are applied AFTER whole-volume reconstruction; they
are not model-input crops. Nine implementation tests pass, not clinical tests.

## 5. Input and execution audits

Pinned official VAE validation transforms reproduced on real120/165 using
isolated MONAI: prepared input arrays exactly equal the probe's arrays,
max difference0. Shapes512x512x104 and512x512x88 after padding. This rules
out a basic mismatch against the original-spacing default validation path,
NOT all official deployment differences. Diffusion embedding creation separately
resizes dimensions to multiples128; production decode may use sliding windows.

Installed MONAI1.5.1 source distinguishes reconstruct(x), which decodes mu,
from forward(x)/encode_stage_2_inputs(x), which sample mu+epsilon*sigma.
Our mean reconstruction matches an official API. It is not a faithful test
of sampled diffusion embeddings.

CPU FP32 contract check with real weights, random input[1,1,8,64,8]:
- Explicit mean encode/decode vs reconstruct: max error0.
- Stage2 embedding vs same-seed explicit sampling: max error0.
-36 convolution layers changed in memory from num_splits1 to4:
  encoded mu max error1.5497e-6, sigma1.7881e-7;
  same-latent decode2.2650e-6; full mean reconstruction2.4140e-6.
- Baseline normalized output scale1.09344; finite checked outputs.
- Sample-versus-mean latent max difference2.75780, sigma mean.535319.

Toy CPU contracts do NOT prove real-image/GPU/FP16/chunk-offload parity.
Posterior sampling can materially change latents; no lesion-level consequence
has been measured. No extra GPU cost for these audits.

## 6. Actual compute, not estimates of future training

| Job | Outcome | Allocation seconds | Physical GPU-hours | Approx charged-equivalent hours |
|---|---|---:|---:|---:|
|3289813|Host-memory OOM; no reconstruction|103|.02861|.05722|
|3289833|Case005 completed|204|.05667|.11333|
|3290494|Case120 completed|162|.04500|.09000|
|3290519|Case165 completed|128|.03556|.07111|
|Total|Includes failed allocation|597|.16583|.33167|

Interactive2x rule; figures derive from actual elapsed allocation, not posted
account debits. First campaign.17056/.25; separately approved additional
campaign.16111/.25, remaining.08889. Do NOT combine caps as a single overrun.
Forward/copy timings005106.89s,120107.89s,16596.11s; CUDA allocation peaks
39.09/39.09/33.09GiB. Allocation time includes startup/metrics and is the
appropriate spending measure. Remaining interactive budget offers insufficient
safe margin for an uncalibrated detector run; seek a new bounded cap first.

## 7. Reader packet and its limits

Prepared blinded A/B packet: six pages, every tumor-containing axial slice
for120/165, full context plus60mm detail, fixed HU[-100,200], no GT outline.
Case order/condition side randomized once per case. Known-location fidelity
review, NOT unprompted detection or a clinical reader study. Texture could
reveal condition; reviewer should record guesses. No independent ratings yet.

Local worktree ZIP: research/compression_probe/blinded_review_20261001.zip.
Private condition key stays remote; not downloaded, committed or included in
ZIP. Image publication was blocked without explicit egress permission; keep
packet/CT-derived images local until user explicitly approves. Code/written
reports can be shared. Do not send an unblinded repository to a blinded reader.

## 8. Alternatives to recruiting a new reader: verified findings

User asks whether already expert-annotated CTs can support a cheap objective
screen. YES, but original-image labels are not ratings of reconstructed images.

### PANORAMA — preferred pancreatic candidate, NOT execution-ready

Official annotation repository describes676PDAC cases;482 have manual lesion
delineations by two trained investigators supervised by an expert radiologist
with20+years pancreatic-cancer experience. Remaining194 have automatically
generated lesion annotations. Select actual manual labels; never silently mix.
Clinical reference standards vary by patient, including histopathology and
CT-only confirmation. Consult per-patient clinical information, not a blanket
claim that all public cases have pathology confirmation.

Public dataset includes MSD and NIH cases as well as Dutch cohorts. Selecting
PANORAMA does not automatically create a new independent dataset. Released
baseline uses two-stage pancreas localization then PDAC detection, publishes
five-fold splits and weights. It targets PDAC, NOT every pancreatic lesion.

Promising alternative: pair manual cases with a model fold that excluded them.
Before using it, verify exact released checkpoint->split mapping, all training
and validation/model-selection exposures, localization-stage exclusions,
clinical subtype and overlap with MAISI pretraining. A validation-fold result
can be useful exploratory evidence, but is NOT untouched-test performance if
the checkpoint was selected on that fold. Full five-fold ensemble generally
includes models trained on the selected public patients; do not use it and
claim patient-disjointness. Fold JSON existence alone proves no exclusion.

### LIDC-IDRI — strong existing multi-reader annotation, different task

Original study provides1018 thoracic CT scans annotated in a two-phase process
by four experienced radiologists per scan. Nodules>=3mm include outlines and
subjective characteristics; smaller marks are not equivalent contour labels.
Reader disagreement is available, and malignancy ratings are not universal
pathology-confirmed cancer diagnoses. Could test small-nodule preservation,
but lung-nodule evidence cannot establish pancreatic-tumor preservation.
Pretraining overlap and detector patient splits still require checking.

### Other detector candidates audited, not run

- MONAI pancreas_ct_dints_segmentation: label semantics matchMSD, but trained
  onMSDTask07; released HF tree lacks dataset_0.json. Recreating intended
  seed123 split is NOT proof of the checkpoint's actual split. Weights~554MB;
  runtime MONAI1.4/torch2.4 differs from shared environment. No download/run.
- PANTHER/PancCTMultiTalentV2: official repository says765 pancreatic-tumor
  cases fromMSD+PANORAMA. Not patient-independent on ourMSD cases without
  actual exclusions. No weights downloaded/run.
- Official MAISI source table listsMSDTask03, notTask07, forVAEv1; diffusion
  DDPM table does list224Task07. Distinguish VAE/diffusion corpora; absence
  from a table is NOT patient-level independence proof.

## 9. Proposed future plan — staged, not launched

1. CPU/read-only feasibility audit: inspect PANORAMA checkpoint packaging,
   both stages' exact folds, manual-case IDs, clinical labels and licenses.
   Pin revisions and map case identity across public datasets. Stop if genuine
   exclusions cannot be established; don't disguise in-sample results.
2. Lock a small case list before outputs: small manually outlinedPDAC lesions
   plus negatives; use physical size, not voxel count alone. State selection
   limits, slice thickness and clinical reference strength. Avoid full-corpus
   download. Preserve PanTS's untouched test evaluation and environments.
3. Define paired screen: original -> preprocessing control -> VAE reconstruction.
   Same frozen detector/runtime/settings/automatic ROI on every condition.
   No GT-centered model crops. Verify GT and prediction world-coordinate maps.
   Fix probability thresholds/lesion matching from released protocol or
   independent calibration, not these cases. Preserve full probabilities.
4. CPU preflight first, then request NEW capped GPU calibration for one case.
   Measure complete detector plusVAE cost and host/GPU memory. Pin weights,
   versions/input hashes and timeout; no unbounded retry or environment change.
5. Primary outcomes: paired lesion detections lost/gained and false positives
   at a fixed threshold; stratify small lesions. Report raw detector misses,
   retained-hit transitions and all cases, not only originally detected cases.
   Secondary tumorDice/surface metrics and contrast; organ averages are not
   success. Small sample establishes feasibility, not population safety.
6. Consider fixed physical Gaussian-smoothing controls to see whether observed
   contrast/CNR behavior is generic smoothing. Not yet implemented. Prespecify
   sigmas; do not tune them on outcomes to manufacture a VAE-specific effect.
7. If objective paired losses repeat beyond controls/runtime variability,
   bring evidence toPI; consider reader confirmation and a representation
   repair only after checking closest lesion-aware/region-aware prior work.
   If no reproducible damage, report negative result and move to noise/depth
   probes only after obtaining compatible pretrained denoiser/config/budget.

Do not assert that latent compression makes repair theoretically impossible:
conditional generation may infer/synthesize detail, but that does not establish
patient-specific preservation. Reconstruction and de-novo generation are
different tasks. Faster inference is not faster training; any eventual method
must report total training/setup/search/evaluation hours at matched quality.

## 10. What Claude should challenge

- Is PANORAMA out-of-fold inference actually possible with released weights,
  including the localization stage and checkpoint-selection exposure?
- Would a different public cohort with verified manual labels avoid overlap
  more cheaply? Keep anatomy/task changes explicit.
- Are posterior mean/native spacing the appropriate diagnostic, and which
  production parity factor deserves the next capped test?
- Are the metrics detecting generic smoothing/intensity shifts rather than
  a specific clinically meaningful failure? What cheaper decisive control exists?
- What is the smallest defensible pilot and negative-result stopping rule?
- Is there a novel question after prior-work review, or only known VAE blur?

No guaranteed top-conference idea or measured training speedup exists yet.
Critique this evidence before proposing a large training campaign.

## 11. Sources and artifact map

Primary sources checked:
- https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/tree/5cb04e82fed71f2fe64a2617ab695be1f5a37fea
- https://huggingface.co/nvidia/NV-Generate-CT/tree/430f2c82a96dce44b455de5876d43a5a9f753cb2
- https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/blob/main/data/README.md
- https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/autoencoderkl.py
- https://github.com/DIAGNijmegen/panorama_labels
- https://zenodo.org/records/10998332 (public imaging record; follow current version)
- https://github.com/DIAGNijmegen/PANORAMA_baseline (both fold files in src/)
- https://www.cancerimagingarchive.net/collection/lidc-idri/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3041807/ (original LIDC study)
- https://github.com/MIC-DKFZ/panther
- https://huggingface.co/MONAI/pancreas_ct_dints_segmentation/tree/1b4b04a0de2cf6236860891bf1c2e9494e1afaf7

Repo research/compression_probe/: PROTOCOL.md, PREPROCESSING_AND_JUDGE_AUDIT.md,
results_20261001/SUMMARY.md, small_results_20261001/SUMMARY.md,
second_results_20261001/SUMMARY.md; their completion/provenance/metric records.
Tooling: prepare_msd.py, run_maisi.py, region_metrics.py, inventory_small_msd.py,
prepare_small_followup.py, contrast_sensitivity.py, audit_preprocessing.py,
audit_execution_contracts.py, prepare_blinded_review.py, test_*.py and bounded
Slurm scripts. Some image artifacts remain local and are deliberately untracked.

Remote isolated root:/projects/bdyo/asanjeev/compression_probe_20261001.
Original/decoded volumes, input manifests, CPU raw audits and private reader
key stay there. Recheck access/live state before remote work; this document
does not authorize MFA bypass, data publishing or new GPU jobs.

## 12. Independent review (Claude, October 1, 2026)

Research-only review. No GPU job, environment change, image upload, PanTS
evaluation change or data download beyond public metadata/fold files and two
ZIP central directories (HTTP range reads, no weights downloaded).
Labels: **[measured]** = read from committed metrics/fold/metadata files or
verified at a primary source in this review; **[hypothesis]** = untested;
**[estimate]** = not measured.

### 12.1 Verification of Sections 4-8

- [measured] Section 4 table matches committed `*/metrics.json` exactly
  (tumor MAE 36.21/18.27/42.12 HU; contrast 43.67->43.32, -65.55->-58.58,
  11.79->11.20 HU). Preprocessing control tumor/boundary MAE is 0 in all three.
- [measured] PANORAMA README: 676 PDAC, 482 manual, 194 automatic; public set
  includes 194 MSD Task07 patients (98 PDAC) and 80 NIH Pancreas-CT patients.
  Correction: the `manual_labels/` folder holds 482 files, of which 3 are
  labelled non-PDAC in `clinical_information.xlsx` (479 PDAC).
- [measured] MAISI data README: CT VAE v1 trained on 37,243 volumes, of which
  NLST chest CT is 31,801 (~85%); abdominal sources are TCIA Colon (1,522) and
  MSD Task03 Liver (104). MSD Task07 is listed for the diffusion model (224),
  not the VAE. PANORAMA/Dutch cohorts are not listed for either model.
- [measured] Section 6 compute figures are internally consistent
  (597 allocation-seconds = 0.1658 physical GPU-h).

### 12.2 Confounds in the current evidence

1. **Lesion error is not shown to be lesion-specific.** [measured] Tumor MAE
   is below whole-image MAE in all three cases (36.2 vs 63.1; 18.3 vs 21.9;
   42.1 vs 69.2 HU). Whole-image MAE mixes air, bone and soft tissue, so it is
   not the right reference either. Needed: *excess* error = lesion MAE minus
   MAE in volume-matched non-lesion pancreas (or matched soft tissue) in the
   same reconstruction.
2. **Global intensity bias masquerades as lesion change.** [measured] Case 165
   tumor and ring both shift about -32 HU (Section 4). MAE counts this as
   damage; contrast cancels it. Report bias-corrected error (subtract the
   local or organ-level mean shift) alongside raw MAE.
3. **Latent grid vs lesion size, driven by slice thickness.** [hypothesis,
   based on the probe's 4-divisible padding, consistent with 4x per-axis
   compression] At native spacing a 4x latent cell is 16 mm along z for case
   120 (4 mm slices) and 10 mm for case 165 (2.5 mm). Both lesions span three
   slices (12 mm and 7.5 mm), i.e. about one latent cell or less along z. Any
   size effect is confounded with slice thickness. Stratify by lesion extent
   *in latent cells*, and add a fixed-spacing condition (resample to one
   isotropic spacing before encoding) so thickness is held constant.
4. **Domain shift vs compression.** [measured + hypothesis] The VAE is ~85%
   chest CT. Abdominal low-contrast lesion loss could reflect training-domain
   mismatch rather than compression capacity. Control: an abdomen-trained
   autoencoder (e.g. DiffTumor's, trained on abdominal CT) on the same cases.
   If it preserves what MAISI loses, the cause is domain, not latent size.
5. **Generic smoothing.** Already noted (Section 9, step 6). Make it the
   primary control, not optional: Gaussian blur matched to the VAE's measured
   global error or noise power, prespecified.
6. **Noise reduction inflates CNR.** [measured] Case 165 CNR rose 17% while
   contrast fell; ring variance dropped. CNR is not a detectability measure
   when noise texture changes. Prefer a model-observer statistic or detector
   output.
7. **Selection and n.** [measured] n = 3, two chosen as the smallest of 120
   inspected masks. Results cannot separate case idiosyncrasy from a size
   effect. Section 4's caveats are correct.
8. **In-sample risk for MSD cases.** [measured] All 194 MSD-sourced PANORAMA
   studies sit in the PANORAMA folds (30-46 per fold) and the metadata gives
   no mapping to original MSD IDs. Any PANORAMA detector run on our MSD cases
   005/120/165 is therefore of unknown exposure. Do not use MSD cases with the
   PANORAMA detector.

### 12.3 Can PANORAMA's released folds give patient-disjoint tests?

Short answer: **yes, with two exclusions and one residual caveat.**

- [measured] Per-fold weights are released separately for BOTH stages
  (Zenodo record 11160381, ZIP central directories read via range request):
  `Dataset103_.../fold_{0..4}/checkpoint_final.pth` (pancreas localization)
  and `Dataset104_.../fold_{0..4}/checkpoint_best_panorama.pth` (detection).
  The default `process.py` ensembles folds 0-4; out-of-fold use requires
  running `-f k` for both stages.
- [measured] The two stages use **identical** validation folds (all five
  folds equal, 448/448/448/447/447 studies, union 2,238, no overlap). Study k
  in fold k is unseen by both fold-k checkpoints, including the localization
  stage.
- [measured] **Folds are split by study, not by patient.** Using
  `PANORAMA_patient_id` from `clinical_information.xlsx`: 2,224 patients,
  11 of whom have studies in more than one validation fold (25 studies).
  Exclude those 11 patients to be patient-disjoint.
- [measured] Excluding those 11 patients and MSD/NIH-sourced studies leaves
  **380 manually annotated PDAC studies** (fold 0-4: 74/81/84/58/83) with
  reference standard histopathology 167, pathology 109, cytology 85,
  radiology 19; and 265-284 clean negatives per fold.
- Residual caveat (selection exposure): the detection checkpoint was chosen
  per fold by AUROC/AP on that fold's validation set, so absolute fold-k
  performance is optimistic. For a **paired** original-vs-reconstruction
  comparison the same checkpoint sees both arms, so this bias largely cancels
  in the difference [hypothesis; standard paired-design reasoning]. Report
  it anyway. The pancreas stage uses `checkpoint_final` (no selection).
- Unverified: the ZIPs contain no `splits_final.json`, so the
  checkpoint-to-fold mapping rests on the README and directory names. Cheap
  CPU check: load one `.pth` and read its stored `init_args` (fold, dataset).
- Unverified: whether the PANORAMA imaging release lets you download
  selected cases without the full corpus.
- [measured] PANORAMA/Dutch cohorts are not in MAISI VAE or diffusion
  training lists, so Dutch-sourced cases are also clean with respect to MAISI.

### 12.4 Cheapest decisive experiment

Run two tiers; stop after Tier A if it is negative.

**Tier A — controlled lesion-insertion dose-response (no detector, no reader).**
- [estimate] Cost: about 8 MAISI VAE passes. Measured passes in Section 6 were
  0.036-0.057 physical GPU-h each, so roughly 0.3-0.5 GPU-h including one
  calibration. Needs a new bounded cap.
- Hosts: 4 PANORAMA-native negative studies from one fold (clean of MAISI and
  of patient leakage), at native spacing AND resampled to one fixed isotropic
  spacing (2 conditions x 4 hosts = 8 passes).
- Inserts: ellipsoidal low-contrast lesions in GT pancreas parenchyma,
  diameters 3/5/8/12/20 mm and contrasts -10/-20/-40/-60 HU, with background
  noise preserved (add the contrast to existing texture, do not paint flat
  values), placed >= 30 mm apart so several fit per volume. Prespecify all
  positions before any reconstruction.
- Arms per host: preprocessing control, MAISI reconstruction, and a
  prespecified Gaussian-blur control matched to the VAE's measured noise power
  in parenchyma away from inserts.
- Outcomes: contrast retention (bias-corrected) and a non-prewhitening
  matched-filter SNR per insert; plot vs diameter in latent cells.
- Limitation: inserts are not real PDAC (no infiltrative margin or desmoplastic
  texture). Tier A tests a necessary condition: can the VAE carry a small,
  faint, known signal at all.
- **GO to Tier B** if, at fixed spacing, inserts <= 8 mm with |contrast|
  <= 40 HU keep < 0.5 of their matched-filter SNR in >= 3 of 4 hosts AND are
  worse than the Gaussian control by more than the host-to-host spread.
- **NO-GO** (compression is not the bottleneck at that scale) if inserts
  >= 5 mm keep >= 0.8 of SNR in >= 3 of 4 hosts. Record the negative result
  and return to the noise/depth probes.
- In between: report as inconclusive; do not tune sizes or contrasts on the
  outcome.

**Tier B — real-lesion paired detector test (only after Tier A GO).**
- Cases: from one PANORAMA fold, ~30 clean manually annotated PDAC with
  small lesions (smallest physical diameter; prespecify cut-off before
  outputs) plus ~30 clean negatives. Out-of-fold checkpoints for both stages.
- Arms: original, preprocessing control, MAISI reconstruction, Gaussian
  control. Same runtime, thresholds from the released protocol
  (`extract_lesion_candidates`, picai_eval matching), full probabilities saved.
- Primary outcome: paired lesion-level detection lost vs gained at a fixed
  threshold, exact McNemar on discordant pairs; false positives on negatives.
- **GO** (bring to PI): losses minus gains >= 5 with one-sided exact McNemar
  p < 0.05 for MAISI vs control, and MAISI loses more than the Gaussian arm.
  (For reference, 7 losses and 0 gains gives p = 0.0078.)
- **NO-GO**: losses minus gains <= 2, or MAISI no worse than the Gaussian
  arm. Then compression damage is either absent or generic smoothing; stop
  the representation-repair idea.
- [estimate] Cost unknown until a one-case calibration of the two-stage
  detector plus VAE; calibrate first under a new cap, as Section 9 says.

### 12.5 Changes to the Section 9 plan

- Step 1: add the patient-ID leakage exclusion (11 patients) and the
  `.pth` `init_args` check; drop MSD cases for any PANORAMA detector use.
- Step 3: add the fixed-spacing condition and bias-corrected error; make the
  Gaussian control mandatory, not "consider".
- New step before 4: Tier A insertion dose-response, which is cheaper than
  any detector run and has exact ground truth.
- Step 5: report lesion size in latent cells as well as mm.
- Add an abdomen-trained-autoencoder arm if Tier A is GO, to separate domain
  shift from compression.

### 12.6 Answers to Section 10

- Out-of-fold PANORAMA inference: possible for both stages with released
  per-fold weights; patient-disjoint after excluding 11 patients; detection
  checkpoint has fold-level selection exposure (largely cancels in paired
  deltas).
- Cheaper cohort: for the *preservation* question, controlled insertion
  (Tier A) is cheaper and more decisive than any new cohort. LIDC remains a
  different anatomy/task.
- Posterior mean / native spacing: mean is right for a reconstruction bound;
  native spacing confounds size with slice thickness, so add fixed spacing.
- Generic smoothing vs specific failure: the matched Gaussian arm is the
  decisive control.
- Smallest pilot / stopping rule: Tier A with the GO/NO-GO above.
- Novelty: "VAEs blur small low-contrast detail" alone is expected and not
  novel. A quantified size x contrast x latent-cell transfer curve for a
  widely used 3D CT foundation VAE, tied to real-lesion detection loss, would
  be a useful measurement; a repair method would need to beat region-aware
  prior work (e.g. MAISI-v2's region-specific contrastive loss).

### 12.7 Sources checked in this review

- PANORAMA baseline repo, README, `src/process.py`, both fold JSONs:
  https://github.com/DIAGNijmegen/PANORAMA_baseline
- PANORAMA baseline weights (ZIP directories read by range request):
  https://zenodo.org/records/11160381
- PANORAMA labels README, `clinical_information.xlsx`, `manual_labels/`:
  https://github.com/DIAGNijmegen/panorama_labels
- MAISI CT training-data README:
  https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/blob/main/data/README.md
- MAISI paper (VAE training corpus, small-organ synthetic-data gap):
  https://arxiv.org/abs/2409.11169
- MAISI-v2 (region-specific contrastive loss, closest repair prior work):
  https://arxiv.org/abs/2508.05772
- Committed probe outputs: `results_20261001/`, `small_results_20261001/`,
  `second_results_20261001/` (`metrics.json`, `contrast_sensitivity.json`).

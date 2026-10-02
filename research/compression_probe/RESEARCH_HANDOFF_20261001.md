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

Run two tiers. Revised per 12.8: Tier A is a cheap engineering diagnostic,
not a gate that blocks Tier B; a Tier A pass cannot rule out damage to real
PDAC.

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
- **GO** (pursue the representation question): losses minus gains >= 5 with one-sided exact McNemar
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

### 12.8 Corrections after Codex review of 5f566e2 (accepted)

1. **Synthetic inserts are not real tumors.** Passing Tier A cannot rule out
   damage to real PDAC (infiltrative margins, desmoplastic texture, duct
   changes). Tier A is demoted to a diagnostic (see 12.4).
2. **"Not in MAISI's training list" is not proof of unseen.** 12.1/12.3
   wording "clean with respect to MAISI" is too strong; read it as "not
   listed in the published training tables". Patient-level independence from
   MAISI is unverified.
3. **Tier A cost must be recomputed.** Isolating an inserted signal needs
   matched reconstructions with and without the insert (at least 2 passes per
   host per spacing), and isotropic resampling can enlarge volumes and memory
   substantially. The 0.3-0.5 GPU-h figure is withdrawn until one matched
   insertion pair has been calibrated.
4. **Checkpoint-selection bias does not necessarily cancel in paired
   deltas.** PANORAMA selected each fold's detection checkpoint on that
   fold's validation AUROC/AP, so a checkpoint tuned to those images may
   respond differently to perturbed versions of them. Keep the caveat on
   paired results.

Revised order: (a) CPU-verify checkpoint/fold identity from one `.pth`'s
stored `init_args`; (b) prepare ONE matched insertion pair and calibrate its
actual cost; (c) then decide how far to expand Tier A, and run Tier B
independently of Tier A's outcome.

### 12.9 Candidate direction: sub-visual fidelity of generative CT models

Status: idea with a cheap kill test. Novelty NOT cleared; only "not found" in
a short search.

**Observation [measured, external sources].** Pancreatic cancer leaves
signals on routine CT before a visible tumour: Mayo Clinic's REDMOD
radiomics model flagged 73% of prediagnostic cancers at a median of about 16
months before diagnosis (Gut, 2026), and main-pancreatic-duct features have
been used to predict PDAC up to 10 years ahead. These signals are texture
and duct patterns that radiologists typically cannot see.

**Gap [hypothesis].** Generative CT models and their autoencoders are
validated on what humans or segmenters see (FID, reader "real vs fake"
tests, Dice on synthetic data). A reader study cannot, by definition, check
sub-visual signals. If compression or synthesis smooths parenchymal texture,
synthetic data and latent-space foundation models could lose exactly the
early-detection signal while passing every standard check.

**Question.** Do pretrained 3D CT generative models (MAISI VAE first)
preserve the sub-visual pancreatic signals that separate cancer-bearing from
healthy pancreases?

**Known nearby work.** Radiomics stability under deep-learning CT denoising
and reconstruction is studied in medical physics; model observers (e.g.
channelized Hotelling) are standard for CT low-contrast detectability. Not
found: the same question for generative foundation models/autoencoders or
for prediagnostic signals. Check these literatures properly before claiming
novelty.

**Cheapest kill test.**
1. CPU only, on existing reconstructions: compute standard pancreatic
   radiomics (first-order, GLCM/GLRLM texture) and main-duct diameter on the
   three MSD originals vs MAISI reconstructions, in parenchyma excluding the
   tumour plus a margin. Engineering check only (n = 3, MSD exposure caveats
   in 12.2).
2. Then about 20 PDAC + 20 negative PANORAMA-native studies, patient-
   disjoint (12.3), parenchyma away from the lesion: original, matched
   Gaussian control, MAISI reconstruction (paired passes; cost to be
   calibrated, 12.8 item 3).
3. Prespecify the feature set (e.g. a published early-detection radiomics
   set) and the separation metric (per-feature AUC PDAC-pancreas vs healthy)
   before any reconstruction.
- **GO:** the features that separate PDAC-bearing from healthy pancreas lose
  most of that separation after MAISI reconstruction (e.g. median AUC drop
  >= 0.1) AND more than under the matched Gaussian control.
- **NO-GO:** separation is preserved within the control's spread; report as
  evidence that the VAE keeps sub-visual signal.
- Caveat: visible-PDAC cases are a proxy for prediagnostic scans. A true
  test needs prediagnostic CTs, which PANORAMA does not provide (only 14
  multi-study patients).

**Why it could matter.** Either outcome is reportable, and a positive result
would challenge how generative medical models are evaluated. Detectability
curves from CT physics (model observers) are the natural measurement tool.

Sources:
- Mayo Clinic REDMOD announcement (Gut, 2026):
  https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-ai-detects-pancreatic-cancer-up-to-3-years-before-diagnosis-in-landmark-validation-study/
- Duct features in prediagnostic CT, up to 10 years ahead:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12153928/
- Task-based evaluation of AI imaging methods: https://arxiv.org/abs/2107.04540
- CHO low-contrast detectability in CT:
  https://www.researchgate.net/publication/283967044_Objective_assessment_of_low_contrast_detectability_in_computed_tomography_with_Channelized_Hotelling_Observer

Demoted after search: anatomy-routed compute for 3D diffusion (deep only near
the pancreas). Structure-adaptive sparse 3D diffusion already reports up to
10x training acceleration (https://pith.science/paper/2604.17773), and
LAW & ORDER learns where to spend compute (https://arxiv.org/abs/2603.04795).

## 13. Execution preparation after731dcad (Codex)

See NEXT_STEPS_PREFLIGHT.md for the actionable checklist and remaining gates.
No new GPU allocation or feature extraction has run. Isolated CPU ROI tool
and three tests added; total12local tests pass. Existing cached120/165 labels
hash-checked and audited with fixed tumor5mm/edge2mm exclusions:16138voxels
(30.165mL) and25526voxels(60.859mL) remain. Both suitable for an initial
engineering feature-availability check, not proven healthy tissue.

New direction needs these safeguards: MSD has no duct label, so its proposed
duct-diameter step is unavailable; geometry from a fixed mask is unchanged by
definition. Duct dilation is not inherently invisible. Three diagnostic
tumor-positive scans cannot measure prediagnostic discrimination orAUC loss.
Generic radiomics changes do not replicateREDMOD. The duct paper's cohort is
available on request, not verified public download; no author contacted.
Mayo announcement inspected, linkedGut full paper returned403 here.

Read-only dependency check:SimpleITK/nibabel present, PyRadiomics absent.
No shared-runtime install. Next is choosing/pinning an isolated validated
feature implementation/settings, identity/phantom tests, then paired descriptive
features. Freeze image grids, sourceROIs and binning; report bias separately.
Do not treat negative tiny pilots as proof of preserved cancer signal.

CPU launcher initially fed/dev/null instead of its here-document and did no
work; corrected and verified both output records plus explicit completion.
Artifacts remain remote/local; publication restrictions unchanged. No change
to PanTS evaluation. New GPU insertion/detector work requires calibrated cap.

## 14. User-approved parallel control round

User subsequently approved a NEW2charged-GPU-hour total round and requested
efficient parallel submissions. Three preflighted controls queued:3290857
matched insertion pair,3290863real-image split4,3290864fixed-seed sampled
posterior. Exact combined hard-limit charge.66667hours, no retries. First
interactive probe RUNNING at last check, regular probes pendingPriority.
No completion/quality result yet. CONTROL_ROUND_20261001.md records bounds,
scientific limits and actual scheduler restriction(one interactive submit/user).
PANORAMA detector/radiomics still NOT launch-ready; no blind larger campaign.

## 15. Completed controls, efficiency audit and Claude's next review

All3short jobs now COMPLETED0:0 with actual output artifacts verified.
Actual new-round spending.143333physical/.212222chargedGPU-hours; NEW2hour
cap remainder1.787778. CONTROL_ROUND_20261001.md now contains complete results,
CPU Gaussian reference and prioritized next actions; running snapshots above
are historical,not current status.

Measured matched insertion response retains43.5645%raw/37.0755%ring-corrected
of nominal-20HU signal. Single8mm rasterized sphere,111voxels,cached tumor-
positive host;NOT truePDAC,detector recall or matched-filterSNR. Real-image
split4 versus cachedsplits1result had0HUdifferences at reported precision,
but memory peak unchanged and timing is not controlled enough for a speedup
claim. Single posterior sample differs from mean by21.0671HUwhole-imageMAE,
21.1752HUtumorMAE;not comparison to originalCTor evidence of clinical harm.

Post-hoc CPU Gaussian insertion references at.5/1/2mmsigma give corrected
retention.90154/.79077/.44995. VAE.37075is lower than these references,
but controls are not noise-matched or preregistered and cannot establish a
VAE-specific medical failure. Ordinary smoothing remains a plausible account.

Verified efficient decisions:reuse cached data/dependencies,CPU preflight,
regular-partition billing for2jobs,parallel short allocations,no retry,
metrics-only output,unchanged fullvolumeFP32scientific control. Audit also
found redundant syntheticROI/modifiedCT preparation in non-insertion modes
and a metrics-only retention tradeoff. Fix CPU waste in next runner version;
do not spendGPUtime repeating completed diagnostics to optimize seconds.

Next:validated CPU paired texture extraction on existing saved reconstructions;
PANORAMA identity/exclusions/data access audit;only then a calibrated real-
lesion detector test or one new prespecified insertion host. Not a40caseAUC
experiment within an unmeasured cost. AskClaude to challenge controls,endpoint
definitions,rasterization/location effects and what truly requires additional
GPU inference. No blanket claim that the stack is maximally optimized.

## 16. Idea selection first, resource-minimal screening

User reiterated the objective: find the strongest research idea using the least
resources, not maximize experiments or spend the remaining authorized budget.
The remaining 1.787778 charged GPU-hours is a ceiling, not a spending target.
No new GPU job was submitted during this follow-up.

Next screen is CPU-only, reusing the existing saved posterior-mean volumes.
Time-box isolated extractor setup to 30 minutes of active engineering; if it
does not pass identity/phantom checks, record the blocker and change route rather
than improvise an unvalidated feature implementation. Read-only inspection found
Python 3.12.9 / NumPy 2.5.3 on the aarch64 host. Official PyPI metadata for pinned
PyRadiomics 3.0.1 has only Python 3.5-3.7 x86 wheels and a source archive: no
matching prebuilt wheel. No dependency was installed and no base environment
was changed. This is a setup risk, not proof that compilation cannot work.
Source: https://pypi.org/pypi/pyradiomics/3.0.1/json

Decision ladder for Claude's review:

1. Measure a small, fixed set of paired features on already-saved cases, with
   identical source masks/grids/binning and separate HU-bias summaries. This is
   descriptive feasibility, not cancer detection, AUC or REDMOD replication.
2. Ask whether an observation is more informative than generic smoothing or
   intensity bias. If all observations are readily explained by those controls,
   deprioritize this pilot as a paper lead; do NOT conclude clinical safety.
3. In parallel, resolve an independent detector's actual checkpoint folds,
   patient exclusions and per-case data access using metadata first. A detector
   that saw the cases is not a cheap decisive test just because it is available.
4. Before another GPU allocation, specify the competing explanations, endpoint,
   controls, charged-hour hard cap and what result would change our next action.
   Prefer one bounded discriminating test over more parameter sweeps or broad
   downloads. No fresh training until evidence supports a specific mechanism.

The current signal-retention result motivates a question; it does not yet make
this a top-conference idea. Novelty, a meaningful downstream consequence and
controls that rule out simpler explanations remain separate requirements.

## 17. Claude review round 3 (October 1, 2026)

Research only. No GPU job, no environment change, no image upload, no PanTS
change. Network reads were public metadata, fold JSONs, and the pickled
headers of the ten PANORAMA checkpoints, streamed by HTTP range request and
parsed with a restricted unpickler (no class import or code execution; tensor
storage never loaded; no weights saved). Labels as in Section 12.

### 17.1 PANORAMA checkpoint identity: resolved (Section 13 step 4 / 15 step 2)

- [measured] Each of the 10 released checkpoints stores `init_args.fold`
  equal to its directory: detection `fold_0..4/checkpoint_best_panorama.pth`
  and pancreas `fold_0..4/checkpoint_final.pth`. All store
  `dataset_json.numTraining = 2238`, the size of the fold-JSON union.
- [measured] Selected detection epochs: 950, 950, 750, 950, 950 (fold 0-4).
  Pancreas stage: epoch 1000 (final, no selection). This confirms
  per-fold checkpoint selection exposure for detection; fold 2 picked 750.
- [measured] The released fold JSON is **exactly** nnU-Net v2's default
  split: KFold(5, shuffle=True, random_state=12345) over the sorted 2,238
  study IDs reproduces all five validation sets (448/448/448/447/447, 100%
  match; numpy reimplementation of the sklearn procedure). This is strong
  evidence the checkpoints were trained on that split, and it explains why
  the split is by study, not patient: the default ignores patient grouping.
  The 11-patient exclusion in 12.3 therefore remains necessary.
- Residual: init_args proves the fold index, not the split file used at
  training time; the exact default-split reproduction makes a mismatch
  unlikely but not impossible.
- Method is reusable and cheap (~a few MB of range reads per checkpoint);
  script kept locally (scratchpad `ckpt_meta4.py`), not committed.

### 17.2 Feature extractor blocker (Section 16): a pure-Python option exists

- [measured] PyRadiomics latest release is 3.1.0 (May 2023), wheels only for
  cp37-cp39 x86_64/win/mac; nothing for cp312 or aarch64.
  https://pypi.org/pypi/pyradiomics/json
- [measured] MIRP 2.7.0 (Aug 2026) ships a `py3-none-any` wheel, requires
  Python >= 3.11, and states IBSI compliance for image processing, features
  and filters. No compilation, so it fits the aarch64 / Python 3.12.9 host in
  an isolated target. https://pypi.org/pypi/mirp/json
- Recommendation: use MIRP in an isolated venv/target (not the frozen shared
  runtime), pin the version, run its IBSI phantom checks first, fix bin width
  in HU and the same source ROI for all conditions. This replaces the
  time-boxed PyRadiomics build attempt.

### 17.3 The decisive cheap question: smoothing or a learned prior?

Current evidence cannot separate "the VAE is just a blur" from "the VAE's
learned prior removes faint structure it treats as noise". The single insert
(37% retention vs 45% for a 2 mm Gaussian) cannot, because any blur can be
made stronger. The discriminating property is **linearity**:

- A linear shift-invariant blur G gives the same fractional retention at
  every contrast and both signs: G(a*s) = a*G(s).
- A nonlinear learned decoder can retain a faint signal less than a strong one
  (contrast-dependent) and treat dark and bright inserts differently.

CT physics precedent: iterative reconstruction's nonlinear regularization makes
spatial resolution depend on contrast and noise, measured as a contrast-
dependent task transfer function (TTF). The same measurement transfers
directly to a generative VAE.
- Yu et al., Med Phys 2015, contrast- and noise-dependent resolution of IR:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4401802/
- Duke (Samei group) task-based TTF across reconstruction algorithms:
  https://fds.duke.edu/db/pratt/BME/faculty/samei/publications/269194

**Proposed contrast-linearity test** [hypothesis; cost is an estimate]:
- Same host (case 120, cached baseline reused), same prespecified location as
  job 3290857, same 8 mm partial-volume insert, contrasts -10, -40, -80 HU
  and +20 HU (the -20 HU point already exists). Four new whole-volume passes
  in one job. One extra pass at -20 HU shifted by 2 voxels (half a 4x latent
  cell) checks location/grid-phase sensitivity.
- [estimate] Measured regular-partition passes were ~105 s forward within
  126-142 s allocations (jobs 3290863/64). Five passes in one job ~ 10-12 min
  allocation, about 0.2 charged GPU-h on regular partition (x2 if
  interactive). Calibrate on the first pass; well inside the remaining
  1.787778-hour ceiling, but still needs explicit approval.
- Endpoint: ring-corrected fractional retention per contrast, plus the
  half-cell shift. Run-to-run determinism reference: the split-4 control
  showed 0 HU difference from the cached baseline, so FP32 repeat noise is
  expected to be negligible; verify on one repeated pass if cheap.
- **GO (learned-prior suppression, worth pursuing):** retention at -10 HU is
  less than 0.7x retention at -80 HU, or +20 vs -20 HU retention differ by
  more than 0.15, beyond the half-cell shift's spread. Prespecify both.
- **NO-GO (generic smoothing):** retention flat across contrasts and sign
  within the shift spread. Then the VAE behaves like a linear blur for this
  construction; the paper lead weakens to "known VAE blur", and the effort
  should move to the noise/depth or detector questions.
- Limits: one host, one location, synthetic sphere, not PDAC; the result
  explains mechanism, not clinical harm. Do not sweep contrasts further
  until this outcome is known.

Insertion realism improvements, cheap and CPU-only: build the insert by
supersampling a sphere and averaging into voxels (partial-volume weights)
instead of binary rasterization; the current 111-voxel mask is a coarse
8 mm sphere on 4 mm slices. Image-domain insertion is accepted practice in CT
low-contrast detectability work, though it ignores scanner blur on the insert.
https://pmc.ncbi.nlm.nih.gov/articles/PMC7338819/

### 17.4 CPU texture step: add the noise power spectrum

For the paired CPU texture step on saved 005/120/165 reconstructions, add the
3D noise power spectrum (NPS) in the fixed parenchymal ROIs (30.2 mL and
60.9 mL already verified for 120/165), original vs reconstruction. NPS is the
standard CT physics description of texture and directly shows which spatial
frequencies the VAE removes. It also gives the defined, auditable rule for a
noise-power-matched Gaussian control that Section 13/NEXT_STEPS asks for
(match integrated or band-limited NPS on separate calibration regions).
Caveat: ROI NPS on patient anatomy mixes anatomy and noise; report it as a
descriptive texture spectrum, not a pure noise measurement.

### 17.5 Novelty update for the sub-visual direction (12.9)

- [measured, literature] Radiomic stability under deep-learning CT
  reconstruction and denoising is well studied (e.g. DLIR vs ASIR-V feature
  reproducibility; generative denoisers improving radiomic concordance), and a
  Cycle-GAN low-dose study reported that classifiers on radiomics from
  GAN-processed images performed worse.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12618174/ ,
  https://arxiv.org/abs/2109.07787 ,
  https://www.nature.com/articles/s41598-023-36712-1
- So "generative processing changes radiomic features" is NOT novel. The
  remaining, narrower question: do latent foundation-model autoencoders used
  for 3D CT synthesis keep the pancreatic early-detection signal, measured
  with a contrast-dependent transfer function and an independent detector.
- [measured, press] BMJ describes REDMOD as detecting "very early, normally
  invisible tissue changes" with automated whole-pancreas segmentation and
  radiomics (test AUC 0.82, sensitivity 73.0%, specificity 81.1%). Full Gut
  paper not retrieved; the "sub-visual" wording is the authors'/publisher's,
  not independently verified (consistent with Codex's caution in Section 13).
  https://bmjgroup.com/ai-model-detects-very-early-normally-invisible-tissue-changes-of-pancreatic-cancer/
- Not yet checked: whether the same group's earlier prediagnostic radiomics
  papers publish a reproducible feature list that could replace a generic set.

### 17.6 Recommended order (resource-minimal)

1. CPU: MIRP isolated setup + IBSI phantom check; paired first-order,
   texture and NPS on saved 005/120/165 reconstructions (no GPU).
2. CPU: PANORAMA per-case download feasibility (checkpoint identity is done).
3. GPU, one small approved job: the 17.3 contrast-linearity test. This is the
   single result most likely to change the next decision.
4. Only if 17.3 is GO: one real-lesion detector calibration case (Tier B),
   with the per-fold checkpoints and 11-patient exclusion.

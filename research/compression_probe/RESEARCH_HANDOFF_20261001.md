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

## 18. CPU texture screen completed; generic texture story demoted

See TEXTURE_SCREEN_20261001.md and reproducible texture_screen.py. Isolated
Python3.12 install failed cleanly; Python3.11 module/source build succeeded
without training-environment changes. Geometry/identity/analytical/shift checks
passed, then all3existing reconstruction pairs completed under a180s CPU bound.
No new GPU job/charge. Fixed sourceROIs,25HU bins,native grid,8features;
no images or reader keys published. This is not REDMOD or early-cancerAUC.

Parenchymal mean shifts005/120/165:-39.90/-8.96/-27.20HU. GLCMcontrast changes
-31.5/-24.4/-26.0%;1mm Gaussian references change it more:-58.7/-42.4/-65.2%.
Entropy and other selected textures likewise change less underVAE than chosen
blur controls. Secondary mean-bias removal leaves similar texture changes.
Case005variance increases underVAE, unlike blur: no full Gaussian-equivalence
claim. Controls are exploratory/notnoise-matched; this cannot establish safety.

Decision:deprioritize generictexture-drift as a standalonepaperlead. Do not
expand features/cohort just to accumulate results. Remaining potentially useful
question is task-specificsignalpreservation. Claude's section17 checkpoint
identity audit arrived while this screen was running: fold indices are now
reported verified by Claude, not independently reproduced here. Patient/data
access and selection-exposure safeguards still matter. No detector download/
inference ornewtraining. RemainingNEWcap1.787778chargedh.

### 18.1 Integration of Claude's proposed next test

The contrast-linearity test is more discriminating than more generic texture
features, but it must not be called proof of a learned prior: encoder/decoder
nonlinearity, normalization, clipping, aliasing and host interaction can all
produce amplitude or sign dependence. It tests whether this local response is
consistent with a fixed linear operator. A flat response likewise does not
prove global linearity or clinical safety.

Keep the binary111voxel insert for the contrast sweep if reusing the existing
-20HU result. Supersampling changes the signal and would require a new matched
baseline point; do not compare it directly with the old binary point. Spatial
phase-shift controls must pass the same ROI/clearance and clipping checks.
One shifted point does not bound phase sensitivity for every amplitude or
location. Proposed0.7/0.15 cutoffs are prioritization heuristics, not calibrated
statistical thresholds. A same-allocation reference pass should check cached
baseline consistency, not infer determinism from a different job.

Do not make synthetic GO a mandatory gate for a realistic lesion test; a
synthetic NO-GO can deprioritize this mechanism without excluding PDAC harm.
Next prepare a bounded multi-contrast runner and CPU contracts/cost estimate,
then use the existing authorized total screening ceiling only if that plan
meets provenance/runtime checks. No extra GPU job submitted in this update.

## 19. Bounded contrast probe preflighted and submitted

User said continue; existing2chargedhour screening authorization covers this
single probe. See CONTRAST_PROBE_20261001.md and reproducible runner/tests.
All3CPU unit tests and realcase/modelCPUpreflight passed. Original111voxel
insert and allcontrasts/shift preserve ROI/no-clipping contracts. Sevenpasses:
baseline,-20repeat,-10,-40,-80,+20,shifted-20. FP32/fullvolume/mean unchanged.
Cachedbaseline max0.001HU gate and first-pass runtime projection stop early
ifparity/timing fails. No automatic retry, no images saved/published.

Heldsubmission3291031 inspected:regularghx4,oneGPU,18min,billing1000,requeue0.
Released onlyafter frozen script/resources/sourcehash checks. LaststatePENDING;
noresultyet. Maxadditional0.30chargedh;newroundactualunchanged0.212222,
actualplusreservedceiling0.512222of2. PanTS evaluation untouched.
Nonlinearity is not unique evidence of a learned prior;flat response is not
clinical safety. Report signed retentions and phase sensitivity, notjustGO.

## 20. October2 live status and storage-guard correction

Bridges2connection reopened byuser;read-onlycheck47320183:PENDING,Priority,
runtime0. Scheduler currentlyestimatesOct4 18:00EDT=15:00Pacific,NOTguaranteed.
All4finaltrainingcheckpoints present. Testtable stillincomplete;queued frozen
evaluation untouched. This is not anewtraining delay orneed to retrain.

Contrast3291031FAILED1:0 after36sec, beforeinference:projectroot freebytes0.
Correct guard preventedoutput loss;manual fix movessmallsource/logs/metrics
to home /u/asanjeev/compression_probe_smalloutputs, weights/dataread-only.
1MiBwrite/fsyncprobe passed,4unit tests andrealcaseCPUpreflightpassed.
Read-onlyhomequotaqueryunsupported(non-Lustre),so rawfree notquota proof.
No data deletion/baseenv changes. New3291159held/resources/frozen script
verified thenreleased:PENDINGlastcheck,18min/0.30chargedhmax,noretry.
Newroundactualestimate0.222222h;actualplusreservation0.522222<=2.

## 21. October2 additional novelty check, no extra allocation

See NOVELTY_DECISION_20261002.md: primary-source review found close MICCAI2026
medical-latent probing and Foundation-VAE-on-MSD work. Genericfeaturedrift,
latentlearnability, VAEsegmentationcomparison andautoencoderscreening are
not defensible standalone novelty claims. Candidate narrows to task-specific
signal-preservation envelope, NOTanestablishednewpaper orfullTTF measurement.
Frozenjudgefailure differsfromirrecoverable informationloss;original-trained
and reconstruction-trained decisionprocedures answer differentquestions.

Existing3291159baseline gate passedmax/mean0HU;repeated-20HU matchesearlier
raw.4356453896/corrected.3707545519. Otherpasses/completion stillpendingat
thissnapshot. Noadditionaljob/training/weights/image download. LatestPANORAMA
recordlink returned429;stopped,percaseaccess unresolved. RequestClaude preserve
its cheaprestrictedmetadataauditscript, don'tdownloadallweights toduplicateit.

## 22. October2 completed contrast curve and selective-access feasibility

3291159 COMPLETED0:0,815s allocation,sevenpasses/completionverified. Corrected
retention -10/-20/-40/-80/+20HU: .313566/.370755/.517142/.720277/.235942.
Shifted-20HU .374746,only+.003991 versusunshifted. Baseline0HUdiff/repeat
parity passed. See CONTRAST_PROBE_20261001.md forraw/corrected values/caveats.
Amplitude heuristic crossed;sign-difference heuristicnotcrossed. One-host
nonlinearresponse NOTlearned-prior proof orrealPDAC endpoint. Noimages saved.
Newroundcharge-equivalentestimate .448611of2 includingfailedattempt,notposted
debit. No furtherGPUjob;prior .522222figurewas reservation ceiling,notspent.

CPU-only ZIPdirectoryprobe on publicZenodo10998332v1 batch1:48,238,512,666byte
archive,554members,49,683bytes/5requests. Tests3pass locally/remotely;final
oversize-byte-budgettightening locallyretested. See SELECTIVE_DATA_ACCESS_20261002.md.
NoindividualCT extractiontested. Publiclabelsrepo hasclinical_information.xlsx.
Nextcheapgate:clinical/manualmask/fold/versionjoin andsingle-member transfer
integrity beforeboundedreal-lesionjudge. No workbookanalysis/checkpointdownload
yet;no largearchive justifiedbydirectorysuccess. Preserveoriginaljudgemisses;
frozenjudge failure doesnotproveirrecoverableinformationloss. PanTSevaluntouched.

## 23. CPU-only public metadata and individual CT access verified

See SELECTIVE_DATA_ACCESS_20261002.md forpinnedSHAs/details. Independentjoin
reproduces380manualPDAC studies,74/81/84/58/83byfold afterMSD/NIH and11
crossfoldpatient exclusions. Batch1contains81(17/16/15/14/19). Two-stagefold
membership matches despite differentJSONlistorder;twofold-contracttestspass.

Fetchedonly21,471,554byteCT100226_00001,smallestpathology-confirmedtransfer
candidate,NOTscientificcohortselection. ZIPCRC/fullhash/pinnedmanualmask,
512x512x58geometry,mmunits/finite/nonemptytumorverified;completiontrue.
Maskinitiallyfailedbinary-onlycheck;actual documented0-6 conventionverified
thenexistingCTCRC/pinnedmaskrecheckedwithoutrepeatCTdownload. ZenodoETagabsent;
noimmutableETagclaim. Bundledruntime lackednibabel,usedexistingisolatedimaging
runtime,noinstall/trainingenvchange. Fivebounded-rangeunit tests nowpass.

Noimages/workbook/readerkeypublished;noGPUjob,PanTSeval47320183untouched.
Chargeestimateunchanged .448611of2. Next:judgepreprocessing/singlefoldboth
stages andprespecifiedpairedprotocol beforeanynewGPUcalibration. Access
resolvedforonecase,notunbiasedcohortorindependentcheckpointselectionexposure.

## 24. Review brief for Claude: how close is this to a strong idea?

**Status: a promising measurement lead, not a validated method or S-tier idea.**
We have moved from speculation to one controlled observation and feasible
real-data access. We have not demonstrated real-lesion detection damage,
identified a unique mechanism, implemented a repair or established novelty.

### Evidence to critique, not embellish

- Seven-pass MAISI mean/FP32 probe completed. One synthetic 8 mm insert in
  one MSD host: ring-corrected retention is 31.4% at -10 HU and 72.0% at
  -80 HU. Repeated -20 HU matches; its one shifted-position control changes
  retention by only 0.399 percentage points. See CONTRAST_PROBE_20261001.md.
- This rejects a fixed linear response for this construction. It does not
  uniquely establish a learned-prior explanation: normalization, clipping,
  encoder/decoder nonlinearities, sampling and context remain alternatives.
- Generic texture drift was deprioritized, not because it proves safety, but
  because it was less discriminating and overlaps prior work. Novelty review
  already found close medical-latent probing and foundation-VAE comparisons.
- CPU metadata audit independently reproduced 380 manually masked PDAC
  candidates after source/patient exclusions; 81 are in batch 1. Individual
  range retrieval, ZIP CRC, pinned mask and geometry were verified on one
  21.5 MB CT. This transport case was chosen by file size, NOT clinical cohort
  criteria. Do not generalize from it or silently use it as a selected cohort.
- Screening-round spending estimate: 0.448611 charged-equivalent GPU-hours
  out of the authorized 2-hour total. Posted accounting debit not verified.
  No further GPU jobs submitted. Protected PanTS evaluation unchanged.

### Cheapest decisive next questions

1. **Judge feasibility before inference:** inspect actual preprocessing,
   intensity conventions, weights/runtime, single-fold execution for BOTH
   pancreas and detector stages, probability outputs and thresholds. Preserve
   full original context under the judge's released protocol. Calibrate cost
   only after code/data provenance and output contracts pass. No default
   five-fold ensemble on cases seen by four folds during training.
2. **Prespecify a tiny paired feasibility pilot:** patient-disjoint eligible
   cases and negatives, original/preprocessing-control/reconstruction/blur
   arms, original misses retained. Define any blur matching on separate
   calibration data; an arbitrarily strong Gaussian is not a fair comparator.
   Do not choose cases, thresholds or lesion sizes after seeing outcomes.
   A tiny screen can prioritize or kill a direction, not establish clinical
   equivalence or provide a powered safety conclusion.
3. **If frozen-detector losses appear:** separate downstream domain shift
   from irrecoverable task information. Do not turn an original-trained
   detector's failures into a claim that no observer can recover the signal.
   A real effect beyond controls is a reason to investigate, not sufficient
   novelty or a guarantee of a top-conference paper.
4. **Method/PI fit before more training:** a useful direction would pair a
   specific failure with a convincing explanation and an efficient remedy.
   Explain how that remedy serves the PI's UNet++/diffusion/faster-training
   objective. A standalone VAE audit is a possible tangent, not automatically
   a contribution to that objective. Do not propose expensive fine-tuning
   merely because a synthetic response curve looks interesting.

### Specific input requested from Claude

- Attack the strongest claim above. Which alternative explanation is cheapest
  to distinguish, and what exact result would change our next action?
- Check primary-source novelty of the narrow measurement-plus-remedy direction,
  especially medical VAE lesion preservation and MAISI-v2 region-aware losses.
  State exact overlap; do not certify novelty from a title-level search.
- Propose ONE highest-information next experiment, including a pessimistic
  charged-hour estimate, confounds, paired outcomes and explicit stop criteria.
  Stay within the remaining authorization; no job submissions for this review.
- Should we pursue this direction or return to adaptive-depth diffusion?
  Compare scientific payoff, PI alignment and decisive-test cost, not excitement.
- Preserve the restricted checkpoint-header audit script in the repo for review;
  fold identity is reported verified by Claude, not independently reproduced
  here from checkpoint headers. Metadata joins have been reproduced here.

### Honest assessment of Claude's contributions so far

Helpful: it proposed the contrast-dose test, identified the patient-level
leakage risk, documented fold/checkpoint selection exposure and provided the
cheap checkpoint-header audit. Its reported cohort/fold counts have now been
independently reproduced. These are concrete scientific/safety contributions.

Not all suggestions should be adopted unchanged: contrast nonlinearity is not
unique proof of a learned prior; synthetic GO must not block real-lesion tests;
the 0.7/0.15 cutoffs are heuristic rather than validated statistical thresholds;
supersampling changes the stimulus and cannot be compared directly to the old
binary point; anatomy spectra are not automatically noise-power spectra.
Checkpoint-header evidence should be preserved as runnable code, not only prose.

Use Claude as a critical second reviewer, not a vote that establishes truth.
Resolve disagreements with primary sources, runnable contracts or measured
controls. Prefer one discriminating experiment over another descriptive sweep.

## 25. Claude review of Section 24 (October 2, 2026)

Research only: no job, no environment change, no image upload, PanTS 2x2
evaluation untouched. Labels: [measured] = from committed results or a primary
source read in this review; [hypothesis]; [estimate].

### 25.1 Attacking the strongest claim

Claim: the seven-pass curve "rejects a fixed linear response" and is a lead.

- The rejection itself holds: corrected retention rises from 0.314 (-10 HU)
  to 0.720 (-80 HU), and the half-cell-ish shift moved -20 HU by only 0.004.
- But amplitude-dependent shrinkage is the generic signature of **any
  denoiser** that separates signal from noise by amplitude (wavelet shrinkage,
  total variation, non-local means, learned denoisers). CT physics already
  documents the clinical version: deep-learning and iterative reconstruction
  lower noise yet lose small low-contrast lesions. [measured, literature]
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12417284/ (DLIR, small liver
  lesions <= 5 mm), https://pmc.ncbi.nlm.nih.gov/articles/PMC4401802/ (IR
  resolution depends on contrast and noise).
- Of Section 24's listed alternatives, two are already largely excluded:
  clipping (CPU preflight passed for every amplitude) and intensity
  normalization (an affine map cannot create amplitude dependence without
  clipping). "Encoder/decoder nonlinearity" is the mechanism, not an
  alternative. The alternative that matters, and is cheapest to test, is
  **generic denoising**: would any denoiser with the same noise reduction show
  the same curve?
- The sign asymmetry is the most VAE-specific-looking feature: +20 HU kept
  0.236 vs -20 HU 0.371 (difference 0.135). A classical denoiser acting around
  a local mean treats small bright and dark perturbations nearly alike, so an
  asymmetry it does not reproduce would point to a learned, data-dependent
  prior. [hypothesis]
- Clinical relevance [measured, literature]: isoattenuating PDAC differs from
  parenchyma by about 10-15 HU or less and is reported in 5-45% of cases;
  small tumours are more often isoattenuating. The probe's worst retention
  (31% at -10 HU) is in exactly that range.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3473757
  One host, one synthetic sphere: relevance, not evidence of PDAC harm.

### 25.2 Novelty against primary sources (exact overlap)

- **Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation**
  (Chen, Ding, Gu, Liu, Bian, Yuille, Zhou, Fu; arXiv 2605.30893, 2026).
  [measured] Evaluates 7 video VAEs plus MedVAE and MAISI on MSD Task06/07,
  LiTS, KiTS19, CT-RATE, ReXGroundingCT with PSNR/SSIM/MSE and nnU-Net
  Dice/NSD trained on reconstructions. States "the reconstruction gap is
  dominated by CT noise, not structural distortion". No lesion size or
  contrast stratification, no controlled insertions, no remedy; limitations
  mention over-smoothed small findings and subtle textures (for conditional
  generation). https://arxiv.org/html/2605.30893v1
  Overlap: dataset, MAISI, downstream segmentation. Gap: whether the "noise"
  being removed includes faint lesion signal. The contrast curve tests that
  paper's central interpretive claim directly. Prof. Zhou is a coauthor, so
  this is PI-relevant; frame it as an extension of that result, not a rebuttal.
- **MAISI-v2** (arXiv 2508.05772). [measured] The region-specific contrastive
  loss trains the **ControlNet**; the VAE is reused frozen "without
  fine-tuning". Reports +6.4% pancreatic-tumour Dice in augmentation; no size
  analysis. https://arxiv.org/html/2508.05772v2
  Implication [hypothesis]: if faint-signal loss happens in the frozen VAE,
  a ControlNet-side loss cannot restore it at decode. Complementary, not
  overlapping.
- **The Learnability Gap in Medical Latent Diffusion** (MICCAI 2026).
  [measured] Studies classifiers on latents (MedVAE, SD, Flux; not MAISI);
  reviewers flagged limited novelty vs prior latent-misalignment work.
  Different question (latent learnability, not reconstruction signal).
  https://papers.miccai.org/miccai-2026/1053-Paper3049.html
- Net: "denoising erases faint lesions" is known for CT reconstruction.
  What is not found: a quantified faint-signal envelope for the frozen VAEs
  used in 3D CT synthesis, shown to be beyond matched generic denoising and
  linked to lesion evidence. Novelty therefore depends on 25.3 coming out GO.

### 25.3 ONE next experiment: matched classical-denoiser control (CPU, 0 GPU-h)

Question: is the VAE's curve explained by generic denoising at the same noise
reduction?

- Inputs: cached case-120 original, its saved MAISI baseline reconstruction,
  and the same reproducible 111-voxel insert at the same location and the
  same five contrasts plus the +2-voxel shift (all already CPU-validated).
- Two prespecified denoiser families, pinned implementations and defaults:
  (a) 3D total variation (Chambolle; implementable in NumPy), (b) 3D non-local
  means or, if unavailable in the isolated imaging runtime, a 3D bilateral
  filter. No installs into shared runtimes.
- Matching rule, fixed before any insert is processed and using the baseline
  only: choose each family's single strength parameter by bisection so that
  the standard deviation of the high-pass residual (image minus 2 mm Gaussian)
  in the verified parenchymal calibration ROI (case 120, 30.2 mL, minus the
  insert, its ring and a 5 mm margin) equals that of the VAE baseline
  reconstruction. Report NPS-band agreement descriptively; do not re-match on it.
- Then apply each matched denoiser to host and host+insert and compute raw and
  ring-corrected retention with the same metric code as job 3291159. Local
  crops with a halo larger than the filter support are exact for these local
  filters and allowed here (they are not the VAE path).
- [estimate] Cost: 0 GPU-hours; minutes to about one CPU-hour, plus about a
  day of engineering. Fits Section 16's resource-minimal rule.

Stop criteria (prespecified; heuristics, not calibrated statistics):
- **STOP the VAE-specific lead** if, for at least one family, the absolute
  retention difference to the VAE is <= 0.05 at -10, -20 and -40 HU AND the
  +20/-20 asymmetry differs by <= 0.05. Record "consistent with generic
  denoising" and return to adaptive-depth (25.4).
- **GO** to the single-case PANORAMA paired judge calibration if the VAE's
  retention at -10 and -20 HU is lower than BOTH matched families by >= 0.10,
  or its sign asymmetry exceeds both by >= 0.10.
- Otherwise: inconclusive. Do not tune parameters or add families after seeing
  outcomes; decide on the real-lesion judge test on its own merits.

Confounds to report, not hide: choice of matching metric; one host with a real
tumour; binary insert; one location; classical filters' secondary parameters;
ring correction assumes the ring is a fair local background.

Why this one: it is free, reuses only already-validated assets, and is the
only cheap result that would change the next action either way (generic ->
stop and pivot; VAE-specific -> spend on the real-lesion judge).

### 25.4 Pursue this or return to adaptive-depth diffusion?

| | Faint-signal VAE lead | Adaptive-depth diffusion |
|---|---|---|
| Scientific payoff | High only if VAE-specific (25.3 GO); otherwise "known denoising" | Premise published (ASE, multi-decoder, TDC); realistic ~1.2-1.5x [estimate] |
| PI alignment | Tests the PI-coauthored Foundation-VAE claim; not the literal "faster training" ask | Literal PI ask (UNet++ backbone, faster training) |
| Next decisive test cost | 0 GPU-h (25.3) | Timing probe < 1 GPU-h, then tens-hundreds GPU-h of training pilots [estimate]; no diffusion code yet |

Bridge that makes the lead serve the PI objective [hypothesis]: if frozen
latent VAEs erase faint lesions, faithful lesion synthesis needs pixel space
near the lesion; 3D pixel-space diffusion is expensive; an efficient
nested/depth-adaptive UNet++ for pixel-space lesion-patch diffusion is then a
motivated remedy, reconnecting to UNet++ and faster training with a reason.

Recommendation: run 25.3 first because it is free. GO -> pursue the faint-
signal lead with the pixel-space bridge. STOP -> return to adaptive depth,
starting with the Gate 1 timing probe.

### 25.5 Other insights for the real-lesion judge pilot

- PANORAMA's stage 1 localizes the pancreas at 4.5 x 4.5 x 9 mm and crops a
  fixed-margin ROI. The VAE shifts parenchymal mean HU by -9 to -40 HU
  (Section 18). Report stage-1 pancreas overlap per arm separately; otherwise
  a localization failure masquerades as a detection loss.
- Add one cheap arm: VAE reconstruction with the measured parenchymal mean
  shift added back. It separates global intensity bias from structural loss in
  the detector's response.
- Checkpoint-header audit is now runnable code:
  `audit_checkpoint_headers.py` (range reads, restricted unpickler, nothing
  saved) with offline safety tests `test_checkpoint_headers.py` (2 pass; a
  pickled `os.system` payload is neutralized). Re-run on fold_2 detection and
  fold_4 pancreas reproduces fold 2 / epoch 750 and fold 4 / epoch 1000,
  numTraining 2238.

## 26. Corrected CPU-only denoiser control prepared and launched

User authorized the proposed next step. See DENOISER_CONTROL_20261002.md and
denoiser_control.py/test_denoiser_control.py. Established pinnedTV/NLM instead
of an unvalidated newTV solver; task-local wheels only, no trainingenv change.
HF-texture matching is NOT pure noise matching. Source-mask spatial fit/holdout
exclude inserts/rings; native anisotropic kernels explicitly documented.

Four tests passed (constant,analyticresponse,NLMcrop,globalsign). Boundedtoy
timingsTV.375s/NLM.391s at96x96x32,notfullrunguarantees. Finalrealpreflight
passedinputhashes,geometry,original111voxelinsert;fit6570/holdout6386. Baseline
only strengths,8bisections,2%fit tolerance;unmatchedfamilies stayunavailable.
TV empirical24/40mmcrop and200/400iterationgates beforecurve interpretation;
perpass cropresponse checks andweak/opposite-signTViterationresponse checks.

Localworkstation detachedhidden,oneCPUaffinity,1800swatchdog,noretry. PID11052
atlaunch; frozenrunner acd6617b4b84887d598cee8966371bc510ab1577edb469ff296a4b2991070395.
Metrics-only uniqueoutput. Noheavydenoising onHPC login,nonewGPUallocation.
Statusatlaunch:running,NOTcompleted. Estimate .448611chargedGPUh unchanged.
ProtectedPanTSeval untouched. No auto scientificGO; numerical/matching failure
meansinconclusive,anddoesnotautomaticallyblockreal-lesiontest.

## 27. CPU denoiser comparison completed, with meaningful limitations

Completed98.547s,zeroGPUhours. SeeDENOISER_CONTROL_20261002.md andcommitted
denoiser_results_20261002/ scalarprovenance. FixedTV/NLM fitting errors.0617%/
.2481%;heldout errors8.92%/11.19%,soNLMfails10%holdoutmatchingcriterion.
Do notclaimbothcontrols matched orrelaxthresholdafteroutcomes. TVcropmax
.000175HU/NLM0;allpasscropresponse discrepancies <=1.5e-9.

Correctedretention -10/-20/-40HU: VAE.314/.371/.517; TV.663/.690/.766;
NLM.635/.707/.832. Weakgapnotreproducedbytheseconfigurations,butonehost and
HFtexturematchingnotpure-noisematching. NLM+/−20asymmetry.198 >VAE.135,
soasymmetryaloneNOTVAE-specific. Do notturncontrolfailureintosafetyresult.

TV200/400capcheckcouldearlystopatsametolerance;post-runnumericalcheckheld
strengthfixed,usedeps2e-5/2e-6,max800 onbaseline/-20/-10/+20. Completed48.219s;
largestretentionshift .002566<.01. Primaryresults/strengthsunchanged. Gapremains,
notglobalconvergenceproof. Numericalvalidation code+scalars preserved.

Decision:prioritizeindependentlyjustifiedreal-lesionjudgecalibration after
protocolaudit;NOTcleanboth-matched-familyGO,learned-priorproof orS-tiermethod.
Noextrafilterfamilies/retuning/newGPUjobs. .448611chargedhestimate unchanged;
PanTSevaluntouched. NeedPIalignment beforeanyrepresentation-repairtraining.

## 28. Real-lesion judge source audit completed before spending

Read pinned PANORAMA process/data utilities, requirements, Dockerfile and README.
See PANORAMA_JUDGE_PROTOCOL_20261002.md for concrete execution gates. Important:
crop margins are 100/50/15 mm despite README cm text; stage-2 mask suppresses
PDAC probabilities outside predicted labels1/4/5 plus voxel dilation. Separate
crop exclusion, mask suppression, raw detector and candidate-extraction effects.
Use held-out fold in BOTH stages, not default five-fold ensemble. Fresh per-arm
directories essential: fixed scan names plus --continue_prediction can reuse
earlier predictions. Empty pancreas is pipeline failure, not negative finding.

Runtime is not reproducibly pinned by upstream Dockerfile; external candidate
extractor/defaults and actual model plans remain unverified. No weights/GPU
jobs/trainer-env changes or protected evaluation changes. Screening budget
estimate unchanged. Next CPU geometry and dependency contracts, then priced
single-case calibration only after remaining gates pass.

## 29. CPU judge guards tested; actual wrapper parity still open

Added judge_contracts.py and eight passing offline tests (0.026s). Covers
physical-grid agreement, XYZ crop bounds vs ZYX arrays, exclusive endings,
invalid/nonfinite outputs, exact audited pancreas-mask dilation, scoring-only
crop coverage, and refusal of existing per-arm output without overwriting.
No new packages or GPU spend. These are helper tests, not upstream-wrapper
execution: local environment lacks SimpleITK, actual resample/read/crop parity
still open. Helpers not yet wired into inference; no claim of launch readiness.
Need actual plans and pinned candidate extractor before detector calibration.

## 30. Published crop code and selected extractor executed on CPU

Task-local SimpleITK2.5.3/report-guided-annotation0.3.4/tqdm4.67.1 only; no trainer
environment change. Nine actual publisher-code toy tests plus eight guards pass
(17,0.050s). Pinned source hashes/version contracts in test_judge_upstream_cpu.py.
Actual NIfTI roundtrip, B-spline resample, rotated/anisotropic physical crop,
masking/expansion verified. Still NOT network/preprocessor/HPC execution parity.

Extractor dynamic-fast has scan-global max/2.5 threshold, discards <=10 voxels,
and doesn't enforce iterative mode's five-candidate cap. Toy unchanged faint
component disappears solely after adding remote stronger peak; preserve raw
maps and report postprocessing effects, don't call this CT/VAE damage evidence.
Complete selected wheel source read; it differs from main, historical container
version unknown. Primary thresholds not changed to make outcomes look better.

Pinned repository plans/dataset JSONs now read: stage1 labels0/1, stage2tumor1
andpancreas4/duct5; both CT/SimpleITKIO. Plans mix legacy stage1 and modern stage2
architecture fields. Must test runtime compatibility and compare archived plans,
not assume current nnUNet loads both. More in PANORAMA_JUDGE_PROTOCOL_20261002.md.
No GPU spending/weightsdownload/eval change. Next bounded weight/metadata access
and isolated actual-load checks before any priced tiny real-lesion calibration.

## 31. User-requested pause before checkpoint transport

Authorized normal public downloads/read-only cluster access resumed. DeltaAI
socket works; PyTorch2.10+cu129/nnUNet2.8.1 verified. Both archive JSON pairs
match pinned repo (3567/4124 fetchedbytes). Actual runtime converts legacy
stage1 plans and resolves stage2 plans/classes/imports; heads2/7. No network
instantiation/checkpointload/preprocessing/inference yet.

Prepared fixed-source fold4 transfer helper: bounded streaming, CRC/hash,
no overwrite/pickle execution/retries, watchdog. Three new+five existing range
tests pass. NOT launched: no weights downloaded or new GPU job. User asked to
pause; next restart at transfer review/launch, then actual-load checks.
Protected PanTS evaluation untouched; .448611of2 screening estimate unchanged.

## 32. Transfer resumed, allocation connection unavailable

User resumed/authorized normal public downloads. Frozen local two-fold4 transfer
launched detached, bounded/no retries. PID22644 at launch, source15b5db7f...
Outputswork/judge_weights_fold4_v1/logsjudge_transfer_v1.*. Last check zero-byte
stage1 partial/no completion. Never treat partial as model. Three transfer/six
range tests pass;4MiB blocks reduce requests,300s/archive+660s overall limits.

Both DeltaAI compression and Bridges2 progress SSH sockets now absent. Asked
user reopen DeltaAI. User requests allocation resources: actual load/inference
on scheduled compute, not login; no GPU reserved for waiting on local download.
No GPU job/eval modification/new spending. Restart by inspecting current local
transfer (do not duplicate), then authenticated staging and capped compute check.

## 33. Both selected model files staged; restricted load smoke prepared

Local zero-byte transfer stopped after verifying own exact command. Partial kept.
Reopened DeltaAI master worked; actual homequota~6.18GBof100GB, enough staging.
Frozen helper copied tohome/judge_stage_fold4_v1; detached remote transfer
completed with CRC/SHA, onlyfold4 members:132172006+246418428bytes; total
archive traffic351596245bytes,115requests. Completiontrue,weightsloadedfalse.
Source/noimages only; sharedprojects and protectedeval untouched. ZeroGPUspend.

judge_load_smoke.py prepared against installedAPI: allocatedGPUonly, hashes,
strictweight/head/fold/meta checks, restrictedtorchload, tinyfiniteFP32output,
no unrestricted fallback. Localsyntaxonly, NOTexecuted/submitted. Seeprotocol
for hashes/source paths. Nextreview+cappedcompute smoke, then realpairedinputs.

## 34. One short model-load job queued, no duplicate or retry

Static metadata audit caught NumPy scalar/dtype compatibility BEFORE allocation.
Narrow float32/64+old/newscalar aliases, weights_only=True retained. Toyroundtrip
passed; no unrestrictedfallback. Source d5fc186b... frozen+hashchecked.

Actualjob3294898 submittedheld, oneGPU/twoCPU/8GB/5min/norequeue. Regular
estimate~week vs near-term interactive; moved SAME heldjob tointeractive,
verifiedfields and released. LastlivePENDINGPriority,elapsed0; no loadresult.
Conservativeinteractive2x max.166667chargedh; prior.448611,maxreservedtotal
.615278of2. HeldReqTRESbilling1000 not assumedactualinteractivecharge; check
runningAllocTRES/accounting. TestonlyIDs3294897/3294901 aren't actualjobs.

Remotehomejudge_load_gate_v1 logload_3294898.log/result_3294898.json.
No protectedevalchange; noautoretry. Nextreadresult+charge, thenrealpairprep
ifloadingpasses. Check existingjob before anynewsubmission.


## 35. Allocated real-checkpoint loading PASSED, 14 seconds

Job3294898 COMPLETED0:0,14s,billing2000. .00777778charge-equivalentGPUh;
campaignestimate~.456389of2, posteddebitunverified. Bothactualfold4model files
loadedwithrestrictedloader/strictstate, archive/checkpointmetadataexactmatch,
finiteexpected2/7head syntheticoutput. Peaks127192064/184667648bytes.
ScalarJSONjudge_load_result_3294898.json committed. Synthetictimingnotbenchmark.

No CTinference/recallresult yet, noautoretry/newevalchange. Nextactual
preprocessing and guardedpaired native/reconstruction pipeline, then separately
cappedreal-casecompute. This closes modelcompatibility, not scientificquality.

## 36. Three-arm public-case pilot released, results pending

DeltaAI **3295187** was verified held and released October 2. Last snapshot:
PENDING. One GPU, two CPUs, 32GB, ghx4-interactive, hard 20-minute limit,
Requeue=0. Maximum charge-equivalent cost 0.666667 GPU-hours; campaign prior
estimate 0.456389, worst-case combined 1.123056/2. Posted debit unverified.
Do not submit a duplicate or automatically retry a failure.

Native CT, clipped CT and whole-volume posterior-mean FP32 MAISI reconstruction
use both official fold4 PANORAMA detectors with frozen publisher crop,
postprocessing and TTA. GT is scoring-only. Raw/masked PDAC and GT-matched
candidates, crop inclusion and mask exclusion are kept separate: a patient
maximum elsewhere is not tumor detection. Clipping has its own control arm.

13 CPU geometry/publisher-contract tests pass; source/import guards and real
CT pixel-identical canonical/native restoration pass. No interpolation in
orientation restoration. Private case and maps remain under home
compression_probe_smalloutputs/paired_judge_pilot_v1, never in GitHub.

One transport-selected OOF/validation-selected case is NOT clinical recall,
external test or cohort evidence. A frozen-detector response drop alone cannot
distinguish information loss from domain shift. Next require completion.json
and Slurm accounting before interpreting results. Evaluation47320183 untouched.

## 37. Pilot API bug fixed; one bounded replacement released

3295187 FAILED1:0 after105 allocation seconds, billing2000: charge-equivalent
0.058333 GPUh. No completion.json. First detector prediction failed because
manual_initialization received parameters=None; installed nnU-Net2.8.1 later
iterates list_of_parameters. This is a harness failure, not scientific evidence.

Fix passes [checkpoint['network_weights']] for the single held-out fold.
CPU regression loads both real checkpoints and executes the installed weight
loop with only sliding-window neural compute mocked: list length1, exactly
one call, returned tensor exact, all loaded weights bit-identical. This test
and four geometry tests passed in1.515s. No GPU cost for regression; full CT
inference remains unverified until replacement completion.

Replacement3295395 verified held then released, last PENDING. Separate private
paired_judge_pilot_v2 preserves failed v1 artifacts, reuses unchanged inputs.
One GPU/32GB/20min interactive/no requeue, frozen SHA and bash syntax pass.
Campaign estimate now0.514722; worst-case with replacement1.181389/2 chargedh.
No threshold/model/geometry changes or protected evaluation changes. No retries
automatically. Next read3295395 logs, completion and actual accounting.

## 38. Real three-arm pilot completed; locked two-patient replication queued

3295395 COMPLETED0:0,126s,billing2000, completion.json verified. Cost0.070
charge-equivalentGPUh, campaign cumulative estimate0.584722/2. Scalar JSON
paired_judge_result_3295395.json preserved; private images/maps stay remote.
Native/control mean tumor probability0.1715648, reconstruction0.0236760
(~86.2% lower). GT-overlapping candidate confidence0.821785 vs0.125032.
All arms retain100% of tumor in crop/postprocess mask; crop bounds identical.
Native/control scores identical. This narrows attribution but proves neither
clinical recall loss nor irreversible information loss: frozen-judge domain
shift remains a real alternative. Pancreas Dice is only a coarse localizer
diagnostic, not the quality endpoint. No S-tier/novelty claim yet.

Locked two additional histopathology cases100430_00001/100259_00001 by smallest
stored batch1 fold4 member size under60MB excluding pilot, BEFORE predictions.
Private pinned clinical join verifies three distinct patients. Size-biased
feasibility cohort, not representative. Download CRC/geometry/finite/mask and
native orientation roundtrip pass. Shapes512x512x123/210, tumors6.283/1.943mL.
Two selection tests pass; five actual-weight-loop/geometry tests pass1.297s.

The32M-voxel default guard is explicit56M for this replication INCLUDING padded
volume: no cropping, resampling or changed model operations. Scaling the pilot
24.25GB peak suggests~88GB for larger volume, NOT a proven bound. Runtime
requires at least110GB actual free GPU memory before any larger-volume VAE.
Resource failure is retained, never silently rescued or replaced.

Job3295549 verified held and released, last PENDING:1GPU/32GB/20min interactive,
Requeue=0;480s per case, sequential separate processes. Max0.666667 chargedh,
campaign worst-case1.251389/2; actual posted debit unverified. Frozen code,
selection and batch SHA pass. No automatic retry or case replacement; include
original misses/failures. Protected evaluation47320183 unchanged.

See JUDGE_REPLICATION_PLAN_20261002.md. Next read both completion files and
accounting. If consistent, prioritize posterior/independent mechanism controls
and PI alignment before larger cohorts, new VAE or remedy training. Mixed
results mean report heterogeneity; no posthoc threshold/selection tricks.

## 39. Memory guard stopped replication; smaller locked case queued safely

3295549 FAILED1:0 in38s,billing2000 =>0.021111charge-equivalentGPUh. Both cases
stopped before VAE at110GB-free gate. No scientific output. Runtime exact
free/total was not logged in v1, so do not invent it. Official NCSA architecture
lists96GB GPU/120GB CPU; Slurm GRES misleadingly says nvidia_gh200_120gb and
software documentation inconsistently says120GB/~97GB usable. Earlier HBM
assumption was insufficiently checked. Safety gate prevented large allocation.

Large100259 remains blocked, no replacement. Small100430 has explicit33M
padded limit, >=65GB actual-free guard, estimated~52GB VAE peak from pilot
scaling (not guaranteed). No crop/resize/model/neural operation changed.
Runtime now prints free/total. New private small_v1 preserves prior sources.
3295577 released after held/SHA/syntax checks, last PENDING,1GPU/32GB/10min,
no requeue,540s timeout. Estimate0.605833/2 used; maximum0.939167 including
new allocation. Protected47320183 unchanged. Report resource failure and
selection bias; do not turn a two-patient feasibility screen into a cohort.

https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/architecture.html
https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/software.html

## 40. Small replication hit Slurm OOM; host-RAM-only correction queued

3295577 FAILED1:0 after114s, step OUT_OF_MEMORY0:125, Slurm oom_kill event,
no completion/arm outputs. Cost0.063333charge-equivalentGPUh. Logged CUDA
free101495996416/total102087458816 bytes;65GBGPU guard passed. Host limit32G.
Sampled MaxRSS17328064K is not the instantaneous peak. CPU-offload source
was inspected directly in installedMONAI1.5.1: GroupNorm for maxdimension>=500
uses cloned CPU tensors and repeated concatenation before returning to GPU.
Likely host-memory pressure; no Python failure-stage traceback survived.

3296011 verified held then released PENDING,96G host RAM instead of32G,
same1GPU/2CPU/billing2000/10min/no requeue/540s timeout. Scientific settings
and locked patient unchanged; added CPU RSS/GPU memory stage logging only.
Frozen hashes/bash syntax pass, five CPU contract/geometry tests pass1.408s.
New private small_v2 preserves OOM artifacts. No case substitution or47320183
changes. Cumulative estimate0.669167/2, worst-case1.002500 with new job;
posted debit unverified. Do not automatically retry if it fails again.

## 41. Second patient completed; interpretation narrowed; split gate queued

3296011 COMPLETED0:0,220s =>0.122222charge-equivalentGPUh. Scalar result and
two CPU saved-map audits preserved. Mean tumor PDAC0.030071 native/control to
0.007681 reconstructed (~74.46% lower). Native AND reconstructed GT candidate
overlap0: no newly missed lesion demonstrated. Patientmax0.732969 native is
elsewhere; dynamic cutoff0.293187 exceeds GTmax0.233829. Posthoc explanation,
not a clinical threshold. Reconstruction crop differs by3 starting slices;
100%GT inclusion does not remove context confound. Pilot still has candidate
overlap after reconstruction too. No clinical recall/irreversible-loss claim.

VAE153.53s/GPUpeak50.029GB; producer CPUpeak40314496KiB (~38.45GiB), above
old32G.96G host fix succeeded. Campaignestimate0.791389/2, posteddebitunknown.
See JUDGE_RESULTS_20261002.md for table/caveats. No additional clinical cases
selected; locked large100259 blocked, protected47320183 untouched.

3296141 released PENDING after CPUtopology/hash/held checks: same pilot,
MaisiConvolution split1to4 only, full-volumeFP32/mean/spacing/context retained.
1GPU/96G/5min/no retry,270s timeout; max0.166667 =>campaignmax0.958056/2.
Compare frozen split1 with maxHUdrift<=0.1 engineering tolerance before any
larger-case rollout, measure memory. Not clinical/detector parity. If drift
fails or memory benefit weak, stop this lever; no automatic pipeline changes.

## 42. Real split parity completed; no memory win, stop lever

3296141 COMPLETED0:0,114s,billing2000 =>0.063333charge-equivalentGPUh.
HUoutputglobal/tumor max/mean differences0 against frozen split1. GPUpeak
24250846720bytes identical. Split4 83.21s vs split1 72.89s (~14.2% longer)
in single checks, not rigorous throughput benchmark. Numerical gate passes,
memory benefit gate fails: do not roll out as a memory optimization or claim
it unlocks big100259. ScalarJSON and result report updated.

Campaignestimate0.854722/2, posteddebitunknown. No more GPU submitted here;
47320183 untouched. Both completed patients show confidence loss, not a
conversion from GTcandidate presence to absence. Need independent/task and
posterior controls before claiming clinical harm/irreversible erasure/new
method. Larger patient blocked, no scientific outcome-driven replacement.

## 43. Cheap image-by-window control released

3296871 released after source/hash/resource checks, last PENDING. One GPU,
2CPU/32G/3min/no-requeue,150s process timeout. Max0.1charge-equivalentGPUh;
campaignestimate0.854722, worst-case0.954722/2, posteddebitunknown.
No VAE rerun/training/newpatient. Frozen100430 native/reconstruction each
tested in BOTH saved publisher windows; GT scoring only. Own-window replay
first must pass maxrawprobabilitydrift<=1e-4 before cross-window interpretation.
Two CPU exact-voxel/physical-grid/rejection tests pass; transitive staging
imports fixed before GPU submission. See FIXED_CROP_CONTROL_20261002.md.
Posthoc mechanism diagnostic only, not clinical recall or proof of erasure.
No protected47320183change, no large-case rescue/replacement. Latest prior
commits confirmed on origin/main; push blocker resolved independently.

## 44. Fixed-window control stopped at native replay gate

3296871 FAILED1:0 in39s, cost0.021667chargedh-equivalent. Cumulativeestimate
0.876389/2, posteddebitunknown. Native replay maxrawprobability drift exceeded
locked1e-4; exactdriftnotlogged, no invented number/no threshold relaxation.
Cross-window comparison not reached; no new scientific result/no retry.
CPU native segmentation audit equalphysicalgrid,4/4164942 labels differ;
this does NOT establish probability parity. Helper source matches producer.
Installed predictor sets cudnn.benchmark=True on CUDA, overriding earlier
False; possible algorithm/history confound, not proven rootcause. Futurelocal
code saves failure diagnostics before raising, not yet GPU-validated.
Need runtime/repeatability audit before further inference. See controlprotocol
and fixed_crop_failure_3296871.json. Protected47320183untouched.

## 45. Same-input repeatability diagnostic released

3296939 released after four CPU contract/geometry tests and held/source/resource
checks.1GPU/2CPU/32G/3min/no-requeue,150s process timeout, max0.1chargedh;
campaignworst0.976389/2, currentestimate0.876389, posteddebitunverified.
Same100430 nativeimage/window x two repeats under publisherdefaults and two
under strictdeterministicpolicy. No VAE/training/newpatient/scientificarm.
Record backendflags before/after, global/tumor drift, historicalmap drift,
policy drift. SharedCUBLAS_WORKSPACE_CONFIG and seededprocess mean default
arm not exacthistoricenvironment. Historical1e-4gateunchanged; no retrospective
pass claim. Unsupported deterministic ops preservefailure,no silentfallback.
See JUDGE_REPEATABILITY_20261002.md. Need result/accounting before claims;
oneprocess study doesn't establish crossprocess/device repeatability.
47320183untouched,no automaticretry.

## 46. Repeats stable within process; historical gate still failed

3296939 COMPLETED0:0,46s,0.025556charge-equivalentGPUh. Campaignestimate
0.901944/2,posteddebitunknown. Default repeats identical; deterministicrepeats
identical. Historical maxdrift0.001598/default,0.001856/deterministic, old1e-4
gate stillFAILED. Default/deterministic mean tumor0.030070854/0.030073991,
~3.14e-6 difference; both identicalrepeats ONLY withinoneprocess. Rootcause
historicaldrift NOT proven. Scalarresult/updatedrepeatabilityprotocolpreserved.
New separate same-process study planned:4image/windowcells eachrepeated,
freshnativecontrol, allclassmaxrepeatdrift<=1e-4 plus identicalsegmentation.
No relaxation/retroactivepass. Max0.1additionalchargedh if preflightpasses;
worstcampaign1.001944/2. See SAME_PROCESS_WINDOW_20261002.md. No47320183change.

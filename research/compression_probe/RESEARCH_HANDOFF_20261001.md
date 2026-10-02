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

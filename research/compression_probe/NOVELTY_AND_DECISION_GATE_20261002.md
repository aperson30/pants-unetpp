# Compression direction: novelty audit and next decisive gate

2026-10-02. Research assessment, not a conference-acceptance prediction.

## Current provisional decision

**GO for a bounded second-model diagnostic; NO-GO for new diffusion/VAE
training or a claimed novel lesion-preserving method yet.** Existing results
show continuous frozen-detector score changes on two selected cases, not
clinical sensitivity loss. One case's three fixed posterior samples vary
substantially and none fully restore its original PANORAMA score. Selecting
the best sample would be invalid; all three remain in the evidence.

Strongest supported candidate question: **can cheap, posterior-aware,
lesion-specific reconstruction audits identify failures hidden by overall
image-fidelity measures before expensive 3D diffusion training?** Currently
this is a falsifiable research question, not a demonstrated new method.
Our archive contains score changes, image statistics and numerical controls;
it does not yet show high global fidelity predicting poor lesion recall
across a cohort, or predict subsequent diffusion outcomes.

## Primary-source novelty checks actually read

- **The Learnability Gap in Medical Latent Diffusion (MICCAI2026):** tests
  five autoencoder families on four classification benchmarks, comparing
  image, reconstruction and latent spaces. It reports largely preserved
  reconstruction-space classification but worse latent-space learnability.
  This substantially overlaps any broad claim that conventional fidelity
  is insufficient to judge medical latents. Our proposed test instead keeps
  image-space judges frozen and investigates local 3D pancreatic lesions
  and posterior sampling. That distinction is a possible opening, not proof
  of novelty or a contradiction of their different task.
  https://papers.miccai.org/miccai-2026/1053-Paper3049.html
  https://papers.miccai.org/miccai-2026/paper/3049_paper.pdf

- **Lesion-Aware Post-Training (MICCAI2025/preprint2510.09056):** already
  adds image-space and lesion-masked objectives to latent diffusion for
  CT-perfusion-to-diffusion-MRI translation, evaluated on817 patients. An
  ordinary lesion-weighted fine-tune is not a defensible new core idea.
  It addresses diffusion post-training, not our frozen VAE posterior probe.
  https://arxiv.org/html/2510.09056v1

- **MAISI-v2:** already uses tumor-weighted ControlNet loss and
  region-specific contrastive loss, with downstream tumor augmentation
  evaluation. Merely replacing whole-image loss with ROI loss is crowded.
  Its weighted loss alone was insufficient for clear tumor appearance.
  https://arxiv.org/html/2508.05772v1

- **MAISI:** evaluates pretrained VAE reconstruction on out-of-distribution
  datasets including MSD pancreas; also reports downstream synthesis utility.
  Reconstruction quality and segmentation-based synthesis evaluation are
  established baselines, not our novelty. The small-organ synthetic-data
  results do not by themselves establish small-tumor compression failure.
  https://arxiv.org/html/2409.11169v3

- **Pathology Image Compression with Pre-trained Autoencoders (MICCAI2025):**
  already repurposes diffusion autoencoders for compression and evaluates
  downstream tasks, with pathology-specific fine-tuning and quantization.
  A generic downstream-aware compression benchmark is not new.
  https://papers.miccai.org/miccai-2025/0671-Paper1570.html

- **MedSegLatDiff:** its abstract describes image/mask VQ-VAE compression
  and weighted mask objectives for small structures in segmentation. Mask
  compression is different from CT-intensity compression, but weakens any
  broad small-structure-preserving latent segmentation claim. Abstract-only
  check here; do not overstate implementation verification.
  https://arxiv.org/abs/2512.01292

- **Diffusional MRI anonymizing with VAE (Quantitative Biology2026):**
  published region-separated lesion/non-lesion VAEs and a fused decoder.
  Simply adding a lesion-only latent branch is not untouched territory.
  https://onlinelibrary.wiley.com/doi/10.1002/qub2.70033

Searches included medical latent reconstruction + small lesions, task-aware
CT compression, rate-distortion + lesions, posterior sampling + preservation,
and MAISI reconstruction. This is a scoped audit as of today, not an
exhaustive novelty certificate. Negative search results are not proof of absence.

## Independent-judge resolution and deliberate scope

DiffTumor appendix E.2 explicitly identifies organ pseudo-labels as coming
from its reference55: Liu et al., CLIP-Driven Universal Model. The latter's
official README releases Swin weights and a pred_pseudo.py inference path.
That resolves the *producer-family lead*, not the exact checkpoint revision
used for the archived120 masks or compatibility of its old runtime.
https://arxiv.org/html/2402.19470v2
https://github.com/ljwztc/CLIP-Driven-Universal-Model

Do not deploy that extra network simply to satisfy a superficial checklist.
The immediate causal question is whether a separately trained tumor network
changes its raw response to the same images. A **full-context raw-head
diagnostic** answers that narrower question without an additional organ
segmenter, extra compute, or GT-input leakage. It is expressly NOT official
DiffTumor postprocessed detection, a clinical recall measurement, or evidence
of patient-independent generalization. The official end-to-end pipeline
remains separately blocked until its exact mask model/runtime are verified.

## Locked cheapest next GPU experiment — not submitted yet

- Reuse existing patient100226 and saved native, clipping-control,
  posterior-mean, seeds0/1/2 images. No new VAE encode/decode or download.
- Frozen strict-loaded public DiffTumor pancreas U-Net; no adaptation,
  checkpoint selection by outcome, GT organ input or input crop.
- Its published image preprocessing: RAS,1mm,HU[-175,250] normalized[0,1],
  minimum96 padding; full image sliding96cube, batch1, overlap.75, Gaussian.
  FP32. Match all settings across arms. Optional CPU output accumulation
  must be disclosed as execution choice, not declared legacy GPU parity.
- Invert logits onto each original image grid; only then compute softmax.
  GTlabel1 used solely AFTER inference to score DiffTumor output-channel2.
  Native and clipping arms must repeat independently; retain all samples.
- Primary scalar: mean tumor probability relative to native. Also record
  tumor maximum, raw argmax tumor overlap, global maxima separately, and
  per-arm image/runtime hashes. No clinical threshold is invented.
- Save repeatability/geometry evidence and private raw probability maps
  for separate CPU audit. Any failure must retain partial outcomes and cost.
- One scheduled GPU; held submission reviewed before release; no automatic
  retry/requeue; hard allocation/process limit. Launch cap <=0.34 charged
  GPUh for first probe, giving campaign worst-case <=1.449722 of2 if a
  10-minute allocation is charged at2x. Verify actual billing before release.
  Count failures and allocation startup, not just neural execution.

## Decision criteria before seeing outcomes

This one-case probe can decide whether to **invest in a small preregistered
cohort**, not whether the method is publishable.

- Numeric gate: native/control equivalence in the tumor; within-process
  repeatability <=1e-4 probability max drift; exact restored grid; finite
  outputs; all arms completed and reported. No tolerance changed after a
  failure. A failed gate yields no scientific second-judge conclusion.
- Support-only GO: native tumor mean>=0.01 (predeclared floor preventing
  near-zero ratio artifacts), mean and **each** of the3samples at least20%
  below native, and decrease exceeds20x measured repeat drift. Thresholds
  are resource-screening heuristics, NOT clinical operating points.
- Mixed result: disagreement across judges or samples => preserve it,
  reject a blanket erasure narrative; possibly study detector domain shift
  or posterior sensitivity, but no more diffusion training on this budget.
- NO-GO for current lesion-erasure story: native response too weak to assess,
  reconstruction largely retains raw response, or controls explain changes.
  This does not prove all compression is safe; it kills this cheap evidence
  route under the current constraints.

## What would justify PI approval afterward

Smallest next serious experiment: prospective fixed-case cohort, at least
two frozen judges, native/clip/mean/all fixed seeds, lesion-specific and
global fidelity metrics, native-response adequacy reported not used for
post-hoc case replacement, plus negative-case false positives. Sample size
and spending must be approved before any new cohort. Include a second codec
or a matched distortion baseline before claiming a VAE-specific mechanism.
If score changes persist, reader evidence or an independently validated
clinical threshold is needed before calling them diagnostic misses.

No training-time speedup has been demonstrated by this exploration. Codec
latent-size reduction is not automatically file bitrate or an actual GPU-hour
saving. Likewise loss of a detector response is not proof information was
irreversibly erased: a different judge, adaptation or conditioning can alter
that conclusion. Hypothesize mechanism only after appropriate controls.

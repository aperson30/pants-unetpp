# Resource-minimal idea screen: nearby work and next decision

Research-only follow-up; no extra job, training, weights download or CT egress.
The existing contrast job3291159 is still incomplete at this snapshot. Its
baseline matches the frozen reconstruction exactly (max/mean0HU), and its
repeated-20HU point matches the previous job's recorded retention exactly:
raw0.4356453896, ring-corrected0.3707545519. Other amplitudes/phase still pending.
This verifies a useful reference, not the proposed nonlinear mechanism.

## Primary-source novelty checks

### The Learnability Gap in Medical Latent Diffusion (MICCAI2026)

Studies classifiers in original-image, reconstructed-image and latent space;
reports a latent learnability gap despite reconstruction-space performance.
Its tested families include MedVAE, SD and Flux, not MAISI. Medical tasks
include CT-RATE classification. Its novelty already covers medical latent
probing; our reconstruction-only screen is different from training classifiers
on latents. Read both paper and public reviews; broad generative consequences
should not be treated as demonstrated by classifier experiments alone.

https://papers.miccai.org/miccai-2026/paper/3049_paper.pdf
https://papers.miccai.org/miccai-2026/1053-Paper3049.html

### Foundation VAEs for3D CT Reconstruction, Augmentation, and Generation

Particularly close prior work, coauthored by ZongweiZhou; do not assume it is
the user's current draft. Evaluates pretrained video/medical VAEs, including
MAISI, on MSD pancreas and other CT tasks. Table2 trains nnU-Net on each
reconstructed dataset, already reporting organ/tumor Dice and NSD. Thus a
generic VAE-on-MSD segmentation comparison duplicates substantial ground.
Our single-model signal probe does not contradict its cross-model cohort
results. Published preprocessing/checkpoints differ until verified.

https://arxiv.org/html/2605.30893v1
https://github.com/qic999/Foundation-VAE

### Other adjacent work

- MedVAE already frames medical compression as computationally efficient
  interpretation with clinically relevant feature preservation.
  https://arxiv.org/abs/2502.14753
- EQ-VAE already addresses autoencoder equivariance deficits. A grid-phase
  artifact alone is not a new general discovery.
  https://arxiv.org/abs/2502.09509
- Domain-specific medical super-resolution work proposes autoencoder-quality
  screening before diffusion training. That generic screening pitch is taken.
  https://arxiv.org/abs/2604.12152

## Stronger candidate, still a hypothesis

**Task-specific signal-preservation envelopes for pretrained3D CT VAEs:**
characterize when faint structure survives as a function of contrast and
sampling phase, then ask whether that predicts paired lesion evidence loss
under a defensible judge. This is narrower than pixel fidelity or broad latent
classification. A signal-response curve is not a full spatial task-transfer
function or an observer detectability model; do not use those names as if their
frequency/noise/observer machinery had already been implemented.

Necessary separation:

1. **Signal transport:** what changed in a controlled input/output pair?
2. **Frozen-model compatibility:** does a model trained on originals fail after
   reconstruction? This could be domain shift, not irrecoverable information loss.
3. **Task information:** can an independently validated decision procedure still
   recover the clinically relevant evidence? Three descriptor measurements or a
   single frozen-detector failure do not answer this.

The published comparison's separately trained segmenters and our prospective
frozen judge answer different questions. Neither result should be substituted
for the other. Do not spend on retraining a probe solely to bridge that gap
before the mechanism and scientific objective justify it.

## Next actions ranked by information per resource

1. Finish and validate the already-budgeted contrast job; no extra allocation.
   Report raw signed curves and phase effect before applying heuristic flags.
2. If amplitude dependence is weak but phase sensitivity is strong, prioritize
   sampling/aliasing explanations instead of claiming a learned prior. If both
   are weak, this mechanism is a weaker lead; no additional amplitude sweep.
3. If interesting, independently audit an eligible judge/case pair and calibrate
   a tightly bounded paired test. A synthetic negative does not rule out realistic
   lesion harm and is not a mandatory gate blocking it.
4. Ask Claude to preserve its restricted checkpoint-header audit code in the
   repository so fold-identity claims are reproducible; that is cheaper than
   independently downloading all checkpoint weights.
5. Defer broad radiomics/NPS expansion: spectrum of patient anatomy is not a
   pure noise measurement, and more descriptive features alone will not settle
   the mechanism or downstream meaning.

## Access check / blockers

Official PANORAMA image record exposes a large ZIP, not a verified individual-
case endpoint. Following its latest-version link returnedHTTP429 in this session;
stopped rather than retrying repeatedly. No48GB archive or image downloaded.
Per-case access remains unresolved. Automatic duct masks in PANORAMA are not
manual radiologist duct truth. MSD label2 is not necessarily histologically
confirmed PDAC; maintain the original engineering-host wording.

https://zenodo.org/records/10998332
https://github.com/DIAGNijmegen/panorama_labels

## Current decision

Do not label any current idea S-tier or publishable yet. The cheapest useful
next evidence is already being collected. Larger studies, a second VAE and
any method/fine-tuning proposal need a new outcome-based plan, PI alignment,
known implementation/data provenance and calibrated charge—not unused budget.

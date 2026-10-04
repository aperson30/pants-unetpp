# Synthetic utility branch: source audit and next investment decision

2026-10-03, continued through 22:51 Pacific. No new GPU allocation, data/weight
download, author-code execution or main PanTS evaluation action. One original
standard-library CPU algebra test executed successfully. The earlier frozen
negative cases and total screening budget remain unchanged.

## Decision first

**Do not implement a new synthetic selector yet.** Generic hard-example mining,
local quality scores, class-influence selection and background-bias diagnosis
have direct competitors. We found a concrete algebraic limitation worth retaining
as a test contract, NOT a selected contribution or S-tier result. No evidence yet
of a new medical failure mechanism or of a cheap, complete downstream-utility paper.

## What the actual author implementations do

NAM source pinned at `4c18850db69e89db3a01dcb3beda6665a7f52405`.
Read complete `hat.py`, `lsrs.py`, `qsf.py`, mitigation README and configuration.
[Pinned source directory](https://github.com/JackCD99/Native-Adversariality-Mining/tree/4c18850db69e89db3a01dcb3beda6665a7f52405/nam/mitigation).

- HAT uses an initial global 95th-percentile adversariality threshold and retries
  over-threshold candidates up to five times, keeping the condition fixed. If no
  retry qualifies it returns the lowest-adversariality retry flagged unaccepted.
  **Correction to a possible hypothesis:** it does not simply delete rare labels
  or reduce condition counts. Within-condition appearance selection can still
  change, but that has not been measured here.
- LSRS compares squared full-vector differences between full, unconditional and
  component predictions, averages across cached timesteps, and takes the minimum
  component score. Default threshold zero, five attempts, best-score fallback
  flagged unaccepted. This is global aggregation, not lesion-region scoring.
  A large region can mathematically dominate the sum; no real-model demonstration
  was run. Also inspect fallback flags when reproducing acceptance rates.
- QSF renders a 2D mask contour and scores teacher-forced "Yes" using MedGemma;
  default acceptance .8, up to five attempts, best-score fallback. Its direct image
  adapter is CHW, not a 3D volume interface. MedGemma is gated; no licence acceptance
  or download attempted. Contour-based shortcutting is a hypothesis, not a finding.

These are code-read findings, not reproduction of the published numbers. Extended
repository features must not all be attributed to the original CVPR paper.

## Additional collisions that change the plan

1. [CompLift, ICML 2025](https://arxiv.org/html/2505.13740v1), section 5.1,
   already computes spatial lift maps and thresholds activated-pixel counts;
   the reported threshold is 250. Its [author page](https://chenningyu.com/complift/)
   explicitly reports rejection of small objects. Generic regional scoring or
   "global scores miss small objects" cannot be claimed as our new contribution.
   Simply normalizing by lesion area would need comparison to local-count and
   conventional normalization baselines, not a novelty claim from absent keywords.
2. [ASOB-Bench, July 2026 preprint](https://arxiv.org/html/2607.03831v1),
   sections 4.2/4.3, already analyzes size/background bias in diffusion classifier
   reconstruction errors. It is natural-image evidence, not evidence of tiny
   medical-lesion failure, but directly overlaps the proposed general mechanism.
3. [C2I, July 2026 preprint](https://arxiv.org/html/2607.12464v1), already steers
   medical augmentation using class-specific validation-gradient influence.
   Its binary squared-gap and multiclass signed-softmax definitions differ.
   Our CPU check below only tests that algebra; it does not reproduce their code.
   The signed alternative is already present, so adding a sign check alone is
   not a new method. RL runs use one A100 40GB; total runtime was not established.
4. [Grad-Mimic](https://arxiv.org/abs/2501.06708) already uses gradient alignment
   toward reference weights for data selection. Abstract verified; implementation
   not audited. Generic cheap gradient utility is not an empty research space.
5. [DiffAug, ICCV 2025 workshop](https://arxiv.org/abs/2508.17844), already uses
   segmentation-based spatial validation for medical synthetic abnormalities.
   Abstract inspected; reported performance not reproduced. A mask-agreement
   filter is a baseline, not automatically a new medical-generation idea.
6. [FROST, September 24 2026 preprint](https://arxiv.org/html/2609.29988v1),
   sections 3.1-3.3, already uses signed real-gradient alignment, an EMA reference,
   history-calibrated batch routing and sample IQR filtering without an external
   verifier or held-out selection set. It retains real samples. Head-only scoring
   is explicitly a proxy for full-model updates; local alignment is not guaranteed
   downstream benefit. Therefore removing the independent-verifier requirement
   or adding online signed utility alone is not a distinct fallback contribution.
7. [Koh and Liang, ICML 2017](https://proceedings.mlr.press/v70/koh17a.html) already
   studies influence-based dataset-error detection; [class-based influence,
   ACL 2023](https://aclanthology.org/2023.acl-short.104/) is another direct lead.
   Abstracts verified, not reproduced. Label-flip controls and class-specific
   influence should not themselves be claimed as new.

## Reproducible zero-GPU algebra contract

Run `python research/influence_sign_contract.py` (standard library only).
Results: [influence_sign_contract_20261003.json](influence_sign_contract_20261003.json).

For cosine influences, reversing a nonzero candidate gradient negates both
class means, leaves both variances unchanged, and therefore leaves a squared
mean-gap score unchanged. In our hand-constructed logistic example, a deliberately
flipped synthetic training label gives the same unsigned score, 11.7238394,
while the actual class-0 loss changes by -7.8771e-5 versus +5.1759e-5 at step 1e-4.
Signed softmax scores instead differ, .7149721 versus .2850279. Assertions also
check negative scaling, first-order signs and finite-step loss signs.

This isolates a known sign ambiguity; it does NOT prove generator labels are
wrong, that C2I's actual implementation uses a particular formula, that published
results are incorrect, or that a deep medical model behaves like this toy.
Validation features are disjoint artificial groups, not patient data. A validity
or utility rule must not confuse unsigned separation with beneficial direction.

## Reusable resources: metadata verified, payloads not downloaded

LeFusion author code pinned at `03dc67bd8169ced5f8bb6a8707d73377f110ebff`.
Read its inference script/config, LIDC launch command, segmentation README,
dataset factory, both LIDC loaders and relevant RePaint mask code.
[Pinned author tree](https://github.com/HINTLab/LeFusion/tree/03dc67bd8169ced5f8bb6a8707d73377f110ebff).

Loader clamps HU to [-1000,400], maps to [-1,1] and crops/pads to (32,64,64).
Positive LIDC mask value 1 denotes synthesis region; RePaint inverts that mask
for background injection. The name `gt_keep_mask` alone is misleading.
Outputs are region-level examples, not evidence about full-volume context or
pancreatic tumor detection. Do not silently apply the original training-grid
context constraint to claim these are equivalent data.

[Public dataset API](https://huggingface.co/api/datasets/YuheLiuu/LeFusion_Preprocessed_Data?blobs=true)
revision `b0516d354c4473ea4f8539cad395dfddb5944215` advertises:

| Archive | Bytes | Advertised SHA256 |
| --- | ---: | --- |
| LIDC-IDRI/Demo.tar | 41,293,312 | eea5b165b14d5ae3b10212fe946466831a9edbbcbc8c236a58bb00997d8f6e47 |
| LIDC-IDRI/Normal.tar | 6,344,704 | 2c3342b79b57b457dd97fbbc1786ef56ce32f81d14e4b3e2caf91b7b03806f02 |

Combined 47,638,016 bytes, not 1.1TB. MIT dataset-card metadata does not establish
all upstream patient-data rights. No archive hash, contents, pairing, affine,
intensity or source-patient overlap validated. We deferred downloading because
the novelty gate failed; small payload size is not a reason to build infrastructure.
The public generator metadata advertises a ~287MB LIDC checkpoint, not retrieved.

### There IS a public task-matched lung detector, with important contracts

[MONAI lung-nodule bundle](https://huggingface.co/MONAI/lung_nodule_ct_detection)
pinned revision `9d6622fda0e52be0a4155fed2e0afb9c17bcd80b`; actual inference and
metadata JSONs read. Public, ungated `models/model.pt` advertised at 83,709,381
bytes, SHA256 `b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01`.
This corrects a broad impression that no reusable lung task model exists.

It is 3D RetinaNet detection trained on LUNA16, NOT lesion segmentation. Config
uses RAS orientation, .703125/.703125/1.25mm spacing, [-1024,300] to [0,1],
default inference ROI [512,512,192], box transformations and NMS. Metadata permits
variable-sized patches divisible by [16,16,8]; small crops are not automatically
unsupported, but native performance on LeFusion regions is unverified. Its
LPS/RAS world-box settings must match actual inputs, not be copied blindly.
Reported dependencies include MONAI1.4/PyTorch2.4/torchvision.19; no install run.

Never feed [-1,1] synthetic values as HU or turn detector boxes into segmentation
Dice. Establish source-patient/fold overlap and native adequacy on an annotation-
selected cohort before interpreting any degradation. Intended use is research,
not diagnosis. No checkpoint retrieval, loading or neural test occurred.

## What evidence would earn the next experiment

Best remaining QUESTION: can a cheap selector separate annotation-consistent,
useful hard samples from inconsistent harmful ones without suppressing small
valid examples? It is not a verified novel claim. Ranking a sample highly is
not proof of validity, and one-step validation improvement is not training utility.

1. Identify a distinction not already covered by signed influence, local lift,
   mask verification, conditional replacement and real-anchored online selection.
   A method that wins only against
   unsigned scores or weak global aggregation is insufficient. Stop this branch
   if its strongest formulation reduces to those baselines.
2. Only then validate bounded public assets, freezes and native task-model
   adequacy. Retain every selected case; do not replace failures after outcomes.
   Pair by patient, not generated variant; avoid treating three histogram variants
   from one patient as three independent successes.
3. Test a fixed generation pool with matched acceptance counts and distributions,
   standard intensity/mask/size controls, signed influence and published selectors.
   Report acceptance by size and actual total scoring/rejection cost. Synthetic
   condition masks alone cannot establish clinical plausibility.
4. Utility claims require matched downstream training with independent evaluation
   and repeated seeds. Generator-free scoring does not remove that cost. Estimate
   the whole study as datasets x seeds x baselines x downstream-run cost plus
   generation/scoring/evaluation; no measured full-paper budget is available yet.

No launch warranted now. Keep unused screening credit rather than manufacture
a positive probe. Old reconstructed spending 1.225833/2 charged-equivalent GPUh
is historical; posted debit/rounding still unverified. This pass added zero.
No commands targeted the protected main test evaluation or production services.

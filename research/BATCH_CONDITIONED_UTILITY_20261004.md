# Batch-conditioned utility: CPU mechanism screen, not a selected paper idea

Local work date: 2026-10-03 PDT; result names use 2026-10-04 UTC.
Status: **PARK as a standalone paper proposal; retain the loss contract.**
No GPU jobs, patient-data downloads or changes to the main experiment/evaluation.

## Question

Our segmentation objective couples four physical patches through batch Dice.
Can a selector that scores each candidate independently get the replacement
decision wrong? Can the coupling be accounted for without separately running
every counterfactual batch through the model?

This is a diagnostic for a possible future augmentation/selection method, NOT an
optimization to apply to the frozen 2x2 grid. Selection changes training exposure.

## Executed screens: keep every seed, including negative results

| Screen | Independent-score sign disagreements | CPU seconds excluding import |
| --- | --- | --- |
| Binary sigmoid, CE alone, 256 seeds | 0/256 | 2.34 for both binary endpoints |
| Binary sigmoid, CE + batch Dice, same seeds | 10/256 | included above |
| 29-channel softmax, CE alone, 256 seeds | 0/256 | 10.28 for initial multiclass run |
| 29-channel softmax, CE + foreground batch Dice | 1/256 | included above |
| Same multiclass seed set, rare-class soft-Dice validation target | 0/256 | 18.94 for complete follow-up |

The binary label rule is the same for every candidate and validation patch;
there are no deliberately flipped labels. Multiclass labels follow one fixed
artificial rule, with category 28 occurring for feature x0 > 2.3. This category
is not a pancreatic tumor and this setup does not emulate lesion anatomy.

The utility is the **additional first-order validation-loss improvement from
replacing one slot**, holding the other three companions fixed. It is NOT the
absolute usefulness of either example, trained-model accuracy or lesion recall.
The follow-up endpoint was chosen after the binary and aggregate screens and is
explicitly exploratory, not a preregistered confirmatory study. No seed subset
was selected and no threshold was tuned to maximize reversals.

Original outputs remain separate:

- `batch_dice_utility_screen_20261004.json`: binary, float64 reductions.
- `batch_dice_stat_contract_20261004.json`: initial multiclass, float32.
- `batch_dice_stat_contract_followup_20261004.json`: multiclass plus rare target.

The multiclass sole aggregate reversal is seed 72: exact -2.4136e-6,
isolated +4.5439e-6. Its fp32 finite-step loss differences at the smallest
learning rates are at numerical resolution, so do not interpret them as verified
training harm. The rare-target follow-up evaluates finite-step losses in float64
to resolve subtraction, while the training gradients remain float32.

## Cheap correction contract

For a foreground class, batch negative Dice is

`D = -(2*T+s)/(P+Y+s)`.

Its derivative in any parameter direction v is

`dD[v] = -2*dT[v]/(P+Y+s) + (2*T+s)*dP[v]/(P+Y+s)^2`.

Each patch contributes additive T, P, Y and their directional derivatives.
Replacing a slot changes the denominator for the companions as well. That is
why subtracting two isolated-example gradients is generally not equivalent to
subtracting the actual full-batch gradients. CE's additive control does match.

The CPU softmax contract reconstructs the validation-gradient directional score
from cached per-patch statistics. Across all 256 multiclass trials, maximum
absolute difference from autograd is 1.1991e-8; assertions passed at
rtol=2e-4, atol=2e-7. Rare-target reconstruction assertions also passed.
This formula is basic calculus, **not a novelty claim**.

The linear model cheaply computes probability directional derivatives from
features and weights. A real backbone needs a JVP/backprop or a justified proxy.
We have NOT shown lower end-to-end cost on a neural segmentation model. Caching
full parameter Jacobians is not an acceptable hidden expense; output-head-only
attribution may miss backbone effects.

## Loss compatibility is limited

Read the full upstream nnU-Net Dice source pinned at
[2edf34680698d3ee62f5e72f3707082d44fb2db0](https://github.com/MIC-DKFZ/nnUNet/blob/2edf34680698d3ee62f5e72f3707082d44fb2db0/nnunetv2/training/loss/dice.py).
The multiclass screen reproduces foreground exclusion, spatial then batch
float32 reductions, smooth=1e-5, and unweighted CE+Dice in its own implementation.
It does not import or execute nnU-Net. This live upstream revision has NOT been
verified as the deployed 2.8.1 loss revision; ignore masks, DDP, deep supervision,
nonlinear backbones and actual optimizer state are not covered. Therefore these
passing assertions are not trainer-parity or clinical-quality certification.

## Closest prior work and novelty exclusions

- [VIF](https://arxiv.org/html/2412.01335v1), section 3.4: already uses gradient
  differences of non-decomposable losses for attribution, with inverse-Hessian
  weighting. General non-decomposable attribution is not our contribution.
- [JEST](https://arxiv.org/abs/2406.17711): joint example selection for multimodal
  learning. Generic batch-aware curation is not new.
- [ACID](https://arxiv.org/html/2411.18674v2), joint batch sampling: scores
  candidates conditioned on already selected companions, following JEST.
- [FROST](https://arxiv.org/html/2609.29988v1): signed gradient-based synthetic
  utility filtering already exists. Our prior source audit describes its scope.
- [Generalized Dice](https://arxiv.org/abs/1707.03237): rare-class imbalance in
  overlap losses is established. Merely adding class weights is not a new idea.

Targeted web searches for Dice data valuation, batch-Dice data selection, and
Dice influence functions also surfaced medical active-learning and brain-tumor
TracIn work. Their full implementations have not been audited. Search results
not returning an exact matching title do not establish absence of prior art.

## Decision and next investment

The mechanism exists, but effect size and rarity benefit are not demonstrated:
the more relevant 29-channel aggregate endpoint has only one reversal and the
rare target has none. **Do not spend GPU hours on this as a method yet.** A
positive binary example is not enough to override the weaker follow-up.

If revisited, require a reason why actual trained gradients should differ from
the toy. Then freeze a development-only comparison against isolated gradients,
exact full-batch counterfactual gradients and simple foreground-count balancing;
measure ranking agreement and total scoring cost, without touching the 901-case
test or altering the four grid cells. Use actual deployed loss and optimizer
contracts. Require a material improvement beyond VIF/JEST-inspired baselines,
not merely nonzero disagreement. Longer downstream quality validation is still
needed before claiming benefit, especially with delayed PanTS tumor learning.

The next search should favor an observable failure in a competent existing
model over another algebraic variant. Reuse a pretrained model and annotated
development examples only after native task adequacy, provenance and overlap
checks pass. The already audited small LeFusion assets are a resource lead,
not an approved experiment or selected idea. No new generator training.

Budget remains the existing 2 charged GPU-hour total cap. This turn spent zero.
No S-tier claim, acceptance promise or new GPU authorization is implied.

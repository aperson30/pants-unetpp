# Generation architecture reading register — October 8, 2026

## Reading honesty

Twenty relevant paper families selected below. This is **not 20 completed
end-to-end readings**, nor a claim each paper remains SOTA. M = targeted
method/experiment reading; A = abstract/source screening; F = full supplied paper
plus implementation audit. Revisions/renamed preprints count once. Reported
results are their experiments, not predictions for our conditional CT task.

| Primary source | Depth | Consequence for our experiment |
| --- | --- | --- |
| [UNet++](https://arxiv.org/html/1807.10165v1) | M | Sections 3–4: nested skips, supervision and pruning are established ingredients; its wide U-Net control addresses parameter-count confounding. |
| [Latent Diffusion](https://arxiv.org/html/2112.10752v2) | M | Section 3: separate autoencoding and diffusion, concatenation/cross-attention conditioning. Hold latent representation and condition encoder fixed. |
| [U-ViT](https://arxiv.org/html/2209.12152v2) | M | Sections 3/5: time/condition/image tokens, skip fusion and latent/CLIP setup. Backbone comparison must control size/representation. |
| [Matryoshka](https://arxiv.org/html/2310.15111v2) | M | Multi-resolution objectives plus progressive training. Appendix D, 8 A100s: nested/ordinary training iteration 0.52/0.50s, inference step 0.41/0.40s. Nesting alone did not accelerate this comparison. |
| [ASE](https://arxiv.org/html/2408.05927v1) | M | Timestep-dependent exits and finetuning/EMA/loss changes; U-ViT/DiT/PixArt on A100. Direct collision with broad depth-versus-noise novelty. |
| [AdaDiff / DeeDiff](https://arxiv.org/html/2309.17074v3) | M | Intermediate heads and timestep-aware learned uncertainty. Same paper family; timing hardware not verified here. |
| [T-Stitch](https://arxiv.org/html/2402.14167v1) | M | Small/large pretrained models switched along trajectory. No extra stitching training does not make both pretrained models free. Timing hardware not verified. |
| [Early-Bird Diffusion](https://arxiv.org/html/2504.09606v1) | M | Timestep-aware structured subnetworks and training acceleration on A100. Count discovery and parallel training in GPU-hours, not just wall time. |
| [Diff-Pruning](https://arxiv.org/html/2305.10924v3) | M | Timestep-filtered Taylor importance and channel-pruning/finetuning; about half FLOPs with 10–20% original training expenditure on DDPM/LDM natural images. Not a CT runtime/fidelity guarantee. |
| [DeepCache](https://arxiv.org/html/2312.00858v2) | M | Reuse deep features across steps; abstract SD1.5 2.3x with small CLIP change. Training-free is not output-exact. Hardware not verified here. |
| [FasterDiffusion](https://arxiv.org/html/2312.09608v2) | M | Encoder propagation and decoder work; A40 48GB inference. Feature-propagation failures warn against assuming details survive caching. |
| [Cache Me If You Can](https://arxiv.org/html/2312.03209v2) | M | Block-wise caching and alignment. Compare at equal wall time to fewer-step sampling; some alignment choices require fitting. Hardware not verified. |
| [DuoDiff](https://arxiv.org/html/2410.09633v1) | M | Independently trained shallow/deep U-ViTs and fixed switch, A100 40GB. Simple routing avoids batching overhead; extra model training still costs. |
| [DPM-Solver++](https://arxiv.org/html/2211.01095v3) | M | Sections 4/7: multistep reuse vs extra singlestep model calls. Latent experiments omit thresholding because latents are unbounded. Natural-image 15–20-step results do not prove CT fidelity. |
| [UniPC](https://arxiv.org/html/2302.04867v3) | M | Sections 3/4: predictor-corrector reuse and same-noise latent convergence versus long DDIM. Decoder perceptual metrics can hide latent convergence error. |
| [Consistency Models](https://arxiv.org/html/2303.01469v2) | M | Sections 5/6: teacher-free training vs pretrained distillation, EMA/discretization and metric choices. Different objective; not a scheduler switch. |
| [Latent Consistency Models](https://arxiv.org/html/2310.04378v1) | M | Guided latent distillation: 4k-update/32 A100-hour fast-convergence demonstration alongside 100k-update main settings. Headline is not cost of every adaptation. |
| [MeanFlow](https://arxiv.org/html/2505.13447v1) | M | Average-velocity training with stop-gradient JVP target. Changes objective, needs training; not a replacement scheduler for epsilon checkpoints. |
| [MAISI](https://arxiv.org/html/2409.11169v2) | M | 3D latent compression, spacing/body-region and ControlNet conditions. Mask-conditioned synthesis cannot assume unknown test tumor masks. |
| [SMILE](https://arxiv.org/abs/2512.07251) | F | Task-matched 2.5D conditional diffusion baseline. Public shell and paper disagree on training recipe. Weights gated; actual reproduction not run. |

## Decisions supported so far

1. Do not claim shallow-at-high-noise/deep-at-low-noise is new. Multiple papers
   cover it. We must demonstrate what conditional nested architecture adds.
2. Establish full-depth competence before schedules. Random small nested versus
   large pretrained baseline confounds architecture, capacity and pretraining.
3. Exact retained-head graph parity is a code contract; selecting another head,
   caching, switching solver or reducing steps changes predictions.
4. Compare speed-quality frontiers, including fewer-step DDIM and a compatible
   strong solver. Add caching/distillation comparisons in stages to limit cost.
5. Whole-image MSE/SSIM and natural-image FID cannot certify tumor preservation.
   Include small lesions, negatives, geometry and slice consistency. Development
   labels may evaluate/supervise, not supply unavailable deployment conditions.
6. Profile VAE, auxiliary networks/losses, assembly and I/O alongside denoiser.
   Report training, inference, setup and total charged cost separately.

## Next gates, cheapest first

- Approved baseline weights and frozen recipe/data; pinned dependencies.
- Deterministic baseline image-in/image-out with units/geometry/coverage checks.
- End-to-end component profile, then matched nested/nondense full-depth pilot.
- Competent exits before fixed/scheduled/reversed-depth comparisons; freeze
  schedule on development data, compare against fewer model evaluations.
- Protected 2x2 test evaluation stays outside architecture development.

All twenty now have targeted method/experimental or full-source reading, but
this remains a selective engineering pass, not twenty full proof/appendix audits.
Hardware marked unverified remains unverified. None establishes our medical
benefit, conference odds or a novelty guarantee. No additional GPU job submitted.

## Current medical acceleration neighbor (additional, not part of the twenty)

[Structure-Adaptive Sparse Diffusion, MICCAI 2026](https://papers.miccai.org/miccai-2026/paper/5369_paper.pdf):
method and experiment pages inspected. Conditional voxel-space 3D denoising/
super-resolution, five selected timesteps, clean-image prediction, velocity-space
loss and anatomy-conditioned time modulation; two B200s, batch four. Its 10x
claim is primarily fewer iterations to baseline quality; table 3 reports relative
training cost without an absolute GPU-hour budget. Restricting eligible noise
timesteps alone does not reduce per-update model calls in ordinary sampled-timestep
training. No direct noncontrast-to-contrast pancreatic tumor result is established
here. Treat as relevant prior art/possible later comparator, not evidence that
our epsilon architecture can adopt the changes without retraining. Reviewers
also demanded stronger non-diffusion baselines; a direct conditional image-to-image
U-Net control is worth considering after the basic pilot.

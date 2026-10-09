# PI-directed generation architecture: preparation, October 8, 2026

## Actual status

Source audit and original CPU prototype complete. Random-weight GPU contract
probe **3344849 passed** on DeltaAI GH200; see `GPU_SMOKE_RESULT.md` and raw log.
**No baseline enhancement run, trained U-Net++ generator, medical fidelity result
or end-to-end generation speedup exists yet.**
The frozen PanTS 2x2 implementation, data, checkpoints and evaluation were not
modified. An isolated copy of this prototype was uploaded to a separate cluster
home directory; production segmentation environments were not modified.

Source audited: https://github.com/MrGiovanni/SMILE at
`23f5a28fe25ed0472024b688e19a79ce119c4833`, cloned separately into
`work/smile-source-audit`. This updates the meeting note: public **training code
does exist**, but the shell script is an author-specific configuration, not a
verified reproduction of the supplied paper.

## What the actual baseline expects

- `SMILEModel/train_text_to_image.py` and `testEnhanceCTPipeline.py` use
  `UNet2DConditionModel`, a Stable Diffusion v1.5 text encoder and a finetuned VAE.
- Three adjacent CT slices become three image channels at 512x512. This is
  2D/2.5D processing, not our nnU-Net 3D volume architecture.
- VAE latent channels = 4, noisy target + source conditioning concatenate to
  **8 denoiser input channels; residual output has 4 channels**.
- Target phase is encoded by CLIP prompt tokens, width 768; timestep conditioning
  is required. The source latent is not noised by the target noise schedule.
- Released multi-volume inference uses DDIM, **200 steps**, source posterior
  sampling, a noise volume shared across overlapping slice triplets, and
  overlap averaging back to a volume. Seeding only the generator supplied to the
  pipeline does not seed earlier `torch.randn` and posterior samples.
- CT is clipped to [-1000,1000] HU, normalized, and written back in HU with the
  source NIfTI geometry. Clipping is baseline behavior, not evidence of preserved
  pathology. Verify geometry/coverage/output units before downstream scoring.
- Segmentation and phase-classification models are frozen auxiliary networks;
  they are **not the denoiser** we are replacing. Preserve gradient flow through
  their input when they supervise generation.

## Issues to resolve before paid baseline reproduction

1. `SMILEModel/train.sh` specifies 200,000 updates, L1 denoising, author-private
   data/checkpoint paths, init step 15,000, resume, different auxiliary weights,
   and a 190,000 final loss-stage threshold. It is not the paper's 100,000-update
   schedule. The Python defaults are different again. Do not silently choose
   one and claim paper reproduction. The code invokes cycle denoising in its
   classification stage, so names of thresholds alone do not describe losses.
2. Requirements pin Torch 2.6/torchvision 0.21; INSTALL requests Torch >=2.7.
   Build an isolated, explicit dependency lock and test imports on the selected
   GPU; do not overwrite the working segmentation environment.
3. Root `download_ckpts.sh` contains Markdown code fences and an interactive
   login. Do not execute it as a reliable unattended download script. Select
   needed VAE/denoiser artifacts directly after checking size/access; downloading
   classifier/segmenter is unnecessary for inference-only smoke tests.
4. Root inference paths are relative, then the script changes directories. Use
   resolved absolute input/output/model paths and a new output directory.
5. `testEnhanceCTPipeline.py` uses `unet.config.in_channels` (8 after expansion)
   to prepare the **noisy latent**, although it should be 4 before concatenation.
   The supplied multi-volume entrypoint explicitly supplies four-channel noise,
   which may bypass the issue. Test both supplied-noise and internally-created
   noise paths against installed diffusers; the latter is not launch-ready.
6. Training offers `v_prediction`, but reconstruction and cycle helpers use the
   epsilon-to-clean formula. Keep parameterization explicit; either use verified
   epsilon mode or correct/test both arms before enabling velocity prediction.
7. Volume loader needs >=3 slices; ensure every slice receives overlap weight.
   Source code sets zero weights to one, which could silently turn uncovered
   voxels into -1000 HU. For normal D>=3, D-2 triplets cover all slices; this is
   a guard requirement, not a demonstrated missing-tail bug.
8. Source LICENSE is CC BY-NC-ND. No SMILE source was modified/copied into our
   prototype. Clarify permission before publishing a modified SMILE integration.
   Code availability does not establish checkpoint/data reuse terms.

## Original prototype and executed checks

`conditional_nested_denoiser.py` implements a small, independently written
conditional nested denoiser with continuous full-resolution exits, sinusoidal
time embeddings, residual affine conditioning, and token cross-attention. Dense
and nondense connections are selectable. Default size is **1,202,536 parameters**;
this is a contract fixture, NOT a capacity-matched SD1.5 model. It uses affine
conditioning, **not an implemented adaLN-zero reproduction** of our draft.
No pretrained SMILE/segmentation weight compatibility is claimed.

For exit j, compute only grid nodes i+k<=j; the retained-node count is
(j+1)(j+2)/2. Default exits retain 3 versus 6 nodes. This is neither a FLOP ratio
nor a GPU speedup: node resolutions/channels differ. It preserves each exit,
not the deepest denoiser output. Additional full-resolution exits may increase
training cost; inference depth savings require quality evidence.

`diffusion_contracts.py` is an epsilon/eta=0 DDIM mathematical fixture, not a
production scheduler. Production must reuse the pinned baseline scheduler.
All betas, timesteps, exits and supervision weights are explicit.

Executed CPU tests: Torch **2.10.0+cpu**, one thread, toy tensors only.
`python -m unittest discover -s research/generation_architecture -v` from repo root:
**12 tests passed**, 6.807 seconds reported by unittest (not application timing).
The command was wrapped in a 90-second subprocess timeout.

- Shapes, scalar/batch timesteps, finite head outputs.
- Exact selected-head output **and parameter-gradient parity** versus full graph.
- Dense/nondense, all three exits in a deeper fixture, odd spatial dimensions.
- No unused segmentation-style output head computation.
- Gradients reach noisy latent, source latent, prompt tokens and time embedding
  for every exit; finite optimizer updates.
- Model+optimizer serialization resumes to identical next-update parameters on
  fixed inputs. Full data/RNG/sampler/AMP resume is **not yet verified**.
- Explicit summed versus normalized supervision scale; malformed inputs rejected.
- Known epsilon reconstructs clean latents; deterministic trajectories preserve
  input tensors and RNG; invalid reverse schedules rejected.

## Cheapest controlled execution sequence

1. Prototype GPU smoke passed. Resolve baseline weights, recipe/data and pinned
   dependencies before running actual enhancement.
   Use non-protected public demo data for implementation checks, not our 901-case
   evaluation. A demo without paired labels cannot establish tumor benefit.
2. Original baseline first: fixed case/phase/noise/VAE draws, record all component
   and end-to-end timings plus peak allocated/reserved GPU and host memory.
   Save output HU geometry/coverage, manifests, versions and source commit.
3. Adapt prototype behind the same latent/token/residual interface. Complete
   diffusers serialization/EMA/optimizer integration and GPU dtype/gradient tests.
   Do not pass random U-Net++ outputs to a detector and interpret its result.
4. Cheap depth-cost screen on the actual GPU: all heads, full head only, each
   pruned exit; forward/backward and denoising rollout cost. Kill this direction
   if viable exits save no meaningful **end-to-end** time. No guessed percentage.
5. Small matched training pilot on authorized training/development data only:
   baseline vs full-depth candidate, then deep supervision vs deepest-only.
   Match patient splits, exposure, optimizer updates, precision, sampler, and
   source/phase/noise draws. Pretraining and capacity differences require controls:
   a random nested model versus a pretrained baseline alone is confounded.
6. Only after heads are competent, measure depth/noise error on development data,
   freeze the schedule, and compare deepest/fixed-depth/scheduled/reversed paths
   with identical step count, RNG and quality targets. Also compare fewer-step
   baseline, since saving denoising steps may beat architecture changes.
7. Medical gate: properly aligned reference fidelity, missed/hallucinated lesions,
   fixed-detector lesion sensitivity/false positives and tumor Dice, negatives and
   small tumors included. Labels may supervise training but never condition an
   inference method that assumes unavailable test tumor masks.

## Efficiency without altering the experiment

Profile before optimizing. Cache fixed prompt embeddings; batch triplets within
measured memory limits. Frozen VAE posterior parameters may be cached only when
input transforms are fixed, while retaining stochastic draws as required; caching
one latent draw forever changes training. Auxiliary segmentation/classification,
cycle generation, VAE encoding/decoding and I/O may dominate total training time.
Measure them before asserting a denoiser-only win. Preserve expensive results
and resume state; no unconditional automatic retry of a known broken startup.

## Genuine remaining gates

Preparation follow-up: `BASELINE_PREFLIGHT.md` records the source-specific noise,
coverage, unit, asset and fair-comparison gates. `baseline_contracts.py` adds seven
CPU tests; **19 total tests passed**, 7.557 seconds locally under a 90-second
timeout. These helpers are not yet wired into SMILE's pipeline. The twenty-paper
register now includes targeted method/experiment reading for every selected
family, not full end-to-end readings; an additional MICCAI 2026 medical-acceleration
neighbor is documented. No further GPU allocation for this follow-up.

We still need the PI-intended training recipe/data. On October 8 the user created
their own Hugging Face account, received SMILE access and personally authorized
the cluster's browser login. Authenticated downloads now succeeded; the earlier
401 gate is resolved. Never request tokens in chat or substitute generic SD
weights for the finetuned SMILE reference. The completed prototype
job used approximately 0.00722 charge-equivalent GPU-hours before accounting
rounding, against the authorized 0.25 smoke cap. Full training needs a separate
budget and frozen recipe. No acceptance or quality guarantee.

## October 8 authenticated asset and environment preflight

Selected pinned SMILE/SD1.5 assets downloaded into the isolated cluster home
directory (about 7.2 GiB reported by `du`), not shared project storage. Omitted
optimizer states, alternate denoisers and auxiliary networks. Downloads were
detached and timeout-bounded; SHA256 inventory saved remotely. This inventory
records local content, not an independent authenticity proof.

New `baseline_venv` inherits the read-only cluster Torch stack, disables user-site
for checks and installs diffusion dependencies locally only. No segmentation
environment or shared module was edited. CPU preflight passed actual lazy model
imports and finetuned checkpoint-header validation: 686 tensors, conv_in
[320,8,3,3], four output latent channels and width-768 conditioning. Torch
2.10.0+cu129, torchvision 0.25.0+cu129, diffusers 0.35.1, transformers 4.56.1,
accelerate 1.10.1, safetensors 0.7.0, huggingface_hub 0.35.3.
See `asset_preflight_20261008.json` for observed package paths and scope.

**Not yet done:** strict full weight loading, CT enhancement, production source
imports, output coverage/geometry verification, matched generator training or
clinical evaluation. No additional GPU job was submitted for this staging.
The new environment is an inference-core preparation, NOT a full SMILE training
requirements reproduction. Full training remains gated by paired training/dev
data, frozen recipe, agreed quality margins and a separate compute budget.

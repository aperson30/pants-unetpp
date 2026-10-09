# Baseline preparation and experiment gate — October 8, 2026

## What is actually ready

Original CPU fixtures in `baseline_contracts.py`, **19 total tests passed** in
7.557 seconds on local Torch 2.10.0+cpu, under a 90-second subprocess timeout.
Seven new tests cover HU conversion, latent channels, complete triplet assembly,
invalid/duplicate inputs, geometry, noise interpolation and checkpoint headers.
These are tested helpers, **not integrated production safeguards**. No weights
loaded, dependencies installed, CT inputs processed or extra GPU allocation.

`inspect_safetensors` reads only bounded JSON headers and checks shapes/byte
ranges. It does not execute pickle, allocate model tensors or establish payload
integrity/authenticity. Approved-download hashes and strict actual model loading
remain required. Zero-sized/unsupported dtype edge cases are not a general
safetensors implementation; production loading uses the maintained library.

## Release/source issues to freeze before inference

Source: [SMILE](https://github.com/MrGiovanni/SMILE), audited revision
`23f5a28fe25ed0472024b688e19a79ce119c4833`, especially
`SMILEModel/inference_mul.py` and `testEnhanceCTPipeline.py`.

1. **Noisy/source/residual contract is 4/4/4 channels.** Concatenate noisy and
   source only at the denoiser, producing eight channels. Internally prepared
   pipeline noise may use the denoiser's eight-channel config incorrectly;
   exercise both explicit-noise and default-noise paths after installing diffusers.
2. **Original noise creation depends on original scan dimensions.** The entrypoint
   creates noise at H/8 x W/8, but images are resized to 512x512. Non-512 scans
   can trigger bilinear resizing to VAE latent resolution. The deterministic CPU
   fixture proves this reduces variance and introduces neighboring correlation.
   This does not prove a clinical failure. Preserve and document released behavior
   for reproduction; direct latent-size white noise is a separately labeled control,
   not an unannounced correction applied to only one architecture.
3. **Seed all random draws, not just the pipeline generator.** Source VAE posterior
   sampling and base-noise creation precede that generator. Fix explicit per-case
   draws shared across comparison arms. Posterior mean is not the same experiment.
4. **Volume assembly must fail if coverage is missing.** Normal D>=3 and all D-2
   starts cover every slice. Do not replace missing weights by one and write a
   convincing but incomplete volume. Reject duplicate starts; preserve HWD order.
5. **Inspect interpolation overshoot and units.** Released code resizes decoded
   [0,1] images using cubic interpolation before converting to HU/int16. Cubic
   overshoot is possible; report it, don't silently add clipping in one arm.
   Preserve float HU for checks, then explicitly record baseline integer conversion.
6. **Paths/cases must resolve before starting the GPU.** Source fallback forms
   a filename without verifying it exists; an empty case selection exits success.
   Wrapper must reject absent/empty inputs, output-input overlap and pre-existing
   output files. No automatic discovery of protected test cases.
7. **Recipe/dependencies remain unverified.** Public training shell != supplied
   paper schedule. Do not invent an authoritative lock before import/runtime tests.
   Source GitHub license and Hugging Face model-card license differ; track which
   terms govern each asset, especially before redistributing modified code/weights.

## Asset manifest to create after authorized access

- SMILE repository/revision, selected denoiser and VAE filenames, byte sizes,
  SHA256, approval/download source; omit optimizer and auxiliary weights for smoke.
- SD1.5 config/tokenizer/text encoder/scheduler assets pinned to one revision.
- Explicit input-case allowlist on non-protected data; source hash/shape/affine,
  phase, preprocessing and slice convention. No inferred patient correspondence.
- Python/OS/CPU/GPU, exact Torch/CUDA/cuDNN/diffusers/transformers/accelerate versions
  and import paths; no dependence on user-site packages or changing shared modules.
- Scheduler class/config, epsilon/v/x0 parameterization, actual model-call count,
  guidance scale, precision, timestep list, VAE scaling and RNG draws/seeds.
- Fresh output directory, output checksums and geometry/units/coverage report;
  completion marker only after checks pass, never merely after files appear.

This manifest is not filled with placeholders and treated as validated. Actual
assets and import checks are blocked until weights/access and dependency setup.

## Minimal matched experimental ladder

| Gate | Comparison | Purpose / stop condition |
| --- | --- | --- |
| 0 | Released SMILE, deterministic approved demo | Correct channels, finite output, HU/geometry/coverage and reproducible draws before any speed claim |
| 1 | Baseline component profile | Determine denoiser share, VAE/text/assembly/I/O cost and available memory; no spec-sheet speed predictions |
| 2 | Matched random-initialized U-Net and nested full-depth | Does the new architecture learn competitively? Pretrained SMILE remains a reference, not the sole fair comparator |
| 3 | Nested deepest-only vs supervised exits; nondense skip ablation | Determine whether heads become usable and whether dense skips help. Nondense nested fixture is NOT a conventional plain U-Net |
| 4 | Full, fixed shallow, scheduled, reversed schedules | Freeze on development data; same scheduler/noise/call count first, then matched end-to-end-time frontier |
| 5 | Fewer-step DDIM and compatible DPM-Solver++/UniPC | Ensure architecture gain beats a strong cheap sampler. Do not apply pixel-space thresholding to unbounded latents |
| 6 | Lesion/negative/slice-consistency gate | Inspect misses/hallucinations and fixed-detector sensitivity/FPs; prettier output or whole-image SSIM is insufficient |

Training: match patient split, source/target phase, tumor exposure, transforms,
optimizer/updates, loss scales, precision and initialization policy. Report
parameters and compute separately; equal updates, equal hours and equal parameter
count are different controls. Match normalized head-loss scale when isolating
deep supervision; paper-literal summed supervision is a separate experiment.

Quality margins, pilot data and full training budget require agreement; don't
select arbitrary tolerances to declare success. No downstream tuning on protected
2x2 test outputs. No automated retries for a known startup failure.

## Cheap-first speed priorities

Profile first. Candidate low-risk work: reuse phase prompt embeddings, batch
triplets within measured memory, fixed-transform VAE posterior-parameter caching
with preserved sampling, and avoid redundant disk work. Verify equivalence and
charge savings before deploying. Cache one stochastic draw permanently only if
the experiment intentionally changes. Approximate feature caches, fewer steps,
distillation and depth schedules are quality experiments, not neutral plumbing.

No inference time or training-hour estimate for the new architecture is justified
by the 1.20M-parameter random-weight GPU fixture. Remaining 0.25 smoke budget is
not a full training authorization.

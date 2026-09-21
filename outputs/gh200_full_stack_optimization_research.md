# GH200 full-stack training optimization audit

Date: 2026-09-20

Scope: reduce total GH200 allocation time for the fixed 1,000-epoch PanTS 2x2 grid without changing
the sampled patches, physical batch size 4, `[64,160,224]` patch, model definitions, losses,
optimizer, learning-rate schedule, augmentation, or number/order of parameter updates. A throughput
or numerical result is not a tumor-quality result; class-28 detection/recall remains the final gate.

## Current measured floor

The deployed stack already includes BF16 autocast, dead-head elimination, complete-loss compilation,
quality-neutral trainer plumbing, validation every five epochs, and cuDNN fixed-shape autotuning.
Real GH200 training-step measurements project the four cells at 72.35 GPU-hours, excluding online
validation, checkpointing, plotting, startup/compilation, and final sliding-window validation.

Measured dead ends must not be reopened without a changed software stack or profiler evidence:

- `channels_last_3d`: 10.5% slower on the real GH200 trainer.
- `max-autotune`: 0.44% faster with 154 seconds of compilation; no-cudagraphs had no gain.
- exhaustive cuDNN plan search: inconsistent and regressed plain U-Net by 2%.
- loader/augmentation/NVMe/DALI work: exposed wait was only 0.03-0.17%.
- fused SGD: about 0.12% of a UNet++ step even when the optimizer call itself was faster.
- activation checkpointing: trades additional recomputation for memory; capacity is not limiting.
- DDP: may reduce elapsed wall time but adds GPU-hours and changes batch-Dice execution details.
- FP8/INT8 and low-bit optimizers: no mature Conv3d training path with a defensible rare-tumor
  accuracy argument, while the optimizer is not a meaningful runtime fraction.

## Ranked candidates

### A1. Compile network plus complete objective as one graph

Current code compiles the network and loss as separate regions. UNet++ then materializes several
full-resolution, 29-class logits at the graph boundary before the separately compiled Dice+CE loss
consumes them. Compiling `loss(network(data), target)` as one callable lets AOTAutograd see the joint
forward/backward graph and gives Inductor fusion and min-cut partitioning a chance to remove boundary
traffic. This directly targets the measured activation-bandwidth bottleneck. It preserves the exact
objective and update sequence.

Expected range before measurement: 2-8% for UNet++, 0-4% for plain U-Net. The range is deliberately
conservative because cuDNN convolution outputs required by backward may still need materialization.

Gate: fresh-process paired timing for all four cells; identical inputs/seeds; compare loss, every
parameter, gradient, and momentum buffer over at least eight updates; reject if any cell regresses by
more than 1% or compilation/memory is operationally unsafe.

**Measured 2026-09-21; rejected.** DeltaAI job 3186346 compared the existing separately compiled
network and loss with a jointly compiled `loss(network(data), target)` callable in fresh processes.
After eight identical synthetic updates, strict state comparison failed for both architectures. For
UNet++, the final loss differed by `2.38e-7`, maximum parameter difference was `7.25e-5`, and maximum
momentum-buffer difference was `4.28e-4`. For plain U-Net, the corresponding values were `5.60e-6`,
`2.29e-4`, and `7.01e-4`. The first paired real-trainer result was also below the prespecified payoff
floor: UNet++ DS-on improved only 0.31%, from 0.43697 to 0.43562 seconds per step, with unchanged
49.335 GiB peak allocated memory. The job was cancelled before redundant cells, after 7m19s
(0.122 GH200-hour). Do not deploy this boundary or spend a long tumor-quality run validating it.

### A2. Profile one actual update with Nsight Systems and PyTorch profiler

The remaining work must be driven by a kernel timeline. DeltaAI explicitly supports GPU-counter jobs
with `--constraint=nvperf`. Capture warmed training updates, not compilation, and attribute time to
Conv3d forward/backward, InstanceNorm, concatenation, compiled loss reductions, gradient clipping,
optimizer, copies, synchronization, and inter-kernel gaps. Also run `TORCH_LOGS=perf_hints,recompiles`
outside the timed region to verify CUDA-graph coverage and absence of unexpected recompiles.

This is diagnostic rather than a speedup, but prevents spending hours on sub-percent components.
Budget: roughly 1-3 minutes of one GH200.

**Measured 2026-09-21.** Jobs 3186419 and 3186466 captured warmed real UNet++ DS-on trainer
updates; the second used CUDA-graph node granularity because Nsight's default graph-level view hides
the replayed convolution nodes. Across three updates, projected GPU work was 443.24 ms/update versus
447.50 ms synchronized wall time. Aggregating all 233 kernel rows gave: convolution kernels 56.24%,
cuDNN NCDHW-to-NDHWC and reverse layout conversions 25.29%, Triton normalization/loss/fused kernels
14.66%, concatenation-bearing kernels 2.45%, multi-tensor optimizer/clipping kernels 0.26%, and other
work 1.11%. Exposed loader time was 0.05%. This confirms that optimizer, loader, and invasive
concatenation rewrites have ceilings too small to prioritize. The layout tax is the main software
soft spot, but the already-measured whole-model `channels_last_3d` regression means it should be
attacked through a newer stack/layout propagation test rather than forcing that memory format on the
current stack. Both profiling jobs together consumed 3m24s, or 0.057 GH200-hour.

### A3. Isolated PyTorch/cuDNN stack A/B

The live venv is PyTorch 2.10.0+cu129 with cuDNN 9.10.2. PyTorch has an open, high-priority silent
BF16 Conv3d correctness issue attributed to cuDNN 9.10.2 for very large spatial tensors. The published
reproducer is much larger than PanTS, so it does not prove this model is affected, but a same-library
comparison cannot detect a shared backend defect. Newer PyTorch builds ship cuDNN 9.17-9.24 and newer
Inductor reduction tuning; PyTorch has also reported substantial Hopper improvements for compiled
normalization/reduction kernels.

Create a separate venv or container overlay; never upgrade the validated live environment in place.
First run a layer-shape Conv3d BF16-versus-FP32 correctness sweep for the actual topology, then the
same four-cell paired benchmark and state comparison. Expected speed range is unknown (0-15%); this
candidate ranks highly because it tests both correctness and performance. Roll back by deleting the
isolated environment only.

**Measured 2026-09-21; throughput passes, quality gate still open.** Same-node job 3186493 first
screened real UNet++ DS-on at 0.43257 s/step on PyTorch 2.10.0/CUDA 12.9/cuDNN 9.10.2 versus
0.42406 s/step on the isolated PyTorch 2.12.0/CUDA 13.0/cuDNN 9.20 module (+1.97%). Job 3186515
then covered all four cells in fresh processes. New-stack changes were: UNet++ DS-on +1.99%, UNet++
DS-off +2.73%, plain U-Net DS-on +0.41%, and plain U-Net DS-off -0.65%. Summing equal-count cell
step times gives a 1.88% grid-level gain, projecting the 72.35 training-step hours to 70.99 hours
(1.36 hours saved). Peak allocation was effectively unchanged.

Controlled eight-update comparisons showed small but nonzero cross-version drift. UNet++ final loss
differed by `1.19e-6`; maximum parameter, gradient, and momentum differences were `7.54e-6`,
`3.53e-5`, and `1.03e-4`. Plain U-Net final loss differed by `2.86e-6`; corresponding maxima were
`3.77e-5`, `2.44e-4`, and `5.62e-4`. The strict comparator passed UNet++ parameters but failed other
groups because near-zero elements exceeded its `2e-5` absolute tolerance. Relative-to-global-max
differences stayed below 0.051%. This is consistent with a backend/kernel change, but it is not a
tumor-quality result. Treat the newer stack as a deployment candidate only; retain rollback to the
validated venv and require the full class-28 detection/recall gate through the delayed-learning
window. The two jobs used 13m10s total (0.219 GH200-hour).

### A4. Quantify online validation, plotting, and checkpoint stalls end to end

Stock nnU-Net performs 250 training and 50 validation iterations each epoch. The existing sparse
trainers preserve every optimizer update and validate every fifth epoch plus the final epoch, but the
72.35-hour projection excludes their actual savings. Time one warmed training epoch, one validation
block, `plot_progress_png`, a periodic checkpoint, and a best checkpoint separately.

If checkpoint stalls are material, prototype asynchronous CPU staging/writing while ensuring only one
save is in flight and joining it before exit or replacement. PyTorch documents asynchronous DCP as a
way to move checkpoint writes off the critical path, but it raises host-memory usage. For nnU-Net's
single-file checkpoint compatibility, a simpler background snapshot may be preferable and must be
crash/reload tested. Plotting can be throttled to real-validation epochs while scalar logs remain
per-epoch. Expected combined gain: 1-8% of total allocation time, dominated by validation; measure it.

### A5. Measure and compile gradient clipping only if the profile justifies it

`clip_grad_norm_` scans every gradient of the roughly 116M-parameter UNet++ each update. PyTorch's
default CUDA path normally chooses foreach kernels, but this pass was not included in the isolated
fused-SGD measurement. Time norm/clipping independently and verify whether clipping activates.
If it is above 1% of a step, compare explicit `foreach=True` and a compiled clipping helper with exact
norm/gradient/update checks. Never remove clipping merely because a short sample did not trigger it.

### A6. DeltaAI placement and accounting guardrails

Use one GPU per job, `--gpu-bind=verbose,closest`, `--cpu-bind=cores`, and `--mem-bind=local`.
Stay within one GH200 allocation unit: DeltaAI documents that excessive CPU memory can charge a
second GH200 and `--exclusive` charges all four GPUs. The current 100 GB / 32 CPU benchmark request is
inside one unit; do not request a whole node. Keep `/projects` as the durable source, and stage only
if a full-epoch trace contradicts the measured 0.03-0.17% loader wait. Expected throughput gain is
0-2%; the larger value is preventing accidental 2x/4x accounting.

### B. Conditional follow-ups after profiling

- Try Inductor reduction tuning only if InstanceNorm or loss reductions are prominent. PyTorch's
  newer normalization work shows that bad reduction launch choices can be severe on Hopper, but this
  is not direct evidence for this exact InstanceNorm3d topology.
- Inspect UNet++ `torch.cat` traffic. If concatenation is more than 10% of GPU time, investigate a
  mathematically equivalent split-weight convolution/accumulation prototype. This is invasive and
  cuDNN efficiency on smaller convolutions may erase the saved copy, so it is not a first-line change.
- Compile more of the training update only after the joint objective test. A full compiled step can
  include backward/clip/SGD, but the epoch-varying polynomial LR can cause guards/recompiles unless
  handled explicitly. Optimizer timing says the ceiling is small.
- Test `torch.inference_mode` and a validation-specific compiled graph only if online or final
  validation is a measured material fraction. Preserve mirroring/TTA and full-volume context.

## Proposed kill-fast sequence and budget

1. Nsight/PyTorch trace: <=0.05 GPU-hour.
2. Joint objective compile across four cells plus state parity: <=0.20 GPU-hour.
3. One real epoch plus one validation/checkpoint instrumentation run: <=0.25 GPU-hour.
4. Isolated newer-stack four-cell A/B and Conv3d correctness sweep: <=0.35 GPU-hour.
5. Profile-directed clipping/reduction follow-up only if justified: <=0.15 GPU-hour.

Total cap: about one GH200-hour. Do not run all arms automatically: stop when a gate fails or the
profile shows the candidate's ceiling is below 1%. No result licenses changing the scientific grid;
only parity-passing runtime changes applied symmetrically can be deployed, followed by the full
1,000-epoch class-28 acceptance check.

## Primary sources

- PyTorch compile end-to-end training:
  https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html
- PyTorch compiler profiling and graph-break interpretation:
  https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
- PyTorch asynchronous checkpointing:
  https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
- PyTorch compiled normalization/reduction performance:
  https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/
- PyTorch BF16 Conv3d/cuDNN 9.10.2 issue:
  https://github.com/pytorch/pytorch/issues/163539
- PyTorch tracker for incomplete CUDA `channels_last_3d` operator support:
  https://github.com/pytorch/pytorch/issues/59168
- PyTorch release/library matrix:
  https://github.com/pytorch/pytorch/blob/main/RELEASE.md
- DeltaAI job binding examples:
  https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/running-jobs.html
- DeltaAI accounting rules:
  https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/job-accounting.html
- DeltaAI GPU profiling support:
  https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/debug-perf.html
- nnU-Net training/validation/checkpoint behavior:
  https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md

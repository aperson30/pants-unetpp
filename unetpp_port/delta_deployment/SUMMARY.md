# Delta calibration and grid launch prep

Measured 2026-09-21 on NCSA Delta (separate NSF ACCESS-CI resource from DeltaAI; pivoted here
because DeltaAI's shared `/projects/bdyo` allocation was at 989/1000GB, 12GB free -- not enough
room for the full ~1.1TB PanTS dataset. Delta's storage is a completely separate pool, 5.3PB free
on `/projects`). Account `bdyo-delta-gpu`, PyTorch 2.12.1+cu130 (Delta's available modules are
`pytorch-conda/2.8` and `pytorch-conda/2.12` -- no 2.10 module exists here, unlike DeltaAI's
validated 2.10/cu129 stack).

Real 10-case matched calibration (same streamed-early-kill technique used for the DeltaAI
calibration), real trainer classes (`nnUNetTrainerUNetPlusPlus`, `nnUNetTrainerBF16`), dead-head-skip
+ bf16 + cuDNN autotuning (via the shared BF16 mixin) + explicitly-applied `reduce-overhead` compile
on the network, 40 measured steps after 8 warmup steps, physical batch 4, real patch [64,160,224].

| GPU | UNet++ s/step | Plain U-Net s/step | Peak mem (UNet++) |
|---|---|---|---|
| A100-SXM4-40GB | 0.6894 | 0.1790 | 35.0 GiB |
| H200 | 0.3168 | 0.0829 | 42.3 GiB |
| (GH200, DeltaAI, for reference) | 0.4518 | 0.1128 | 40.8 GiB |

H200 is ~1.4x faster than GH200 here; A100 is ~1.5-1.6x slower than GH200. At 1000 epochs / 250
updates each, paired training-only projection: H200 ~55.5 GH200-hours total, bottleneck cell (the
UNet++ arms) ~22h wall-clock -- comfortably inside a single 48h job. A100 projects to ~120.6 hours
total, bottleneck ~47.9h wall-clock -- right at the partition's time cap, would almost certainly
need at least one checkpoint/requeue cycle. **Decision: H200 for the real 4-cell grid.**

## Open item not yet reconciled with DeltaAI's own findings

This calibration ran on PyTorch 2.12.1+cu130, not the 2.10/cu129 stack DeltaAI's own numbers are
based on. Per `coordination/CODEX_STATUS.md`'s 2026-09-21 entries, an isolated 2.12/CUDA13/cuDNN9.20
test on DeltaAI showed a small (1.88%) throughput gain over the validated 2.10 stack, but with
measurable numerical drift (UNet++ loss differed by 1.19e-6, max gradient/momentum differences up
to 1.03e-4) -- gated behind the class-28 delayed-onset tumor detection test before being trusted,
not yet deployed. Delta's calibration/planned launch uses 2.12 by necessity (it's what's available
here, not a chosen upgrade), but the same numerical-drift question hasn't been separately checked
for Delta's specific PyTorch/CUDA/cuDNN combination (2.12.1+cu130 -- a different minor version pairing
than DeltaAI's isolated 2.12.0/CUDA13/cuDNN9.20 test). Flagging for review rather than assuming it's
fine by analogy.

## Infrastructure note: Delta job-priority stall

An earlier combined download+convert+preprocess job (single sbatch, `gpuA100x4`, 32 CPUs) sat
PENDING with `Reason=Priority` for 12+ hours straight -- `sprio` showed its priority (626) far below
other jobs actually running/queued on the same partition (3000-5555), driven by this being a new
account with minimal fairshare history, not a resource shortage (`AllocTRES=(null)` the entire time).
Fixed by splitting into a small download+convert job on the `-interactive` partition variant (whose
built-in partition-priority weight of 1000 raised the job's total priority to 1355, comparable to
mid-pack) followed by a preprocess-only job with a reduced CPU footprint (16 vs 32) for better
backfill odds, chained via `--dependency=afterok`. Worth knowing if launching further jobs here.

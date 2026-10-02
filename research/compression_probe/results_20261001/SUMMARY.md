# MSD compression-only pilot — October 1, 2026

## What actually ran

One public MSD Task07 case, pancreas_005, native 512x512x104 CT, 3609 tumor
voxels / 7602.96mm3 (7.60mL). Selected before reconstruction as smallest tumor
burden among four prespecified available cases; NOT a voxel-tiny lesion and
NOT a representative or independent held-out benchmark.

Pinned pretrained MAISI VAE, deterministic posterior MEAN, FP32 including
normalization, original spacing, orientation-only RAS transformation and
center padding to a multiple of four. Whole-volume encode/decode, no tiles,
no diffusion, no training, no segmenter inference. Mirror/input/model hashes
are in provenance.json; third-party mirror equivalence to official MSD bytes
not independently established. No claim of public-model training-set exclusion.

## Actual time/memory, one DeltaAI GH200 120GB

- First job 3289813: failed host-memory limit, elapsed 103s, requested 32G,
  MaxRSS 33839296K. No reconstruction. No auto-retry and no extended allocation.
- Retry 3289833: COMPLETED, elapsed 204s, requested 96G, still exactly one GPU
  and interactive billing=2000. Slurm's sampled MaxRSS 22960192K missed the
  in-process high-water mark; Python reported 37327232KiB (~35.60GiB).
- Actual forward/copy timing: 106.892s. GPU peak allocated 41973353984 bytes
  (~39.09GiB); reserved 41987080192 bytes. Total internal run 175.28s.
- Both job allocations together: 307/3600 = 0.08528 physical GPU-hours.
  Interactive partition 2x charge => ~0.17056 allocation-equivalent hours.
  This is computed from accounting/charge rule, not a posted balance debit.
- Initial total cap 0.25 hours respected even including failed attempt.

## Measurements and cautious interpretation

Preprocessing-only control: tumor HU MAE and boundary-band MAE both zero;
tumor-to-local-ring mean contrast unchanged. Whole-image HU MAE 7.09 because
official intensity clipping changes values outside [-1000,1000].

Reconstruction vs control:

- Tumor HU MAE: 36.21; boundary-band HU MAE: 36.09.
- Mean local contrast: 43.67 -> 43.32HU (ratio 0.99188).
- Local CNR: 0.56165 -> 0.54443 (~3.07% decrease).
- Whole-image HU MAE: 63.08.

The saved fixed-window slice visibly looks smoother after reconstruction.
Average contrast is approximately preserved, but region averages can hide
spatial/boundary loss or shared intensity shifts. The ring is surrounding tissue,
not a pure healthy-pancreas reference. No clinical realism, detection recall,
boundary-location accuracy or quality-equivalence conclusion follows.

The red contour is ORIGINAL ground truth, not a prediction or proof that the
reconstructed image contains a realistic lesion. One maximum-lesion-area slice
does not establish three-dimensional lesion fidelity. No lesion-erasure claim.

## Next decision

Pipeline works with realistic host-memory request. Do not build an adaptive-depth
method from this one example. First identify genuinely small lesions and repeat
paired compression diagnostics, including multiple slices / blinded inspection
and an appropriate independent detection check if warranted. Preserve preprocessing
control. Count posterior-sampling, setup/evaluation costs in any larger campaign.

PanTS training/evaluation artifacts and environments remained untouched.

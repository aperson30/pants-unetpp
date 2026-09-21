# GH200 fixed-shape cuDNN autotuner screen

Measured 2026-09-20 on NCSA DeltaAI GH200, PyTorch 2.10.0+cu129, cuDNN 9.10.2, physical batch 4,
patch `[64,160,224]`, BF16 autocast, complete compiled Dice+CE loss, SGD/Nesterov, gradient clipping,
and the real 9-case PanTS calibration dataloader. Every arm ran in a fresh process through the
actual trainer class. This is a throughput/numerical gate, not tumor-quality evidence.

## Adopted: `torch.backends.cudnn.benchmark = True`

Job 3185187, 30 measured steps per arm:

| Trainer | Off (s/update) | On (s/update) | Time reduction | Peak allocation off -> on |
|---|---:|---:|---:|---:|
| UNet++ DS-on | 0.457499 | 0.424652 | 7.18% | 40.77 -> 49.34 GiB |
| UNet++ DS-off | 0.439721 | 0.410493 | 6.65% | 34.52 -> 43.08 GiB |
| Plain U-Net DS-on | 0.112168 | 0.111339 | 0.74% | 10.96 -> 10.98 GiB |
| Plain U-Net DS-off | 0.096867 | 0.095378 | 1.54% | 10.82 -> 10.82 GiB |

At 250 updates x 1,000 epochs per cell, the paired training-only projection falls from 76.82 to
72.35 GH200-hours, saving 4.47 hours (5.82%). Validation/checkpoint/startup time is excluded.

The real pipeline probe measured exposed augmentation/loader wait at only 0.03-0.17% across the
four cells. Worker tuning, DALI, file-open caching, and NVMe staging therefore have no meaningful
ceiling in the measured stack and were not pursued.

## Numerical gate

After eight identical full-patch BF16 updates from the same seed/input/target, complete-state
comparison covered 116,319,249 model elements and 45,562,164 gradient and momentum elements:

- final loss absolute difference: `7.63e-6`;
- maximum parameter difference: `4.91e-6` (`4.84e-6` of the maximum reference magnitude);
- maximum gradient difference: `6.10e-5` (`1.24e-4` of the maximum reference magnitude);
- maximum momentum difference: `9.35e-5` (`2.38e-5` of the maximum reference magnitude).

The model tensors pass `rtol=5e-3, atol=2e-5`; gradients/momentum fail that strict elementwise test
only around near-zero entries. Kernel reduction order is not bit-identical. Apply the flag to every
grid cell and retain the full 1,000-epoch class-28 tumor recall/detection acceptance gate.

## Rejected

- `max-autotune`: 0.45839 versus 0.46041 s/update (0.44%) while first-step compilation rose from
  11.2 to 154.3 seconds. The search selected cuDNN over every Triton 3D-convolution candidate.
- `max-autotune-no-cudagraphs`: 0.46063 s/update, no improvement, 62.8-second first step.
- `channels_last_3d` plus cuDNN autotuning: 0.47688 versus 0.43149 s/update, 10.5% slower, with
  longer compilation and more reserved memory.
- `cudnn.benchmark_limit=0`: synthetic timing suggested about 1%, but the paired real-trainer run
  was inconsistent across architectures/runs, regressed plain U-Net by 2%, and added minutes of
  search. Keep PyTorch 2.10's default limit of 10.

## Resource use

Production-queue benchmark jobs 3185136, 3185187, 3185246, 3185285, and 3185357 consumed about
0.44 GH200-hour total. A replaced job (3185131) ran for 19 seconds before cancellation, about
0.0053 GH200-hour. All jobs used one GPU, hard per-arm timeouts, isolated output directories, and
left the calibration dataset unchanged.

Primary references:

- https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- https://docs.pytorch.org/docs/stable/backends.html

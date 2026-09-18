# Conservative PGPS patch curve on GB10

Measured 2026-09-17 on `bdmap2`, NVIDIA GB10, PyTorch 2.10.0+cu130. Every point used the real
PanTS planner topology, physical batch size 4, eager FP16 autocast, SGD with momentum, five warm-up
steps and twenty measured forward/backward/update steps. Every point ran in a fresh process, and
host memory returned to approximately 116 GiB available between points.

The patch path deliberately keeps the complete 64-slice axis for anatomical context and grows one
in-plane axis at a time. These are throughput measurements, not evidence that tumour detection is
preserved.

| Patch | Relative voxels | UNet++ s/step | UNet++ alloc/reserved GiB | Plain U-Net s/step | Plain alloc/reserved GiB |
|---|---:|---:|---:|---:|---:|
| 64x128x160 | 57.1% | 2.279 | 22.69 / 29.25 | 0.565 | 6.34 / 9.64 |
| 64x128x192 | 68.6% | 2.816 | 27.14 / 35.05 | 0.673 | 7.55 / 11.54 |
| 64x160x192 | 85.7% | 3.588 | 33.82 / 43.74 | 0.844 | 9.37 / 14.37 |
| 64x160x224 | 100% | 4.232 | 39.38 / 50.99 | 0.988 | 10.88 / 16.62 |

## Conservative schedule projection

A simple schedule that reaches the full patch by epoch 450 and retains it for the final 550 epochs:

- epochs 0-149: 64x128x160
- epochs 150-299: 64x128x192
- epochs 300-449: 64x160x192
- epochs 450-999: 64x160x224

Using the measured eager step times, its weighted average is:

- UNet++: 3.630 instead of 4.232 s/step, 14.2% less step time or a 1.166x speedup.
- Plain U-Net: 0.856 instead of 0.988 s/step, 13.4% less step time or a 1.155x speedup.

This projection isolates patch-size effects. It must not be multiplied blindly into the existing
compiled timing without measuring the same shapes under the final compile configuration. Validation
iterations and dataloader rebuilds also need to be accounted for in a trainer-level timing.

## Decision

The curve supports implementing a conservative PGPS trainer for one long validation run. It does
not justify applying PGPS to the comparison grid yet. The change remains tumour-risky until at least
epoch 500 and must be judged by tumour detection/recall and whole-lesion misses, not aggregate Dice.

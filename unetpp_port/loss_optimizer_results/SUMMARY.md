# Full-loss compilation and fused-SGD benchmark (GB10)

Measured on `bdmap2` (NVIDIA GB10, PyTorch 2.10.0+cu130), physical batch 4, real PanTS patch
`[64,160,224]`, 29 output channels, BF16 autocast, and the real planned network topology. Each
timing arm ran in a fresh process with a 15-minute timeout and a 70 GiB host-memory launch floor.
The synthetic target was overwhelmingly background with a tiny class-28 focus; this is a
throughput/parity benchmark, not tumour-accuracy evidence.

## Result

| Architecture | Current loss path | Complete compiled loss | Time reduction | Speedup |
|---|---:|---:|---:|---:|
| UNet++ DS-on | 3.6995 s/step | 3.5105 s/step | 5.11% | 1.054x |
| Plain U-Net DS-on | 0.8562 s/step | 0.8308 s/step | 2.96% | 1.031x |

The current path compiles only `MemoryEfficientSoftDiceLoss`; the new path compiles the complete
Dice + cross-entropy + `DeepSupervisionWrapper` after branch weights are applied. nnU-Net's source
left complete loss compilation disabled because PyTorch 2.2.2 crashed on CE. That failure did not
reproduce on this project's PyTorch 2.10.0 environment.

UNet++ peak allocated memory was unchanged (40.813 vs 40.811 GiB), while peak reserved memory fell
from 56.564 to 50.533 GiB. Plain U-Net fell from 11.664 to 11.096 GiB allocated and 18.568 to
15.531 GiB reserved. Process exit returned the node to 116 GiB `MemAvailable` after every arm.

## Numerical gates

The complete compiled loss was compared against the exact current path using four full-resolution
UNet++ branches, all 29 classes, and a tiny class-28 target:

- FP32: identical scalar loss; maximum gradient difference `2.73e-12`.
- BF16 autocast: scalar loss difference `1.91e-6`; maximum gradient difference `1.19e-7`
  (`0.645%` of the maximum reference-gradient magnitude).
- Trainer contract test confirmed the final wrapper, not only Dice, is compiled for plain U-Net,
  default UNet++, and paper UNet++ while preserving their exact branch weights.
- The full architecture smoke suite passed after installation into editable nnU-Net.

## Fused SGD

PyTorch's default CUDA SGD already selects its foreach implementation. Explicit `fused=True`
reduced the isolated optimizer call from 9.236 ms to 4.645 ms (1.99x), but the absolute saving is
only 4.59 ms, about 0.12% of a current UNet++ step. Five Nesterov updates differed by at most
`2.38e-7` in parameters and `2.38e-6` in momentum buffers. It remains benchmark-only: the tiny
end-to-end ceiling does not justify changing the optimizer path for multi-day comparison runs.

## Compile-mode check

The real trainer's bare `torch.compile(model)` measured 3.7133 s/step versus 3.6995 s/step for
`mode="reduce-overhead"` with the same current loss, only a 0.37% difference. This is within the
size of a low-value tuning change, so network compile mode is left unchanged. The complete-loss
compile result above is the meaningful new lever.

Raw logs in this directory contain the warning/header and JSON result for every arm.

## GB10/PGPS interaction

`reduce-overhead` may retain a CUDA-graph workspace for each compiled shape. That is safe for the
fixed-patch grid, and every isolated benchmark returned the node to 116 GiB `MemAvailable`. It is
not safe to assume the same for PGPS if four patch sizes are visited in one Python process: compiled
graphs/workspaces may remain referenced across stage changes and compound in the GB10's unified
memory pool. If PGPS is promoted to a long run, make each patch-size stage a separate process that
checkpoints and resumes the exact optimizer/scheduler state. Process exit is the verified memory
reclamation boundary; `empty_cache()` alone is not an adequate guarantee here.

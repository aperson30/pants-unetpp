# Quality-neutral plumbing microbenchmark — DeltaAI GH200

Measured 2026-09-20 on one NVIDIA GH200 120GB with PyTorch 2.10.0+cu129. The Slurm job requested
one GPU, four CPU cores, and 32GB RAM and completed in 15 seconds (job 3180511). Shape was the real
PanTS training shape: batch 4, patch `[64, 160, 224]`, 29 classes.

| Component | Stock path | Optimized path | Component speedup | Peak allocated |
|---|---:|---:|---:|---:|
| Exclusive-label validation counts | 6.989 ms | 0.715 ms | 9.78x | 2.121 -> 0.150 GiB |
| Four identical UNet++ target transfers | 0.256 ms | 0.060 ms | 4.24x | 0.070 -> 0.018 GiB |

The validation comparison reproduces nnU-Net's dense prediction/target expansion and TP/FP/FN/TN
arithmetic versus exact label `bincount` arithmetic. The target comparison transfers four distinct
full-resolution int16 tensors versus one tensor referenced four times.

These are component timings, **not an epoch-speed claim**. Validation is already sparse and the
absolute savings are milliseconds per applicable iteration. The main value of the count path is
removing about 1.97 GiB of transient allocation. End-to-end trainer timing still needs the complete
DeltaAI nnU-Net environment and real-data calibration.

The failed predecessor job 3180505 consumed four seconds because `/tmp` was node-local and the
compute node could not see the login-node script. No data was touched. Total GPU allocation used by
both attempts was about 19 seconds (0.0053 GPU-hour).

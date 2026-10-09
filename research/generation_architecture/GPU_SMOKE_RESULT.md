# Original architecture GPU smoke — October 8, 2026

PASS: DeltaAI job **3344849**, GH200 node gh021, COMPLETED/exit 0,
13 seconds allocation elapsed. Random-weight original prototype only: no SMILE
checkpoint, CT enhancement, trained quality or lesion preservation result.
Protected PanTS evaluation was untouched.

Request: one GPU, two CPUs, 8 GB RAM, two minutes, no requeue. Held-job preflight
confirmed billing=2000 before release. Estimated charge-equivalent usage:
13 x 2 / 3600 = **0.007222 GPU-hours**. Posted rounding unverified; not a receipt.
Authorized cap: 0.25 charged GPU-hours for smoke testing.

Torch 2.10.0+cu129, CUDA 12.9, cuDNN 91002, aarch64, GH200 120GB.
Existing Python used with user-site/PYTHONPATH isolation; no environment install.
Fixture: 1,202,536 parameters, batch 1, four-channel noisy/source latents 64x64,
77x768 prompt tokens. Much smaller than Stable Diffusion.

| FP32 forward | ms | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: | ---: |
| Shallow | 2.7291 | 45,766,144 | 67,108,864 |
| Deepest | 4.8709 | 47,142,400 | 67,108,864 |
| All heads | 4.9251 | 47,142,400 | 67,108,864 |

Shallow saves about 44% of this fixture's forward time. Selecting a different
head changes the prediction: **not quality-neutral pruning of the deepest
output**, and not an end-to-end trained generation speedup.

All 12 CPU contracts passed on cluster (unittest 2.157 seconds). GPU selected-head
output parity and finite FP32/BF16 gradients passed. Full parameter-gradient
parity is covered by CPU contracts, not separately proven across GPU dtypes.
Random-target losses: FP32 1.08953154, BF16 1.08950818; not accuracy metrics.
Probe elapsed 5.953 seconds; allocation elapsed includes startup.
Raw stdout: `probe_3344849.log`. Remote isolated directory:
`/u/asanjeev/generation_architecture_20261008_v1`.

Baseline enhancement remains blocked by approved weights. Continue CPU/source
work without spending GPU time waiting on unavailable assets.

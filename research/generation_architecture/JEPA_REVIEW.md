# JEPA relevance — October 8, 2026

## Decision

Keep the PI-directed diffusion architecture experiment. A pretrained feature
teacher is a plausible later training aid, not an established CT speedup or a
novel contribution by itself. No JEPA job, teacher download or integration run.

## Primary sources

- [I-JEPA, CVPR 2023](https://arxiv.org/abs/2301.08243): non-generative prediction
  of image-block embeddings. It does not directly produce pixel/HU output.
- [D-JEPA, ICLR 2025](https://arxiv.org/abs/2410.03755): generative extensions
  combining autoregressive prediction and per-token diffusion already exist.
- [REPA, ICLR 2025](https://arxiv.org/abs/2410.06940): aligns denoiser features
  with frozen visual features. Its ImageNet SiT-L/2 ablation already includes
  I-JEPA-H; DINOv2 did better there. No automatic transfer of these results to CT.
- [U-REPA, NeurIPS 2025](https://arxiv.org/abs/2503.18414): alignment in U-shaped
  diffusion models, with layer selection, resolution mapping and relational loss.
  Targeted method/experiment reading does not establish savings on our CNN port.
  Hardware not verified in accessed text; no hardware-specific speed claim.
- [iREPA official code](https://github.com/End2End-Diffusion/iREPA): emphasizes
  spatial structure via projection/normalization. Another close neighbor, not
  evidence of tumor protection. This search is not exhaustive novelty clearance.

## Cheapest useful future test (not submitted)

First establish competent plain/nested generators on the same approved split.
Then compare no alignment against a frozen domain-relevant teacher with explicit
spatial alignment. Do not assume a natural-image JEPA teacher is best. Keep data,
precision, exposure, optimizer and loss-scale controls fixed; count teacher/cache
cost in total time to an agreed generation and lesion-quality target. Steps are
not GPU-hours. Feature caching requires fixed transforms. Report any teacher
pretraining cost. Adding an objective is a separate ablation, not neutral plumbing.

Risk hypothesis: semantic features may ignore tiny HU/texture differences that
matter to tumors. Require small-lesion and negative-case checks with a frozen
detector; feature agreement alone cannot certify preservation. Training labels
must never become an unavailable inference condition. No CT gain demonstrated.

## Executed progress

Job 3345369 passed: GH200, 35s allocation, exit0. Actual SMILE weights loaded,
synthetic two-step denoising and finite decode passed. Raw log saved. Estimated
charge 0.019444h; with prior 13s prototype smoke, cumulative estimate 0.026667h
before provider rounding. Not whole-volume enhancement or clinical validation.

Added an original conventional single-decoder control using the same conditioning
blocks as the nested fixture. Equal widths != equal parameter count; neither is
a capacity-matched SD1.5. All 22 CPU tests passed in 4.320s under a 90s timeout.

Public center-triplet job 3345403 released after held-job checks: 1GPU/3min,
no requeue, billing2000, maximum extra estimated0.1h. Last pending. Uses inspected
public input, 200 DDIM steps, phase prompt and CFG7.5; not released overlapping
volume inference, paired evaluation or tumor preservation. No protected data.

# PI meeting: generation architecture direction

Recorded October 8, 2026 from user-confirmed PI meeting guidance. Supersedes treating independent segmentation/RL as the primary assignment. No GPU jobs submitted or protected evaluation modified.

## Assignment

Architecture design for faster image generation. Application: non-contrast CT -> synthetic contrast enhancement -> tumor segmentation. Adapt the enhancement diffusion denoiser from U-Net to U-Net++; existing segmentation weights are not automatically a denoiser. First implementation may fail; use controlled iterations. Tumor-positive exposure must not rely blindly on random selection. Retain negatives/context and match exposure across arms; no ratio was specified. Labels used for training/evaluation are not assumed available at deployment.

## Proposed cheap-first sequence, not completed experiments

1. Pin intended baseline code, training availability, checkpoint, dependencies, source/phase conditioning, output units/geometry, permitted data and compute budget.
2. Isolated image-in/image-out baseline smoke demonstration on non-protected data. Measure total time including preprocessing, autoencoder and postprocessing. A demo does not prove efficacy.
3. Separate conditional U-Net++ denoiser: verify timestep/source conditioning, output parameterization, shapes, finite gradients/updates, each branch and retained-head pruning parity before training.
4. Compare full-depth baseline/candidate with matched data, training updates, sampler and exposure; include capacity/cost controls where necessary. No assumed speedup.
5. Test supervision and depth-versus-noise behavior before scheduled-depth inference. Compare fixed depths and matched-cost reversed/alternative schedules. Exact retained-head computation is not full-denoiser output equivalence.
6. Measure enhancement fidelity with appropriately aligned references and fixed-segmenter tumor Dice, lesion sensitivity and false positives. Include missed tumors and negative scans. Do not tune protected test outcomes or treat appearance/confidence as evidence of cancer preservation.

## Assets and limits

Local search found segmentation code and prior generation probes, not a verified ready-to-train CT U-Net++ enhancement pipeline. Public author repo https://github.com/MrGiovanni/SMILE advertises enhancement checkpoints, demo and inference; CTVerse is request-based. This is a baseline candidate, not confirmation of PI-intended stack, available training code or successful loading. Inspect scripts before execution; do not change an existing cluster trainer environment.

The CVPR draft describes depth-scheduled generation, starts with CIFAR-10 and explicitly has no results. Reconcile initial demonstration with medical application. Merely swapping U-Net for U-Net++ is not sufficient novelty; need a meaningful measured quality-speed advantage and mechanism. No acceptance probabilities or strong-paper guarantees.

Frozen PanTS 2x2 remains separate. Older screening-budget permission is not assumed to cover new full enhancement training. Source/CPU preparation can begin now; require baseline training entrypoint, permitted data and capped GPU smoke budget before submission.

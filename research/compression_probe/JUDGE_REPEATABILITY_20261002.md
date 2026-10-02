# Locked same-input repeatability diagnostic

Motivation:3296871 stopped at historical native replay max probability drift
>1e-4. Exact drift wasn't logged. Do not change that failed gate or silently
replace its evidence. Geometry matched;4/4164942 segmentation labels differed.

Source audit: nnUNetPredictor constructor enables cudnn.benchmark=True on CUDA;
sliding-window inference uses autocast. No other benchmark/deterministic/seed
override found in the installed predictor source. Autotuning is a plausible
inter-run numerical confound, not a proven root cause.
PyTorch2.10 documents that benchmarking noise can change algorithm selection,
and that disabling benchmarking alone doesn't guarantee deterministic ops:
https://docs.pytorch.org/docs/2.10/notes/randomness.html

Four full official stage-2 predictions of the SAME original100430 image in the
SAME saved publisher-native crop. No VAE, reconstructed-image comparison,
new patient, training, changed TTA, GT-selected crop, or tuned clinical cutoff.
GT label1 is only used to summarize tumor-local differences after predictions.

1. Publisher default policy: benchmark=True, deterministic=False, two calls.
2. Diagnostic policy: benchmark=False, cudnn.deterministic=True,
   torch.use_deterministic_algorithms(True,warn_only=False), two calls.
Set CUBLAS_WORKSPACE_CONFIG=:4096:8 BEFORE torch/CUDA startup for both policies.
This shared environment setting and seeded process mean the default arm is
not an exact recreation of every historical execution condition. No claim of
cross-process/cross-device reproducibility from one same-process diagnostic.

Freeze weights, native pixels/window, preprocessing, AMP/mirroring and export.
Log actual backend flags before/after every call; fail if they change within
prediction. Compare global/tumor max/mean drift, signed tumor score change,
each call against historical map, and deterministic-vs-default drift.
Keep max1e-4 engineering gate unchanged and explicitly distinct from medical
safety. Save per-call scalar records before moving on; unsupported deterministic
operations raise and preserve partial failure, no warn-only fallback/retry.

Decision: if same-input drift itself is material, fix the measurement tool
before new scientific comparisons. If a stable diagnostic policy emerges,
prepare a SEPARATE same-process image-by-window study, with fresh native
controls/repeats; do not claim the historical replay gate now passed. Even
zero drift would not prove tumor-quality loss, information erasure, or novelty.

One GPU/2CPU/32G, hard3min,150s process timeout,no-requeue. Max0.1
charge-equivalentGPUh at billing2000. Prior estimate0.876389/2,
worst-case0.976389/2, actual posted debit unknown. Submit held, inspect fields
and frozen hashes before release. No automatic retry or main47320183change.

3296939 released after held/source/resource checks; last PENDING.
Four CPU comparison/geometry tests passed. Frozen source manifests retained.

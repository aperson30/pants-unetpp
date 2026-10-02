# Fixed-window mechanism control (posthoc, not clinical recall)

Question: does the three-slice change in the publisher's pancreatic window
explain the lower frozen-detector tumor probability in completed patient100430?
This question arose AFTER the replication; label it a posthoc diagnostic,
not a prespecified clinical hypothesis or independent replication.

Use the original image and saved whole-volume MAISI posterior-mean reconstruction
from3296011. Use both frozen original/reconstructed stage-1 crop bounds.
Run all four image x window combinations through unchanged official fold4
stage-2 detector, preprocessing, mirroring and candidate extraction. No VAE
rerun, new patient, trainable update, GT-selected window, threshold tuning or
change to the main PanTS experiment. GT label1 is scoring-only here (NOT PanTS28).

First replay each image in its original window. Maximum raw probability-map
drift must be <=1e-4 against saved output, or stop interpreting cross-window
results. This is an engineering reproducibility tolerance, NOT medical safety.
CPU crop tests check exact voxel values, physical origin, spacing, direction,
out-of-range rejection and shifted-grid rejection before GPU submission.

Then compare the reconstruction/native tumor probability ratio WITHIN each
fixed window; separately report window effect WITHIN each image. Report all
four outputs and retain original misses. If a decrease remains in both fixed
windows, window change alone cannot explain it on this case. If it disappears
or reverses, prioritize the localization/preprocessing interaction rather than
claim tumor erasure. Mixed responses are reported, not tuned away.
Even consistent scores do NOT establish irreversible information loss,
clinical sensitivity, external generalization or novelty. Frozen-model
domain shift and posterior-mean vs sampled latents remain unresolved.

This control intentionally holds a detector's existing window constant; it is
not a proposal to reduce spatial context in training or rescue a large VAE case.
The larger locked patient remains resource-blocked, with no replacement.

One scheduled GPU,2CPU,32G host,hard3-minute limit,150-second process timeout,
no requeue or automatic retry. At billing2000, maximum charge-equivalent0.1h;
prior estimate0.854722/2h, worst-case0.954722/2h. Posted debit unverified.
Submit held, check requested resources/source hashes, then release once.
Keep all images/maps private; commit only code and scalar summaries.

Supporting caution: domain shift is a documented failure mode for medical
image models; it is an alternative explanation, not evidence that it caused
our observed score changes.
https://arxiv.org/abs/2102.09508

## Submission

3296871 released after held resource/source inspection; last state PENDING.
Two CPU geometry tests pass. Missing transitive staging helpers were caught
and staged before submission (zero GPU cost). Source checksums frozen remotely.
No automatic continuation. Prior commits2b3284a/6fd2d1f independently verified
on origin/main; earlier reported push blocker is resolved.

## Outcome: replay gate failed; no cross-window evidence

3296871 FAILED1:0 after39 allocation seconds at billing2000:
0.021667charge-equivalentGPUh, cumulative estimate0.876389/2.
Native own-window maximum probability drift exceeded locked1e-4 before
cross-window inference. Exact drift was not logged in this version; do NOT
invent it or change the threshold after seeing the failure. No completion.json.

CPU audit of saved native segmentations: equal shape, spacing, origin and
direction;4 of4164942 labels differ. This does NOT bound probability drift.
Scalar failure_audit is preserved; private maps/images remain remote.
Copied detector helper source is identical to the producer's.

Installed nnUNetPredictor.__init__ sets cudnn.benchmark=True on CUDA,
overriding the harness's earlier False. Autotuning/algorithm and execution
history is a plausible replay confound, NOT a diagnosed cause. The paired
producer did this too; original score decreases are not automatically
invalidated, but inter-run reproducibility remains unquantified.

Local follow-up code now writes failed replay drift/metrics/backend flags
BEFORE raising. This patch has NOT been GPU-validated or resubmitted.
Keep completed observations separate from this failed control. No retry or
protected evaluation change. Next: audit actual runtime policy and establish
same-process repeatability before any new cross-window/posterior claim; don't
spend on a larger cohort or remedy training while this gate is unresolved.

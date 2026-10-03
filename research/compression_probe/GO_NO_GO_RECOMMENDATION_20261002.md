# Decision: stop new training; retain a narrow, conditional audit direction

2026-10-02. End-of-screen recommendation for the PI, not a clinical result.

## Bottom line

**NO-GO for spending on a new lesion-preserving VAE/diffusion method now.**
**Conditional GO for inexpensive expert review of already completed images.**
We found an interesting, reproducible detector-response phenomenon, but
neither a demonstrated clinical failure nor a sufficiently distinct remedy.
Calling this an S-tier idea or an established training speedup is unjustified.
Do not spend the unused GPU budget merely to find a favorable judge or seed.

The strongest supported research question is whether *posterior- and
judge-robust, lesion-specific fidelity audits* can serve as a useful acceptance
gate before investing in medical diffusion efficiency changes. It is a
question to validate, not a new benchmark/method already established here.
It remains PI-aligned with preserving tiny tumors while making generation
cheaper, but our probes did NOT yet demonstrate a training acceleration.

## Decisive additional experiment: second frozen network

Job3297419 completed0:0. Reused the same100226 native/clip/mean/seeds0,1,2
images; no new autoencoder inference, training, GT input or input crops.
Public DiffTumor U-Net, exact strict weights, FP32 whole-image96cube sliding
windows, Gaussian .75 overlap, RAS/1mm published preprocessing. Output logits
accumulated on CPU, restored to native grid before softmax. This is expressly
a **raw-head diagnostic**, NOT official organ-masked DiffTumor detection.

All six arms repeated exactly across all3channels; argmax maps identical
on repetition. Saved-map hashes and scalar metrics verified separately on
CPU. Ground truth PANORAMAlabel1 used only in that CPU scorer; tumor output
is DiffTumorchannel2. All1756 annotated tumor voxels included; full context.

| Arm | Mean probability on annotated tumor | Fraction of native | Tumor argmax voxels within annotation |
|---|---:|---:|---:|
| Native | .00329347 | 1.000 | 0 |
| Clip control | .00329347 | 1.000 | 0 |
| Posterior mean | .00098717 | .300 | 0 |
| Seed0 | .00132893 | .404 | 0 |
| Seed1 | .00089546 | .272 | 0 |
| Seed2 | .00552814 | 1.679 | 0 |

**The predeclared adequacy floor was .01; native is below it.** Therefore
the study's predeclared decision is NO_GO_CURRENT_ROUTE, not a second-judge
confirmation of loss. Even without that floor, seed2's increased score
contradicts a uniform score-suppression narrative. It does NOT prove improved
clinical visibility, and should not be cherry-picked as a successful remedy.

Native/control tumor-map maximum difference2.42e-8; numerical repetition
drift0. Global native tumor probability maximum .99991 was elsewhere, while
no argmax tumor voxel overlapped the known lesion. Thus patient-wide maxima
cannot substitute for scoring the actual lesion. Off-lesion predictions are
not adjudicated clinical false positives here.

Prior PANORAMA results are NOT discarded: mean/seed0/seed1/seed2 retained
13.80/10.80/47.53/35.29% of its native tumor mean on this patient. That network
had a stronger native response. The disagreement is evidence of judge and
posterior sensitivity; neither judge establishes irreversible information
loss, patient-independent generalization or clinical recall change.

## What the evidence rules out, and what it does not

- Numerical drift, swapped axes and variable publisher windows have explicit
  controls. These controls do not eliminate learned-detector domain shift.
- A separately trained model is not necessarily patient-independent. Released
  DiffTumor header lacks training patient/fold manifest; overlap unresolved.
- The original DiffTumor organ-mask producer family is CLIP-Driven Universal
  Model, identified from appendixE.2. Exact archived checkpoint/runtime are
  unverified. Supplying GT organs would be invalid; we did not do that.
- A posterior sample can change detector behavior substantially. Three seeds
  are not a distribution estimate or three independent patients.
- Two PANORAMA cases and three earlier MSD reconstructions are selected
  feasibility screens, not a representative cohort or a causal size study.
- Reconstruction may change a detector response without deleting clinical
  evidence. A label on the original CT is not a reader rating of reconstruction.

## Novelty assessment

The broad solutions are crowded: lesion-aware image-space diffusion
post-training already exists; MAISI-v2 already uses tumor-focused objectives;
medical latent learnability and downstream-aware pathology compression are
already studied. Region-separated lesion latents also exist. Details and
primary links: NOVELTY_AND_DECISION_GATE_20261002.md.

The remaining possible distinction is a controlled **3D tiny-lesion audit
of posterior uncertainty AND judge sensitivity before diffusion training**,
with prospective cases, independent clinical evidence, negative controls and
an actionable efficiency/fidelity tradeoff. Our screen does not establish
that distinction as novel, broad enough or useful enough for a top conference.
A diagnostic result alone must not be dressed up as a new loss function.

## Smallest next experiment for PI approval

**Use the existing blinded6-page packet for MSD120/165.** Ask an experienced
pancreatic-CT reader to compare known-location lesion conspicuity and boundary
fidelity on the paired conditions, noting confidence and artifacts. No new
GPU allocation or training is needed. Rough reader time15–30minutes is an
estimate, not an agreed commitment. Preserve condition blinding; do not send
the unblinded repo/results to that reader before review.

This is a preliminary known-location fidelity check, NOT a clinical reader
study or unprompted sensitivity test. Both cases were selected for small
burden, and the packet only tests native versus mean, not posterior sampling.
If expert inspection finds no meaningful change, stop the clinical-erasure
story. If it finds degradation, request a separately approved prospective
cohort and fix the clinical endpoint before any remedy training.

Only then consider a fixed independent-patient cohort with adequate-native
judges, all seeds reported, negative cases, a second codec or distortion-
matched baseline, and global/lesion fidelity measurements. Report inadequate
native responses rather than replacing cases post hoc. Do not promise a
GPU-hour cost without real runtime calibration on that cohort.

## Spending and safety

3297419:209allocationseconds atbilling2000 => .116111charge-equivalentGPUh.
Campaign estimate **1.225833/2**, remaining **.774167**. Includes earlier
failures. Posted allocation debit unverified; use elapsed allocation, not
181.12s neural/script runtime, for accounting. No automatic retry.
PeakGPUallocated82,947,072B; stepMaxRSS6,549,120K;32Ghost limit respected.
No pending/running DeltaAI screening jobs at final check. Main PanTS evaluation
47320183 untouched; no claim about its live state is made here.

All code/scalar results and failed/negative findings preserved in GitHub.
Images, weights and raw maps remain private. The frozen screening settings
and goals did not change after the negative outcome.

## Completion evidence for the requested investment decision

| Requirement | Evidence/status |
|---|---|
| Novelty audit | Primary-source inventory and scope limits in novelty decision doc |
| Independent-judge checks | Hash/strict-load CPU pass;7geometry/audit tests; exact runtime repeats; source/fold/license limitations retained |
| Cheapest decisive experiment | Reused images, one209s job; completed artifact plus independent saved-map audit |
| Budget including failures | Campaign ledger + Slurm elapsed/billing, below2h; posted debit uncertainty explicit |
| Preserve experiment quality | No crop/GTinput/adaptation, all seeds, fixed gates, repeatability and image-grid contracts |
| Preserve main evaluation | No commands or mutations targeting47320183 in this screen |
| Negative findings | Weak native judge response and seed2increase retained, not rerouted to favorable judge |
| Go/no-go and strongest idea | No new training; conditional posterior/judge fidelity-audit question, not proven contribution |
| Next experiment for PI | Existing blinded packet, expert known-location review; no GPU spend without new approval |

Data artifacts: difftumor_result_3297419.json,
difftumor_audit_3297419.json; existing posterior_control_result/audit3297185.
This completes the bounded investment screen, not a paper or clinical claim.

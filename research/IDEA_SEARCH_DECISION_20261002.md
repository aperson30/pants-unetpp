# PI decision: no new method training on the present evidence

2026-10-02. This is the conclusion of the resource-minimal idea search, not a
finished paper, an exhaustive proof of absent novelty, or a clinical result.
Detailed primary-source comparisons and execution evidence are in
[RESOURCE_MINIMAL_IDEA_SEARCH_20261002.md](RESOURCE_MINIMAL_IDEA_SEARCH_20261002.md).

## Recommendation

**NO-GO for committing to a new top-conference method or spending the remaining
screening budget to force one.** None of the investigated candidates clears
novelty, decisive evidence and modest full-paper cost together. Preserve the
existing PanTS comparison and negative results; do not change its experiment.

The strongest supported QUESTION is: **how much real adaptive-depth compute
can be saved while retaining absolute rare-lesion detection, rather than merely
matching a possibly inadequate deepest predictor?** It is PI-aligned and could
reuse trained weights, but is not a new method established here. UNet++ pruning,
class-aware thresholds, calibrated early exits and whole-lesion risk are prior
art. A useful future result must explain or overcome a reproducible limitation
that those baselines do not, not simply rename their losses for tumors.

### The decisive free probe failed its predeclared adequacy gate

Read the saved development summary through Bridges-2; exact frozen SHA256
matched. No inference, staging, fitting or GPU allocation. The four positives
were selected by smallest annotation count, not favorable prediction outcomes.

| Frozen case | Tumor annotation voxels | Overlapping predicted tumor voxels | Off-annotation predicted voxels |
| --- | ---: | ---: | ---: |
| PanTS_00003548 | 36 | 0 | 936 |
| PanTS_00006854 | 66 | 0 | 0 |
| PanTS_00000133 | 126 | 0 | 0 |
| PanTS_00005782 | 126 | 0 | 535 |
| PanTS_00000016 | 0 | 0 | 0 |
| PanTS_00000023 | 0 | 0 | 0 |

Deepest-model any-overlap detection is **0/4 positive cases**. This is an extreme
small-tumor stress set, NOT overall PanTS accuracy or clinical sensitivity.
Counts do not adjudicate off-annotation predictions clinically. All cases are
retained in [frozen_exit_native_outcomes_20261002.json](frozen_exit_native_outcomes_20261002.json).

The prior stop rule says inadequate native response stops the relative-safety
screen. A cheap exit could match zero detections without preserving any tumor.
We therefore do NOT run the proposed oracle/controller GPU probe or swap in
easier cases. Shallow rescue is unmeasured; even if later found, overthinking
and branch ensembles already study that phenomenon. This result does not prove
adaptive depth is impossible, or that all 174 development positives are missed.

## Candidate ranking and why we did not invest

Ranks indicate present scientific value per execution cost, not acceptance odds.
The detailed document retains each proposed claim, alternatives and sources.

| Rank / candidate | Potential significance | Defensible novelty now | Cheapest discriminating evidence / decision | Minimum credible full-paper cost |
| --- | --- | --- | --- | --- |
| 1. Absolute rare-lesion risk versus adaptive-depth savings | High if a generalizable limit or remedy is found | Low/unproven against UNet++, CBT, Fast yet Safe, CLS and MESS | Trained-exit CPU contracts pass; frozen native outcome gate fails. STOP this probe. | Inference-only only if compatible trained exits and independent data exist. A design of 2000 all-head volume passes costs 5.6/16.7/33.3 physical GPUh at hypothetical 10/30/60s each, before repeats/router profiling/staging/billing. Medical runtime and second backbone unverified. |
| 2. Unbiased multi-depth diffusion gradients | High if actual matched-quality training cost falls | Low/unproven against control variates, multifidelity estimation and timestep/subnetwork allocation | Exact CPU expectation/variance/clipping checks expose deep-only parameter and optimizer problems. No GPU training warranted yet. | Two small tasks, six configurations, three seeds: 36R physical GPUh plus 3D transfer/tuning, with R measured per full run. Sensitivity R=2..10h gives 72..360h, NOT a runtime forecast. |
| 3. Rare-structure-preserving structured pruning | High application value | Low: class-aware saliency already in CRISP; diffusion timestep importance already studied | Needs a distinct causal mechanism before comparing saliency ranks and actual dense execution. NO-GO for generic combination. | At least two tasks/two backbones with dense-small and class-aware controls, repeated seeds. Even a 12-setting fine-tune subset costs 12F plus inference; F=.5..2h gives 6..24h hypothetically, not measured. Additional dense training could dominate. |
| 4. Posterior/judge-robust compression audit | Potentially useful diagnostic | Unproven; lesion-aware objectives and task-aware evaluation exist | Completed frozen second-judge probe failed native adequacy; a posterior seed increased response. NO-GO for clinical-erasure/new-VAE story. | The measured 108..110s altered-volume pass gives about 3 physical GPUh per 100 passes. Multiple codecs, conditions, seeds and judges multiply this; setup, scoring and independent clinical evidence are extra. |
| 5. Resolution-consistent small-lesion auxiliary targets | Useful engineering/interpretation | Direct overlap with Park2021 foreground labels and KG-Seg2026 validity masking | Reject from primary literature before any GPU test. | A minimal causal 4-condition x 2-task x 3-seed design is 24R plus evaluation; actual 3D training must be calibrated. Not an inference-only shortcut. |

These are transparent cost sensitivities, not promised budgets. None of the
unmeasured candidates has a validated final-paper runtime. Cheap screening is
not evidence a complete paper is cheap. Lower-cost endpoints must not silently
replace clinical claims or be presented as matched-quality training speedups.

## Strongest confounds and required baselines

- Depth effects can come from poorly trained exits, unequal supervision weights,
  label reduction, scale/normalization, small calibration sets or domain shift.
  Need fixed-exit, entropy, class-aware threshold and risk-control baselines;
  standalone truncated latency is only an oracle lower bound. Pay actual router,
  preceding-head and recomputation costs.
- Unbiased raw gradients do not give identical momentum/Adam/clipped updates.
  The residual variance and optimizer can erase computational savings. Existing
  gradient-control work already reports weak U-Net benefits.
- Mask sparsity is not dense-kernel speedup. Compare a smaller dense model and
  existing worst-class/class-aware importance before attributing a pruning gain.
- A pretrained detector response is not clinical visibility. Domain shift,
  native inadequacy, posterior variability and classical distortion controls
  prevent interpreting our compression response changes as lesion erasure.
- Original UNet++ v1 README advertises liver/tumor weights, but actual contents,
  fold provenance and all-head supervision are not verified. A public HTML200
  OneDrive page is not successful checkpoint retrieval. Its current source has
  the documented branch/loss mismatch; do not impute that to unseen weights.

## Minimal PI-ready execution plan

1. **Now, zero GPU:** take this NO-GO and the preserved negative evidence to the
   PI. Finish the main comparison under its existing protocol; this idea search
   neither queries nor alters its protected test evaluation. Do not retrofit a
   new training recipe into completed cells.
2. **Only if the PI wants to revisit adaptive compute:** define an absolute
   annotation-based endpoint, target population and allowed risk first. For an
   empirical model-behavior study no clinical expert is required. A clinical
   sensitivity/visibility claim requires separate clinical evidence later.
3. **Before any new allocation:** freeze an independent annotation-selected
   cohort, not an outcome-selected replacement for these four failures; verify
   native adequacy, label geometry, preprocessing, trained exits and model/data
   overlap. Obtain an independently usable second medical model/dataset or
   budget its training honestly. Merely using our plain-U-Net lower-resolution
   heads does not create a second genuine anytime backbone.
4. **Only after those gates pass:** preregister a bounded all-head/oracle timing
   probe with FP burden and absolute detection; then compare deployment routing
   against the cheapest qualifying fixed exit and published baselines. Keep
   full context, official resampling/TTA and all selected cases. A >=20% saving
   threshold was a resource heuristic, not a clinical tolerance.
5. **Full paper only with new evidence:** require a substantive distinction
   beyond known thresholding/calibration/overthinking, replication across tasks
   and architectures appropriate to the scope, held-out evaluation and measured
   end-to-end cost. If these assets require long new training, reconsider the
   project rather than calling it a cheap paper.

The statistical limits also constrain scope: 174 independent positives with
zero failures give a 1.707% one-sided95% upper bound for ONE fixed policy, not1%.
299 zero-failure positives are needed for1%. A hypothetical87positive holdout
gives3.385%; three simultaneous29case strata give13.167% each. These are known
binomial design calculations, not model outcomes or conformal expected-risk
guarantees. Do not borrow the protected901test to tune a new idea.

What would change the decision is **new distinguishing evidence**, not another
fashionable combination: a reproducible oracle benefit with an adequate native
baseline, a label-free controller that beats the strongest alternatives at the
same absolute lesion/FP risk, or a genuinely cheap gradient surrogate reaching
deep-only parameters with measured cost-variance and matched-quality benefit.
We have not obtained any of those. No claim that no such idea exists elsewhere.

## Resources, preservation and completion audit

Live2026-10-02 DeltaAI accounting for all17 listed screening allocations,
including FAILED jobs, again matches4413 weighted seconds: **1.225833/2
charge-equivalent GPUh**, reconstructed remainder.774167. Posted debit and
rounding policy remain unverified. No new GPU job was submitted in this reopened
search; no new debit was inferred from neural step time. Spending uncertainty
still blocks future submission, not literature/CPU work.

The final zero-GPU outcome read took approximately3.72s remote-command wall
time. Earlier measured preparation: ~238MB public paired inputs, about43s
bounded label/image processing plus network, .429s small random CPU contract,
5.89s real-checkpoint synthetic CPU contract; local environment setup took a
few minutes. These are component measurements, NOT total researcher time.
Source/library audits and web searches took additional unmetered human/agent
effort; no complete wall-clock log exists, so no false total is reported.

| Objective requirement | Authoritative evidence / scope |
| --- | --- |
| Current documents and completed/negative work inspected | Public fork pulled; local research commits, historical idea brief and previous GO_NO_GO inspected; no repeated GPU probes |
| Primary novelty/code/data/weight search | Detailed source inventory, author code contracts and reuse table; access/provenance limits retained |
| Serious candidates, alternatives, discriminating tests and paper cost | Five candidate sections plus ranking/cost table above; measured facts separated from hypothetical sensitivity |
| CPU/literature rejection before GPU | Estimator and binomial checks, label/hash audits, strict trained-exit CPU parity, direct prior-art rejection |
| Highest-information frozen probe | Exact-hash six-row development read; all selected cases retained, 0/4 native-positive gate fails, no inference substituted |
| Budget/failures/quality protection | Live17job accounting includes failures; no new allocation; no commands targeting protected47320183, no production edits/test fitting |
| Evidence kept updated | Frozen manifest, native-outcome JSON, cohort-design JSON, detailed search and this final decision |
| Strongest supported recommendation and PI plan | Conditional absolute-risk/compute question, NO-GO for new method investment, concrete evidence needed to reopen |

This finishes the investment decision requested by this search. It does NOT
finish the main PanTS paper/evaluation, prove impossibility, or certify S-tier
novelty. Preserve the unused budget rather than spend it to obtain a positive.

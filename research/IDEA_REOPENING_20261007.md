# Reopened idea search: eliminate weak-baseline discoveries cheaply

October 7, 2026. Research is reopened by the user. No protected evaluation
commands, changes to the four trained models, GPU submissions, weight/data
downloads, or new spending. **No S-tier idea established.** Earlier failed
native-adequacy and novelty gates remain valid for their original hypotheses.

## Fresh primary-source screen

| Proposed opening | Closest evidence inspected | Decision in this pass |
| --- | --- | --- |
| Faster diffusion training by timestep/gradient allocation | [Adaptive Non-uniform Timestep Sampling, CVPR2025](https://arxiv.org/html/2411.09998v1); [Min-SNR, ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf) | Generic allocation/gradient-conflict contribution already occupied; no training probe. |
| Size/frequency-aware diffusion noising | [Simple Diffusion](https://proceedings.mlr.press/v202/hoogeboom23a/hoogeboom23a.pdf); [Improved Noise Schedule, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf) | Resolution-SNR adjustment and training schedule design are established. A lesion-specific observation would need a distinct mechanism, not just a medical application. |
| Preserve rare outcomes through better few-step schedules | [Align Your Steps](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/); [INDIS, CVPR2026](https://arxiv.org/html/2603.17671v1); [Gaussian Mixture Solvers](https://arxiv.org/html/2311.00941v1) | Global/instance schedule optimization and multimodal few-step failures have close prior art. Executed the exact-score CPU baseline screen below; no new method warranted. |
| Apparent delayed tumor learning versus delayed hard-label decisions | [Decoupling Representation and Classifier, ICLR2020](https://iclr.cc/virtual_2020/poster_r1gRTCVFvB.html); [RankSEG, JMLR2023](https://www.jmlr.org/beta/papers/v24/22-0712.html) | Useful diagnostic question, not novel thresholding/classifier calibration. Historical probabilities/checkpoints are needed to distinguish mechanisms; final masks and pseudo-Dice alone cannot answer it. |

These are targeted exclusions, not an exhaustive novelty proof. Search-returned
descriptions of other medical/spectral methods were leads only; no implementation
or empirical reproduction claimed. Inspection of INDIS's main formulation and
synthetic section confirms input-conditioned scheduling already exists. Its
finite-prior mismatch discussion is NOT a statement about minority-class erasure.

## Executed zero-GPU discriminating screen

Files: `rare_mode_solver_screen.py`, `rare_mode_solver_screen_20261007.json`.
Local Python environment: existing evaluation-audit venv, NumPy 2.5.3,
SciPy 1.18.1. Reported CPU execution 9.207 seconds excludes interpreter/import
and investigator time. 216 configurations, all retained; no favorable subset
selection or learned schedule.

- One-dimensional two-Gaussian VE diffusion with exact analytic score.
- Fixed component weights .01/.1/.5, separations 2/4, component standard
  deviation .35; sigma 10 to .01.
- Exact finite-sigma initial quantiles avoid approximating the prior with an
  unrelated Gaussian. Quantile conservation supplies exact terminal references.
- 4096 midpoint quadrature points, not independent subjects or random seeds.
- Euler, Heun and RK4; linear/geometric/rho7 grids; matched total score-call
  budgets 16/32/64/128, counting intermediate stages.
- Contracts passed: analytic score versus finite differences; identical-component
  single-Gaussian transport versus exact scaling; quantile-CDF residual <1e-12;
  finite outputs and exact score-call accounting.

Illustrative fixed configuration: weight .01, separation 4, 32 score calls,
rho7 grid. Positive region means terminal x > 2, not clinical lesion identity.

| Solver | Global endpoint MSE | Positive-region endpoint MSE | Exact-positive trajectories ending outside positive region | Absolute positive-mass error |
| --- | --- | --- | --- | --- |
| Euler | .00681972 | .646583 | 12.1951% | .00121094 |
| Heun | .0000700641 | .000797070 | 0% | .00000976015 |
| RK4 | .00000631878 | .000479500 | 0% | .00000976015 |

The rare subset can have much larger error than the average, but ordinary
higher-order baselines remove the illustrated region misses at the same NFE.
**Do not sell Euler's failure as evidence for a necessary new rare-mode sampler.**
Heun/rho7 at 32 calls has zero positive-region losses on all six tested mixtures,
but some have extra negative-to-positive crossings and nonzero mass error.
Thus this is NOT perfect distribution recovery, a safety certificate, a proof
about all rare modes, or a general claim that higher order solves the problem.
Mass resolution is 1/4096; no statistical confidence bound is inferred.

The probe does NOT implement DPM-Solver, AYS, INDIS, learned scores, spatially
small lesions, classifier guidance, or 3D CT synthesis. Small component prevalence
and small lesion spatial support are different concepts. No practical speedup or
training-quality result is established. The code implements standard numerical
methods, not a proposed paper contribution.

## Next investment: locate a distinguishing failure, not another combination

The most useful remaining question is whether acceleration loses an annotated
local condition that strong ordinary numerical baselines preserve poorly even
when their global output error is small. This is a QUESTION, not a discovered
gap; task-aware sampling, instance adaptation and spatial allocation are obvious
competitors. It becomes interesting only if an inexpensive observable exposes
and repairs a failure those methods do not already resolve.

Before spending GPU hours:

1. Audit a public conditional generator's actual available weights, conditioning,
   source split and synthesis evaluation. Prefer code/assets already audited;
   do not equate a supplied conditioning mask with verified generated pathology.
2. Specify an independently measurable condition and strong no-training controls.
   A detector is usable only after native adequacy, coordinate/units and split
   checks; earlier failed judges are not revived or swapped for favorable cases.
3. Require same total score-call budget and include higher-order, optimized-global
   and instance-conditioned scheduling baselines as applicable. Count setup,
   calibration, rejection, routing and metric evaluation costs.
4. Keep a CPU/small-pretrained-mechanism screen separate from a complete paper.
   No final-paper GPU-hour forecast is defensible until actual runtime, available
   assets and necessary replication are known. No training from scratch yet.

The delayed-onset diagnostic is a separate potentially cheap opening ONLY if
historical checkpoints/probabilities exist. It needs lesion localization at a
fixed false-positive burden, not merely a lowered threshold producing any tumor
voxel. No protected test fitting or changes to the original comparison.

Read-only Bridges-2 filename inventory on this pass found only `checkpoint_best.pth`
and `checkpoint_final.pth` in each of the four standard training result folders
(eight files total). No epoch-indexed sequence was found there. Best-checkpoint
epochs were not loaded and other backup locations were not exhaustively searched;
this does NOT prove no historical weights exist elsewhere. The current folders
alone do not support a temporal representation-versus-decision experiment.
Do not propose retraining 1000 epochs just to manufacture that missing evidence.

Current decision: **do not commit to either as a new method yet.** This pass
saved a likely weak-baseline GPU investigation; it did not produce an S-tier
claim. The earlier 2 charged GPU-hour cap is unchanged; this pass used zero.
Historical reconstructed spend is not refreshed posted allocation accounting.

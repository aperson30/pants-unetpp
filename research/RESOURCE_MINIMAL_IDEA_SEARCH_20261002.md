# Resource-minimal idea search: reopened after negative compression gate

Status: active investigation, not a selected contribution or novelty certificate.
Main PanTS evaluation is protected. No GPU job submitted in this search.

## Authoritative starting evidence

The previous decision is preserved in compression_probe/GO_NO_GO_RECOMMENDATION_20261002.md.
Its second raw-head network had weak native response (.00329347 < frozen .01
floor); seed2 increased response 1.679x. No clinically introduced miss was
established. Do not repeat that screen or find a more favorable judge.
The contrast-response experiment did show amplitude dependence, but a matched
classical denoiser also has amplitude-dependent response. Therefore nonlinearity
alone is not a learned-prior discovery or proof of erased task information.

Campaign estimate including failures:1.225833/2 charged-equivalent GPU-hours;
remaining .774167. Posted debit remains unverified. This new goal does NOT reset
the cap. Before any submission, refresh actual spending and charge factors.

## Initial candidates, ordered by the next cheap decision (not acceptance odds)

### 1. Rare-structure risk versus adaptive depth's attainable compute savings

Claim to test: inexpensive early features may fail to distinguish confident
small-lesion misses from genuinely easy negatives, imposing a practical limit
on safe adaptive-depth inference. A useful contribution would explain and
predict this limit, or beat it without GT at deployment—not merely add exits.
Importance: aggregate confidence can conceal the rare failure we care about.
Closest work: UNet++ already prunes through supervised branches; CRC already
controls false-negative losses; risk-aware early-exit controllers already exist;
ReCIRC studies local/worst-group risk. Generic calibrated early exit is NOT new.
Alternative explanation: poorly trained exits, inadequate calibration sample
size, class imbalance or distribution shift—not a fundamental depth limit.
Cheap test: audit whether our existing checkpoint actually trained shallow
full-resolution exits, then use existing validation predictions if available.
Compute an oracle speed/lesion-risk frontier before fitting any controller.
If the oracle cannot save useful compute, kill the deployment route immediately.
No main test-set policy fitting. Whole-volume context, positives and negatives.
Minimum credible paper: two datasets, two genuinely multi-exit architectures,
oracle plus entropy/calibrated/class-aware baselines, held-out calibration/test,
lesion-size strata and confidence bounds, measured latency and abstention cost.
Cost feasibility: reuse could make it inference-only, but absent trained exits
it requires training and is no longer a cheap idea. No numeric GPU-hour estimate
is credible until checkpoint/data inventory and per-case runtime are inspected.

### 2. Unbiased multi-depth gradient estimation for diffusion training

Claim to test: a cheap nested network's gradient plus infrequent paired full-depth
corrections could preserve the full objective in expectation at lower cost.
This is a candidate mechanism, NOT a new theorem: control variates and
multifidelity Monte Carlo are established. Novelty must survive those baselines.
Closest work: Early-Bird Diffusion trains timestep-specific subnetworks;
adaptive timestep sampling optimizes updates; DAR modifies depth-wise routing;
Taylor-control-variate diffusion work is a particularly close unresolved check.
Importance: asks whether compute can be reduced without quietly replacing the
learning objective with a shallow-head curriculum.
Alternative: correction variance and clipping/optimizer nonlinearities erase
the savings; the surrogate gradient may be badly aligned during tumor onset.
CPU-first rejection calculation: ghat=gs+B/p*(gf-gs), B~Bernoulli(p).
Conditional expectation is gf; added covariance=(1/p-1)*dd^T, d=gf-gs.
Expected unbiased gradients do NOT imply identical SGD trajectories, unbiased
clipped gradients, or unchanged Adam updates. Full correction has real cost.
Cheap decisive probe, only after novelty/code check: paired gradients with
matched input/noise across depth, estimate residual variance and measured cost;
compare cost-times-variance to full and random routing. No training-speed claim
from a forward benchmark or a few updates.
Minimum paper: paired training from scratch on two small public tasks, 3seeds,
full and routing/pruning/variance-reduction baselines, matched time/quality,
wall-clock overhead and one relevant 3D transfer. Budget model:6configs*3seeds*
2tasks*hours-per-run =36R hours BEFORE transfer and tuning. If R is not small,
reject as compute-heavy. R must be calibrated, not guessed from specifications.

### 3. Rare-structure-preserving structured pruning of frozen models

Claim to test: preserve lesion-specific functional responses at a useful
measured structured-pruning speedup, without retaining the whole model.
Closest work: CRISP already uses class-aware gradient saliency; Diff-Pruning
already studies timestep-aware importance. Combining them for medical CT is
not automatically a publishable contribution.
Alternative: ordinary worst-class objectives or a smaller dense model explain
all improvement; mask sparsity gives no actual runtime improvement.
Cheap test: compare global/class-aware sensitivity ranks on a fixed calibration
split and evaluate distinct held-out lesions. Do not select the split by scores.
Minimum paper: 2tasks/2backbones, dense-small and pruning baselines, 3seeds,
lesion-level outcomes and latency. Even fine-tuning12settings for F hours each
costs12F plus inference; diffusion quality adds generation cost. Not budget-ready.
Status: deprioritize until a distinction beyond CRISP is identified.

### 4. Posterior/judge-robust tiny-lesion compression audit

Current evidence supports detector sensitivity, not clinical erasure. Foundation
VAEs already evaluate MSD organ/tumor segmentation; medical latent learnability
and lesion-focused objectives are established. Keep the previous negative gate.
A no-reader study could measure paired response/geometry/contrast, but cannot
rename those endpoints clinical visibility. No new samples or VAE sweep yet.
Minimum credible audit would require fixed cohort, multiple codecs/judges,
all seeds and domain-shift/distortion controls. Existing runtime is around
108–110s per altered volume in the prior contrast experiment; 100 such passes
alone are ~3 GPU execution hours, excluding judge/setup/billing. Not free.
Status: reserve, not the favored direction.

## Primary sources inspected this round

- Early-Bird Diffusion, CVPR2025: timestep-aware sparse subnetworks already
  accelerate training. https://openaccess.thecvf.com/content/CVPR2025/html/Whalen_Early-Bird_Diffusion_Investigating_and_Leveraging_Timestep-Aware_Early-Bird_Tickets_in_Diffusion_CVPR_2025_paper.html
- UNet++: supervised branches already enable pruning.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7357299/
- Adaptive Non-uniform Timestep Sampling: adaptive update allocation exists.
  https://arxiv.org/html/2411.09998v2
- Diffusion-Adaptive Routing: learned timestep-aware cross-layer aggregation
  in DiTs exists; this does not prove the same effect in CNNs.
  https://arxiv.org/html/2605.20708v1
- CRISP: class-aware gradient pruning exists.
  https://arxiv.org/html/2311.14272v1
- Conformal Risk Control: monotone task losses and false-negative examples.
  https://arxiv.org/abs/2208.02814
- ReCIRC: local-risk reparameterization and worst-group evaluation.
  https://arxiv.org/abs/2609.38112
- Author repository for risk-aware early exit, not independently reproduced:
  https://github.com/okdlibri/risk-aware-early-exit
- Taylor-control-variate diffusion PDF surfaced, but OpenReview returned a
  verification page. Contents NOT inspected; must resolve before asserting
  candidate2 is novel. https://openreview.net/pdf?id=YqFIzHAfbk

## Next actions

1. Inspect latest prior idea-screen documents and real checkpoint/exit contracts.
2. Resolve closest control-variate/multifidelity and risk-aware-exit papers/code.
3. Do CPU rejection calculations and artifact-availability checks before GPU.
4. Only preregister a probe if it distinguishes a plausible new claim, fits the
   remaining campaign cap, and does not require new expensive training.

No candidate currently clears the top-conference investment bar. Initial rank
is provisional; candidate1 has the best reuse potential, candidate2 the clearest
training-efficiency mechanism but substantial variance and novelty risk.

## Follow-up: code audit and zero-GPU rejection calculation

Read the prior work/diffusion_idea_screen_20261001.md as historical context;
its budget/no-experiments statements are superseded by the completed campaign
ledger. It already covers multi-decoder diffusion, ASE, TDC, gradient surgery,
noise scheduling and compression/noise/depth separation. Do not rediscover
those as new ideas. Its latest snapshot also explicitly says our PanTS models
are segmentation checkpoints, not pretrained diffusion denoisers.

### Candidate1 reuse caveats actually verified in source

nnUNetTrainerUNetPlusPlus.py hardcodes skip_shallowest_deep_supervision_head=True;
the shallowest full-resolution head is omitted because its conceptual weight
is zero. Other retained heads are supervised with decaying weights. DS-off is
not an anytime-trained model; do not infer trained exits from state-dict keys.
unet_plusplus.py computes the full encoder and ALL nested decoder columns before
selecting any outputs. Returning a shallow tensor from that forward gives no
truncated-network speedup. True exit execution requires dependency-aware
truncation and activation/memory timing; encoder computation is already paid.
No trainer or existing evaluation changed. This source audit does not prove
what trainer/config a remote checkpoint used; metadata still needs inspection.

Closer competing work found: Fast yet Safe (NeurIPS2024) already calibrates
early exits for semantic segmentation AND diffusion image generation. It uses
public ADP-C HRNet/Cityscapes checkpoints and evaluates GTA5 too. A general
segmentation risk-control proposal duplicates it. A lesion-level/positive-case
failure study could be a distinction, but must beat class/group-aware baselines
and be more than changing the risk's definition.
https://arxiv.org/html/2405.20915v1
https://github.com/liuzhuang13/anytime

### Candidate2 CPU gate completed

multidepth_estimator_gate.py ran locally:3 exact-enumeration/analytic tests pass
in .113s, zero GPU hours. It verifies unbiased RAW Bernoulli-corrected gradients,
added conditional variance, analytic cost-variance optimum, and a clipping
counterexample. No trained gradients or measured surrogate cost used.

Let V=trace(Cov(gf)), r=E||gf-gs||^2/V, shallow cost a, incremental correction
cost b. Relative efficiency proxy is (a+p*b)*(1+(1/p-1)*r), normalizing V and
full cost to1. For 0<r<1 its interior optimum is sqrt(a*r/[b*(1-r)]), capped
at1; r>=1 favors always full depth. Deep-only parameters have gs=0, so their
residual second moment is at least their full-gradient variance. They cannot
gain sampling efficiency through this estimator unless a genuine cheap
gradient surrogate reaches them. Shared parameters might still benefit; this
does not rule out total savings, but rules out a blanket all-parameter claim.

Illustrative assumptions (not measurements): a=.25, r=.1 -> proxy .5598;
a=.25,r=.7 -> .9969 (essentially no advantage). At a=.5,r=.3 -> .9583.
Any recomputation overhead raises costs and worsens this screen. Equal proxy
values do NOT imply matched optimization or generation quality.
Clipping example: full gradient2 clipped at1 is1. Shallow0 with p=.25 produces
raw0 or8, unbiased mean2; clipping produces expected update.25, not1.
Thus this is NOT quality-neutral plumbing for our existing training runs.

Resolved blocked paper through its arXiv version: Taylor-control-variate work
already tests diffusion-gradient variance reduction, reports weak U-Net/MNIST
variance reduction and no loss convergence benefit. Its surrogate is different
from a nested branch, but both efficiency and novelty need proof, not slogans.
https://arxiv.org/html/2408.12270v1

### Decision update and next test

Do NOT submit a correction-gradient GPU experiment yet. Cheap gates expose
deep-parameter and optimizer problems, and there is no verified pretrained
diffusion model with compatible trained exits. Candidate2 demoted below1.
Candidate1 remains a potential cheap DIAGNOSTIC, not an accepted method idea.
Next: metadata-only checkpoint availability audit and the lesion-risk sample
size/monotonicity requirements. If existing exits/data cannot support it cheaply,
use the public ADP-C checkpoint as an optional nonmedical falsification test,
not as proof of pancreatic-tumor validity. No external expert is required.
Campaign spending unchanged; no new GPU work. Goal remains active.

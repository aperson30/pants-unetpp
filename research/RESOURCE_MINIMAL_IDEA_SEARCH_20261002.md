# Resource-minimal idea search: reopened after negative compression gate

Status: investment decision concluded NO-GO on current evidence; not a selected
contribution or novelty certificate. Latest decision and minimal PI plan:
[IDEA_SEARCH_DECISION_20261002.md](IDEA_SEARCH_DECISION_20261002.md).
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

## Checkpoint reuse and rare-risk feasibility resolved on CPU

audit_local_exit_metadata.py inspected local September29 safety backups using
the existing inert restricted metadata parser: no tensor loading or execution
of pickled classes, fixed8MiB metadata bound, all four SHA256s match manifest.
Each checkpoint records epoch1000/fold0 and the expected sparse-validation
trainer. This independently verifies artifacts, not their latest remote state.
The saved debug configuration records DS-on/skip-shallowest=True for UNet++.
All five seg-head keys exist in BOTH UNet++ files; this alone cannot establish
training. Current trainer semantics plus saved configuration support j2..j5
supervision for DS-on; j1 is omitted. DS-off has no auxiliary supervision.
Do not use its head keys as evidence of learned shallow exits.

For six stages, conceptual loss weights are j5:8/15,j4:4/15,j3:2/15,j2:1/15,
j1:0. This is not the equal-weight Paper configuration. To test actual compute
savings, exit-j dependency closure can in principle truncate to encoder stages
0..j and decoder columns1..j; current production forward does NOT do this.
We will not modify existing training/evaluation to add it. First validate an
isolated inference copy against all relevant full-graph logits.

Checkpoint hashes verified:
- UNet++ DS-on:81f9598fb86d90e18676c89eb3cc9a91379a7cb932e35cfad116e54bbf52d4a5
- UNet++ DS-off:6f1eac58af27784c7eb94539216d72670f0135c884df8ea2bc2bb3ecaff5e27a
- Plain DS-on:9b27d6ee87771eddb254300916bd1b836c6640a7f21cb98fd84723d0d0cc8df2
- Plain DS-off:e453225ad7cfe0a58d7ab47c6af9279dbf51f2e319651874fcf62c1bff8b8bf4

### Statistical rejection gate

rare_risk_feasibility.py:3CPUchecks passed. Known exact binomial zero-failure
bound gives29/59/299 independent positive cases for10%/5%/1% upper miss bounds
at95% confidence, for ONE prespecified fixed policy. Ten positives and zero
misses still permit25.9% miss risk. These are NOT new theory or CRC calibration
sample sizes, nor guarantees after selecting thresholds on the same cases.
At10% prevalence,590 random cases yield59 positives only IN EXPECTATION—not
with a guaranteed count. Use actual counts, separate policy-selection data,
and patient rather than patch/lesion independence. Correct policy multiplicity.

Synthetic counterexamples: marginal4% miss can hide40% conditional miss at10%
prevalence; a perfect10000-voxel lesion plus missed10-voxel lesion gives99.9%
voxel recall but50% lesion recall. These illustrate established metric gaps,
not discoveries in our models. Relative additional misses and absolute recall
must both be reported; the deepest branch itself may miss tumors.

Closer competing primary paper: Conformal Lesion Segmentation (2510.17897)
calibrates3D false-negative constraints, tested across6datasets/5backbones.
Its formal loss counts missed foreground voxels within each case, not a new
lesion-object-matching speed controller. This narrows but does not establish
our novelty. Combining its score with Fast yet Safe is a baseline, not a core
contribution. Inspect implementations before stating their precise guarantees.
https://arxiv.org/html/2510.17897v1
Fast yet Safe official code: https://github.com/metodj/RC-EENN

### Candidate1 minimal decisive probe: predeclared before execution

Purpose: determine whether supervised nested branches have a useful oracle
lesion-recall/compute frontier. No claim of deployable routing from an oracle.
Use a fixed validation-only feasibility manifest selected by IDs/annotation
presence, never by favorable branch responses. Keep main test901 untouched.
Report every chosen positive/negative, lesion burden, all supervised branches,
and repeat errors. Preserve full spatial context/sliding windows; no GT crop
or test-time GT routing. Oracle sees labels only in offline analysis.
Define lesion matching/FP treatment before inference. A tiny feasibility set
does NOT satisfy the statistical sizes above or clinical generalization.

Kill if full native response is inadequate to assess additional misses, exits
offer no meaningful oracle benefit, or real truncated execution saves<20%
latency after overhead. The20% is a resource-screen heuristic, not a clinical
tolerance. Retain inadequate-native cases instead of replacing them post hoc.
Only GO to cohort if paired-repeat/grid contracts pass and the oracle can save
>=20% measured time without additional whole-lesion misses on the fixed probe.
Even a GO is NOT proof a label-free controller can attain the oracle frontier.

No GPU submission: exact posted campaign spending is still unverified and
the user explicitly requires verified spending before jobs. CPU/source/public
artifact work can continue. No need for external clinical readers for this
model-behavior question; do not call it clinically safe pruning.

### Whole-paper cost sensitivity, not measured estimates

For a modest inference-only paper, two datasets *500held-out/calibration cases
*two multi-exit backbones =2000 all-head image passes. At hypothetical measured
c=10/30/60seconds per pass, that is5.6/16.7/33.3 physical GPUh BEFORE repeats,
truncated-latency profiling, staging and billing multipliers. This is a cost
sensitivity model, NOT a forecast; our3D c is unknown. Three repeat passes
would triple those figures. Existing full-depth outputs do not supply branches.
Reuse is realistic for UNet++ DS-on; plain-U-Net downsampled heads are NOT the
same adaptive-depth architecture. A second public backbone avoids retraining
only if compatible weights/data are available and license/overlap are checked.

Ranking:1 remains best reuse/cost candidate, but currently a diagnostic with
weak novelty;2 deferred for objective/optimizer/variance issues;3 duplicates
class-aware-pruning premises;4 retains previous NO-GO on tumor-erasure claims.
Next cheap work: audit primary baseline code/public second-backbone availability
and lesion matching. No candidate warrants full method training yet.

## Isolated exit contracts and official baseline implementation audit

Added research/isolated_unetpp_exit.py without production edits. It truncates
encoder and decoder dependency closure, returns individual logits, rejects
training/omittedj1/non-DS/ensemble settings. Flags do not prove supervised
checkpoint provenance. Installed PlainConvEncoder source confirmed independent
sequential stages. No full-encoder computation shortcut was assumed.

Actual torch2.10+cu129 CPU/GPU-disabled one-thread tests:2tests pass in.429s,
2D4stage,3D4stage,3D6stage anisotropic final stride[1,2,2]. All retained depths
2..L match full-graph logits bit-for-bit. Hooks prove no deeper encoder stages
or out-of-closure decoder nodes run. Six-stage depth2 executes3encoder/3decoder
blocks versus6/15 at full depth; node counts are NOT FLOPs or measured latency.
Hashes/results: exit_cpu_contract_20261002.json. Small random weights only:
real checkpoint/full-patch/GPU/BF16 parity and tumor accuracy unverified.
No large network or patient-volume CPU inference ran on a login node.

Read author RC-EENN code at commit3db9a52dfbbe8edb5d9b787b6f0712d4e580b9f4:
rc/risk_control.py already implements naive/CRC/LTT/UCB over precomputed loss
matrices and calibration/test splits. CRC finite-sample correction and UCB
safe-prefix thresholding are baselines, not new inventions. sem_seg/README
documents ADP-C and GTA5-tuned weights. Root README leaves license and some
dependencies TODO: do not redistribute its code without resolving permission.
https://github.com/metodj/RC-EENN/blob/3db9a52dfbbe8edb5d9b787b6f0712d4e580b9f4/rc/risk_control.py
https://github.com/metodj/RC-EENN/blob/3db9a52dfbbe8edb5d9b787b6f0712d4e580b9f4/sem_seg/README.md

ADP-C original README offers W18/W48 multi-exit weights via Google Drive;
code MIT. Actual weight retrieval/hash/license scope/runtime still unverified.
Cityscapes has its own data access terms. This optional nonmedical backbone
is not proof of pancreatic validity or frictionless free data availability.
https://github.com/liuzhuang13/anytime

New GPU hours0, protected main evaluation untouched. Next: authoritative
campaign charge-ledger refresh and validation-only case availability; only
then decide whether a capped trained-model probe satisfies all gates.
Goal active; execution feasibility is not a novel method contribution.

## Live accounting and stronger novelty rejection (2026-10-02)

Re-queried all17 allocation IDs through the live DeltaAI master. Every record
is terminal; FAILED allocations are included. ElapsedRaw times AllocTRES billing,
normalized by1000, totals4413weighted seconds =1.225833charge-equivalent GPUh,
leaving.774167under the unchanged2hour cap. Machine-readable records:
screening_allocation_ledger_20261002.json. This is verified Slurm reconstruction,
NOT independently verified posted account debit or billing-rounding policy.
No new allocation. Previous4e81cb2 push completed, then pull was up-to-date.

The official MICCAI2026 CLS page now strengthens the novelty warning: its
author feedback explicitly discusses presence-level FNR and complete misses
as well as voxel-level FNR. Its formal guarantee targets per-case voxel-FNR
tolerance under exchangeability, not deterministic safety of every test case.
Therefore do NOT claim nobody studies whole-lesion misses, or that changing
voxel risk into an object metric is sufficient novelty. Author feedback also
warns of scanner/site shift; our contextual-risk framing must beat real baselines.
https://papers.miccai.org/miccai-2026/0194-Paper2825.html

Fast yet Safe itself warns that a weak final predictor can make the relative
performance-gap risk trivially easy to control (AppendixD.2, GTA5). Thus our
native-response adequacy gate is not a novel discovery. It is essential hygiene.
https://arxiv.org/html/2405.20915v1

New compute-accounting gate for candidate1: standalone truncated-exit timing
is only an oracle lower bound. A deployed router must also pay for preceding
exit checks, heads, routing and potentially recomputation. Incremental execution
may reuse the nested dependency graph, but needs a separately verified runtime;
do not assume separate forward_exit calls reuse state. No savings claim from
standalone depth timing or node counts alone. Compare against cheapest FIXED
exit satisfying identical lesion/false-positive criteria, not only deepest exit.
Always report absolute native recall alongside newly introduced misses. Existing
project any-overlap detection is a reproducibility convention, not a clinical
matching standard; giant foreground predictions need explicit FP/volume checks.
No production scorer or901-case evaluation protocol changed.

Data gate: nine-case DeltaAI calibration is tumor-negative, so cannot support
rare-lesion claims. Existing Bridges2 research socket refused connection; no
fresh login was attempted. Requested a new user-authenticated master to locate
development-validation summaries only. No main evaluation job was inspected
or modified. Candidate1 remains an unproven diagnostic, not an S-tier method;
next decision requires real positive-case availability and runtime parity.

New bridges2-idea master now works. Read only development-validation summary
schema/reference counts, not model response metrics. Source SHA256fecb2318...
contains1800cases/174class28positives. Fixed feasibility manifest records first
four positives by(n_ref,ID):3548/6854/133/5782, with36/66/126/126GT voxels,
and first two negative IDs16/23. These tiny cases are a stress screen, not a
representative cohort or clinical sample. No outcomes-based replacement.
Manifest:exit_validation_feasibility_manifest_20261002.json.

CT/GT availability is still unverified: usual raw/preprocessed paths under
the Bridges project root are absent (a separate nine-case calibration subtree
exists). Therefore do not infer full training data survived node-local staging
and do not submit a probe until exact reusable data or bounded selective access
is established. No need to redownload1.1TB just to run this screen.

Commit0bfba1b records accounting/novelty gates locally. Its noninteractive push
failed because GitHub credentials could not be obtained; findings are NOT yet
published. No credential changes or alternate login bypass attempted.

## Bounded public validation data reuse completed (2026-10-02)

The user's Ocean area has no usual persistent full-training raw/preprocessed
directory; bounded inventory did not locate one. Instead verified individual
public iPanTSMini files at immutable revision
0254c148f3c05fd3913e42d16982b010a048167c. Dataset card declaresCC-BY-NC-SA4.0;
use for noncommercial research with attribution, retain redistribution terms.
https://huggingface.co/datasets/BodyMaps/iPanTSMini/tree/0254c148f3c05fd3913e42d16982b010a048167c

Actual CPU/network audit:6combined masks plus6individual pancreatic_lesion
masks,7441862bytes/20.640s. Every combined class28mask equals individual lesion
mask voxel-for-voxel, matches header geometry, and reproduces saved development
reference counts36/66/126/126/0/0. No model predictions used for selection.
CT staging:230730733bytes/21.953s. All6CT headers match corresponding label
shape/affine (absolute1e-6); all downloads match pinned LFS SHA256/size. These
checks support paired-case reuse, NOT original archive byte identity or full
class-map equivalence. Intensity/preprocessing/checkpoint parity remains open.

Scripts:audit_exit_public_labels.py,stage_exit_public_images.py. Exclusive new
output directories, bounded files/total bytes, TLS verification, no retries or
overwrites; public data/actual audit JSON remain outsideGit in sibling
exit_label_audit_20261002_v1 and exit_image_audit_20261002_v1. Python compilation
passes, actual6case audits completed. Source-data payload238172595bytes total;
no1.1TBdownload, GPU hours0, protected evaluation untouched.

Additional primary competitors found:
- ICML2019Shallow-Deep Networks already studies destructive overthinking and
  confidence/disagreement exits. Shallow branches rescuing deepest errors alone
  is NOT novel. https://proceedings.mlr.press/v97/kaya19a.html
- MIDL2026Adaptive Inference for Medical Vision Transformers already profiles
  medical data and learns to select token reduction/early exit across5datasets.
  Abstract reports71.4%averageFLOPs reduction/.1ppaccuracy loss, NOT a measured
 3DCTlatency comparison on our hardware. This further rules out a generic
  medical adaptive-compute pitch. Abstract audit only, full code not reviewed.
  https://proceedings.mlr.press/v315/byun26b.html

Ranking unchanged only for cheapest next evidence: candidate1 has reusable
weights/data but weak novelty;2/3stilldeprioritized and4negative preserved.
Next: actual isolated checkpoint/preprocessing compatibility, then decide
whether runtime/lesion oracle probe is worth the remaining verified budget.
Do not make a new conference-quality claim from data-readiness checks.

## Trained-weight CPU contract and stricter competitors (2026-10-02)

Actual local WindowsCPU test with the hash-verified1000epoch DS-on checkpoint,
full trained6stage architecture/32..320channels and29outputs: strict state-dict
load passes. Depthj2..j5each bit-identical to its full-graph branch on fixed
synthetic32cubed input; deepest DS branch also bit-identical to ordinary
single-output inference. One CPUthread, PyTorch2.10.0+cpu, DNA0.4.2,5.890s
including load/init/checks. See test_trained_exit_cpu.py and
trained_exit_cpu_contract_20261002.json. Trusted checkpoint hash checked BEFORE
pickle loading. No source checkpoint writes. This is stronger execution evidence
than random small weights, but NOT fullpatch/GPU/patient/lesion quality evidence.
Separate local exit-cpu-venv; no existing environment/cluster login compute
modified. Official CPUwheel113.7MB; setup took roughly a few minutes of wall
time, no charged allocation. Full package versions available via pipfreeze.
https://download.pytorch.org/whl/cpu/torch/

Inspected actual installed DeltaAI nnUNetPredictor methods (CUDAdisabled,
source only). initialize_from_trained_model_folder buildsDS-off and restores
parameters; predict_logits_from_preprocessed_data reloads parameters by fold.
A wrapper that adds state-dict prefixes will fail unless explicitly handled.
predict_single_npy_array warns that NibabelXYZ differs from SimpleITKZYX;
preserve trained SimpleITKIO rather than freestyle array ordering. Predicted
logits require official native-shape probability-space conversion. Do not skip
TTA, context, resampling or class competition to make this probe cheaper.
Follow-up source query lostSSH; no new login/bypass attempted. Both prior
DeltaAI and Bridges sockets now closed/refused; requested fresh DeltaAI master.

Stronger competing work, primary fullHTML checked:
Class Based Thresholding (CBT,2210.15621) already learns class-specific pixel
exit thresholds from class-conditioned training probabilities, usingADP-C.
Its rule selects the threshold by predictedargmax class (Section2), not unknown
GT class. Thus a confident tumor-to-background error receivesBACKGROUND
threshold: increasing only the tumor threshold does not guarantee protection.
This is an inference from the explicit rule, not an observed failure in our data
or a novel theorem. Formalrisk/ordinarybackground-threshold baselines still
needed. Table1W48:387.80->299.10GFLOPs,81.31->80.69mIoU at[.99,.998]; these
are CityscapesFLOP/accuracy values, NOT latency on our3Dhardware or lossless.
https://arxiv.org/html/2210.15621v1

ADS_UNet (2304.04567,authorabstract) already proposes stage-wise additive
training/resource-efficient shallow supervision/performance-weighted nested
sub-UNets. Generic depth-curriculum/weighted-nested ensemble is therefore not
new either. Its34%trainingtime claim is vsTransformers in histopathology, NOT
our optimizedUNet++baseline; fullHTML unavailable, implementation not audited.
https://arxiv.org/abs/2304.04567

Additional related authorrepository okdlibri/risk-aware-early-exit describes
margin/Mahalanobis/entropy-change logistic routing and survivor-conditioned
predictor training. Read README only; paper-code attribution, pretrained-weight
retrieval and formalguarantees remain unverified. Treat as a lead, not confirmed
benchmark evidence. https://github.com/okdlibri/risk-aware-early-exit

No new GPUhours/production edits/protected evaluation action. Generic
class-aware early exit and generic depth-curriculum ideas further rejected;
candidate1 needs a substantive empirical distinction beyond these baselines.
Next inspect sliding-window/mirroring/channel accumulation before any parallel
head extraction optimization. Source/code feasibility is not an S-tier result.

## Literature-first rejection and cheaper next decision (2026-10-02)

No new allocation or inference in this update. The previous access-refresh
turn did not advance the research decision; this turn checks a different,
tempting mechanism before building more inference infrastructure.

### Candidate5: prevent auxiliary labels from teaching away tiny lesions

Proposed claim: nearest/thresholded label reduction can turn a true tiny lesion
into background, so suppress unreliable negative auxiliary supervision or use
foreground-preserving targets. This could explain a DS interaction without
inventing another architecture. It would matter if it prevented missed lesions
at unchanged inference cost. However, this mechanism/remedy is already studied:

- Park et al., NeuroImage2021, multi-scale highlighting foregrounds uses
  max-pooled auxiliary labels, compares average-pooled hard labels and loss
  weighting, with repeated cross-scanner experiments. Primary indexed Section3.3
  inspected; full-page opening encountered a verification page. This is a real
  competing lesion-supervision method, not a new pooling idea of ours.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8382044/
  https://pubmed.ncbi.nlm.nih.gov/33957235/
- KG-Seg, Electronics2026, Section3.4.2 explicitly constructs a max-pool
  occupancy reference and masks auxiliary negatives where downsampling has
  removed a lesion (equations21-23). Primary indexed section inspected; direct
  full-page opening unavailable. This directly overlaps the proposed validity
  mask. Performance/code/checkpoint reproduction NOT checked here.
  https://www.mdpi.com/2079-9292/15/18/4227
- Deep Feature Surgery already tackles shared-exit gradient conflict and
  training cost in CIFAR100/ImageNet (author abstract checked, already present
  in the historical brief). Generic conflict correction is not a fallback
  novelty claim. Its reported training reduction is NOT measured on our3DCT.
  https://arxiv.org/abs/2407.13986

Alternative explanations for a PanTS DS effect remain loss magnitude, different
branch weights, model capacity, augmentation/preprocessing and optimization;
label disappearance alone would not identify the causal training effect.
Our UNet++ full-resolution branches also differ from plain U-Net's lower-
resolution auxiliary heads. The4-cell comparison cannot isolate all of these.

Cheapest distinguishing experiment would first count per-component label loss
under the ACTUAL trained preprocessing, augmentations and DS scales, then
compare nearest/occupancy/validity-mask gradients on matched batches. A native-
grid mask rescaling is not that experiment. Those measurements could be useful
for interpreting the existing paper but would not defeat the above prior art.
Minimum causal paper would require resolution-matched supervision ablations,
two tasks, repeated seeds, those target baselines and class-specific outcomes.
Even a lower-bound4conditions*2tasks*3seeds is24training runs, plus evaluation;
cost=24R GPUh with R calibrated on the respective tasks, not assumed to equal
our main grid. This exceeds an inference-only reuse plan in researcher effort.
Decision: REJECT as a new main method direction before any GPU expenditure.
Do not modify the completed scientific grid to try this known remedy.

### Stronger allocation decision: inspect saved native outcomes first

Candidate1 remains the strongest NEXT EVIDENCE choice, not a top-conference
method endorsement. Before implementing a stacked-head patient predictor,
inspect the saved development-validation outcomes for the frozen6case manifest.
The selected IDs and reference counts were frozen using annotations only;
retain all cases, including deepest failures. A summary with TP>0 establishes
only any-overlap CASE detection, not lesion-wise matching or clinical adequacy.
If all4selected positives have TP=0, additional miss risk versus the deepest
model is vacuous on this screen: stop the relative-safety probe instead of
replacing cases with easier ones. Any shallow rescue would need a separately
justified question and the existing overthinking/ensemble baselines; it is not
permission to rename this negative gate a positive method result.

At least one native detection merely makes a relative-loss calculation
non-vacuous; it is NOT a success gate or evidence of acceptable absolute recall.
Full-head results, FP burden, actual deployed cost, preprocessing parity and
novelty remain separate gates. The most useful next action is thus a free read
of6saved rows, not GPU inference, a router implementation or another download.
Bridges2 master currently absent; requested user reopening. DeltaAI read-only
hostname check succeeds. Main evaluation47320183 remains untouched.

Ranking after this pass:1conditional diagnostic;2training mechanism deferred;
3class-aware pruning insufficiently distinct;4compression method NO-GO;
5auxiliary-label remedy rejected by direct prior art. None clears the full-
paper investment bar. No new spending; last verified Slurm reconstruction
remains1.225833charge-equivalent GPUh, NOT newly verified posted debit.

## Reusable-model inventory and whole-paper statistical cost (2026-10-02)

Rechecked the official repositories, not just search-result descriptions.
Our own DS-on checkpoint remains the only fully loaded, provenance-audited
medical multi-exit model in this workspace. No new model downloaded or loaded.

| Resource | Evidence actually checked | What remains unproven |
| --- | --- | --- |
| Our PanTS UNet++ DS-on | Local strict trained-weight CPU parity for j2..j5 | Patient/GPU runtime, all branch outcomes, deployment router |
| Official UNet++ v1 MSD Liver models | PyTorch README advertises a SharePoint model folder and per-branch liver/tumor results | SharePoint fetch failed; actual files/hashes/folds/trainer revision/exit supervision unknown |
| ADP-C HRNet W18/W48 | Official MIT-code README has Drive model links, four-exit commands and Cityscapes metrics | Drive fetch unavailable through web tool; checkpoint integrity, data access and runtime not reproduced |
| UNet3+ | Author README/source repository exists | No reusable supervised-exit checkpoint identified in the inspected README; not evidence none exists anywhere |
| STU-Net | Official README offers pretrained TotalSegmentator models and downstream fine-tuning | Pretrained anatomical labels do not establish tumor heads or genuine trained adaptive-depth exits; not a drop-in substitute |

Sources:
https://github.com/MrGiovanni/UNetPlusPlus/tree/master/pytorch
https://github.com/liuzhuang13/anytime
https://github.com/ZJUGiveLab/UNet-Version/blob/master/README.md
https://github.com/uni-medical/STU-Net

Important correction to a broad no-model impression: the official UNet++
PyTorch README DOES advertise pretrained MSD Liver/tumor models. It reports
branch-wise validation results. This is a potentially useful second DATASET,
not an independent second backbone. Published numbers are not proof that the
accessible weights match current repository training code. FINDINGS.md already
documents a branch/loss mismatch in the v1 source. Current author trainer still
sets ds_loss_weights=None and passes it to MultipleOutputLoss2. Do not infer
all-head supervision or silently repair a downloaded checkpoint's provenance.
No checkpoint was retrieved, so no claim about those weights' actual behavior.
https://raw.githubusercontent.com/MrGiovanni/UNetPlusPlus/master/pytorch/nnunet/training/network_training/nnUNetPlusPlusTrainerV2.py

MESS (ECCV2022) is another strong deployment baseline, not a new idea of ours:
its author abstract already describes two-stage multi-exit segmentation training
and post-training architecture/policy search. The advertised <1GPUh is search
cost, NOT total model training/paper cost. Public usable weights/code not verified
in this pass; do not transfer its speed claims to3Dlesions.
https://arxiv.org/abs/2106.03527

### The cohort, not just runtime, constrains a cheap credible safety claim

Executed our existing exact-binomial CPU functions on the frozen174positive
case count. New values recorded in exit_cohort_design_limits_20261002.json;
no extra patient data, prediction inspection, inference or GPU allocation.

- Even if ALL174independent positives were an untouched validation set and
  zero additional misses occurred, a fixed-policy one-sided95% upper bound
  would be1.707%, not1%. One1% claim needs299zero-failure positives.
- A hypothetical87calibration/87held-out-positive split gives3.385% upper
  bound with zero failures on the holdout. This is a design illustration,
  NOT a split already performed or evidence zero misses occurred.
- Three hypothetical equal29case strata from that holdout give13.167% per-
  stratum upper bounds using delta=.05/3 for simultaneous95% coverage. Three
  simultaneous5% bounds need80zero-failure positives PERstratum (240total).

These are known statistical calculations, not novel theory and not a clinical
tolerance recommendation. Nonzero failures, threshold selection, unequal
size strata and patient/scanner dependence only make the simplistic design
less reassuring. CRC expected-risk calibration and binomial holdout validation
are different guarantees; don't rename one as the other. The main901test
remains protected, not a convenient extra calibration dataset.

Consequences for the PI plan: choose an honest empirical endpoint first; don't
promise a1%tiny-lesion risk certificate with this cohort. Full-paper feasibility
must include independent positive-case count, source/model overlap and actual
checkpoint availability, not only2000inferencepasses. A smaller descriptive
audit may be feasible, but is not automatically a top-conference contribution.

Next highest-information action remains reading the6frozen saved native rows.
Bridgesidea socket authoritatively missing on recheck; local progress folder
contains only SUMMARY.md/checkpoint_manifest.json, not per-case outcomes.
No new job, new spending, production change or47320183action. Goal active;
conditional diagnostic recommendation unchanged, full-paper commitment NO-GO.

## Decisive frozen native-outcome gate completed (2026-10-02)

User reopened Bridgesidea master. Read ONLY the frozen development-summary
file, size bounded12MiB, exact SHA256fecb2318... matched,1800unique IDs. All6
prespecified rows retain expected reference counts; nonnegative integral
TP/FP/FN/TN and n_pred/n_ref identities pass. All4positive cases haveTP=0.
Off-annotation voxels:3548=936,5782=535,others0; both fixednegatives predict0.
Allselectedrows retained in frozen_exit_native_outcomes_20261002.json.

This is0/4any-overlap detection on four SMALLEST annotation-positive cases,
not overall174positive accuracy, clinical sensitivity or a representative
cohort. It triggers the predeclared inadequate-native STOP for the current
relative-safety probe. Do NOT build/submit oracle/controllerGPUinference or
replace failures with easiercases. No shallow results measured; potential
rescue would be a different question with knownoverthinking/ensemblebaselines.

LiveDeltaAIsacct recheck of17campaign IDs matches all terminalstates/elapsed/
billing and4413weightedseconds=1.225833charge-equivalentGPUh. No newallocation;
posteddebitstillunverified. No47320183/main901evaluationread or mutation.
Ordinarypublic author-modelHEAD/GET reached200OneDriveHTML(~395kcharacters)
withoutlogin or payloaddownload; thisdoesNOTverifycheckpointaccess/provenance.

Finaldecision/ranking/alternativeexplanations/wholepapercostsensitivities/
PIexecutionplan/completionevidence:IDEA_SEARCH_DECISION_20261002.md.
No candidateclearsinvestmentbar; strongestremainingquestionis absolute
rare-lesionrisk/compute, notanestablishednewmethod. Finishboundedsearchwith
honestNO-GO ratherthanconsumeunusedbudget; mainpaperworkremainsseparate.

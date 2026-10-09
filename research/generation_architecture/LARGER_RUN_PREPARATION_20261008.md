# Larger-run preparation: PI direction, not automatic launch

User priorities: move quickly, preserve experiment quality, minimize charged GPU
time. Current explicit cap0.5h covers architecture feasibility including startup
and failures, not an unlimited production sweep. Protected 2x2 remains untouched.

## Draft versus current test fixtures

Re-read all8pages of supplied CVPR draft, visually inspected method page2.
Its central claim is depth-scheduled sampling at matched quality, not superior
full-depth architecture quality. It calls for usable supervised heads at each
depth, epsilon MSE, AdaLN-zero conditioning, a noise-bin relative-depth-error
measurement, exact pruning, measured wall-time and trajectory compute, plus a
reversed schedule control. CIFAR/transformer experiments in that draft are not
permission to abandon the PI's later conditional CT architecture instructions.

Current paid pilots use small FiLM/group-normalized cross-attention prototypes,
deepest-head epsilon L1, not the full draft recipe or pretrained SMILE denoiser.
Do not infer depth scheduling or generation-speed gains from their results.

Original convolutional AdaLN-zero adapter prepared locally: channel LayerNorm,
zero timestep modulation/gates, zero prediction heads; same nested topology and
cross-attention interface. This is a convolutional adaptation, NOT an exact DiT
or SMILE implementation. CPU tests cover projected-skip identity at initialization,
all-head nonzero supervision gradients, and nontrivial shallow-head bitwise parity
with exactly the retained graph. GPU behavior and useful learning not validated.

## Cost-ordered next work

1. Complete consolidated conditioning/capacity pilot3345990: two seeds,1000updates,
   conditioned plain34/68/136 versus nested32/64/128 and phase-only plain control.
   Cached features/images loaded once, no new downloads, six-minute hard cap,
   independent saved per-arm reports/checkpoints, no retries. Development conditions
   correct/mismatched-patient/absent at identical noisy targets and fixed t100/500/900.
   One20step DDIM generation per arm/seed and a copy-source baseline. Generated
   pixel L1 is explicitly unregistered engineering evidence, not a tumor metric.
2. Use those outcomes to identify the actual limitation: conditional signal, model
   training sufficiency, or generation chain. Do not promote nested simply because
   its epsilon loss is lower. Failure of1000update tiny models is NOT a proof the
   PI's architectural idea fails, and cannot justify training everything longer.
3. Before a longer pilot, wire/test the intended supervised heads and conditioning;
   explicitly fix MSE weights and total loss scale. Count active versus instantiated
   parameters, including unused heads. Re-measure runtime on real inputs, not FLOPs
   alone; choose schedule using development bins, never protected test results.
4. For medical claims, verify patient identity/phase/correspondence, independent
   splits, annotation scope, protected overlap exclusion, positives and negatives.
   Current single tumor candidate and conflicting mask extents do NOT clear this.
   General-pair enhancement training followed by separate labeled NC tumor-dev
   testing is possible as a separately declared endpoint; perfect registration is
   needed for voxel-level paired fidelity, not every conceivable learning recipe.
5. Prepare fixed-detector input/output mapping, geometry-preserving preprocessing,
   phase-negative controls, lesion sensitivity/false positives, tumor Dice and size
   strata. Do not run the detector on clearly failed generated images just to produce
   a number. Do not invent an acceptable cancer-recall loss margin on the PI's behalf.

## Production launch checklist (currently NOT cleared)

- Actual production architecture/trainer, not a small adapter smoke fixture.
- Certified input manifest and appropriate clinical/evaluation scope.
- Correct-source benefit and generated-fidelity evidence on independent development.
- Frozen safety criteria and tumor preservation/detection evidence where claimed.
- End-to-end preemption/resume at the target production recipe, atomic checkpoints,
  failure propagation, no staging on paid GPUs, dependency/version assertions.
- Explicit finite additional production budget and current hardware charge check.

`larger_run_gate.py` fails closed on absent gates/budget/data flags. It is a CPU
validation scaffold, NOT a certificate of clinical adequacy or runnable production
launcher. No pretend-ready sbatch file or long allocation is submitted while the
actual model, data and medical gates are unverified.

The gate explicitly distinguishes general architecture-only development training
from tumor-preserving production. General-pair architecture training does NOT
require tumor-positive pairs or claim cancer preservation; it still needs explicit
budget, an independently verified data split, appropriate generation/conditioning
checks and production implementation/resume evidence. The medical endpoint has
the additional tumor-pair/safety gates. Do not turn a tumor endpoint's restrictions
into a universal assertion that every architecture experiment needs tumor labels.

## Completed controlled pilot

Job3345990 COMPLETED,ExitCode0:0,279 allocated seconds,billing2000.
No retries; all6arms finished1000updates and exact serialized GPU next-update replay.
Current0.5h cap cumulative usage: (37+22+279)*2/3600 = **0.187778estimatedcharged h**
before provider rounding. Derived result JSON saved; checkpoint/CT/image arrays
remain isolated remotely, generated montage inspected locally outsideGit.

| Seed | Near-capacity arm | Correct-source epsilon L1 | Wrong-source epsilon L1 | Training seconds |
|---|---|---:|---:|---:|
|1729|Plain|.335918|.352454|38.306|
|1729|Nested|.310612|.327553|42.774|
|2718|Plain|.336982|.353595|37.600|
|2718|Nested|.321994|.343134|42.819|

Counts1,200,976plain and1,202,536nested,~0.13%difference; unused heads included.
Source benefit appears in both seeds under matched forward interventions. Against
the independently trained phase-only plain model, plain conditioned is slightly
worse (.335918/.336982 versus .333904/.333190), whereas nested has lower epsilon
error. This remains one training candidate patient and one development candidate,
four correlated triplets and48fixed noise/timestep probes, not48independent cases.

20step DDIM images FAIL enhancement readiness: generated-target unregistered L1
.5274–.7762versuscopy-source .08096; VAEtarget reconstruction .01070. Best nested
montage visually shows noise, not plausible preserved anatomy. VAE reconstruction
is a diagnostic lower-level control, not an achievable oracle clinical baseline.
Neither epsilon ranking nor source sensitivity clears generation or tumor safety.
This short-training failure is not proof the architecture direction is invalid;
it is a reason not to scale this fixture or score cancer on its noisy outputs.

50CPUtests passed6.480s before scope-separation follow-up. Prepared AdaLN-zero
adapter has CPU contracts only; no paid training for that adapter this turn.
No production or frozen-detector job submitted. Remaining work includes verifying
the intended learning/sampling recipe and general-pair development dataset, then a
small intended-architecture check—not a blind longer sweep on these same fixtures.

CPU-only general venous inventory proposal is also prepared:18candidate patient
groups,19pairs,14train/4development candidate patients. Group aliases stay on the
same side and public scan IDs are checked against cross-split duplication. This is
NOT a locked experiment split or identity/phase/overlap certification; no new CT
files downloaded and no training approved from this proposal. It supplies a concrete
small general-pair data option without falsely requiring every pair to have cancer.
Patient aliases are discovery metadata, not independently verified patient identity.

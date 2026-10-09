# Authorized tiny architecture-learning pilot

User cap: **0.5 charged GPU-hours total**, including startup and failures.
This is an engineering feasibility check following the PI's U-Net versus U-Net++
diffusion direction, not SMILE reproduction, tumor preservation evidence, a
capacity-matched comparison, or a paper-quality speedup result.

## Prepared design

- One GH200, ten-minute Slurm cap, no automatic retry. At the previously observed
  billing factor two, the maximum is one-third charged GPU-hour before provider
  rounding. Recheck the current charge rule and rounding before submission;
  authorization is a spending ceiling, not permission to assume billing details.
- Two small original conditional denoisers, 200 AdamW updates each, same explicit
  posterior/noise/timestep draws, input order, phase prompts and optimizer recipe.
  Same channels do not mean equal parameter counts; report both counts.
- Fixed-image frozen VAE posterior parameters and text embeddings are cached;
  posterior samples remain fresh. No clinical losses, EMA, compilation or pruning.
- Exact saved-checkpoint next-update replay is required on the actual GPU.
- Eight training triplets from two provisional phase-pair groups; four development
  triplets from a separate provisional group. Both phase prompts occur in training.
  Candidate patient identity, registration, annotation scope and overlap exclusions
  are **not certified**. No protected PanTS evaluation data are accessed.

## Verified now

CPU staging of four additional public CT files completed remotely (452,042,142
bytes), with pinned hashes and finite normalized arrays. The first staged NPZ hash
is `87220c9eca7bb8c0dfd29a896676b60f61f2d730f4a05a4ecea6afe45d63f91e`.
The improved split assembler has **not run** remotely yet. Candidate montage has
not been retrieved/visually inspected. All 42 local CPU tests pass (2.111 seconds).
GPU learning and GPU checkpoint replay are **not validated yet**.

## Before release

1. Reconnect using the user-authenticated WSL master, without autonomous MFA.
2. Retrieve and visually inspect staged candidate images; assemble and hash the
   improved split on CPU; preserve all uncertainty flags.
3. Verify isolated imports, offline assets, shell syntax, exact resources and
   current charging/rounding. Submit held, inspect, then release only if affordable.
4. Collect finite learning curves, actual elapsed/accounting data and exact GPU
   replay result. A failed check is a failed pilot, not permission to relax it.

No learning job was submitted as of this note. The requested
`deltaai-pilot-oct8.sock` failed the fail-closed connection check. Existing 2x2
training/evaluation and original data remain untouched.

## Subsequent verified launch, October 8 client date

Connection reopened. CPU-only input/import/config preflight passed on Torch
2.10.0+cu129, diffusers0.35.1 and transformers4.56.1. Improved arrays assembled;
SHA256 `c781e6f41143b0ca22180f1a9161bde9f790693a85add72d173e40f9ea2d83d1`.
Derived manifest is saved alongside this note, not the CT/NPZ files.

Actual montage review: first training pair covers abdomen; second covers
neck/chest at these slice indices. Within-pair anatomy broadly corresponds,
but motion/registration are not certified. This mixed-region tiny pilot remains
engineering-only, not a pancreas-specific training or fidelity comparison.
Development candidate montage shows corresponding abdominal anatomy; its mask
scope discrepancy remains unresolved and masks are not used in this pilot.

Live partition billing weights and the official guide both indicate interactive
factor two. Held submission **3345959** verified account `bdyo-dtai-gh`, one
GPU, two CPUs,16G RAM, billing2000, ten minutes, Requeue0; then released.
Verified RUNNING on gh013; this denotes allocation, not yet confirmed updates.
No other job was changed. Final elapsed/curve/replay results still pending.

## Learning result

Job3345959 COMPLETED, ExitCode0:0,37 allocated seconds, billing2000.
Estimated charge37*2/3600 = **0.020556 GPU-hours**, excluding provider rounding.
Both arms completed200 updates and passed bitwise exact saved-checkpoint
next-update replay. Source JSON is `architecture_learning_result_3345959.json`.

| Small prototype | Initial development epsilon L1 | Final epsilon L1 | Learning+checks seconds | Parameters |
|---|---:|---:|---:|---:|
| Plain |0.86082|0.68127|6.7193|1,096,516|
| Nested |0.84122|0.60985|7.6709|1,202,536|

Nested final epsilon error is10.48% lower, timing14.16% higher, parameter count
9.67% higher. One seed/four correlated development triplets and unequal capacity
do not support a method-win claim. Denoising loss is NOT image reconstruction
quality, Dice, cancer sensitivity or demonstration of faster image generation.

## Cheap follow-up and next decision

Frozen conditioning probe3345965 submitted held, verified billing2000, oneGPU,
three-minute limit, Requeue0, then released. Maximum additional estimated charge
0.1h; cumulative maximum with completed learning pilot ~0.12056h before rounding,
inside the0.5h authorization. Probe changes only input-source/phase conditions;
never updates weights or writes original checkpoints. Zero source is deliberately
out-of-distribution, so sensitivity is a diagnostic, not causal/clinical evidence.

Before a scientifically meaningful architecture experiment:

1. Verify source and phase sensitivity, and add a proper condition-permutation
   control at fixed noise/timestep. No sensitivity is a warning, not proof the
   architecture cannot eventually learn conditioning.
2. Implement the CVPR draft's intended production conditioning design, explicitly
   distinguish it from this small FiLM/cross-attention fixture, and test contracts.
3. Predeclare capacity-matched and compute-matched controls; use at least multiple
   seeds and paired timing repeats. Same widths alone do not isolate architecture.
4. Obtain/certify independent abdominal paired data, correspondence and annotation
   scope, with protected evaluation overlap exclusion. Use real clinical outputs
   only after this gate; mixed-region tiny inputs are engineering fixtures.
5. Test decoded CT enhancement and frozen tumor-segmentation performance, then
   assess error versus actual end-to-end generation time. Do not extrapolate this
   pilot's epsilon loss or step timing into a clinical/speed claim.

## Frozen conditioning result: warning, not a success claim

Job3345965 COMPLETED, ExitCode0:0,22 allocated seconds, billing2000.
Both jobs total59 allocated seconds; estimated cumulative charge **0.032778h**
before provider rounding. No additional jobs submitted or automatic retries.
Results: `architecture_conditioning_result_3345965.json`.

| Prototype | Normal epsilon L1 | Zero-source epsilon L1 | Other-phase epsilon L1 |
|---|---:|---:|---:|
| Plain |0.681266|0.677698|0.679750|
| Nested |0.609855|0.605254|0.610349|

Predictions change under both interventions, but neither arm demonstrates useful
source conditioning: zero-source has slightly lower error here. Wrong-phase
effects are tiny/mixed. Four correlated samples, fixed noise draws,200 updates
and out-of-distribution zero source do not establish statistical significance or
failure of conditional diffusion. They DO prevent claiming this pilot has learned
CT enhancement simply because epsilon loss fell. No inference on clinical data,
no tumor metric or enhancement-fidelity claim, no segmentation job touched.

The next learning experiment should explicitly evaluate correct versus mismatched
source on independent verified abdominal pairs at fixed noise/timestep, compare
against an unconditional control, and include decoded image fidelity. Predeclare
capacity/compute controls before expanding training. Do not simply increase model
size or run a long study on the strength of the lower nested epsilon loss.

CPU parameter-count preparation (no training performed): plain widths(34,68,136)
has1,200,976 parameters versus nested(32,64,128)1,202,536 (~0.13% difference).
Plain(32,64,128)1,096,516 versus nested(30,60,120)1,092,678 (~0.35% difference).
These give two near-capacity-matched control candidates rather than assuming
identical widths are fair. Counts include instantiated unused heads; production
analysis must separately report active and total parameters. This does not match
actual FLOPs, wall-time, initialization, or prove equal representational capacity.
Compute-matched comparisons remain a separate requirement.

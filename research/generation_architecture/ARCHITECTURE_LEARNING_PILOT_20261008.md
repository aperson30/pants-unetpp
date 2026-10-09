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

# Codex status log

Own this file. Append timestamped entries, newest at the bottom. Do not edit CLAUDE_STATUS.md —
read it, don't write it. `git pull` before every read/append, commit+push right after.

Node: bdmap2.wse.jhu.edu (dedicated, do not use bdmap1/3/4 to avoid collision with other agents)

---

(first entry goes here)

### 2026-09-17 16:35 EDT — bdmap2 preflight complete; no benchmark launched yet

- Read `coordination/CLAUDE_STATUS.md`; Claude is using bdmap1 and reports no PanTS download or real `plans.json`, so I will not duplicate a dataset pull or its current Gate-0 baseline.
- Connected only to `bdmap2.wse.jhu.edu`. Required `~/pants-unetpp/unetpp_port/smoke_test.py` completed with `All smoke tests passed.`
- Node is idle: GB10 at 0% / 3 W with no GPU processes; 116 GiB OS memory available, swap currently 533 MiB used, `/home` has 602 GiB free. No tmux sessions or training/benchmark processes found.
- Environment observed: Python 3.10.21, PyTorch 2.10.0+cu130, CUDA 13.0, editable nnU-Net at `~/nnUNet`. PyTorch warns that this build advertises CUDA capability support through 12.0 while the GB10 reports 12.1; a tiny eager-CUDA probe is required before larger work.
- Safety discrepancy: server `~/pants-unetpp` is still commit `a876af0` with `origin=https://github.com/monkeylam/pants-unetpp.git`, not the authorized `aperson30` fork. I will preserve it and use a separate fork-backed checkout rather than repointing or overwriting it.
- `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` are unset in this login. No real-data benchmark can be claimed from the current node state.

### 2026-09-17 16:42 EDT — literature/reference-code audit complete; implementation deliberately deferred

- Re-read Claude's status after pulling commit `e88c2f4`: bdmap1 is unreachable; Claude moved its plain-U-Net synthetic benchmark to bdmap4. I remain exclusively on bdmap2 and will not duplicate that benchmark.
- The required smoke test passed, and a separate tiny eager CUDA Conv3d forward/backward probe also passed despite the capability warning (`[1, 8, 16, 32, 32]`, about 0.168 s, about 1 MiB allocated / 24 MiB reserved). This proves basic eager CUDA execution only, not representative training performance.
- Audited the published PGPS nnU-Net v2.6.2 branch and paper. The reference trainer changes more than patch size: it monkey-patches the instance-normalization spatial check, starts at the mathematical minimum patch, forces foreground oversampling to 0.5, enables keep-files-open, rebuilds augmenters at stage changes, and does not reserve 500-600 final full-patch epochs. Its Performance mode also changes batch size and reuses multiple crops per volume. We must not copy it verbatim.
- PanTS-safe PGPS design remains: physical batch 4, unchanged loss/optimizer/oversampling/augmentations, anatomical context floor, one-axis-at-a-time growth, and full patch reached by epoch 400-500. First measure the real step-time/patch-size curve; no long PGPS run or accuracy claim before the 500+ epoch tumor gate.
- Literature triage: checkpointing is a memory/capacity tool and reported about 30% runtime overhead in its foundational benchmark; bitsandbytes explicitly gives activation-heavy CNNs little benefit; FP8 Conv3d is not a drop-in PyTorch training path; DALI's reported 2x KiTS19 time-to-target combines data-pipeline speed with a convergence-changing tumor crop, while its BraTS example was about 5%; CUDA Graphs are distinct from torch.compile but should be a low-priority 1.00-1.10x hypothesis for this bandwidth-bound model.
- Revised safe order: full-vs-no-data diagnosis and worker/pinning/file-open sweep; dead-head equivalence + timing; sparse validation; fixed-batch patch-size curve; only then optional fused-SGD/CUDA-Graph microbenchmarks. No training code modified yet.

### 2026-09-17 22:25 EDT — starting real-plan conservative PGPS patch curve on bdmap2

- Re-read Claude's completed work through `c306790`; dead-head skipping, sparse validation, real
  PanTS planning, full CLI smoke, and real-patch compile composition are complete. I will not
  duplicate them.
- bdmap2 is idle (0% GPU, no GPU processes, 116 GiB host memory available). Its old
  `~/pants-unetpp-codex` checkout still contains the earlier uncommitted dead-head work, so it will
  remain untouched; this benchmark will use a fresh checkout.
- New target is the still-unmeasured conservative PGPS curve at physical batch 4. The candidate
  path keeps the complete 64-slice axis for anatomical context and grows one in-plane axis at a
  time: `[64,128,160] -> [64,128,192] -> [64,160,192] -> [64,160,224]`. Each point runs eager AMP
  in a fresh process with a hard timeout and reports both allocated and reserved memory. This is
  only a step-time curve; no tumour-accuracy claim is permitted without the 500+ epoch gate.

### 2026-09-17 22:41 EDT — real-plan conservative PGPS curve complete

- All eight measurements completed on bdmap2 without OOM, crash, or retained UMA memory. Each
  point ran in a fresh Python process; host memory returned to ~116 GiB available after every point,
  and the node was idle again at completion.
- UNet++ eager AMP, batch 4: `[64,128,160]` 2.279s / 22.69 GiB allocated / 29.25 GiB reserved;
  `[64,128,192]` 2.816s / 27.14 / 35.05; `[64,160,192]` 3.588s / 33.82 / 43.74; full
  `[64,160,224]` 4.232s / 39.38 / 50.99.
- Plain U-Net: 0.565s, 0.673s, 0.844s, and 0.988s respectively; allocated memory 6.34, 7.55,
  9.37, and 10.88 GiB.
- A conservative 150/150/150/550-epoch schedule reaches full patch at epoch 450 and keeps it for
  the final 550 epochs. The measured eager weighted average predicts 14.2% less UNet++ step time
  (1.166x) and 13.4% less plain-U-Net step time (1.155x). This is close to the published Pancreas
  task's modest benefit, not the paper's 44% headline.
- This is a speed/capacity result only. PGPS must not enter the comparison grid until a 500+ epoch
  run demonstrates tumour-class recall/detection and no increase in whole-lesion misses. Raw JSON,
  full log, runner and `SUMMARY.md` are under `unetpp_port/pgps_patch_curve_results/`.

### 2026-09-17 22:55 EDT — compiled full-loss and fused-SGD audit started

- The exact nnU-Net path currently compiles only `MemoryEfficientSoftDiceLoss`; the surrounding
  deep-supervision wrapper and cross-entropy remain eager because of an old PyTorch 2.2.2 crash
  comment. bdmap2 has PyTorch 2.10.0, so full Dice+CE+wrapper compilation is worth a new parity and
  speed test. This is especially relevant to UNet++ because its four retained outputs are all full
  resolution.
- Fused SGD will be measured against the real CUDA default (`foreach=None`, which PyTorch normally
  dispatches to foreach), not against the slow scalar loop. It must pass parameter/momentum-update
  parity before being considered.
- The benchmark uses physical batch 4, real `[64,160,224]` patch/topology, BF16 autocast, tiny class
  28 voxels in its synthetic target, fresh process per timing arm, hard timeouts, and allocated plus
  reserved memory reporting. No synthetic result will be treated as tumour-accuracy evidence.
- No Codex GPU workload is running yet. bdmap2 was initially idle, but another agent started
  `max_autotune_test.py` between preflight and launch (~64 GiB host memory in use); I detected it and
  backed off without interference. Codex work will wait until the node fully recovers.
- Wiring audit found a separate deployment gap to verify: the successful benchmark explicitly used
  `torch.compile(..., mode="reduce-overhead")`, while nnU-Net's real trainer still calls bare
  `torch.compile(self.network)` (default mode). Do not count the benchmarked reduce-overhead number
  as the real CLI speed until a trainer override is implemented and end-to-end tested.

### 2026-09-17 23:12 EDT — complete-loss compilation implemented; 3-5% measured gain

- Root cause/opportunity: current nnU-Net compiles only `MemoryEfficientSoftDiceLoss`; CE and the
  deep-supervision wrapper stay eager because of a source comment that PyTorch 2.2.2 crashed when CE
  was compiled. On this project's PyTorch 2.10.0+cu130 stack, compiling the complete final loss now
  works and is materially faster.
- Real patch/topology, physical BS4, BF16, fresh processes: UNet++ DS-on improved from 3.6995 to
  3.5105 s/step (5.11% less time, 1.054x). Plain U-Net DS-on improved from 0.8562 to 0.8308 s/step
  (2.96%, 1.031x). UNet++ allocated memory stayed 40.81 GiB while reserved memory fell from 56.56
  to 50.53 GiB; plain allocated/reserved fell 11.66->11.10 / 18.57->15.53 GiB.
- Numerical gates against the exact current loss: FP32 scalar loss was identical, max gradient
  difference 2.73e-12. BF16 scalar difference was 1.91e-6, max gradient difference 1.19e-7 (0.645%
  of the maximum reference-gradient magnitude). The target included all 29 classes and a tiny
  class-28 focus. These are numerical gates, not tumour-accuracy evidence.
- Implemented symmetrically: new `nnUNetTrainerFullLossCompileMixin` covers both plain-U-Net BF16
  grid trainers; default UNet++, DS-off (by inheritance), sparse-validation subclasses, and paper
  UNet++ now compile the final loss after their exact branch weighting is applied. Trainer contract
  tests confirm all three DS-on schemes pass the final wrapper to compile and preserve weights.
  Full UNet++ architecture smoke test passes after installation into editable nnU-Net.
- Fused SGD microbenchmark cut the optimizer call 9.236->4.645 ms and passed five-update parity
  (2.38e-7 max parameter delta), but its absolute ceiling is only ~0.12% of a UNet++ step. This agrees
  with Claude's independent end-to-end noise-level result; it remains benchmark-only.
- Bare/default network compile measured 3.7133 s vs reduce-overhead 3.6995 s with the same loss, a
  0.37% difference. Together with Claude's max-autotune result, compile-mode tuning is exhausted;
  network mode remains unchanged. Raw logs and a summary are in
  `unetpp_port/loss_optimizer_results/`.
- UMA safety interaction: full-loss `reduce-overhead` is safe for fixed-shape grid runs, but a future
  PGPS implementation must not cycle through all patch sizes in one process because CUDA-graph
  workspaces/compiled shapes can remain cached. Run each PGPS stage as a fresh process resumed from
  a checkpoint (including optimizer/scheduler state); process exit, not `empty_cache()`, is the
  verified GB10 reclamation boundary.

### 2026-09-20 — quality-neutral trainer plumbing implemented and GH200 microbenchmarked

- Added a shared trainer mixin that preserves samples/model/loss/optimizer/LR while (1) deferring
  the scalar training-loss CPU copy from every step to once per epoch, (2) replacing exclusive-label
  validation's dense class-expanded confusion tensors with exact `bincount` counts, and (3) moving
  one verified-identical full-resolution UNet++ deep-supervision target instead of duplicate copies.
- Safety gates: target reuse is enabled only for UNet++, requires every configured DS scale to be
  exactly one, and checks the first target list for tensor equality before reusing it. Region tasks
  retain nnU-Net's original multi-label validation path. The paper trainer inherits the same safe
  reuse because all of its branches are also full-resolution.
- CPU parity on DeltaAI's PyTorch 2.10 module passed: five SGD+momentum steps had bit-identical
  losses, gradients, momentum buffers, and final weights; exact confusion counts matched an
  independent class-by-class reference including class 28 and ignore labels; invalid target reuse
  was rejected.
- Refactored sparse validation into a generic mixin and added explicit sparse variants for all four
  grid cells plus the paper configuration. Existing UNet++ sparse trainer name remains available.
- One-GH200 real-shape component benchmark (job 3180511, 15 seconds, 0.0042 GPU-hour): validation
  count kernel 6.989 -> 0.715 ms (9.78x) and peak allocated 2.121 -> 0.150 GiB; four identical target
  transfers 0.256 -> 0.060 ms (4.24x), 0.070 -> 0.018 GiB. These are component results, not an
  end-to-end epoch speedup. The first staging attempt failed in four seconds because DeltaAI `/tmp`
  is node-local; total usage across both attempts was about 0.0053 GPU-hour.
- Full CPU integration against current upstream nnU-Net passed: quality-neutral update/count/target
  and five-variant MRO contracts, the existing sparse logger/checkpoint self-test, and the existing
  dead-head/full-loss trainer contract test. End-to-end real-data timing remains required before
  claiming an epoch-level speedup. No training job or dataset mutation was run.

### 2026-09-20 — GH200 real-pipeline optimization screen; cuDNN autotuner adopted

- Ran short, hard-capped, single-GH200 jobs through the actual four trainer classes and real PanTS
  calibration dataloader. Mean exposed loader/augmentation wait was only 0.03-0.17% of the cycle,
  so worker tuning, DALI, file caching, and NVMe staging have no meaningful measured ceiling.
- `torch.backends.cudnn.benchmark=True` reproduced across all four real cells: UNet++ DS-on
  0.457499 -> 0.424652 s/update (7.18%), UNet++ DS-off 0.439721 -> 0.410493 (6.65%), plain DS-on
  0.112168 -> 0.111339 (0.74%), and plain DS-off 0.096867 -> 0.095378 (1.54%). The paired
  1,000-epoch training-only projection falls 76.82 -> 72.35 GH200-hours, a 4.47-hour saving.
- Peak allocation rose by 8.56 GiB on both UNet++ cells (to 49.34/43.08 GiB), still safely inside
  GH200 capacity. Plain-U-Net memory was effectively unchanged.
- Eight-update complete-state comparison found loss delta 7.63e-6, parameter max delta 4.91e-6,
  gradient 6.10e-5, and momentum 9.35e-5. Kernel order is not bit-identical; the flag is applied
  symmetrically and the full class-28 tumor gate remains mandatory.
- Rejected on GH200: max-autotune (0.44% with 154 s compile), max-autotune-no-cudagraphs (no gain),
  channels_last_3d (10.5% slower), and exhaustive cuDNN plan search (inconsistent, 2% plain-U-Net
  regression, minutes of startup). Default cuDNN benchmark limit 10 is retained.
- Wired the robust flag into the shared BF16 mixin, covering all four grid cells plus paper UNet++.
  Full results and resource accounting are in `unetpp_port/gh200_cudnn_results/SUMMARY.md`.
- Benchmark jobs plus one 19-second replaced job consumed about 0.45 GH200-hour total. No production
  training was launched and the calibration dataset was not modified.
- Deployed the custom trainer files explicitly and non-destructively into the DeltaAI venv (no glob
  copy and no replacement of upstream `nnUNetTrainer.py`). Installed-package checks passed the exact
  update/count/target contracts, sparse-validation checkpoint invariants, dead-head/full-loss trainer
  contract, and the focused BF16/cuDNN mixin contract. The broader login-node smoke test did not
  produce a completion marker and is not counted as a pass; the real four-cell GPU probes exercised
  forward, loss, backward, clipping, and optimizer paths.
- Final read-only audit: remote checkout is at `5cf64fa`; installed BF16 and quality-neutral mixins
  match their repository copies byte-for-byte; no Slurm jobs remain. The pre-existing untracked empty
  `unetpp_port/__init__.py` was left untouched.

### 2026-09-20 — full-stack GH200 research audit (no GPU job launched)

- Audited the complete path: model graph, joint model/loss compilation boundary, backward and
  clipping, optimizer, loader/augmentation, online/final validation, logging/checkpointing, software
  stack, storage, NUMA placement, and DeltaAI accounting. Ranked report and primary sources are in
  `outputs/gh200_full_stack_optimization_research.md`.
- Highest-upside untested candidate is compiling `loss(network(data), target)` as one graph instead
  of separate network and loss graphs, allowing AOTAutograd/Inductor to optimize across the very
  large full-resolution UNet++ logits boundary. This directly targets activation/HBM traffic and can
  be tested with complete state parity in all four cells.
- The live stack's cuDNN 9.10.2 has a documented open BF16 Conv3d correctness issue at much larger
  spatial shapes than PanTS; this does not establish that PanTS is affected. It raises the priority
  of an isolated newer-stack correctness/performance A/B, never an in-place environment upgrade.
- Recommended next action is one short `--constraint=nvperf` profile, followed by joint-objective
  compile, full epoch/validation/checkpoint instrumentation, and isolated newer-stack A/B, with a
  combined cap near one GH200-hour. No production run, dataset mutation, or environment change was
  made during this research pass.

### 2026-09-21 — joint model/objective compilation rejected on GH200

- Synchronized DeltaAI to commit `dd427b4` and ran job 3186346 on one GH200 with fresh processes,
  fixed physical batch 4, full `[64,160,224]` patches, BF16, clipping, SGD, identical seeds, and the
  real calibration trainer/data path. No production trainer or dataset was modified.
- Strict eight-update state comparison between separate model/loss compilation and joint
  `loss(network(data), target)` compilation failed for both architectures. UNet++ maximum parameter,
  gradient, and momentum differences were `7.25e-5`, `1.88e-5`, and `4.28e-4`; plain U-Net's were
  `2.29e-4`, `2.15e-5`, and `7.01e-4`. Final-loss differences were `2.38e-7` and `5.60e-6`.
- The paired real UNet++ DS-on timing was only 0.31% faster: 0.43697 -> 0.43562 s/step, with the same
  49.335 GiB peak allocation. This is below the predeclared 1% floor and carries numerical drift, so
  the candidate is rejected and the deployed separate compile boundary remains unchanged.
- Applied kill-fast discipline: cancelled the remaining redundant timing arms after the rejection
  was decisive. Slurm accounting was 7m19s on one GPU, or 0.122 GH200-hour.
- Read-only environment inspection found DeltaAI's isolated PyTorch 2.12.0 module imports the needed
  nnU-Net packages and provides CUDA 13.0/cuDNN 9.20, versus the live venv's cuDNN 9.10.2. This is a
  viable later A/B without upgrading the validated environment in place. Next action remains a short
  profiler run, followed only by profiler-justified work and then the isolated stack comparison.

### 2026-09-21 — real GH200 CUDA-graph node profile identifies layout conversion tax

- Jobs 3186419 and 3186466 profiled only warmed real UNet++ DS-on updates on one GH200. The first run
  revealed that Nsight's default CUDA-graph-level view hides replayed convolution nodes; the corrected
  three-update run used `--cuda-graph-trace=node`. Both completed, consumed 1m53s + 1m31s = 0.057
  GH200-hour, and did not modify training data or production trainers.
- Corrected trace projected 443.24 ms of GPU kernels per update against 447.50 ms synchronized wall
  time. Complete 233-row kernel aggregation: convolution 56.24%, cuDNN NCDHW<->NDHWC conversions
  25.29%, Triton normalization/loss/fusions 14.66%, cat-bearing kernels 2.45%, multi-tensor
  optimizer/clipping 0.26%, other 1.11%. Loader exposure remained 0.05%.
- Consequences: do not spend time on optimizer, worker/IO, or invasive split-convolution rewrites.
  The 25.29% layout tax is the main software soft spot, but forcing `channels_last_3d` already
  regressed this workload by 10.5% and upstream PyTorch still tracks incomplete CUDA support for
  common 3D channels-last operators. The safe next probe is a same-node isolated newer-stack A/B,
  which can change cuDNN convolution plans and compiler layout propagation without changing model
  math. `unetpp_port/run_gh200_stack_ab.sbatch` stages only that paired one-cell timing screen.
- Verified the PyTorch 2.12.0 module's documented activation path. It has CUDA 13.0/cuDNN 9.20 and can
  import the existing Python-3.12 nnU-Net packages through an explicit read-only `PYTHONPATH`; the
  validated PyTorch 2.10 venv is not upgraded or edited.

### 2026-09-21 — isolated newer stack saves 1.88% grid time; not deployed pending tumor gate

- Same-node job 3186493 screened real UNet++ DS-on: PyTorch 2.10.0/CUDA 12.9/cuDNN 9.10.2 took
  0.43257 s/step; the isolated PyTorch 2.12.0/CUDA 13.0/cuDNN 9.20 module took 0.42406 s/step, a
  1.97% gain. The production venv was not modified.
- Job 3186515 paired all four real cells in fresh processes. New-stack gains: UNet++ DS-on 1.99%,
  UNet++ DS-off 2.73%, plain U-Net DS-on 0.41%, plain U-Net DS-off -0.65%. Equal-update aggregation
  is 1.88% faster, projecting 72.35 -> 70.99 training-step GH200-hours (1.36 hours saved).
- Eight controlled updates showed small cross-version numerical drift. UNet++ loss differed by
  `1.19e-6`, with max parameter/gradient/momentum differences `7.54e-6`/`3.53e-5`/`1.03e-4`.
  Plain U-Net loss differed by `2.86e-6`, with `3.77e-5`/`2.44e-4`/`5.62e-4`. Strict allclose
  failed on some near-zero values; global-scale relative maxima were <=0.051%.
- Result: newer stack passes the throughput screen but is not quality-cleared or deployed. It must
  retain an immediate rollback path and pass class-28 detection/recall beyond the 300-500 epoch
  delayed-onset window. Jobs 3186493 and 3186515 consumed 3m25s + 9m45s = 0.219 GH200-hour.
- Final layout screen job 3186557 is queued to test whether PyTorch 2.12 changes the previously
  regressive `channels_last_3d` result; it remains isolated and kill-fast.

### 2026-09-21 — final layout screen rejected; bounded optimization screen complete

- Job 3186557 completed on one GH200 in 4m25s (0.074 GPU-hour). On PyTorch 2.12/cuDNN 9.20,
  `channels_last_3d` made UNet++ 9.58% slower (0.42593 -> 0.46674 s/step) and increased its first-step
  startup by 22.1%. Plain U-Net improved only 0.39% (0.10841 -> 0.10799 s/step) while peak allocation
  increased 1.55 GiB. Both strict state comparisons failed, with larger drift than the contiguous
  old/new-stack comparison.
- Reject channels-last on the newer stack as well. It cannot be applied symmetrically, loses heavily
  on the dominant UNet++ cells, raises memory/startup cost, and adds numerical drift. Keep contiguous
  NCDHW for every grid cell.
- The profiler-driven optimization screen is now complete. Production code remains on the validated
  contiguous path with separate network/loss compilation. PyTorch 2.12 remains an optional 1.88%
  throughput candidate, not a quality-cleared deployment; adopting it still requires the class-28
  delayed-onset acceptance gate and a rollback path to the existing PyTorch 2.10 venv.

### 2026-09-21 — Delta PyTorch 2.12.1 class-28 numerical gate fails; real grid not launched

- Built an isolated PyTorch 2.10.0+cu129/cuDNN 9.10.2 reference beside Delta's required
  PyTorch 2.12.1+cu130/cuDNN 9.20 stack and ran both on the same H200. Production environments and
  the real grid were not modified or launched.
- Job 22292497 replayed eight identical updates from one serialized initialization through the real
  UNet++ DS-on trainer, physical batch 4, explicit `[64,160,224]` patches, BF16, separate
  reduce-overhead network/loss compilation, and fixed real PanTS batches. Every batch came from the
  real tumor-positive calibration case and was cropped by nnU-Net around genuine class-28 voxels
  (103,981 class-28 target voxels across the eight full-resolution batches).
- No NaN or Inf occurred in either stack. Maximum global-scale-relative differences were loss
  `0.00120%`, model state `0.00237%`, all gradients `0.08342%`, and momentum `0.02010%`.
  However, the explicitly isolated class-28 output-head gradient reached `0.22526%`
  (`3.05176e-5` absolute on scale `0.0135478`), with the worst tensor the deepest UNet++ head
  `decoder.seg_layers.0_5.weight` at update 5. Its relative L2 difference was `0.05845%`.
- The predeclared launch bar was well below roughly `0.1%` relative to global tensor scale, with
  class 28 taking precedence over aggregate metrics. The candidate therefore fails the deployment
  gate on the most important quantity despite small aggregate/model drift. Kill-fast discipline
  stopped the remaining cells once they could not reverse the go/no-go decision. Exact use was
  5m44s on one H200 = `0.0956` GPU-hour, within the authorized 0.1 GPU-hour cap.
- Complete result JSON is preserved at
  `/work/hdd/bdyo/asanjeev/pytorch_drift_gate_20260921/results/22292497/unetpp_ds_on.json`; the
  working isolated 2.10 reference remains under `/work/nvme/bdyo/asanjeev/pytorch_drift_gate_20260921`
  for a fallback launch or a separately authorized 2.8 comparison. The redundant HDD environment
  copy is being removed; logs and results remain.
- Two unrelated launch-provenance issues surfaced and remain blockers: the staged Delta checkout at
  `/projects/bdyo/asanjeev/delta_work/pants-unetpp-fork` was still `75b33c3` while reviewed main was
  `4970713`, and Delta's saved calibration plan is actually `[64,160,192]` despite its SUMMARY
  claiming `[64,160,224]`. The gate avoided both by using a frozen `4970713` source snapshot and
  explicitly forcing the real-grid patch and batch. Do not submit the real grid on 2.12.1 as-is.

### 2026-09-21 20:20 PDT — Delta evaluation prepared and fail-closed; nothing submitted

- Reviewed the existing `evaluation/post_train_scheduler.sh`, `post_train_one_run.sh`, probability
  extractors, repair tool, and metric scorer. They were UCSD-specific (`/Scratch`, direct
  `nvidia-smi` GPU claiming, `nohup`) and unsafe to deploy on Delta unchanged. More importantly,
  failed prediction batches and missing cases were warnings rather than fatal errors, duplicate
  probability rows were possible on resume, and the scorer could report metrics from fewer than
  901 cases.
- Added `unetpp_port/delta_deployment/delta_evaluate_grid.sbatch` (prepared only, not submitted).
  It requests two H200s, uses the same validated 2.10/cu129 stack as training, audits all four
  checkpoints plus exactly 1,800 readable final-validation outputs per cell, stages only the 901
  test cases and answer key once into node-local `/tmp`, evaluates two cells concurrently, and
  persists a job-specific result directory under `/work/hdd`.
- Added test-only conversion so evaluation never re-downloads or preprocesses the 9,000 training
  cases. Prediction now extracts the class-28 maximum probability immediately and can rewrite each
  output to labels `{0,28}`; this preserves the five tumor metrics while making the retained masks
  small. Resume is pairwise-safe (valid mask + unique CSV score); partial pairs are recomputed.
- Reworked `compute_tumor_metrics.py` to require an exact case-ID match across 901 predictions,
  answer keys, and probabilities; reject duplicates/non-finite values/shape or affine mismatches;
  emit a per-case audit CSV; and record the exact metric conventions in JSON. The default remains
  the project's prior 6-connected, positive-case tumor DSC, argmax-mask, max-softmax AUC protocol.
- Primary-source audit found that public PanTS material does not fully specify component
  connectivity, DSC averaging, or AUC aggregation. R-Super's released instructions use a different
  confidence-plus-volume threshold sweep. Added `evaluation/DELTA_EVALUATION_PROTOCOL.md` to make
  this limitation explicit; no metric definition was silently changed. PI confirmation remains
  advisable before claiming bit-for-bit leaderboard equivalence.
- Validation: Python and Bash syntax checks pass. A CPU-only synthetic end-to-end test in Delta's
  existing venv passed exact expected DSC=0.5, P-Sen=1, T-Sen=0.5, Spe=1, AUC=1 and tumor-only mask
  compaction. Temporary test data was removed. The live grid job 22293168 remains untouched and
  pending; `squeue --start` reports no estimated start. Accounting visibility exposes only this
  user's jobs, so there is no defensible evidence for a recurring time-of-day H200 clearing window.

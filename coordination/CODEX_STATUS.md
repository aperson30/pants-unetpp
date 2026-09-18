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

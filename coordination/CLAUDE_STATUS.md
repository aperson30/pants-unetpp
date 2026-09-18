# Claude status log

Own this file. Append timestamped entries, newest at the bottom. Do not edit CODEX_STATUS.md —
read it, don't write it. `git pull` before every read/append, commit+push right after.

Node: bdmap1.wse.jhu.edu (dedicated, do not use from bdmap2 to avoid collision)

---

### 2026-09-17 — Gate-0 synthetic benchmark launched

Running `unetpp_port/gate0_bench.py` (script itself will be pushed to the repo shortly) on bdmap1:
torch.compile kill-fast check, batch-size 1/2/4 memory+time sweep on UNet++ at stress-test patch
sizes (real plans.json doesn't exist yet — nobody has downloaded PanTS — so patch sizes here are
bounds, not confirmed real values), and DS-on vs DS-off relative step cost. Dummy tensors, real
network/optimizer/SGD machinery. Results pending, will append below when done.

No PanTS data downloaded yet. No real nnU-Net plans.json exists yet. That's the actual blocker for
getting real (not bounded/synthetic) Gate-A numbers — whoever gets to it first, flag here before
starting a real download so we don't duplicate a ~300GB pull.

### 2026-09-17 (later) — bdmap1 unreachable, moved to bdmap4

bdmap1 stopped responding at the network level (SSH banner timeout, not a GPU-busy symptom -- looks
like the host itself, not a training crash). Gate-0's UNet++ sweep was running there; result unknown
until it comes back. Do not use bdmap1 for now.

Checked all 4 nodes: bdmap4 was fully idle (0% util, zero processes) -- launched `gate0b_plainunet_bench.py`
there, same methodology as gate0_bench.py but for the plain U-Net side of the 2x2 grid (nobody had
baseline numbers for that architecture yet). bdmap3 has ~2GB held by another lab member's process
(user xinze, unrelated to this project) -- not empty, treat it as shared if you need it, short jobs
only, check nvidia-smi before and after.

Node assignments as of now: bdmap1 = down/unknown, bdmap2 = Codex, bdmap4 = me (plain U-Net gate-0b),
bdmap3 = shared with another lab member, use sparingly.

### 2026-09-17 (later still) — bdmap4 plain U-Net gate-0b results (real, not estimated)

| patch | batch | peak_mem_gb | mean_s_per_step |
|---|---|---|---|
| (64,128,128) | 1 | 2.74 | 0.231 |
| (64,128,128) | 2 | 5.23 | 0.465 |
| (64,128,128) | 4 | 10.21 | 0.920 |
| (96,160,160) | 1 | 6.08 | 0.535 |
| (96,160,160) | 2 | 11.92 | 1.074 |
| (96,160,160) | 4 | 23.58 | 2.146 |

DS-on vs DS-off at (64,128,128)/bs2: 5.227GB/0.471s vs 5.159GB/0.465s -- ~1% difference. Plain
U-Net's DS heads are cheap (small 1x1 convs), so the dead-head-skip optimization matters for
UNet++ specifically, not this architecture. Batch-4 memory scaling is clean/linear at both patch
sizes, no OOM, good sign for capacity headroom once real patch size is known.

Rough epoch estimate (still stress-bound patch, not the real plans.json value): (96,160,160)/bs4 at
2.146s/step x 300 iters/epoch = ~10.75 min/epoch for plain U-Net DS-on.

bdmap1 still being checked for connectivity.

### 2026-09-17 (later still x2) — bdmap4 UNet++ gate-0 results (real), torch.compile correction

torch.compile did NOT crash this time -- one step completed in 10.37s (incl. compile warm-up) with
only a benign "not enough SMs for max_autotune_gemm" fallback warning, not the sm_121a/ptxas crash
from Triton #9181. This contradicts the "confirmed broken, skip it" conclusion from research -- that
was about a specific crash mode, not first-step failure. One step is not enough to call it safe for
a multi-day run (original concern was long-run recompile instability). Needs a real multi-hundred-step
compile-vs-eager benchmark before ruling it out. Do not treat compile as settled either direction.

| patch | batch | peak_mem_gb | mean_s_per_step |
|---|---|---|---|
| (64,128,128) | 1 | 10.20 | 0.945 |
| (64,128,128) | 2 | 20.01 | 1.933 |
| (64,128,128) | 4 | 39.62 | 3.859 |
| (96,160,160) | 1 | 23.38 | 2.234 |
| (96,160,160) | 2 | 46.37 | 4.531 |
| (96,160,160) | 4 | 92.35 | 9.137 |

UNet++ is consistently ~4.2x slower per step than plain U-Net at matched patch/batch (e.g. 3.86s vs
0.92s at (64,128,128)/bs4), and memory scales proportionally worse too. At (96,160,160)/bs4, UNet++
hits 92.3GB of 128GB unified -- just model+activations, before CPU workers/dataloader
queue/CUDA context/OS share the same pool. This is a real capacity risk at the larger stress-bound
patch size, consistent with the ~17GB/patch A5000 reference number, now with GB10 numbers behind it.

DS-on vs DS-off at (64,128,128)/bs2, CURRENT UNPATCHED CODE: 20.01GB/1.933s (DS-on) vs
17.93GB/1.862s (DS-off) -- ~10% memory, ~4% time difference, all from heads that get computed but
discarded. Once Codex's dead-head-skip fix lands, re-run this exact config for a real before/after
delta instead of an estimate.

Both gate-0 and gate-0b are done on bdmap4. bdmap1 still unreachable.

### 2026-09-17 (evening) — dead-head-skip optimization: complete, pushed (424c298)

User ran out of Codex usage mid-task. Found Codex's work on its dedicated bdmap2 checkout
(~/pants-unetpp-codex, separate from the shared checkout it deliberately left untouched):
unet_plusplus.py's skip_shallowest_deep_supervision_head was fully implemented and verified --
bit-identical logits/loss/input+parameter gradients vs the unoptimized path (CPU and real GB10 CUDA),
plus real measured numbers (dead_head_bench_results.json): DS-off ~2.7% faster / 0.88GiB less peak
memory, default-DS ~4% faster / 0.22GiB less, at (64,128,128)/batch4.

Codex had correctly identified but not yet fixed the remaining piece when it ran out: the trainer
files (nnUNetTrainerUNetPlusPlus.py, nnUNetTrainerUNetPlusPlusPaper.py) were still completely
unmodified -- the optimization existed in the network class but nothing wired it up for real
training, and Codex's own note flagged the exact hazard ("the paper trainer must explicitly retain
all supervision targets") without having applied the fix yet.

Completed that piece: default trainer now passes skip_shallowest_deep_supervision_head=True and gets
a _build_loss()/_get_deep_supervision_scales() pair that computes the original decaying weights over
the full conceptual output count, then drops the already-zero entry (not re-derives weights over a
shorter list, which would zero out the wrong branch). Paper trainer overrides the flag back to False
as a class attribute specifically to avoid silently inheriting True via normal subclassing -- this is
the exact hazard Codex caught. Verified with a new trainer_dead_head_contract_test.py (checks scale
counts, loss weight_factors, and the network flags on both trainers) plus reruns of the existing
equivalence test on both CPU and bdmap2's real GPU. All pass.

Everything now pushed to main. Next real gap: get actual PanTS data through the real planner to
replace the stress-bound patch sizes gate-0/gate-0b have been using with real numbers.

### 2026-09-17 (night) — real plans.json obtained, batch-4 verified, full download launched

Codex is down for a while (user's usage ran out); bdmap2 is free, using it directly now alongside
bdmap3/bdmap4. bdmap1 still unreachable, no ETA.

Streamed a 41-case real PanTS sample (images+labels) onto bdmap3 by piping curl directly into tar
and killing the stream once enough cases were extracted, instead of downloading full chunks --
1.2GB images + 184MB labels vs the naive 49.8GB (34.2GB chunk 1 + 15.6GB full label archive), all
41 image/label pairs matched. Ran the REAL nnU-Net planner on it (after re-applying the known
fix_affine_orthonormality.py fix -- one case needed it). Real plans:

  patch_size = [64, 160, 224], n_stages = 6, features_per_stage = [32,64,128,256,320,320]
  strides = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,2,2]]  (note: last stride is anisotropic, not uniform)
  default batch_size = 2 (must override to 4, per FINDINGS.md's documented defect)

Verified batch 4 physically fits at this REAL patch size on one GB10: 39.4GB allocated / 51.0GB
reserved out of 128GB unified -- comfortable headroom. Real step time (dead-head-skip applied, no
compile yet): 4.8s/step, AMP. This is noticeably slower than the (96,160,160) stress-test proxy
suggested (~2.1s/step) despite similar voxel count -- shape/anisotropy matters, not just volume.
Don't trust proxy-patch extrapolations for anything beyond rough bounds going forward.

compile "reduce-overhead" mode (CUDA graphs) measured for the first time: 1.171x over eager,
marginally better than default compile mode's 1.131-1.152x (two separate runs, consistent
ballpark). Use reduce-overhead in the final config.

Important context found in training/retrain_wave1.sh: the ORIGINAL reference run used 4 GPUs per
config via DDP (not 1), completing 1000 epochs in ~39-50h that way. We only have 4 total GB10 nodes
(not 8), and multi-node DDP over plain lab ethernet between separate physical machines is a real
engineering risk given this model is already memory-bandwidth-bound within one GPU -- decided to
stick with 1-GPU-per-node, 4-configs-in-parallel (no DDP) as the safe, already-validated path,
flagged the tradeoff to the user rather than silently picking.

Full PanTS download (all 9 image chunks + full label archive, ~350GB) launched in the background on
bdmap3 (setsid, nohup-safe). This is the long pole now -- everything else should happen in parallel
with it, not wait for it serially where avoidable.

### 2026-09-17 (night, cont.) — combined-stack composition confirmed real, not assumed

dead-head-skip + compile together measured (proxy patch 64,128,128, bs4, AMP): 1.163x over legacy
eager, vs a naive multiplicative prediction of 1.142x -- composes slightly super-additively, no
negative interaction. Safe to stack both in the real trainers.

Applying to the REAL patch-size baseline (4.8s/step at [64,160,224]/bs4): ~4.1s/step with both
optimizations. With sparse validation's iteration-count cut (260 avg vs 300): ~17.55 min/epoch ->
~146h (~6.1 days) to 500 epochs, ~293h (~12.2 days) to 1000, for the bottleneck config (UNet++
DS-on). Plain U-Net configs will be substantially cheaper per the earlier ~4.2x gate0/gate0b ratio,
though that ratio was measured at proxy patch sizes and hasn't been reverified at the real one yet.

### 2026-09-17 (night, cont. 2) — full real CLI pipeline validated end-to-end, launch-ready

Created nnUNetPlansBS4 from the real 41-case plans.json (batch_size 4, patch [64,160,224]).
Wrote nnUNetTrainerUNetPlusPlusQuickSmoke (2 epochs x 3 train / 2 val iterations, save_every=1) to
smoke-test the REAL nnUNetv2_train CLI end-to-end before trusting it for multi-day runs. One bug
caught and fixed immediately: nnU-Net's base __init__ does locals()-based introspection for
checkpoint reproducibility, so a trainer subclass MUST declare the exact named params
(plans, configuration, fold, dataset_json, device) -- *args/**kwargs breaks it with a KeyError.
Matches the pattern nnUNetTrainerUNetPlusPlusNoDeepSupervision already used correctly.

Result: full pipeline ran clean end to end on real data -- training, checkpoint_best/final,
automatic post-training validation (real sliding-window inference on held-out cases, currently
running). No crashes, no NaNs. Real per-step time: epoch 1 (warmed up, 5 real steps) = 22.7s =
4.54s/step, essentially matching the 4.8s/step measured with dummy tensors earlier -- strong
evidence we are NOT meaningfully I/O-bound even with the real augmentation pipeline, so the
worker/dataloader-tuning lever is probably low-value here (GPU compute is the real bottleneck, not
data loading). Trust the earlier time estimates.

Full download still running in the background (~23GB / ~350GB total when last checked). Pipeline is
launch-ready the moment full data finishes preprocessing -- nothing else is blocking.

### 2026-09-17 (night, cont. 3) — full stack verified at REAL patch size, both architectures

UNet++ (bdmap2), real patch [64,160,224], bs4, AMP:
  noskip_eager:   4.816s/step, 39.87GB peak  (clean baseline, first arm run)
  noskip_compile: 4.151s/step, 42.90GB peak
  skip_eager:     5.527s/step, 39.55GB peak  (ANOMALY -- slower than noskip_eager, see caveat below)
  skip_compile:   3.600s/step, 40.81GB peak  (the actual deployed config)
  -> full-stack speedup (noskip_eager -> skip_compile): 1.337x
  -> 15.60 min/epoch -> 500 epochs: 130.0h, 1000 epochs: 260.0h (better than the ~146h/293h
     estimated earlier from the proxy-patch measurement)

Plain U-Net (bdmap4), same patch/batch:
  noskip_eager:   0.955s/step, 10.88GB peak
  noskip_compile: 0.758s/step, 11.58GB peak
  -> full-stack speedup: 1.259x -> 3.28 min/epoch -> 500 epochs: 27.4h, 1000 epochs: 54.7h

CAVEAT on the UNet++ skip_eager anomaly: it ran slower than noskip_eager, which shouldn't happen
(dead-head-skip only removes computation). Likely cause: real_patch_full_stack_bench.py runs all 4
arms sequentially in ONE process and only calls torch._dynamo.reset() before compile arms, not
after -- so compiled/dynamo state from the preceding noskip_compile arm may have leaked into the
skip_eager timing. Does not affect the trustworthiness of the headline number (compares the clean
FIRST arm against the actual deployed LAST arm), but don't trust the middle two data points without
a cleaner isolated-process rerun if precise per-arm numbers matter later.

Both bdmap2 and bdmap4 now free again. bdmap1 still unreachable. Download continuing on bdmap3
(was at 165GB/~9 image chunks, 4/9 done, when last checked).

Separately: user's PI got them an ACCESS-CI allocation on NCSA Delta (4x A100/node, NVLink) and
DeltaAI (4x GH200/H100 96GB, node). NCSA account provisioning pending, ETA next day per email.
Once live, plan is to benchmark there for real and likely migrate the actual 2x2+Paper grid runs
there given the large expected speedup (order of magnitude compute/bandwidth over GB10) and the
ability to replicate the original reference run's 4-GPU-DDP-per-run topology, which single-GPU-per-
node GB10 cannot do without risky multi-node DDP over plain ethernet.

### 2026-09-17 (late night) — bf16 vs fp16 settled, full dataset conversion in progress

bf16 vs fp16 AMP tested for real, at the real patch size, on top of dead-head-skip + reduce-overhead
compile (both architectures):
  UNet++:      fp16 3.639s/step, bf16 3.621s/step -> 1.005x (noise-level, confirms identical
               throughput as expected -- same bit width on this hardware)
  Plain U-Net: fp16 0.759s/step, bf16 0.761s/step -> 0.998x (same)

Recommendation: switch real trainers to bf16 autocast. Free -- zero speed cost confirmed on both
architectures -- and removes GradScaler's loss-scale mechanism entirely, which can silently skip
optimizer steps on overflow. That matters specifically for the tumor class, which is already
starved for gradient signal (~10% prevalence) -- can't observe this benefit in a synthetic
random-data benchmark, only in real training, but there's no downside to making the swap given
speed is a wash either way.

Full dataset conversion (9000 train + 901 test cases, PanTS -> nnU-Net format) running on bdmap3,
was at 1274/9000 last checked, steady progress. Once done: fix_affine_orthonormality.py pass (same
as the 41-case run needed), then the real nnUNetv2_plan_and_preprocess on the full training set
(~30h per README's own estimate), then create nnUNetPlansBS4 from the real full-dataset plan (should
closely match the 41-case sample's [64,160,224] but worth confirming, not assuming, once available).

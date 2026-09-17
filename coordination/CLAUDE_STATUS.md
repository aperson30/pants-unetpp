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

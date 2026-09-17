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

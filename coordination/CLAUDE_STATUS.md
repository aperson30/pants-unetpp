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

# Bridges-2 deployment status

Target: PSC Bridges-2 `GPU-shared`, H100-80GB, account `cis260296p`. The production comparison is
the unchanged 1,000-epoch, physical-batch-4 2x2 grid. Each cell uses one GPU; two cells run in
parallel, so there is no DDP or gradient-accumulation change to the experiment.

## Real H100 gate (2026-09-22)

Stack: PyTorch 2.10.0+cu126, cuDNN 9.10.2, nnU-Net v2 2.8.1, real nine-case PanTS preprocessing,
patch `[64,160,224]`, physical batch 4. The sample contained a real class-28-positive case.

| Cell | Median synchronized train step | Peak allocated | Numerical checks |
|---|---:|---:|---|
| UNet++ DS-on | 0.427 s | not persisted by the first harness revision | passed in-process |
| UNet++ DS-off | 0.401 s | 43.08 GiB | passed |
| Plain U-Net DS-on | 0.140 s | 10.97 GiB | passed |
| Plain U-Net DS-off | 0.094 s | 10.82 GiB | passed |

“Numerical checks” means finite losses, parameters and optimizer momentum plus finite, nonzero
class-28 segmentation-head gradients. Job 46755946 completed the first arm and then hit a harness
state-reuse bug before arm two; reaching arm two proves the first arm returned through all those
checks. The fixed harness deep-copies plans per arm and persists each result incrementally. Job
46756922 then passed the other three arms and wrote the retained JSON result.

The serial and six-worker PanTS converters were also compared on all nine real cases: copied CTs
were byte-identical, while merged-label voxels, affines, headers, case lists and `dataset.json` were
exactly equal. The parallel path is therefore allowed only as a data-staging optimization.

## Production safety contract

`submit_grid.sh` refuses to launch without a reviewed `H100_GATE_APPROVED.json`, a clean checkout,
and a pinned commit. `grid_coordinated.sbatch` freezes that source revision, asserts 9,000 training
cases, class 28 and physical batch 4, and requires 9,000 preprocessed cases. It stages only into the
job's node-local `$LOCAL`, preserves checkpoints/validation on Ocean, and uses a bounded `afterany`
successor chain for 48-hour recovery. A cell is complete only when both `checkpoint_final.pth` and
all 1,800 fold-validation predictions plus a readable summary exist; a final checkpoint with an
incomplete validation triggers `--val`, not a false done marker.

Delta job 22293168 must remain queued while Bridges-2 is queued or merely staging. Cancel it only
after a Bridges-2 trainer is visibly executing real epochs, per the user's explicit race rule.

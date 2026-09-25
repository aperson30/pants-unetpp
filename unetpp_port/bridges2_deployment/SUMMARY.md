# Bridges-2 deployment status

## September 23 staging incidents — old frozen jobs must not be reused

Production job `46810860` reached full-data conversion, then failed before preprocessing or any
training epoch. The script was still in `$SOURCE/PanTS/data` when it removed `$SOURCE`; the next
nnU-Net/PyTorch import therefore ran from a deleted working directory and emitted a misleading
`Intel oneMKL FATAL ERROR: Cannot load .../libtorch_cpu.so`. A bounded zero-GPU compute-node
import test (`46842686`) passed from a valid directory. A disposable deleted-cwd reproduction
failed with the same oneMKL message and exit 2; repeating the cleanup after `cd "$LOCAL_ROOT"`
passed with exit 0. The launch script now makes that directory change before cleanup. The
unsubmitted DeltaAI fallback had the same bug and received the same fix.

The automatic successor `46835559` was held, then canceled after a new corrected job was safely
queued. It pointed to the original commit-frozen script, which still contained the bug. Delta job
`22293168` remained queued and untouched.

Later on September 23, corrected job `46842964` finished preprocessing 9,000 cases but its
launcher then looked for `*.npz` files. The validated nnU-Net 2.8.1 output uses `.b2nd`, so the
guard falsely reported zero and stopped before training. Its two retries were canceled with user
approval. The guard now compares exact case IDs across raw CTs, raw labels, preprocessed image
`.b2nd`, preprocessed `_seg.b2nd`, and metadata `.pkl`; the case-ID version passed on the existing
real nine-case Bridges-2 calibration, and the final nonempty-file version passed six synthetic
positive/negative tests. After the Bridges-2 SSH master was restored, the exact final guard also
passed on the real nine-case calibration from the corrected remote checkout. This corrected
launcher has **not** been submitted; job `46842964` and its frozen retries remain unusable.

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
job's node-local `$LOCAL` and preserves checkpoints/validation on Ocean. Automatic `afterany`
retries are disabled for this backup launch: an unexpected failure stops for diagnosis, while a
time-limit continuation can be reviewed and submitted manually from persistent checkpoints.
A cell is complete only when both `checkpoint_final.pth` and
all 1,800 fold-validation predictions plus a readable summary exist; a final checkpoint with an
incomplete validation triggers `--val`, not a false done marker.

The original launcher backgrounded two separate `srun --exclusive --exact` steps. That pattern
serialized on Delta, so the Bridges-2 backup now launches one two-task `srun` step per model pair.
Bridges-2 H100 probe `47030369` completed in 22 seconds: ranks 0 and 1 both started at
08:51:05 UTC, both ended at 08:51:25 UTC, and reported distinct GPU UUIDs. This verifies the
task/GPU-binding pattern, not the complete training run. The production launcher rechecks distinct
physical GPU UUIDs on its assigned node before any dataset staging. Before starting either trainer, the
launcher generates nnU-Net's exact default five-fold split once with seed 12345, avoiding a
concurrent write to `splits_final.json`; the same recipe exactly matched the existing real
nine-case calibration split (`EXACT_SPLIT_PARITY=True`).

Delta job 22293168 must remain queued while Bridges-2 is queued or merely staging. Cancel it only
after a Bridges-2 trainer is visibly executing real epochs, per the user's explicit race rule.

# DeltaAI two-GH200 grid fallback — prepared, not submitted

This fallback is for the same four 1,000-epoch, physical-batch-4 cells as the Delta and
Bridges-2 jobs. It does not implement the separate Paper configuration. It uses the same
sparse-validation trainers, full-dataset planner, fold 0, nnU-Net CLI, and phase ordering
as the reviewed Bridges-2 launcher. Two cells use one GH200 each; no DDP, accumulation,
sampling, or loss change is introduced.

## Status and gates

- **Not submitted.** `submit_grid.sh` refuses to launch without a reviewed, commit-matched
  `DELTAI_GATE_APPROVED.json` under `/work/nvme/bdyo/asanjeev/pants_grid_deltaai/`.
- The remote checkout is currently at `19cc861d0330d2dc73197305d9ad4b59c13059af`, with an
  untracked `unetpp_port/__init__.py`. Do not delete or overwrite that file. The launcher
  freezes a Git commit, so untracked material is ignored; the remote checkout still needs
  a deliberate fast-forward to this fallback's reviewed commit before submission.
- Existing Delta `22293168` and Bridges-2 `46810860` must retain their queue positions.
  Never point this fallback at their results directories. Before any submission, inspect
  both jobs and decide whether a third race is still useful. The user's recurring monitor
  is paused; no automatic cross-cluster cancellation is active.
- Recheck `/tmp` free space and `/work/nvme` capacity at launch. Node-local `/tmp` is
  shared and ephemeral. The script requires 3,000 GiB free before staging, stages all
  9,000 cases, and keeps checkpoints and 1,800 validation predictions per cell in an
  isolated persistent run directory under `/work/nvme`.
- The full 9,000-case conversion/preprocessing and final validation have never run on
  DeltaAI. Short calibration showed GH200 0.452 s/step UNet++ and 0.113 s/step plain
  U-Net, but those are not end-to-end or tumor-recall results. The full-data patch
  size is determined by the real planner, not hard-coded from a calibration subset.
- Verify trainer imports from the exact frozen source, PyTorch 2.10.0+cu129,
  nnU-Net 2.8.1, batch 4, 9,000 preprocessed cases, and class label 28 before
  approving a launch gate. `bash -n`, `sbatch --test-only`, and a source-diff review
  are necessary but not sufficient.
- Confirm outbound access to the Hugging Face and JHU dataset endpoints from a
  DeltaAI **compute node**, not merely the login node. The full download on a
  DeltaAI compute node has not yet been tested; a denied connection would consume
  an allocation without reaching training. Also confirm the installed trainer
  overlay will not conflict with anyone else's active use of the shared venv.
- The previous `*.npz` staging assertion was wrong for this nnU-Net 2.8.1 stack:
  the real Bridges-2 calibration uses `.b2nd`. The fallback now shares an exact
  case-ID guard across raw images/labels and preprocessed image, segmentation,
  and metadata files. Its case-ID version passed against the Bridges-2 nine-case
  real calibration; the final nonempty-file version passed synthetic tests but
  awaits a real-data recheck because the Bridges-2 SSH master expired. The DeltaAI
  calibration also could not be rechecked because its own master was unavailable.
- A cell is complete only when its final checkpoint, readable validation summary,
  and all 1,800 fold-0 validation NIfTIs exist. A final checkpoint with incomplete
  validation runs `--val`. A timeout successor is bounded; a failed trainer or failed
  staging requires human diagnosis instead of repeated blind GPU allocations.

## Why a fallback, not a replacement

DeltaAI's September 22 non-submitting two-GH200 estimate moved from September 23
12:00 CDT to September 24 08:28 CDT on a later `sbatch --test-only` check. Neither
check submitted a job; `squeue -u asanjeev` remained empty on DeltaAI. These are
volatile forecasts. At measured step rates, the paired cells need about 39.2 training-only
hours on GH200, versus about 27.8 on Delta H200, before staging and validation.
Therefore an earlier DeltaAI start need not mean an earlier completed table.

The existing Delta job is eligible for both H200 and A100 nodes. It and the
Bridges job remain the first choices if either begins real training. If a vetted
DeltaAI fallback is eventually submitted, cancel no other job merely because it
is queued or downloading; coordinate on observed trainer epochs and fresh Slurm
states so duplicate training/output writes cannot occur.

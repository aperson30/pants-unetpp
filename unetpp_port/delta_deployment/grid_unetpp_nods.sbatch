#!/bin/bash
# Template for one grid cell. Placeholders to fill in per-cell before submission:
#   gpuH200x8   e.g. gpuA100x4 or gpuH200x8
#   nnUNetTrainerUNetPlusPlusNoDeepSupervision     e.g. nnUNetTrainerBF16, nnUNetTrainerBF16NoDeepSupervision,
#                        nnUNetTrainerUNetPlusPlus, nnUNetTrainerUNetPlusPlusNoDeepSupervision
#   unetpp_nods     short label for logs, e.g. default_ds, default_nods, unetpp_ds, unetpp_nods
#SBATCH --account=bdyo-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --job-name=pants_grid_unetpp_nods
#SBATCH --output=/projects/bdyo/asanjeev/delta_work/grid_logs/unetpp_nods_%j.log

set -e
SELF="$0"
OUTDIR=/projects/bdyo/asanjeev/delta_work/nnUNet_results/Dataset001_PanTS/nnUNetTrainerUNetPlusPlusNoDeepSupervision__nnUNetPlansBS4__3d_fullres/fold_0
CKPT="$OUTDIR/checkpoint_latest.pth"
DONE_MARKER="$OUTDIR/checkpoint_final.pth"

# Requeue-before-timeout handler: on SIGUSR1 (sent 180s before the hard limit),
# submit a fresh job for the same cell (picks up --c automatically next run since
# checkpoint_latest.pth will already exist) and exit cleanly rather than being SIGKILLed mid-write.
requeue_self() {
    echo "[requeue] SIGUSR1 received, checkpoint should be safely on disk (save_every=50 epochs)"
    if [ ! -f "$DONE_MARKER" ]; then
        echo "[requeue] not finished yet, resubmitting: sbatch $SELF"
        sbatch "$SELF"
    else
        echo "[requeue] checkpoint_final.pth already exists, not resubmitting"
    fi
    exit 0
}
trap requeue_self USR1

module load pytorch-conda/2.12 >/dev/null 2>&1
cd /projects/bdyo/asanjeev/delta_work
source pants_venv/bin/activate

export nnUNet_raw=/projects/bdyo/asanjeev/delta_work/nnUNet_raw
export nnUNet_preprocessed=/projects/bdyo/asanjeev/delta_work/nnUNet_preprocessed
export nnUNet_results=/projects/bdyo/asanjeev/delta_work/nnUNet_results
export PYTHONPATH=/projects/bdyo/asanjeev/delta_work/pants-unetpp-fork:$PYTHONPATH

CONTINUE_FLAG=""
if [ -f "$CKPT" ]; then
    echo "checkpoint_latest.pth found, resuming with --c"
    CONTINUE_FLAG="--c"
fi

if [ -f "$DONE_MARKER" ]; then
    echo "checkpoint_final.pth already exists -- this cell is already done, nothing to do"
    exit 0
fi

nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerUNetPlusPlusNoDeepSupervision -p nnUNetPlansBS4 $CONTINUE_FLAG &
NNUNET_PID=$!
wait $NNUNET_PID

echo "GRID_CELL_unetpp_nods_DONE"

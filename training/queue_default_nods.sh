#!/bin/bash
# =============================================================================
# Waits for the unetpp_ds run (GPUs 0-3) to finish, then automatically launches
# the last training run of the 2x2 grid: default architecture, deep supervision
# OFF (nnUNetTrainerNoDeepSupervision), on the same 4 GPUs with the same
# nnUNetPlansBS4 config as every other leg.
#
# HOW IT DECIDES unetpp_ds IS DONE
#   It waits for `checkpoint_final.pth` to appear in that run's fold_0 folder.
#   nnU-Net writes that file ONLY when all 1000 epochs have completed, so it is
#   a reliable "training finished" marker -- unlike the process exiting, which
#   also happens when the post-training validation phase crashes on NCCL
#   timeouts (that crash is expected and harmless; the checkpoint is already
#   written by then).
#
#   SAFETY GUARD: if the training process disappears WITHOUT that checkpoint
#   existing, the run died early. In that case this script does NOT launch
#   anything -- it logs the problem and exits, so a broken run doesn't silently
#   trigger the next one before anyone has looked at what went wrong.
#
# THEN it waits for GPUs 0-3 to actually be released (the crashing validation
# phase can hold them for a while after training finishes) before launching.
#
# Run with:
#     nohup ~/pants_phase7_scoring/queue_default_nods.sh > queue_default_nods.log 2>&1 &
#     disown
# Check on it with:
#     tail -20 ~/pants_phase7_scoring/queue_default_nods.log
# =============================================================================

set -u

RESULTS=/Scratch/enl014/nnUNet_results/Dataset001_PanTS
WATCH_CKPT="$RESULTS/nnUNetTrainerUNetPlusPlus__nnUNetPlansBS4__3d_fullres/fold_0/checkpoint_final.pth"
WATCH_PROC="nnUNetTrainerUNetPlusPlus -p nnUNetPlansBS4"   # note: does NOT match the NoDeepSupervision variant
GPUS="0,1,2,3"
MEM_IDLE_MIB=1000
POLL=300   # 5 minutes

cd ~/pants_phase7_scoring || exit 1

echo "$(date): watching for unetpp_ds to finish (waiting on $WATCH_CKPT)"

# ---- phase 1: wait for training to complete ---------------------------------
while true; do
    if [ -f "$WATCH_CKPT" ]; then
        echo "$(date): checkpoint_final.pth found -- unetpp_ds training completed."
        break
    fi
    if ! pgrep -f "$WATCH_PROC" > /dev/null 2>&1; then
        echo "$(date): ERROR -- the unetpp_ds process is gone but checkpoint_final.pth does not exist."
        echo "                 That means it died before finishing 1000 epochs."
        echo "                 NOT launching default_nods. Investigate retrain_unetpp_ds.log first."
        exit 1
    fi
    sleep "$POLL"
done

# ---- phase 2: wait for the GPUs to actually free up -------------------------
# (the post-training validation phase may still be holding them, and it often
#  dies on an NCCL timeout rather than exiting cleanly)
echo "$(date): waiting for GPUs $GPUS to be released..."
while true; do
    busy=0
    for m in $(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits); do
        if [ "$m" -ge "$MEM_IDLE_MIB" ]; then busy=1; fi
    done
    if [ "$busy" -eq 0 ]; then
        echo "$(date): GPUs $GPUS are free."
        break
    fi
    sleep "$POLL"
done

# ---- phase 3: launch the final training run ---------------------------------
export nnUNet_n_proc_DA=6
export nnUNet_def_n_proc=4

echo "$(date): launching default_nods (nnUNetTrainerNoDeepSupervision, nnUNetPlansBS4, 4 GPUs)"
CUDA_VISIBLE_DEVICES="$GPUS" nohup nnUNetv2_train 1 3d_fullres 0 \
    -tr nnUNetTrainerNoDeepSupervision \
    -p nnUNetPlansBS4 \
    -num_gpus 4 \
    > retrain_default_nods.log 2>&1 &

echo "$(date): launched, pid $!. Log: retrain_default_nods.log"
echo "$(date): this completes the 2x2 grid. Remaining afterwards: single-GPU --val for each run,"
echo "         then PanTS-te predictions and the 5 benchmark metrics."

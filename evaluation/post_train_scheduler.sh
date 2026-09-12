#!/bin/bash
# =============================================================================
# Opportunistic scheduler: as soon as a GPU frees up, start the post-training
# pipeline (validation -> PanTS-te prediction -> 5 metrics) for whichever
# trained model has not been processed yet. One GPU per model, several models
# processed in parallel if several GPUs are free.
#
# Run with:
#     nohup ~/pants_phase7_scoring/post_train_scheduler.sh > post_train_scheduler.log 2>&1 &
#     disown
# Watch with:
#     tail -30 ~/pants_phase7_scoring/post_train_scheduler.log
#
# WHY IT WAITS BEFORE CLAIMING ANYTHING
#   queue_default_nods.sh is separately waiting for GPUs 0-3 so it can launch
#   the final training run. If this scheduler grabbed one of those GPUs first,
#   that training would never start. So this script does nothing at all until
#   default_nods training is actually running. In practice that costs no time:
#   every GPU is busy until then anyway.
#
#   If you are running this AFTER all four trainings are already going or done,
#   set SKIP_WAIT=1 to bypass that guard:
#       SKIP_WAIT=1 nohup ./post_train_scheduler.sh > post_train_scheduler.log 2>&1 &
#
# HOW IT DECIDES A MODEL IS READY
#   `checkpoint_final.pth` exists in that run's fold_0 folder. nnU-Net writes it
#   only after all 1000 epochs complete, so a model that crashed early is never
#   picked up. Note the post-training validation crashing under DDP does NOT
#   matter here -- the checkpoint is written before that happens.
#
# MARKER FILES
#   `.post_train_started_<tag>` is created when a model is dispatched, so a
#   model is never started twice. Delete a marker by hand if you need to re-run
#   that model's pipeline.
# =============================================================================

set -u

RESULTS=/Scratch/enl014/nnUNet_results/Dataset001_PanTS
PLANS=nnUNetPlansBS4
MEM_IDLE_MIB=1000
POLL=300
SKIP_WAIT="${SKIP_WAIT:-0}"

# tag -> trainer class
declare -A TRAINERS=(
    [unetpp_ds]=nnUNetTrainerUNetPlusPlus
    [unetpp_nods]=nnUNetTrainerUNetPlusPlusNoDeepSupervision
    [default_ds]=nnUNetTrainer
    [default_nods]=nnUNetTrainerNoDeepSupervision
)

cd ~/pants_phase7_scoring || exit 1

echo "$(date): post-training scheduler started."

# ---- guard: don't steal GPUs 0-3 from the queued default_nods training -------
if [ "$SKIP_WAIT" != "1" ]; then
    echo "$(date): waiting until default_nods training has started before claiming any GPU"
    echo "         (set SKIP_WAIT=1 to bypass this)"
    while ! pgrep -f "nnUNetTrainerNoDeepSupervision -p $PLANS -num_gpus" > /dev/null 2>&1; do
        sleep "$POLL"
    done
    echo "$(date): default_nods training is running -- scheduler is now active."
fi

# ---- main loop ---------------------------------------------------------------
while true; do

    # how many models still need processing?
    remaining=0
    for tag in "${!TRAINERS[@]}"; do
        [ -f ".post_train_started_${tag}" ] || remaining=$((remaining + 1))
    done
    if [ "$remaining" -eq 0 ]; then
        echo "$(date): all four models dispatched. Scheduler exiting."
        echo "         (individual pipelines may still be running -- check post_train_<tag>.log)"
        exit 0
    fi

    # which GPUs are free right now?
    free_gpus=()
    while read -r idx mem; do
        idx="${idx%,}"
        [ "$mem" -lt "$MEM_IDLE_MIB" ] && free_gpus+=("$idx")
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')

    if [ "${#free_gpus[@]}" -gt 0 ]; then
        for tag in "${!TRAINERS[@]}"; do
            [ "${#free_gpus[@]}" -eq 0 ] && break
            [ -f ".post_train_started_${tag}" ] && continue

            trainer="${TRAINERS[$tag]}"
            ckpt="$RESULTS/${trainer}__${PLANS}__3d_fullres/fold_0/checkpoint_final.pth"
            [ -f "$ckpt" ] || continue    # training not finished yet

            gpu="${free_gpus[0]}"
            free_gpus=("${free_gpus[@]:1}")

            touch ".post_train_started_${tag}"
            echo "$(date): dispatching $tag ($trainer) to GPU $gpu"
            nohup ./post_train_one_run.sh "$trainer" "$tag" "$gpu" \
                > "post_train_${tag}.log" 2>&1 &
            disown
            sleep 30   # let it claim the GPU before we re-read nvidia-smi
        done
    fi

    sleep "$POLL"
done

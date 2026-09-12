#!/bin/bash
# Runs the rest of Phase 6 + all of Phase 7, unattended, in order:
# 1. Detect however many GPUs are ACTUALLY free right now (not guess a fixed number)
# 2. Re-predict the cleaned-out (previously corrupted) cases, spread across all of them
# 3. Wait for all of them to finish
# 4. Re-check for corruption (should be zero this time)
# 5. Run the final Phase 7 metrics script
# Everything logs to its own file so you can review what happened when you wake up.

set -x  # print each command as it runs, into the log, for easier debugging later

cd ~/nnUNet

# List every GPU's current memory usage, and keep only the ones near-idle (under 500 MiB used --
# a normal idle GPU only has the ~10MiB baseline system display process, so anything above that
# threshold means someone else is actively using it).
mapfile -t FREE_GPUS < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}'
)
NUM_GPUS=${#FREE_GPUS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No idle GPUs detected -- falling back to GPU 0 only (will be slower, may share with others)."
    FREE_GPUS=(0)
    NUM_GPUS=1
fi

echo "Using $NUM_GPUS free GPU(s): ${FREE_GPUS[*]}"

PIDS=()
for i in "${!FREE_GPUS[@]}"; do
    GPU_ID=${FREE_GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU_ID nnUNetv2_predict \
        -i "$nnUNet_raw/Dataset001_PanTS/imagesTs" \
        -o /Scratch/enl014/PanTS_predictions_test \
        -d 1 -c 3d_fullres -tr nnUNetTrainerUNetPlusPlus -p nnUNetPlansBS4 -f 0 \
        --save_probabilities --continue_prediction \
        -num_parts "$NUM_GPUS" -part_id "$i" \
        > "predict_part${i}_resume2.log" 2>&1 &
    PIDS+=($!)
done

wait "${PIDS[@]}"  # pause here until every one of the launched jobs above has finished

cd ~/pants_phase7_scoring
python check_npz_integrity.py /Scratch/enl014/PanTS_predictions_test > integrity_check_2.log 2>&1

python compute_pants_metrics.py \
    --predictions-dir /Scratch/enl014/PanTS_predictions_test \
    --ground-truth-dir /Scratch/enl014/PanTS_test_answer_key \
    > final_metrics.log 2>&1

echo "OVERNIGHT PIPELINE FINISHED" >> final_metrics.log

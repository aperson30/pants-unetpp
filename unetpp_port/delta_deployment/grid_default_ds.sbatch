#!/bin/bash
# v2: node-local /tmp staging (bypasses the shared ~1TB /projects/bdyo quota, which we hit and
# confirmed is genuinely shared between Delta and DeltaAI), dependency-chain resubmission instead
# of a reactive signal trap (Slurm guarantees the successor cannot start until this job has fully
# terminated -- no race window where two trainers could touch the same checkpoint dir), and
# explicit error propagation on every download/extract (the v1 job silently lost all 9 image
# chunks + labels to a swallowed tar failure because `wget -q`/`curl -s` hid the real error and
# a backgrounded subshell's failure didn't trip `set -e`).
#
# Placeholders to fill in per-cell before submission:
#   gpuH200x8   gpuH200x8
#   nnUNetTrainerBF16SparseValidation     nnUNetTrainerBF16SparseValidation, nnUNetTrainerBF16NoDeepSupervisionSparseValidation,
#                   nnUNetTrainerUNetPlusPlusSparseValidation, nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation
#   default_ds     default_ds, default_nods, unetpp_ds, unetpp_nods
#SBATCH --account=bdyo-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=pants_grid_default_ds
#SBATCH --output=/projects/bdyo/asanjeev/delta_work/grid_logs/default_ds_%j.log

set -e
SELF=/projects/bdyo/asanjeev/delta_work/grid_launch/grid_default_ds.sbatch
OUTDIR=/projects/bdyo/asanjeev/delta_work/nnUNet_results/Dataset001_PanTS/nnUNetTrainerBF16SparseValidation__nnUNetPlansBS4__3d_fullres/fold_0
DONE_MARKER="$OUTDIR/checkpoint_final.pth"

echo "=== provenance ==="
hostname
date -u
git -C /projects/bdyo/asanjeev/delta_work/pants-unetpp-fork rev-parse HEAD
git -C /projects/bdyo/asanjeev/delta_work/pants-unetpp-fork status --porcelain

if [ -f "$DONE_MARKER" ]; then
    echo "checkpoint_final.pth already exists -- this cell is already done, nothing to do"
    exit 0
fi

# Chain the successor BEFORE training starts, not reactively on a near-timeout signal: Slurm's
# afterany dependency guarantees the successor cannot begin until this job has fully exited, by
# any means (normal completion, timeout, kill) -- no window where two trainer processes could
# both be alive and writing to the same checkpoint directory.
sbatch --dependency=afterany:$SLURM_JOB_ID "$SELF"

module load pytorch-conda/2.12 >/dev/null 2>&1
source /projects/bdyo/asanjeev/delta_work/pants_venv/bin/activate
python3 -c "import torch; print('torch', torch.__version__, 'cudnn', torch.backends.cudnn.version())"
nvidia-smi -L

LOCAL_ROOT=/tmp/asanjeev_pants
LOCAL_RAW=$LOCAL_ROOT/nnUNet_raw
LOCAL_PREPROCESSED=$LOCAL_ROOT/nnUNet_preprocessed
STAGE_MARKER=$LOCAL_PREPROCESSED/Dataset001_PanTS/nnUNetPlansBS4.json

if [ ! -f "$STAGE_MARKER" ]; then
    echo "=== staging full dataset into node-local /tmp (not previously staged on this node) ==="
    mkdir -p "$LOCAL_ROOT/PanTS/data/ImageTr" "$LOCAL_ROOT/PanTS/data/ImageTe" "$LOCAL_RAW" "$LOCAL_PREPROCESSED"
    cd "$LOCAL_ROOT/PanTS/data"

    wget -O metadata.xlsx "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/metadata.xlsx?download=true" || true

    PIDS=()
    for i in 1 2 3 4 5 6 7 8 9; do
      start=$(printf "%08d" $(( (i - 1) * 1000 + 1 )))
      end=$(printf "%08d" $(( i * 1000 )))
      file="PanTSMini_ImageTr_${start}_${end}.tar.gz"
      url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/${file}?download=true"
      ( wget -O "$file" "$url" && tar -xzf "$file" -C ImageTr && rm "$file" && echo "[$i/9] image chunk done" ) &
      PIDS+=($!)
    done
    ( curl -sSL -f "https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz" -o PanTSMini_Label.tar.gz \
        && mkdir -p LabelAll && tar -xzf PanTSMini_Label.tar.gz -C LabelAll && rm PanTSMini_Label.tar.gz \
        && echo "labels done" ) &
    PIDS+=($!)

    FAIL=0
    for pid in "${PIDS[@]}"; do
      wait "$pid" || FAIL=1
    done
    if [ "$FAIL" -ne 0 ]; then
      echo "ERROR: one or more downloads/extracts failed -- aborting, not proceeding to preprocess" >&2
      exit 1
    fi

    mkdir -p LabelTr LabelTe
    cd LabelAll
    find . -maxdepth 1 -type d \( -name 'PanTS_0000[0-8]*' -o -name 'PanTS_00009000' \) -print0 | xargs -0 -r mv -t ../LabelTr/
    find . -maxdepth 1 -type d -name 'PanTS_00009*' -print0 | xargs -0 -r mv -t ../LabelTe/
    cd ..
    rmdir LabelAll 2>/dev/null || true

    file="PanTSMini_ImageTe_00009001_00009901.tar.gz"
    wget -O "$file" "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/$file?download=true"
    tar -xzf "$file" -C ImageTe && rm "$file"

    echo "train cases: $(ls ImageTr | wc -l)  test cases: $(ls ImageTe | wc -l)"
    echo "STAGE_DOWNLOAD_DONE"

    cd /projects/bdyo/asanjeev/delta_work
    python3 pants-unetpp-fork/data_conversion/convert_pants_to_nnunet.py \
      --pants-root "$LOCAL_ROOT/PanTS/data" \
      --nnunet-dataset-dir "$LOCAL_RAW/Dataset001_PanTS" \
      --test-answer-key-dir "$LOCAL_ROOT/PanTS_test_answer_key"
    echo "STAGE_CONVERT_DONE"

    export nnUNet_raw=$LOCAL_RAW
    export nnUNet_preprocessed=$LOCAL_PREPROCESSED
    export nnUNet_results=$LOCAL_ROOT/nnUNet_results_scratch
    nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity -npfp 12 -np 12
    python3 -c "
import json
p='$LOCAL_PREPROCESSED/Dataset001_PanTS/nnUNetPlans.json'
plans=json.load(open(p))
plans['plans_name']='nnUNetPlansBS4'
plans['configurations']['3d_fullres']['batch_size']=4
json.dump(plans, open('$STAGE_MARKER','w'), indent=2)
print('patch_size:', plans['configurations']['3d_fullres']['patch_size'])
"
    echo "STAGE_PREPROCESS_DONE"
else
    echo "found existing staged+preprocessed data in /tmp on this node, reusing"
fi

echo "train cases (final): $(ls $LOCAL_RAW/Dataset001_PanTS/imagesTr | wc -l)"
echo "expected: 9000"

export nnUNet_raw=$LOCAL_RAW
export nnUNet_preprocessed=$LOCAL_PREPROCESSED
export nnUNet_results=/projects/bdyo/asanjeev/delta_work/nnUNet_results
export PYTHONPATH=/projects/bdyo/asanjeev/delta_work/pants-unetpp-fork:$PYTHONPATH
mkdir -p "$nnUNet_results"

CONTINUE_FLAG=""
if [ -f "$OUTDIR/checkpoint_latest.pth" ]; then
    echo "checkpoint_latest.pth found on persistent storage, resuming with --c"
    CONTINUE_FLAG="--c"
fi

nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerBF16SparseValidation -p nnUNetPlansBS4 $CONTINUE_FLAG

echo "GRID_CELL_default_ds_DONE"

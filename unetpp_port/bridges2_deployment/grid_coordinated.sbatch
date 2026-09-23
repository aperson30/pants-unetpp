#!/bin/bash
# Four quality-matched 1,000-epoch cells on one Bridges-2 node, two H100s at a time.
# The full dataset is staged once per allocation into node-local $LOCAL. Checkpoints and validation
# outputs remain on persistent Ocean so a successor job can safely resume on a different node.
#SBATCH --account=cis260296p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --job-name=pants_grid_h100
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/grid_%j.log

set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
REPO=$ROOT/repo
VENV=$ROOT/venv
RESULTS_ROOT=$ROOT/results
MAX_RETRIES=4
RETRY_COUNT=${GRID_RETRY_COUNT:-0}
: "${GRID_COMMIT:?submit through submit_grid.sh so GRID_COMMIT is pinned}"
: "${GRID_RUN_ID:?submit through submit_grid.sh so GRID_RUN_ID is stable}"
[[ "$GRID_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid GRID_COMMIT" >&2; exit 1; }
[[ "$GRID_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid GRID_RUN_ID" >&2; exit 1; }

mkdir -p "$ROOT/grid_logs" "$RESULTS_ROOT" "$ROOT/frozen"
echo "=== Bridges-2 PanTS 2x2 provenance ==="
hostname
date -u
echo "job=$SLURM_JOB_ID retry=$RETRY_COUNT/$MAX_RETRIES commit=$GRID_COMMIT run=$GRID_RUN_ID"
nvidia-smi -L
"$VENV/bin/python" -c "import torch; assert torch.__version__.startswith('2.10.0+cu126'), torch.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"
git -C "$REPO" cat-file -e "$GRID_COMMIT^{commit}"

if [ "$RETRY_COUNT" -gt "$MAX_RETRIES" ]; then
    echo "ERROR: retry count exceeded $MAX_RETRIES; refusing an unbounded resubmission chain" >&2
    exit 1
fi

# Materialize the exact submitted source revision once. Successor jobs never execute a moving main.
FROZEN=$ROOT/frozen/$GRID_COMMIT
if [ ! -f "$FROZEN/.frozen_commit" ]; then
    TMP_FROZEN=$ROOT/frozen/.${GRID_COMMIT}.${SLURM_JOB_ID}.tmp
    if [ -e "$FROZEN" ]; then
        rm -rf "$FROZEN"
    fi
    rm -rf "$TMP_FROZEN"
    mkdir -p "$TMP_FROZEN"
    git -C "$REPO" archive "$GRID_COMMIT" | tar -x -C "$TMP_FROZEN"
    printf '%s\n' "$GRID_COMMIT" > "$TMP_FROZEN/.frozen_commit"
    mv "$TMP_FROZEN" "$FROZEN"
fi
test "$(cat "$FROZEN/.frozen_commit")" = "$GRID_COMMIT"

# nnU-Net's CLI discovers trainers only inside its installed trainer package. Overlay an explicit,
# reviewed file list from the immutable source snapshot; never copy a trainer glob.
TRAINER_DIR=$("$VENV/bin/python" - <<'PY'
from pathlib import Path
import nnunetv2.training.nnUNetTrainer as package
print(Path(package.__path__[0]))
PY
)
TRAINER_FILES=(
  unet_plusplus.py
  nnUNetTrainerBF16Mixin.py
  nnUNetTrainerQualityNeutralOptimizationMixin.py
  nnUNetTrainerFullLossCompileMixin.py
  nnUNetTrainerSparseValidationMixin.py
  nnUNetTrainerBF16.py
  nnUNetTrainerBF16NoDeepSupervision.py
  nnUNetTrainerBF16SparseValidation.py
  nnUNetTrainerBF16NoDeepSupervisionSparseValidation.py
  nnUNetTrainerUNetPlusPlus.py
  nnUNetTrainerUNetPlusPlusNoDeepSupervision.py
  nnUNetTrainerUNetPlusPlusSparseValidation.py
  nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation.py
)
for file in "${TRAINER_FILES[@]}"; do
    install -m 0644 "$FROZEN/unetpp_port/$file" "$TRAINER_DIR/$file"
done
"$VENV/bin/python" - <<'PY'
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBF16SparseValidation import nnUNetTrainerBF16SparseValidation
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBF16NoDeepSupervisionSparseValidation import nnUNetTrainerBF16NoDeepSupervisionSparseValidation
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlusSparseValidation import nnUNetTrainerUNetPlusPlusSparseValidation
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation import nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation
print('all four pinned trainer classes import successfully')
PY

fold_dir() {
    echo "$RESULTS_ROOT/Dataset001_PanTS/$1__nnUNetPlansBS4__3d_fullres/fold_0"
}
validation_count() {
    local folder=$1
    find "$folder/validation" -maxdepth 1 -type f -name 'PanTS_*.nii.gz' 2>/dev/null | wc -l
}
cell_done() {
    local folder=$1
    [ -s "$folder/checkpoint_final.pth" ] || return 1
    [ -s "$folder/validation/summary.json" ] || return 1
    [ "$(validation_count "$folder")" -eq 1800 ] || return 1
    "$VENV/bin/python" -c "import json; json.load(open('$folder/validation/summary.json'))" >/dev/null
}

T1=nnUNetTrainerUNetPlusPlusSparseValidation
T2=nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation
T3=nnUNetTrainerBF16SparseValidation
T4=nnUNetTrainerBF16NoDeepSupervisionSparseValidation
D1=$(fold_dir "$T1")
D2=$(fold_dir "$T2")
D3=$(fold_dir "$T3")
D4=$(fold_dir "$T4")

if cell_done "$D1" && cell_done "$D2" && cell_done "$D3" && cell_done "$D4"; then
    echo "all four cells already have final checkpoints and complete 1,800-case validation"
    exit 0
fi

# Establish a non-overlapping recovery job before expensive staging/training. Slurm's afterany
# dependency prevents concurrent chains. A successful run cancels this pending successor.
SUCCESSOR=""
if [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; then
    SUCCESSOR=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
      --export="ALL,GRID_RETRY_COUNT=$((RETRY_COUNT + 1)),GRID_COMMIT=$GRID_COMMIT,GRID_RUN_ID=$GRID_RUN_ID" \
      "$FROZEN/unetpp_port/bridges2_deployment/grid_coordinated.sbatch")
    echo "recovery successor: $SUCCESSOR"
fi

LOCAL_ROOT=$LOCAL/pants_grid_$GRID_RUN_ID
case "$LOCAL_ROOT" in
    "$LOCAL"/pants_grid_*) ;;
    *) echo "ERROR: unsafe local staging path: $LOCAL_ROOT" >&2; exit 1 ;;
esac
LOCAL_RAW=$LOCAL_ROOT/nnUNet_raw
LOCAL_PREPROCESSED=$LOCAL_ROOT/nnUNet_preprocessed
STAGE_MARKER=$LOCAL_PREPROCESSED/Dataset001_PanTS/nnUNetPlansBS4.json

check_space() {
    local available_kb available_gib
    available_kb=$(df --output=avail "$LOCAL" | tail -1)
    available_gib=$((available_kb / 1024 / 1024))
    echo "[local space] ${available_gib}GiB available"
    if [ "$available_gib" -lt 2500 ]; then
        echo "ERROR: less than 2.5TiB free on node-local storage" >&2
        exit 1
    fi
}

if [ ! -f "$STAGE_MARKER" ]; then
    echo "=== stage and preprocess 9,000 training cases once ==="
    SOURCE=$LOCAL_ROOT/source
    DATA=$SOURCE/PanTS/data
    rm -rf "$LOCAL_RAW" "$LOCAL_PREPROCESSED" "$SOURCE"
    mkdir -p "$DATA/ImageTr" "$DATA/LabelTr" "$DATA/ImageTe" "$DATA/LabelTe" \
      "$LOCAL_RAW" "$LOCAL_PREPROCESSED" "$LOCAL_ROOT/empty_test_answer_key"
    check_space

    cd "$DATA"
    for pair in "1 2" "3 4" "5 6" "7 8" "9"; do
        PIDS=()
        for index in $pair; do
            start=$(printf '%08d' $(( (index - 1) * 1000 + 1 )))
            end=$(printf '%08d' $(( index * 1000 )))
            archive="PanTSMini_ImageTr_${start}_${end}.tar.gz"
            url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/${archive}?download=true"
            ( wget --tries=5 --retry-connrefused -O "$archive" "$url" && \
              tar -xzf "$archive" -C ImageTr && rm "$archive" && \
              echo "image archive $index/9 complete" ) &
            PIDS+=($!)
        done
        failed=0
        for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
        [ "$failed" -eq 0 ] || { echo "ERROR: image download/extraction failed" >&2; exit 1; }
        check_space
    done

    curl --fail --location --retry 5 --show-error \
      "https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz" \
      -o PanTSMini_Label.tar.gz
    mkdir -p LabelAll
    tar -xzf PanTSMini_Label.tar.gz -C LabelAll
    rm PanTSMini_Label.tar.gz
    cd LabelAll
    find . -maxdepth 1 -type d \( -name 'PanTS_0000[0-8]*' -o -name 'PanTS_00009000' \) \
      -print0 | xargs -0 -r mv -t ../LabelTr/
    cd ..
    rm -rf LabelAll

    n_images=$(find ImageTr -mindepth 1 -maxdepth 1 -type d | wc -l)
    n_labels=$(find LabelTr -mindepth 1 -maxdepth 1 -type d | wc -l)
    [ "$n_images" -eq 9000 ] && [ "$n_labels" -eq 9000 ] || {
        echo "ERROR: expected 9000 images/labels, got $n_images/$n_labels" >&2; exit 1;
    }

    "$VENV/bin/python" "$FROZEN/data_conversion/convert_pants_to_nnunet.py" \
      --pants-root "$DATA" --nnunet-dataset-dir "$LOCAL_RAW/Dataset001_PanTS" \
      --test-answer-key-dir "$LOCAL_ROOT/empty_test_answer_key" --workers 24
    "$VENV/bin/python" "$FROZEN/data_conversion/fix_affine_orthonormality.py" \
      "$LOCAL_RAW/Dataset001_PanTS"
    # We are still inside $SOURCE/PanTS/data. Deleting the process's cwd makes
    # the following PyTorch import fail with a misleading oneMKL/libtorch_cpu.so error.
    cd "$LOCAL_ROOT"
    rm -rf "$SOURCE"
    check_space

    export nnUNet_raw=$LOCAL_RAW
    export nnUNet_preprocessed=$LOCAL_PREPROCESSED
    export nnUNet_results=$LOCAL_ROOT/preprocess_results_scratch
    "$VENV/bin/nnUNetv2_plan_and_preprocess" -d 1 -c 3d_fullres \
      --verify_dataset_integrity -npfp 24 -np 24
    "$VENV/bin/python" - <<PY
import json
from pathlib import Path
raw = Path('$LOCAL_RAW/Dataset001_PanTS')
pre = Path('$LOCAL_PREPROCESSED/Dataset001_PanTS')
dataset = json.loads((raw / 'dataset.json').read_text())
assert dataset['numTraining'] == 9000, dataset['numTraining']
assert dataset['labels']['pancreatic_lesion'] == 28
source = pre / 'nnUNetPlans.json'
plans = json.loads(source.read_text())
plans['plans_name'] = 'nnUNetPlansBS4'
plans['configurations']['3d_fullres']['batch_size'] = 4
(pre / 'nnUNetPlansBS4.json.tmp').write_text(json.dumps(plans, indent=2))
print('full-dataset patch:', plans['configurations']['3d_fullres']['patch_size'])
PY
    data_id=$("$VENV/bin/python" -c "import json; p=json.load(open('$STAGE_MARKER.tmp')); print(p['configurations']['3d_fullres']['data_identifier'])")
    n_preprocessed=$(find "$LOCAL_PREPROCESSED/Dataset001_PanTS/$data_id" -maxdepth 1 -type f -name '*.npz' | wc -l)
    [ "$n_preprocessed" -eq 9000 ] || {
        echo "ERROR: expected 9000 preprocessed npz files, got $n_preprocessed" >&2; exit 1;
    }
    mv "$STAGE_MARKER.tmp" "$STAGE_MARKER"
    echo "FULL_DATASET_STAGE_DONE"
else
    echo "reusing complete staged dataset at $LOCAL_ROOT"
fi

# Recheck the complete-stage contract on every retry, including same-node reuse.
"$VENV/bin/python" - <<PY
import json
p=json.load(open('$STAGE_MARKER'))
assert p['configurations']['3d_fullres']['batch_size'] == 4
d=json.load(open('$LOCAL_RAW/Dataset001_PanTS/dataset.json'))
assert d['numTraining'] == 9000
assert d['labels']['pancreatic_lesion'] == 28
print('staged dataset contract passed: 9000 cases, class 28, physical batch 4')
PY
data_id=$("$VENV/bin/python" -c "import json; p=json.load(open('$STAGE_MARKER')); print(p['configurations']['3d_fullres']['data_identifier'])")
[ "$(find "$LOCAL_PREPROCESSED/Dataset001_PanTS/$data_id" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 9000 ]

export nnUNet_raw=$LOCAL_RAW
export nnUNet_preprocessed=$LOCAL_PREPROCESSED
export nnUNet_results=$RESULTS_ROOT
# Ten augmentation workers plus the trainer process fit inside each 12-CPU srun allocation without
# oversubscribing both concurrent cells.
export nnUNet_n_proc_DA=10
export nnUNet_compile=true
mkdir -p "$LOCAL_ROOT/torchinductor_cache"

LAST_PID=""
run_cell() {
    local trainer=$1 log_name=$2 folder=$3
    LAST_PID=""
    if cell_done "$folder"; then
        echo "[$trainer] final checkpoint and validation complete; skipping"
        return 0
    fi
    local args=(1 3d_fullres 0 -tr "$trainer" -p nnUNetPlansBS4)
    if [ -s "$folder/checkpoint_final.pth" ]; then
        echo "[$trainer] final checkpoint exists but validation is incomplete; running --val"
        args+=(--val)
    elif [ -s "$folder/checkpoint_latest.pth" ]; then
        echo "[$trainer] resuming checkpoint_latest.pth"
        args+=(--c)
    else
        echo "[$trainer] starting from epoch 0"
    fi
    {
      echo "=== job $SLURM_JOB_ID retry $RETRY_COUNT $(date -u) ==="
      export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/torchinductor_cache/$trainer"
      mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
      srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
        --cpus-per-task=12 --mem=100G "$VENV/bin/nnUNetv2_train" "${args[@]}"
    } >> "$ROOT/grid_logs/${log_name}.log" 2>&1 &
    LAST_PID=$!
}

run_phase() {
    local t_a=$1 log_a=$2 dir_a=$3 t_b=$4 log_b=$5 dir_b=$6
    local failed=0
    run_cell "$t_a" "$log_a" "$dir_a"; local pid_a=$LAST_PID
    run_cell "$t_b" "$log_b" "$dir_b"; local pid_b=$LAST_PID
    if [ -n "$pid_a" ]; then wait "$pid_a" || failed=1; fi
    if [ -n "$pid_b" ]; then wait "$pid_b" || failed=1; fi
    [ "$failed" -eq 0 ] || return 1
    cell_done "$dir_a" && cell_done "$dir_b"
}

echo "=== phase 1: UNet++ DS-on and DS-off ==="
run_phase "$T1" unetpp_ds_on "$D1" "$T2" unetpp_ds_off "$D2"
echo "PHASE1_UNETPP_TRAIN_AND_VALIDATION_DONE"

echo "=== phase 2: plain U-Net DS-on and DS-off ==="
run_phase "$T3" plain_ds_on "$D3" "$T4" plain_ds_off "$D4"
echo "PHASE2_PLAIN_TRAIN_AND_VALIDATION_DONE"

cell_done "$D1" && cell_done "$D2" && cell_done "$D3" && cell_done "$D4"
if [ -n "$SUCCESSOR" ]; then
    scancel "$SUCCESSOR" || true
    echo "cancelled unused recovery successor $SUCCESSOR"
fi
echo GRID_2X2_ALL_TRAINING_AND_VALIDATION_DONE

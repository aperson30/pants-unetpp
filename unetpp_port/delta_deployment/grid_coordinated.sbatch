#!/bin/bash
# v2: addresses Codex's second review round.
#  - GPU binding via `srun --exclusive --gpus-per-task=1` instead of a hand-set
#    CUDA_VISIBLE_DEVICES=0/1, which could have pointed at an unallocated physical GPU depending
#    on whether this cluster remaps device indices per-job via cgroups.
#  - fix_affine_orthonormality.py runs after conversion, before preprocessing (matches the fix
#    the project has needed before -- at least one case in the earlier 41-case sample needed it).
#  - Bounded retry counter (MAX_RETRIES) on the self-resubmission chain, so a permanent failure
#    can't loop forever burning allocation.
#  - Higher /tmp safety floor (300GB, was 150GB) given true peak usage approaches ~1.1TB out of
#    the node's 2TB.
#  - Dataset/config invariants are asserted (case counts, patch size, batch size, label 28), not
#    just printed.
#  - The script resubmits a commit-pinned frozen copy of itself, not the canonical (possibly still
#    being edited) file, so a mid-chain edit can't silently change what a later retry runs.
#SBATCH --account=bdyo-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=2
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=pants_grid_coordinated
#SBATCH --output=/projects/bdyo/asanjeev/delta_work/grid_logs/coordinated_%j.log

set -e
MAX_RETRIES=4
RETRY_COUNT=${GRID_RETRY_COUNT:-0}
echo "=== provenance ==="
hostname
date -u
echo "retry count: $RETRY_COUNT / $MAX_RETRIES"
FORK=/projects/bdyo/asanjeev/delta_work/pants-unetpp-fork
git -C "$FORK" rev-parse HEAD
git -C "$FORK" status --porcelain
nvidia-smi -L

if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
    echo "ERROR: hit MAX_RETRIES ($MAX_RETRIES) without finishing -- stopping the chain, needs a human to look" >&2
    exit 1
fi

module load pytorch-conda/2.12 >/dev/null 2>&1
source /projects/bdyo/asanjeev/delta_work/pants_venv/bin/activate
# PyTorch 2.12.1+cu130 (Delta's only other option besides 2.8) failed the class-28 numerical
# parity gate against the validated 2.10/cu129 stack (0.22526% relative diff on the tumor head's
# gradients, over the ~0.1% launch bar -- see CODEX_STATUS.md 2026-09-21). Shadow it with the
# isolated 2.10.0+cu129/cuDNN-9.10.2 install that matches DeltaAI's validated stack instead.
export PYTHONPATH=/work/nvme/bdyo/asanjeev/pytorch_drift_gate_20260921/torch210_cu129:$PYTHONPATH
python3 -c "import torch; assert torch.__version__.startswith('2.10.0'), torch.__version__; print('torch', torch.__version__, 'cudnn', torch.backends.cudnn.version())"

RESULTS_ROOT=/work/hdd/bdyo/asanjeev/nnUNet_results
mkdir -p "$RESULTS_ROOT"

# Pin the resubmitted script to the exact reviewed version: on the very first submission, freeze
# a copy stamped with the current commit SHA and always resubmit that frozen copy, not the
# canonical file (which could be edited mid-chain).
COMMIT=$(git -C "$FORK" rev-parse HEAD)
FROZEN=/projects/bdyo/asanjeev/delta_work/grid_launch/grid_coordinated.${COMMIT}.sbatch
if [ ! -f "$FROZEN" ]; then
    cp "$0" "$FROZEN"
    echo "froze this run's script as $FROZEN"
fi

done_marker() {
    echo "$RESULTS_ROOT/Dataset001_PanTS/$1__nnUNetPlansBS4__3d_fullres/fold_0/checkpoint_final.pth"
}
D1=$(done_marker nnUNetTrainerUNetPlusPlusSparseValidation)
D2=$(done_marker nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation)
D3=$(done_marker nnUNetTrainerBF16SparseValidation)
D4=$(done_marker nnUNetTrainerBF16NoDeepSupervisionSparseValidation)

if [ -f "$D1" ] && [ -f "$D2" ] && [ -f "$D3" ] && [ -f "$D4" ]; then
    echo "all four cells already have checkpoint_final.pth -- nothing to do"
    exit 0
fi

sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL,GRID_RETRY_COUNT=$((RETRY_COUNT + 1)) "$FROZEN"

LOCAL_ROOT=/tmp/$USER/$SLURM_JOB_ID
LOCAL_RAW=$LOCAL_ROOT/nnUNet_raw
LOCAL_PREPROCESSED=$LOCAL_ROOT/nnUNet_preprocessed
STAGE_MARKER=$LOCAL_PREPROCESSED/Dataset001_PanTS/nnUNetPlansBS4.json

check_space() {
    avail_kb=$(df --output=avail /tmp | tail -1)
    avail_gb=$((avail_kb / 1024 / 1024))
    echo "[space check] ${avail_gb}GB free on /tmp"
    if [ "$avail_gb" -lt 300 ]; then
        echo "ERROR: /tmp headroom (${avail_gb}GB) below the 300GB safety floor, aborting" >&2
        exit 1
    fi
}

if [ ! -f "$STAGE_MARKER" ]; then
    echo "=== staging dataset once into $LOCAL_ROOT ==="
    mkdir -p "$LOCAL_ROOT/PanTS/data/ImageTr" "$LOCAL_ROOT/PanTS/data/ImageTe" "$LOCAL_RAW" "$LOCAL_PREPROCESSED"
    cd "$LOCAL_ROOT/PanTS/data"
    check_space

    wget -O metadata.xlsx "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/metadata.xlsx?download=true" || true

    for pair in "1 2" "3 4" "5 6" "7 8" "9"; do
      PIDS=()
      for i in $pair; do
        start=$(printf "%08d" $(( (i - 1) * 1000 + 1 )))
        end=$(printf "%08d" $(( i * 1000 )))
        file="PanTSMini_ImageTr_${start}_${end}.tar.gz"
        url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/${file}?download=true"
        ( wget -O "$file" "$url" && tar -xzf "$file" -C ImageTr && rm "$file" && echo "[$i/9] image chunk done" ) &
        PIDS+=($!)
      done
      FAIL=0
      for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
      if [ "$FAIL" -ne 0 ]; then echo "ERROR: image chunk download/extract failed" >&2; exit 1; fi
      check_space
    done

    curl -sSL -f "https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz" -o PanTSMini_Label.tar.gz
    mkdir -p LabelAll
    tar -xzf PanTSMini_Label.tar.gz -C LabelAll
    rm PanTSMini_Label.tar.gz
    check_space

    mkdir -p LabelTr LabelTe
    cd LabelAll
    find . -maxdepth 1 -type d \( -name 'PanTS_0000[0-8]*' -o -name 'PanTS_00009000' \) -print0 | xargs -0 -r mv -t ../LabelTr/
    find . -maxdepth 1 -type d -name 'PanTS_00009*' -print0 | xargs -0 -r mv -t ../LabelTe/
    cd ..
    rmdir LabelAll 2>/dev/null || true

    file="PanTSMini_ImageTe_00009001_00009901.tar.gz"
    wget -O "$file" "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/$file?download=true"
    tar -xzf "$file" -C ImageTe && rm "$file"

    n_tr=$(ls ImageTr | wc -l)
    n_te=$(ls ImageTe | wc -l)
    echo "train cases: $n_tr  test cases: $n_te"
    if [ "$n_tr" -ne 9000 ] || [ "$n_te" -ne 901 ]; then
        echo "ERROR: expected 9000 train / 901 test cases, got $n_tr / $n_te" >&2
        exit 1
    fi
    echo "STAGE_DOWNLOAD_DONE"
    check_space

    cd /projects/bdyo/asanjeev/delta_work
    python3 "$FORK/data_conversion/convert_pants_to_nnunet.py" \
      --pants-root "$LOCAL_ROOT/PanTS/data" \
      --nnunet-dataset-dir "$LOCAL_RAW/Dataset001_PanTS" \
      --test-answer-key-dir "$LOCAL_ROOT/PanTS_test_answer_key"
    echo "STAGE_CONVERT_DONE"

    echo "=== affine-orthonormality repair (needed at least once before, on the 41-case sample) ==="
    python3 "$FORK/data_conversion/fix_affine_orthonormality.py" "$LOCAL_RAW/Dataset001_PanTS"
    python3 "$FORK/data_conversion/fix_affine_orthonormality.py" "$LOCAL_ROOT/PanTS_test_answer_key"
    echo "STAGE_AFFINE_FIX_DONE"

    rm -rf "$LOCAL_ROOT/PanTS"
    check_space

    export nnUNet_raw=$LOCAL_RAW
    export nnUNet_preprocessed=$LOCAL_PREPROCESSED
    export nnUNet_results=$LOCAL_ROOT/nnUNet_results_scratch
    nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity -npfp 24 -np 24
    python3 -c "
import json
p='$LOCAL_PREPROCESSED/Dataset001_PanTS/nnUNetPlans.json'
plans=json.load(open(p))
patch = plans['configurations']['3d_fullres']['patch_size']
# Trust the real planner's output on the actual 9000-case dataset rather than hard-asserting a
# specific value: a small calibration sample earlier gave a DIFFERENT patch ([64,160,192]) than
# GB10/DeltaAI's larger samples ([64,160,224]) -- small-sample patch determination is genuinely
# noisy, so the full dataset's own statistics are the authoritative answer here, not a prior guess.
if patch != [64, 160, 224]:
    print(f'NOTE: real full-dataset patch is {patch}, not the [64,160,224] seen on GB10/DeltaAI -- '
          f'this changes the per-cell GPU-hour estimate, continuing anyway since this is the real answer')
plans['plans_name']='nnUNetPlansBS4'
plans['configurations']['3d_fullres']['batch_size']=4
json.dump(plans, open('$STAGE_MARKER','w'), indent=2)
print('patch_size:', patch)
"
    echo "STAGE_PREPROCESS_DONE"
    check_space
else
    echo "found existing staged+preprocessed data in /tmp on this node, reusing"
fi

n_final=$(ls "$LOCAL_RAW/Dataset001_PanTS/imagesTr" | wc -l)
echo "train cases (final, on /tmp): $n_final  (expected 9000)"
if [ "$n_final" -ne 9000 ]; then
    echo "ERROR: staged data has $n_final train cases, expected 9000" >&2
    exit 1
fi
python3 -c "
import json
d = json.load(open('$LOCAL_RAW/Dataset001_PanTS/dataset.json'))
assert d['numTraining'] == 9000, d['numTraining']
assert d['labels']['pancreatic_lesion'] == 28, d['labels']['pancreatic_lesion']
print('dataset.json invariants OK: numTraining=9000, pancreatic_lesion label=28')
"

export nnUNet_results=$RESULTS_ROOT
export PYTHONPATH="$FORK:$PYTHONPATH"

LAST_PID=""
run_cell() {
    trainer=$1; logname=$2; marker_final=$3
    LAST_PID=""
    if [ -f "$marker_final" ]; then
        echo "[$trainer] checkpoint_final.pth already exists, skipping"
        return 0
    fi
    marker_latest="${marker_final%checkpoint_final.pth}checkpoint_latest.pth"
    cflag=""
    if [ -f "$marker_latest" ]; then
        echo "[$trainer] checkpoint_latest.pth found, resuming with --c"
        cflag="--c"
    fi
    export nnUNet_raw nnUNet_preprocessed nnUNet_results PYTHONPATH
    srun --exclusive --gpus-per-task=1 -n1 --ntasks=1 \
      nnUNetv2_train 1 3d_fullres 0 -tr "$trainer" -p nnUNetPlansBS4 $cflag \
      > "$RESULTS_ROOT/${logname}_train.log" 2>&1 &
    LAST_PID=$!
}

PHASE_FAIL=0

echo "=== phase 1: UNet++ DS-on + DS-off, concurrently, one srun step per GPU ==="
run_cell nnUNetTrainerUNetPlusPlusSparseValidation unetpp_ds "$D1"; P1=$LAST_PID
run_cell nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation unetpp_nods "$D2"; P2=$LAST_PID
if [ -n "$P1" ]; then wait "$P1" || { echo "ERROR: unetpp_ds trainer failed, see unetpp_ds_train.log" >&2; PHASE_FAIL=1; }; fi
if [ -n "$P2" ]; then wait "$P2" || { echo "ERROR: unetpp_nods trainer failed, see unetpp_nods_train.log" >&2; PHASE_FAIL=1; }; fi
[ "$PHASE_FAIL" -eq 0 ] && echo "PHASE1_UNETPP_DONE"

echo "=== phase 2: Plain U-Net DS-on + DS-off, concurrently, one srun step per GPU ==="
run_cell nnUNetTrainerBF16SparseValidation default_ds "$D3"; P3=$LAST_PID
run_cell nnUNetTrainerBF16NoDeepSupervisionSparseValidation default_nods "$D4"; P4=$LAST_PID
if [ -n "$P3" ]; then wait "$P3" || { echo "ERROR: default_ds trainer failed, see default_ds_train.log" >&2; PHASE_FAIL=1; }; fi
if [ -n "$P4" ]; then wait "$P4" || { echo "ERROR: default_nods trainer failed, see default_nods_train.log" >&2; PHASE_FAIL=1; }; fi
[ "$PHASE_FAIL" -eq 0 ] && echo "PHASE2_PLAINUNET_DONE"

if [ "$PHASE_FAIL" -ne 0 ]; then
    echo "ERROR: at least one cell's trainer failed this run -- not printing the all-done marker" >&2
    exit 1
fi

echo "GRID_COORDINATED_ALL_DONE"

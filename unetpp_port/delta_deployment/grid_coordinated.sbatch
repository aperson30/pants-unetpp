#!/bin/bash
# One coordinated 2-GPU job, per Codex's review: four independent single-GPU /tmp staging jobs
# would duplicate ~1.4TB of downloads + 4x preprocessing, and risked two of them landing on the
# same 8-GPU H200 node and colliding on /tmp capacity (2TB/node, Slurm doesn't treat it as a
# reservable resource). This stages the dataset ONCE into a job-specific /tmp path, then runs the
# two UNet++ cells concurrently (the bottleneck, ~22h), then the two Plain-U-Net cells concurrently
# (~6h) -- ~28h total on 2 GPUs, ~56 raw GPU-hours, close to the 55.5h theoretical floor.
# Checkpoints go to /work/hdd (no enforced quota there, confirmed via lfs quota; verified live
# 88GB free on the underlying filesystem right now), not the nearly-full /projects.
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
echo "=== provenance ==="
hostname
date -u
git -C /projects/bdyo/asanjeev/delta_work/pants-unetpp-fork rev-parse HEAD
git -C /projects/bdyo/asanjeev/delta_work/pants-unetpp-fork status --porcelain
nvidia-smi -L

module load pytorch-conda/2.12 >/dev/null 2>&1
source /projects/bdyo/asanjeev/delta_work/pants_venv/bin/activate
python3 -c "import torch; print('torch', torch.__version__, 'cudnn', torch.backends.cudnn.version())"

RESULTS_ROOT=/work/hdd/bdyo/asanjeev/nnUNet_results
mkdir -p "$RESULTS_ROOT"
SELF=/projects/bdyo/asanjeev/delta_work/grid_launch/grid_coordinated.sbatch

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

# Chain the successor before doing any work, same reasoning as the per-cell design: Slurm's
# afterany dependency guarantees it can't start until this job has fully exited by any means, so
# a timeout mid-run automatically continues (restaging on /tmp, resuming whichever cells still
# need it via --c) without a manual intervention or a signal-handler race condition.
sbatch --dependency=afterany:$SLURM_JOB_ID "$SELF"

LOCAL_ROOT=/tmp/$USER/$SLURM_JOB_ID
LOCAL_RAW=$LOCAL_ROOT/nnUNet_raw
LOCAL_PREPROCESSED=$LOCAL_ROOT/nnUNet_preprocessed
STAGE_MARKER=$LOCAL_PREPROCESSED/Dataset001_PanTS/nnUNetPlansBS4.json

check_space() {
    avail_kb=$(df --output=avail /tmp | tail -1)
    avail_gb=$((avail_kb / 1024 / 1024))
    echo "[space check] ${avail_gb}GB free on /tmp"
    if [ "$avail_gb" -lt 150 ]; then
        echo "ERROR: /tmp headroom (${avail_gb}GB) below safety floor, aborting" >&2
        exit 1
    fi
}

echo "=== staging dataset once into $LOCAL_ROOT ==="
mkdir -p "$LOCAL_ROOT/PanTS/data/ImageTr" "$LOCAL_ROOT/PanTS/data/ImageTe" "$LOCAL_RAW" "$LOCAL_PREPROCESSED"
cd "$LOCAL_ROOT/PanTS/data"
check_space

wget -O metadata.xlsx "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/metadata.xlsx?download=true" || true

# Download 2 chunks at a time (not all 9 at once) to control peak transient disk use and reduce
# simultaneous-connection load on the source servers, per Codex's review.
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

echo "train cases: $(ls ImageTr | wc -l)  test cases: $(ls ImageTe | wc -l)"
echo "STAGE_DOWNLOAD_DONE"
check_space

cd /projects/bdyo/asanjeev/delta_work
python3 pants-unetpp-fork/data_conversion/convert_pants_to_nnunet.py \
  --pants-root "$LOCAL_ROOT/PanTS/data" \
  --nnunet-dataset-dir "$LOCAL_RAW/Dataset001_PanTS" \
  --test-answer-key-dir "$LOCAL_ROOT/PanTS_test_answer_key"
echo "STAGE_CONVERT_DONE"

# Free the raw PanTS source tree now that conversion succeeded -- convert_pants_to_nnunet.py
# copies into nnUNet_raw, it doesn't need the source afterward.
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
plans['plans_name']='nnUNetPlansBS4'
plans['configurations']['3d_fullres']['batch_size']=4
json.dump(plans, open('$STAGE_MARKER','w'), indent=2)
print('patch_size:', plans['configurations']['3d_fullres']['patch_size'])
"
echo "STAGE_PREPROCESS_DONE"
check_space

echo "train cases (final, on /tmp): $(ls $LOCAL_RAW/Dataset001_PanTS/imagesTr | wc -l)  (expected 9000)"

export nnUNet_results=$RESULTS_ROOT
export PYTHONPATH=/projects/bdyo/asanjeev/delta_work/pants-unetpp-fork:$PYTHONPATH

# NOTE: deliberately not using $(...) command substitution to launch these -- command
# substitution always forks a subshell, and a background job started inside one becomes
# orphaned from this script's job table the moment the subshell exits, making `wait` on it
# fail. LAST_PID is a plain global set directly in this shell's own process, so the background
# job stays a real child of this script and `wait` works correctly.
LAST_PID=""
run_cell() {
    trainer=$1; gpu=$2; logname=$3; marker_final=$4
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
    CUDA_VISIBLE_DEVICES=$gpu nnUNetv2_train 1 3d_fullres 0 -tr "$trainer" -p nnUNetPlansBS4 $cflag \
      > "$RESULTS_ROOT/${logname}_train.log" 2>&1 &
    LAST_PID=$!
}

PHASE_FAIL=0

echo "=== phase 1: UNet++ DS-on + DS-off, concurrently on the 2 allocated GPUs ==="
run_cell nnUNetTrainerUNetPlusPlusSparseValidation 0 unetpp_ds "$D1"; P1=$LAST_PID
run_cell nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation 1 unetpp_nods "$D2"; P2=$LAST_PID
if [ -n "$P1" ]; then wait "$P1" || { echo "ERROR: unetpp_ds trainer failed, see unetpp_ds_train.log" >&2; PHASE_FAIL=1; }; fi
if [ -n "$P2" ]; then wait "$P2" || { echo "ERROR: unetpp_nods trainer failed, see unetpp_nods_train.log" >&2; PHASE_FAIL=1; }; fi
[ "$PHASE_FAIL" -eq 0 ] && echo "PHASE1_UNETPP_DONE"

echo "=== phase 2: Plain U-Net DS-on + DS-off, concurrently on the 2 allocated GPUs ==="
run_cell nnUNetTrainerBF16SparseValidation 0 default_ds "$D3"; P3=$LAST_PID
run_cell nnUNetTrainerBF16NoDeepSupervisionSparseValidation 1 default_nods "$D4"; P4=$LAST_PID
if [ -n "$P3" ]; then wait "$P3" || { echo "ERROR: default_ds trainer failed, see default_ds_train.log" >&2; PHASE_FAIL=1; }; fi
if [ -n "$P4" ]; then wait "$P4" || { echo "ERROR: default_nods trainer failed, see default_nods_train.log" >&2; PHASE_FAIL=1; }; fi
[ "$PHASE_FAIL" -eq 0 ] && echo "PHASE2_PLAINUNET_DONE"

if [ "$PHASE_FAIL" -ne 0 ]; then
    echo "ERROR: at least one cell's trainer failed this run -- not printing the all-done marker" >&2
    exit 1
fi

echo "GRID_COORDINATED_ALL_DONE"

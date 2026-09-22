#!/bin/bash
# Prepared only. Do not submit until PAPER_CONFIG_REVIEW.md's final checklist passes.
# This job intentionally waits for the 2x2 grid's UNet++ validation outputs so it can reuse the
# exact fold-0 IDs and assert plan parity instead of merely assuming both independent runs agree.
#SBATCH --account=bdyo-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=pants_unetpp_paper
#SBATCH --output=/work/hdd/bdyo/asanjeev/nnUNet_results/paper_%j.log

set -euo pipefail

MAX_RETRIES=3
RETRY_COUNT=${PAPER_RETRY_COUNT:-0}
FORK=/projects/bdyo/asanjeev/delta_work/pants-unetpp-fork
RESULTS_ROOT=/work/hdd/bdyo/asanjeev/nnUNet_results
DATASET_RESULTS=$RESULTS_ROOT/Dataset001_PanTS
TRAINER=nnUNetTrainerUNetPlusPlusPaperSparseValidation
REFERENCE_TRAINER=nnUNetTrainerUNetPlusPlusSparseValidation
PLANS=nnUNetPlansBS4
PAPER_FOLD=$DATASET_RESULTS/${TRAINER}__${PLANS}__3d_fullres/fold_0
REFERENCE_MODEL=$DATASET_RESULTS/${REFERENCE_TRAINER}__${PLANS}__3d_fullres
REFERENCE_VALIDATION=$REFERENCE_MODEL/fold_0/validation
DONE_MARKER=$PAPER_FOLD/paper_training_and_validation_complete.json
LOCAL_ROOT=/tmp/$USER/pants_paper_$SLURM_JOB_ID
LOCAL_RAW=$LOCAL_ROOT/nnUNet_raw
LOCAL_PREPROCESSED=$LOCAL_ROOT/nnUNet_preprocessed
STAGE_MARKER=$LOCAL_PREPROCESSED/Dataset001_PanTS/${PLANS}.json
SUCCESSOR_JOB=""

cleanup() {
    if [[ "$LOCAL_ROOT" == /tmp/"$USER"/pants_paper_* ]]; then
        rm -rf "$LOCAL_ROOT"
    fi
}
trap cleanup EXIT

verify_completion() {
    test -s "$DONE_MARKER" || return 1
    test -s "$PAPER_FOLD/checkpoint_final.pth" || return 1
    test "$(find "$PAPER_FOLD/validation" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)" -eq 1800 || return 1
    python3 - "$DONE_MARKER" "$PAPER_FOLD/checkpoint_final.pth" <<'PY'
import hashlib, json, sys
from pathlib import Path
marker, checkpoint = map(Path, sys.argv[1:])
record = json.loads(marker.read_text())
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert record["checkpoint_sha256"] == digest
assert record["validation_cases"] == 1800
PY
}

mkdir -p "$RESULTS_ROOT"
if verify_completion; then
    echo "Paper configuration already has a verified completion marker; nothing to do."
    exit 0
fi
if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
    echo "ERROR: exhausted $MAX_RETRIES retries without verified training+validation completion" >&2
    exit 1
fi

echo "=== provenance ==="
hostname
date -u
echo "retry $RETRY_COUNT / $MAX_RETRIES"
COMMIT=$(git -C "$FORK" rev-parse HEAD)
test -z "$(git -C "$FORK" status --porcelain)" || { echo "ERROR: fork checkout is dirty" >&2; exit 1; }
echo "commit $COMMIT"
nvidia-smi -L

echo "=== require completed reference validation before any shared-environment write ==="
test -s "$REFERENCE_MODEL/plans.json" || {
    echo "ERROR: reference UNet++ plans.json is not available yet" >&2; exit 1;
}
test -s "$REFERENCE_MODEL/fold_0/checkpoint_final.pth" || {
    echo "ERROR: reference UNet++ final checkpoint is not available yet" >&2; exit 1;
}
test "$(find "$REFERENCE_VALIDATION" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)" -eq 1800 || {
    echo "ERROR: reference UNet++ final validation is not exactly 1,800 cases" >&2; exit 1;
}

module load pytorch-conda/2.12 >/dev/null 2>&1
source /projects/bdyo/asanjeev/delta_work/pants_venv/bin/activate
export PYTHONPATH=/work/nvme/bdyo/asanjeev/pytorch_drift_gate_20260921/torch210_cu129:$FORK:${PYTHONPATH:-}
export nnUNet_compile=true
export nnUNet_n_proc_DA=8
export nnUNet_def_n_proc=4
python3 - <<'PY'
import torch
assert torch.__version__.startswith("2.10.0"), torch.__version__
assert torch.cuda.is_available()
print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
PY

python3 - "$REFERENCE_VALIDATION" <<'PY'
import sys
from pathlib import Path
import nibabel as nib

folder = Path(sys.argv[1])
files = sorted(folder.glob("*.nii.gz"))
assert len(files) == 1800
for path in files:
    image = nib.load(str(path))
    assert len(image.shape) == 3 and all(size > 0 for size in image.shape), path
print("reference checkpoint plus 1,800 readable validation outputs confirmed")
PY

echo "=== source install and contract checks ==="
TRAINER_PACKAGE=$(python3 - <<'PY'
from pathlib import Path
import nnunetv2.training.nnUNetTrainer as package
print(Path(next(iter(package.__path__))).resolve())
PY
)
for file in \
  unet_plusplus.py \
  nnUNetTrainerBF16Mixin.py \
  nnUNetTrainerQualityNeutralOptimizationMixin.py \
  nnUNetTrainerSparseValidationMixin.py \
  nnUNetTrainerUNetPlusPlus.py \
  nnUNetTrainerUNetPlusPlusPaper.py \
  nnUNetTrainerUNetPlusPlusPaperSparseValidation.py; do
    install -m 0644 "$FORK/unetpp_port/$file" "$TRAINER_PACKAGE/$file"
    test "$(sha256sum "$FORK/unetpp_port/$file" | cut -d' ' -f1)" = \
         "$(sha256sum "$TRAINER_PACKAGE/$file" | cut -d' ' -f1)"
done
python3 "$FORK/unetpp_port/trainer_dead_head_contract_test.py"
python3 "$FORK/unetpp_port/trainer_quality_neutral_optimization_test.py"
cd "$FORK/unetpp_port"
python3 paper_inference_contract_test.py
python3 smoke_test.py

# Establish a bounded after-any successor before long staging/training. It cannot overlap this job.
FROZEN=/projects/bdyo/asanjeev/delta_work/grid_launch/paper_config.${COMMIT}.sbatch
if [ ! -f "$FROZEN" ]; then
    install -m 0644 "$0" "$FROZEN"
fi
SUCCESSOR_JOB=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID \
  --export=ALL,PAPER_RETRY_COUNT=$((RETRY_COUNT + 1)) "$FROZEN")
echo "scheduled bounded successor $SUCCESSOR_JOB"

check_space() {
    avail_kb=$(df --output=avail /tmp | tail -1)
    avail_gb=$((avail_kb / 1024 / 1024))
    echo "[space] ${avail_gb}GB free on /tmp"
    test "$avail_gb" -ge 300 || { echo "ERROR: less than 300GB /tmp headroom" >&2; exit 1; }
}

echo "=== stage the 9,000 training cases once; test data is intentionally omitted ==="
mkdir -p "$LOCAL_ROOT/PanTS/data/ImageTr" "$LOCAL_RAW" "$LOCAL_PREPROCESSED"
cd "$LOCAL_ROOT/PanTS/data"
check_space
for pair in "1 2" "3 4" "5 6" "7 8" "9"; do
    pids=()
    for i in $pair; do
        start=$(printf "%08d" $(( (i - 1) * 1000 + 1 )))
        end=$(printf "%08d" $(( i * 1000 )))
        archive="PanTSMini_ImageTr_${start}_${end}.tar.gz"
        url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/${archive}?download=true"
        (wget -O "$archive" "$url" && tar -xzf "$archive" -C ImageTr && rm "$archive") &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    test "$failed" -eq 0 || { echo "ERROR: image download/extract failed" >&2; exit 1; }
    check_space
done

curl -sSL -f "https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz" \
  -o PanTSMini_Label.tar.gz
mkdir LabelAll LabelTr
tar -xzf PanTSMini_Label.tar.gz -C LabelAll
rm PanTSMini_Label.tar.gz
find LabelAll -maxdepth 1 -type d \( -name 'PanTS_0000[0-8]*' -o -name 'PanTS_00009000' \) \
  -print0 | xargs -0 -r mv -t LabelTr/
rm -rf LabelAll
test "$(find ImageTr -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 9000
test "$(find LabelTr -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 9000

cd "$FORK"
python3 data_conversion/convert_pants_train_only.py \
  --pants-root "$LOCAL_ROOT/PanTS/data" \
  --nnunet-dataset-dir "$LOCAL_RAW/Dataset001_PanTS" --expected-cases 9000
python3 data_conversion/fix_affine_orthonormality.py "$LOCAL_RAW/Dataset001_PanTS"
rm -rf "$LOCAL_ROOT/PanTS"
check_space

export nnUNet_raw=$LOCAL_RAW
export nnUNet_preprocessed=$LOCAL_PREPROCESSED
export nnUNet_results=$LOCAL_ROOT/nnUNet_results_scratch
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity -npfp 12 -np 12
python3 - "$LOCAL_PREPROCESSED/Dataset001_PanTS/nnUNetPlans.json" "$STAGE_MARKER" \
  "$REFERENCE_MODEL/plans.json" <<'PY'
import json, sys
from pathlib import Path
source, destination, reference = map(Path, sys.argv[1:])
plans = json.loads(source.read_text())
plans["plans_name"] = "nnUNetPlansBS4"
plans["configurations"]["3d_fullres"]["batch_size"] = 4
reference_plans = json.loads(reference.read_text())
if plans != reference_plans:
    current = plans["configurations"]["3d_fullres"]
    expected = reference_plans["configurations"]["3d_fullres"]
    differences = {key: (current.get(key), expected.get(key))
                   for key in current.keys() | expected.keys()
                   if current.get(key) != expected.get(key)}
    raise RuntimeError(f"Paper plan differs from 2x2 reference: {differences}")
destination.write_text(json.dumps(plans, indent=2))
print("exact plan parity confirmed; patch", plans["configurations"]["3d_fullres"]["patch_size"])
PY

echo "=== pin fold 0 to the exact 1,800 reference validation case IDs ==="
python3 - "$LOCAL_RAW/Dataset001_PanTS/imagesTr" "$REFERENCE_VALIDATION" \
  "$LOCAL_PREPROCESSED/Dataset001_PanTS/splits_final.json" <<'PY'
import json, sys
from pathlib import Path
images, validation, destination = map(Path, sys.argv[1:])
all_ids = sorted(path.name[:-12] for path in images.glob("*_0000.nii.gz"))
val_ids = sorted(path.name[:-7] for path in validation.glob("*.nii.gz"))
assert len(all_ids) == 9000 and len(val_ids) == 1800
assert set(val_ids) <= set(all_ids)
train_ids = sorted(set(all_ids) - set(val_ids))
assert len(train_ids) == 7200
destination.write_text(json.dumps([{"train": train_ids, "val": val_ids}], indent=2))
print("fixed fold 0:", len(train_ids), "train /", len(val_ids), "validation")
PY

python3 - "$LOCAL_RAW/Dataset001_PanTS/dataset.json" <<'PY'
import json, sys
dataset = json.load(open(sys.argv[1]))
assert dataset["numTraining"] == 9000
assert dataset["labels"]["pancreatic_lesion"] == 28
PY

export nnUNet_results=$RESULTS_ROOT
mkdir -p "$PAPER_FOLD"
if [ -s "$PAPER_FOLD/checkpoint_final.pth" ]; then
    echo "final checkpoint exists but verified completion does not; rerunning final validation only"
    srun --exclusive --exact --gpus-per-task=1 --cpus-per-task=12 -n1 --ntasks=1 \
      nnUNetv2_train 1 3d_fullres 0 -tr "$TRAINER" -p "$PLANS" --val \
      > "$RESULTS_ROOT/paper_validation_${SLURM_JOB_ID}.log" 2>&1
else
    continuation=()
    if [ -s "$PAPER_FOLD/checkpoint_latest.pth" ]; then continuation=(--c); fi
    srun --exclusive --exact --gpus-per-task=1 --cpus-per-task=12 -n1 --ntasks=1 \
      nnUNetv2_train 1 3d_fullres 0 -tr "$TRAINER" -p "$PLANS" "${continuation[@]}" \
      > "$RESULTS_ROOT/paper_train_${SLURM_JOB_ID}.log" 2>&1
fi

echo "=== fail-closed completion audit ==="
test -s "$PAPER_FOLD/checkpoint_final.pth"
test "$(find "$PAPER_FOLD/validation" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)" -eq 1800
python3 - "$PAPER_FOLD/validation" "$REFERENCE_VALIDATION" "$PAPER_FOLD/checkpoint_final.pth" \
  "$DONE_MARKER" "$COMMIT" <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path
paper, reference, checkpoint, marker = map(Path, sys.argv[1:5])
commit = sys.argv[5]
paper_ids = sorted(path.name for path in paper.glob("*.nii.gz"))
reference_ids = sorted(path.name for path in reference.glob("*.nii.gz"))
assert len(paper_ids) == 1800 and paper_ids == reference_ids
for path in paper.glob("*.nii.gz"):
    import nibabel as nib
    image = nib.load(str(path))
    assert len(image.shape) == 3 and all(size > 0 for size in image.shape), path
record = {
    "commit": commit,
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "validation_cases": len(paper_ids),
    "reference_validation_ids_equal": True,
    "trainer": "nnUNetTrainerUNetPlusPlusPaperSparseValidation",
    "epochs": 1000,
    "physical_batch_size": 4,
}
temporary = marker.with_suffix(marker.suffix + ".tmp")
temporary.write_text(json.dumps(record, indent=2))
temporary.replace(marker)
print(json.dumps(record, indent=2))
PY

verify_completion
scancel "$SUCCESSOR_JOB"
echo "PAPER_TRAINING_AND_VALIDATION_ALL_DONE"

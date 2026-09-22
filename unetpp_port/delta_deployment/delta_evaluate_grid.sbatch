#!/bin/bash
# Prepared only: submit manually after the final grid job/successor exits cleanly.
# See evaluation/DELTA_EVALUATION_PROTOCOL.md before use.
#SBATCH --account=bdyo-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=2
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=pants_grid_eval
#SBATCH --output=/work/hdd/bdyo/asanjeev/nnUNet_results/evaluate_%j.log

set -euo pipefail

FORK=/projects/bdyo/asanjeev/delta_work/pants-unetpp-fork
RESULTS_ROOT=/work/hdd/bdyo/asanjeev/nnUNet_results
EVAL_ROOT=$RESULTS_ROOT/evaluation_grid
EVAL_RUN=$EVAL_ROOT/run_$SLURM_JOB_ID
LOCAL_ROOT=/tmp/$USER/pants_eval_$SLURM_JOB_ID
LOCAL_RAW=$LOCAL_ROOT/nnUNet_raw/Dataset001_PanTS
LOCAL_GT=$LOCAL_ROOT/PanTS_test_answer_key
LOCAL_OUTPUT=$LOCAL_ROOT/predictions
PLANS=nnUNetPlansBS4
EXPECTED_TEST=901
EXPECTED_VALIDATION=1800

mkdir -p "$EVAL_RUN" "$LOCAL_ROOT"
cleanup() { rm -rf "$LOCAL_ROOT"; }
trap cleanup EXIT

module load pytorch-conda/2.12 >/dev/null 2>&1
source /projects/bdyo/asanjeev/delta_work/pants_venv/bin/activate
export PYTHONPATH=/work/nvme/bdyo/asanjeev/pytorch_drift_gate_20260921/torch210_cu129:$FORK:${PYTHONPATH:-}
export nnUNet_raw=$LOCAL_ROOT/nnUNet_raw
export nnUNet_preprocessed=$LOCAL_ROOT/nnUNet_preprocessed_unused
export nnUNet_results=$RESULTS_ROOT
export nnUNet_n_proc_DA=6
export nnUNet_def_n_proc=4

python3 - <<'PY'
import torch
assert torch.__version__.startswith("2.10.0"), torch.__version__
print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
PY
echo "commit $(git -C "$FORK" rev-parse HEAD)"
if [ -n "$(git -C "$FORK" status --porcelain)" ]; then
    echo "ERROR: evaluation checkout is dirty" >&2
    exit 1
fi
nvidia-smi -L

declare -A TRAINERS=(
  [unetpp_ds]=nnUNetTrainerUNetPlusPlusSparseValidation
  [unetpp_nods]=nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation
  [default_ds]=nnUNetTrainerBF16SparseValidation
  [default_nods]=nnUNetTrainerBF16NoDeepSupervisionSparseValidation
)

echo "=== readiness: complete checkpoint plus 1,800 validation outputs per cell ==="
for tag in unetpp_ds unetpp_nods default_ds default_nods; do
    trainer=${TRAINERS[$tag]}
    fold=$RESULTS_ROOT/Dataset001_PanTS/${trainer}__${PLANS}__3d_fullres/fold_0
    test -s "$fold/checkpoint_final.pth"
    n_val=$(find "$fold/validation" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
    if [ "$n_val" -ne "$EXPECTED_VALIDATION" ]; then
        echo "ERROR: $tag validation has $n_val/$EXPECTED_VALIDATION predictions" >&2
        echo "Do not infer completion from checkpoint_final.pth; final validation must be repaired first." >&2
        exit 1
    fi
    python3 - "$fold/validation" "$EXPECTED_VALIDATION" <<'PY'
import sys
from pathlib import Path
import nibabel as nib
folder, expected = Path(sys.argv[1]), int(sys.argv[2])
files = sorted(folder.glob("*.nii.gz"))
assert len(files) == expected
for path in files:
    image = nib.load(str(path))
    assert len(image.shape) == 3 and all(n > 0 for n in image.shape), path
print(f"validated {len(files)} readable cross-validation outputs in {folder}")
PY
done

echo "=== stage only PanTS-te and its answer key into node-local storage ==="
mkdir -p "$LOCAL_ROOT/PanTS/data/ImageTe" "$LOCAL_ROOT/PanTS/data/LabelTe" "$LOCAL_OUTPUT"
cd "$LOCAL_ROOT/PanTS/data"
test_archive=PanTSMini_ImageTe_00009001_00009901.tar.gz
wget -O "$test_archive" "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/$test_archive?download=true"
tar -xzf "$test_archive" -C ImageTe
rm "$test_archive"

curl -sSL -f "https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz" -o PanTSMini_Label.tar.gz
mkdir LabelAll
tar -xzf PanTSMini_Label.tar.gz -C LabelAll
rm PanTSMini_Label.tar.gz
find LabelAll -maxdepth 1 -type d -name 'PanTS_00009*' ! -name 'PanTS_00009000' -print0 \
  | xargs -0 -r mv -t LabelTe/
rm -rf LabelAll

test "$(find ImageTe -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq "$EXPECTED_TEST"
test "$(find LabelTe -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq "$EXPECTED_TEST"

cd "$FORK"
python3 data_conversion/convert_pants_test_only.py \
  --pants-root "$LOCAL_ROOT/PanTS/data" \
  --images-ts-dir "$LOCAL_RAW/imagesTs" \
  --test-answer-key-dir "$LOCAL_GT" \
  --expected-cases "$EXPECTED_TEST"
python3 data_conversion/fix_affine_orthonormality.py "$LOCAL_RAW/imagesTs"
python3 data_conversion/fix_affine_orthonormality.py "$LOCAL_GT"
rm -rf "$LOCAL_ROOT/PanTS"

test "$(find "$LOCAL_RAW/imagesTs" -maxdepth 1 -type f -name '*_0000.nii.gz' | wc -l)" -eq "$EXPECTED_TEST"
test "$(find "$LOCAL_GT" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)" -eq "$EXPECTED_TEST"

LAST_PID=""
launch_prediction() {
    tag=$1
    trainer=${TRAINERS[$tag]}
    out=$LOCAL_OUTPUT/$tag
    mkdir -p "$out"
    srun --exclusive --exact --gpus-per-task=1 --cpus-per-task=12 -n1 --ntasks=1 \
      python3 -u evaluation/predict_and_shrink.py \
        --images-dir "$LOCAL_RAW/imagesTs" --output-dir "$out" \
        --dataset 1 --config 3d_fullres -tr "$trainer" -p "$PLANS" -f 0 \
        --max-probs-csv "$out/max_tumor_probs.csv" \
        --batch-size 20 --expected-cases "$EXPECTED_TEST" \
        --tumor-class 28 --shrink-segmentations-to-tumor \
        > "$EVAL_RUN/predict_${tag}.log" 2>&1 &
    LAST_PID=$!
}

score_and_persist() {
    tag=$1
    out=$LOCAL_OUTPUT/$tag
    python3 evaluation/compute_tumor_metrics.py \
      --pred-dir "$out" --labels-dir "$LOCAL_GT" --tumor-class 28 \
      --probs-csv "$out/max_tumor_probs.csv" \
      --out-json "$out/metrics.json" --per-case-csv "$out/per_case.csv" \
      --expected-cases "$EXPECTED_TEST" --connectivity 1 \
      > "$EVAL_RUN/metrics_${tag}.log" 2>&1
    destination=$EVAL_RUN/$tag
    test ! -e "$destination"
    mkdir -p "$destination"
    cp "$out"/*.nii.gz "$destination/"
    cp "$out/max_tumor_probs.csv" "$out/metrics.json" "$out/per_case.csv" "$destination/"
    test "$(find "$destination" -maxdepth 1 -name '*.nii.gz' | wc -l)" -eq "$EXPECTED_TEST"
}

run_pair() {
    first=$1
    second=$2
    launch_prediction "$first"; pid_first=$LAST_PID
    launch_prediction "$second"; pid_second=$LAST_PID
    fail=0
    wait "$pid_first" || { echo "ERROR: prediction failed for $first" >&2; fail=1; }
    wait "$pid_second" || { echo "ERROR: prediction failed for $second" >&2; fail=1; }
    test "$fail" -eq 0
    score_and_persist "$first"
    score_and_persist "$second"
}

run_pair unetpp_ds unetpp_nods
run_pair default_ds default_nods

python3 - "$EVAL_RUN" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
tags = ("unetpp_ds", "unetpp_nods", "default_ds", "default_nods")
summary = {tag: json.loads((root / tag / "metrics.json").read_text()) for tag in tags}
(root / "grid_metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "GRID_EVALUATION_ALL_DONE"

#!/bin/bash
# Prepared only: submit manually after paper_config.sbatch has a verified completion marker.
# Uses the same test-set conversion, prediction, and class-28 metric protocol as the 2x2 grid.
# See evaluation/DELTA_EVALUATION_PROTOCOL.md before use.
#SBATCH --account=bdyo-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=pants_paper_eval
#SBATCH --output=/work/hdd/bdyo/asanjeev/nnUNet_results/paper_evaluate_%j.log

set -euo pipefail

FORK=/projects/bdyo/asanjeev/delta_work/pants-unetpp-fork
RESULTS_ROOT=/work/hdd/bdyo/asanjeev/nnUNet_results
TRAINER=nnUNetTrainerUNetPlusPlusPaperSparseValidation
PLANS=nnUNetPlansBS4
FOLD=$RESULTS_ROOT/Dataset001_PanTS/${TRAINER}__${PLANS}__3d_fullres/fold_0
COMPLETION_MARKER=$FOLD/paper_training_and_validation_complete.json
EVAL_ROOT=$RESULTS_ROOT/evaluation_paper
EVAL_RUN=$EVAL_ROOT/run_$SLURM_JOB_ID
LOCAL_ROOT=/tmp/$USER/pants_paper_eval_$SLURM_JOB_ID
LOCAL_RAW=$LOCAL_ROOT/nnUNet_raw/Dataset001_PanTS
LOCAL_GT=$LOCAL_ROOT/PanTS_test_answer_key
LOCAL_OUTPUT=$LOCAL_ROOT/predictions
EXPECTED_TEST=901
EXPECTED_VALIDATION=1800

mkdir -p "$EVAL_RUN" "$LOCAL_ROOT"
cleanup() {
    case "$LOCAL_ROOT" in
        /tmp/"$USER"/pants_paper_eval_"$SLURM_JOB_ID") rm -rf "$LOCAL_ROOT" ;;
        *) echo "ERROR: refusing unsafe cleanup path: $LOCAL_ROOT" >&2; exit 1 ;;
    esac
}
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

CURRENT_COMMIT=$(git -C "$FORK" rev-parse HEAD)
echo "commit $CURRENT_COMMIT"
if [ -n "$(git -C "$FORK" status --porcelain)" ]; then
    echo "ERROR: evaluation checkout is dirty" >&2
    exit 1
fi
nvidia-smi -L

echo "=== readiness: verified Paper completion, checkpoint, and validation ==="
test -s "$COMPLETION_MARKER"
test -s "$FOLD/checkpoint_final.pth"
MARKER_COMMIT=$(python3 - "$COMPLETION_MARKER" "$FOLD/checkpoint_final.pth" "$TRAINER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
expected_trainer = sys.argv[3]
marker = json.loads(marker_path.read_text())
assert marker["trainer"] == expected_trainer, marker
assert marker["epochs"] == 1000, marker
assert marker["physical_batch_size"] == 4, marker
assert marker["validation_cases"] == 1800, marker
assert marker["reference_validation_ids_equal"] is True, marker
h = hashlib.sha256()
with checkpoint_path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
assert marker["checkpoint_sha256"] == h.hexdigest(), "checkpoint hash differs from completion marker"
print("completion marker and checkpoint provenance verified")
print(marker["commit"])
PY
)
MARKER_COMMIT=$(printf '%s\n' "$MARKER_COMMIT" | tail -1)
test "$CURRENT_COMMIT" = "$MARKER_COMMIT" || {
    echo "ERROR: checkout $CURRENT_COMMIT differs from trained commit $MARKER_COMMIT" >&2
    exit 1
}

echo "=== install and hash-check the exact trained inference sources ==="
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
python3 "$FORK/unetpp_port/paper_inference_contract_test.py"

n_val=$(find "$FOLD/validation" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
if [ "$n_val" -ne "$EXPECTED_VALIDATION" ]; then
    echo "ERROR: Paper validation has $n_val/$EXPECTED_VALIDATION predictions" >&2
    exit 1
fi
python3 - "$FOLD/validation" "$EXPECTED_VALIDATION" <<'PY'
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

echo "=== stage all 901 held-out test cases and answer keys locally ==="
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

echo "=== predict and score the Paper configuration ==="
mkdir -p "$LOCAL_OUTPUT"
srun --exclusive --exact --gpus-per-task=1 --cpus-per-task=12 -n1 --ntasks=1 \
  python3 -u evaluation/predict_and_shrink.py \
    --images-dir "$LOCAL_RAW/imagesTs" --output-dir "$LOCAL_OUTPUT" \
    --dataset 1 --config 3d_fullres -tr "$TRAINER" -p "$PLANS" -f 0 \
    --max-probs-csv "$LOCAL_OUTPUT/max_tumor_probs.csv" \
    --batch-size 20 --expected-cases "$EXPECTED_TEST" \
    --tumor-class 28 --shrink-segmentations-to-tumor \
    > "$EVAL_RUN/predict_paper.log" 2>&1

python3 evaluation/compute_tumor_metrics.py \
  --pred-dir "$LOCAL_OUTPUT" --labels-dir "$LOCAL_GT" --tumor-class 28 \
  --probs-csv "$LOCAL_OUTPUT/max_tumor_probs.csv" \
  --out-json "$LOCAL_OUTPUT/metrics.json" --per-case-csv "$LOCAL_OUTPUT/per_case.csv" \
  --expected-cases "$EXPECTED_TEST" --connectivity 1 \
  > "$EVAL_RUN/metrics_paper.log" 2>&1

DESTINATION=$EVAL_RUN/paper
test ! -e "$DESTINATION"
mkdir -p "$DESTINATION"
cp "$LOCAL_OUTPUT"/*.nii.gz "$DESTINATION/"
cp "$LOCAL_OUTPUT/max_tumor_probs.csv" "$LOCAL_OUTPUT/metrics.json" \
   "$LOCAL_OUTPUT/per_case.csv" "$DESTINATION/"
test "$(find "$DESTINATION" -maxdepth 1 -name '*.nii.gz' | wc -l)" -eq "$EXPECTED_TEST"
python3 - "$DESTINATION/metrics.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1]))
print(json.dumps(metrics, indent=2))
PY

echo "PAPER_EVALUATION_ALL_DONE"

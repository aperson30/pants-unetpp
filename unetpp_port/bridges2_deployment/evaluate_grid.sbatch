#!/bin/bash
# Prepared evaluation of the 2x2 grid. Submit through submit_evaluation.sh only after training.
# Test predictions and scores are persistent per case so a later job can resume safely.
#SBATCH --account=cis260296p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --job-name=pants_grid_eval_h100
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/evaluate_%j.log

set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
VENV=$ROOT/venv
RESULTS=$ROOT/results
EXPECTED_VALIDATION=1800
EXPECTED_TEST=901
PLANS=nnUNetPlansBS4
: "${TRAIN_COMMIT:?submit through submit_evaluation.sh}"
: "${EVAL_COMMIT:?submit through submit_evaluation.sh}"
[[ "$TRAIN_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid TRAIN_COMMIT" >&2; exit 1; }
[[ "$EVAL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid EVAL_COMMIT" >&2; exit 1; }
FROZEN=$ROOT/frozen/$TRAIN_COMMIT
test "$(cat "$FROZEN/.frozen_commit")" = "$TRAIN_COMMIT"
REPO=$ROOT/eval_frozen/$EVAL_COMMIT
test "$(cat "$REPO/.frozen_commit")" = "$EVAL_COMMIT"
EVAL_RUN=$RESULTS/evaluation_grid/$TRAIN_COMMIT
mkdir -p "$EVAL_RUN"
exec 9>"$EVAL_RUN/.evaluation.lock"
flock -n 9 || { echo "another evaluation job is active for $TRAIN_COMMIT" >&2; exit 1; }

echo "train_commit=$TRAIN_COMMIT evaluation_commit=$EVAL_COMMIT job=$SLURM_JOB_ID"
date -u
hostname
nvidia-smi -L
export PATH=$VENV/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export nnUNet_results=$RESULTS
export nnUNet_def_n_proc=4
export nnUNet_compile=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
"$VENV/bin/python" - <<'PY'
import importlib.metadata
import torch
assert torch.__version__ == '2.10.0+cu126', torch.__version__
assert torch.backends.cudnn.version() == 91002
assert importlib.metadata.version('nnunetv2') == '2.8.1'
print('evaluation torch', torch.__version__, 'cuda', torch.version.cuda,
      'cudnn', torch.backends.cudnn.version())
PY

# A restart must not mix prediction/scoring code versions within one persistent evaluation directory.
"$VENV/bin/python" - "$EVAL_RUN" "$REPO" "$TRAIN_COMMIT" "$EVAL_COMMIT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, repo = Path(sys.argv[1]), Path(sys.argv[2])
train_commit, eval_commit = sys.argv[3:5]
paths = (
    'unetpp_port/bridges2_deployment/evaluate_grid.sbatch',
    'evaluation/predict_and_shrink.py',
    'evaluation/compute_tumor_metrics.py',
    'data_conversion/convert_pants_test_only.py',
    'data_conversion/convert_pants_to_nnunet.py',
    'data_conversion/fix_affine_orthonormality.py',
    'evaluation/verify_grid_ready.py',
    'evaluation/score_grid_cpu.py',
    'evaluation/safe_artifacts.py',
    'evaluation/audit_test_artifacts.py',
    'unetpp_port/bridges2_deployment/progress_20260929/checkpoint_manifest.json',
)
hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths}
manifest = root / 'evaluation_manifest.json'
if manifest.exists():
    previous = json.loads(manifest.read_text())
    assert previous['training_commit'] == train_commit, 'training commit changed'
    assert previous['source_hashes'] == hashes, 'evaluation source changed during resume'
else:
    value = {'training_commit': train_commit, 'first_evaluation_commit': eval_commit,
             'source_hashes': hashes, 'protocol': 'project tumor metrics, 901 PanTS test cases'}
    temporary = manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(manifest)
print('evaluation source manifest verified')
PY

# nnU-Net discovers trainers through its installed package. Install only the source snapshot used
# for training; evaluation utilities come from the separately frozen evaluation commit.
TRAINER_DIR=$("$VENV/bin/python" - <<'PY'
from pathlib import Path
import nnunetv2.training.nnUNetTrainer as package
print(Path(package.__path__[0]))
PY
)
TRAINER_FILES=(
  unet_plusplus.py nnUNetTrainerBF16Mixin.py
  nnUNetTrainerQualityNeutralOptimizationMixin.py nnUNetTrainerFullLossCompileMixin.py
  nnUNetTrainerSparseValidationMixin.py nnUNetTrainerBF16.py
  nnUNetTrainerBF16NoDeepSupervision.py nnUNetTrainerBF16SparseValidation.py
  nnUNetTrainerBF16NoDeepSupervisionSparseValidation.py nnUNetTrainerUNetPlusPlus.py
  nnUNetTrainerUNetPlusPlusNoDeepSupervision.py
  nnUNetTrainerUNetPlusPlusSparseValidation.py
  nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation.py
)
for file in "${TRAINER_FILES[@]}"; do
    install -m 0644 "$FROZEN/unetpp_port/$file" "$TRAINER_DIR/$file"
    cmp -s "$FROZEN/unetpp_port/$file" "$TRAINER_DIR/$file"
done

declare -A TRAINERS=(
  [unetpp_ds]=nnUNetTrainerUNetPlusPlusSparseValidation
  [unetpp_nods]=nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation
  [default_ds]=nnUNetTrainerBF16SparseValidation
  [default_nods]=nnUNetTrainerBF16NoDeepSupervisionSparseValidation
)
for tag in unetpp_ds unetpp_nods default_ds default_nods; do
    trainer=${TRAINERS[$tag]}
    fold=$RESULTS/Dataset001_PanTS/${trainer}__${PLANS}__3d_fullres/fold_0
    test -s "$fold/checkpoint_final.pth"
    test -s "$fold/validation/summary.json"
    test "$(find "$fold/validation" -maxdepth 1 -type f -name 'PanTS_*.nii.gz' | wc -l)" -eq "$EXPECTED_VALIDATION"
    "$VENV/bin/python" - "$fold/validation" "$EXPECTED_VALIDATION" <<'PY'
import json, sys
from pathlib import Path
import nibabel as nib
folder, expected = Path(sys.argv[1]), int(sys.argv[2])
json.loads((folder / 'summary.json').read_text())
files = sorted(folder.glob('PanTS_*.nii.gz'))
assert len(files) == expected
for path in files:
    image = nib.load(str(path))
    assert len(image.shape) == 3 and all(n > 0 for n in image.shape), path
print('readable validation outputs', folder, len(files))
PY
done
echo FOUR_COMPLETE_VALIDATIONS_VERIFIED
"$VENV/bin/python" "$REPO/evaluation/verify_grid_ready.py" --root "$ROOT" \
  --manifest "$REPO/unetpp_port/bridges2_deployment/progress_20260929/checkpoint_manifest.json" \
  > "$EVAL_RUN/checkpoint_validation_provenance.json"

# Match the already measured paired-training placement, not two exclusive steps.
gpu_map=$(srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --gpus-per-task=1 \
  --gpu-bind=single:1 --cpus-per-task=12 --mem=200G bash -c '
    set -euo pipefail
    mapfile -t ids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [ "${#ids[@]}" -eq 1 ]
    printf "%s %s\n" "$SLURM_PROCID" "${ids[0]}"
  ')
printf '%s\n' "$gpu_map"
test "$(printf '%s\n' "$gpu_map" | awk '{print $1}' | sort)" = "$(printf '0\n1')"
test "$(printf '%s\n' "$gpu_map" | awk '{print $2}' | sort -u | wc -l)" -eq 2

# PSC's $LOCAL is available only during this allocation. No 9,000-case restaging is needed.
: "${LOCAL:?PSC node-local storage is required}"
LOCAL_ROOT=$LOCAL/pants_eval_$SLURM_JOB_ID
test "$(df -Pk "$LOCAL" | awk 'NR==2 {print $4}')" -ge 209715200 || {
    echo 'less than 200 GiB local staging headroom' >&2; exit 1;
}
case "$LOCAL_ROOT" in "$LOCAL"/pants_eval_*) ;; *) exit 1 ;; esac
DATA=$LOCAL_ROOT/PanTS/data
LOCAL_RAW=$LOCAL_ROOT/nnUNet_raw/Dataset001_PanTS
LOCAL_GT=$LOCAL_ROOT/PanTS_test_answer_key
mkdir -p "$DATA/ImageTe" "$DATA/LabelTe" "$LOCAL_RAW" "$LOCAL_GT"
export TMPDIR=$LOCAL_ROOT/prediction_tmp
mkdir -p "$TMPDIR"
export nnUNet_raw=$LOCAL_ROOT/nnUNet_raw
export nnUNet_preprocessed=$LOCAL_ROOT/nnUNet_preprocessed_unused

echo STAGING_901_CASE_TEST_SPLIT
cd "$DATA"
test_archive=PanTSMini_ImageTe_00009001_00009901.tar.gz
wget --tries=5 --retry-connrefused -O "$test_archive" \
  "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/3b1cd61108116b58ea5c1ddb3512c1847d965f96/$test_archive?download=true"
sha256sum "$test_archive" > "$EVAL_RUN/test_image_archive.sha256"
tar -xzf "$test_archive" -C ImageTe
rm "$test_archive"
curl --fail --location --retry 5 --show-error \
  'https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz' \
  -o PanTSMini_Label.tar.gz
mkdir LabelAll
sha256sum PanTSMini_Label.tar.gz > "$EVAL_RUN/test_label_archive.sha256"
tar -xzf PanTSMini_Label.tar.gz -C LabelAll
rm PanTSMini_Label.tar.gz
find LabelAll -maxdepth 1 -type d -name 'PanTS_00009*' ! -name 'PanTS_00009000' -print0 \
  | xargs -0 -r mv -t LabelTe/
rm -rf LabelAll
"$VENV/bin/python" - "$DATA" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
expected = {f'PanTS_{i:08d}' for i in range(9001, 9902)}
for name in ('ImageTe', 'LabelTe'):
    actual = {p.name for p in (root / name).iterdir() if p.is_dir()}
    assert actual == expected, (name, len(actual), sorted(expected - actual)[:5], sorted(actual - expected)[:5])
print('exact 901 test image and label IDs verified')
PY
cd "$REPO"
"$VENV/bin/python" data_conversion/convert_pants_test_only.py \
  --pants-root "$DATA" --images-ts-dir "$LOCAL_RAW/imagesTs" \
  --test-answer-key-dir "$LOCAL_GT" --expected-cases "$EXPECTED_TEST"
"$VENV/bin/python" data_conversion/fix_affine_orthonormality.py "$LOCAL_RAW/imagesTs"
"$VENV/bin/python" data_conversion/fix_affine_orthonormality.py "$LOCAL_GT"
test "$(find "$LOCAL_RAW/imagesTs" -maxdepth 1 -type f -name '*_0000.nii.gz' | wc -l)" -eq "$EXPECTED_TEST"
test "$(find "$LOCAL_GT" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)" -eq "$EXPECTED_TEST"
echo TEST_CONVERSION_COMPLETE

# Retain the compact answer key for CPU-only scoring after GPUs are released.
mkdir -p "$EVAL_RUN/test_ground_truth"
"$VENV/bin/python" -m evaluation.safe_artifacts \
  --source "$LOCAL_GT" --destination "$EVAL_RUN/test_ground_truth"
"$VENV/bin/python" - "$LOCAL_GT" "$EVAL_RUN/test_ground_truth" <<'PY'
import hashlib, sys
from pathlib import Path
source, saved = map(Path, sys.argv[1:])
for path in source.glob('*.nii.gz'):
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256((saved / path.name).read_bytes()).digest(), path
PY

run_pair() {
    export FIRST_TAG=$1 SECOND_TAG=$2 EVAL_RUN LOCAL_RAW VENV PLANS
    export FIRST_TRAINER=${TRAINERS[$1]} SECOND_TRAINER=${TRAINERS[$2]}
    srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --gpus-per-task=1 \
      --gpu-bind=single:1 --cpus-per-task=12 --mem=200G --kill-on-bad-exit=1 \
      --export=ALL bash -c '
        set -euo pipefail
        if [ "$SLURM_PROCID" -eq 0 ]; then
          tag=$FIRST_TAG; trainer=$FIRST_TRAINER
        else
          tag=$SECOND_TAG; trainer=$SECOND_TRAINER
        fi
        out=$EVAL_RUN/$tag
        mkdir -p "$out"
        echo "job=$SLURM_JOB_ID rank=$SLURM_PROCID tag=$tag visible=$CUDA_VISIBLE_DEVICES"
        "$VENV/bin/python" -u evaluation/predict_and_shrink.py \
          --images-dir "$LOCAL_RAW/imagesTs" --output-dir "$out" \
          --dataset 1 --config 3d_fullres -tr "$trainer" -p "$PLANS" -f 0 \
          --max-probs-csv "$out/max_tumor_probs.csv" --batch-size 5 \
          --preprocess-workers 1 --export-workers 1 --batch-timeout 7200 \
          --expected-cases 901 --tumor-class 28 --shrink-segmentations-to-tumor \
          >> "$EVAL_RUN/predict_${tag}.log" 2>&1
      '
}
run_pair unetpp_ds unetpp_nods
run_pair default_ds default_nods

echo GRID_TEST_PREDICTIONS_DONE_CPU_SCORING_REQUIRED

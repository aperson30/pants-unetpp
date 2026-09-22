#!/bin/bash
# CPU-only preparation: no GPU resources are requested or charged.
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
# RM-shared enforces at most 2,000 MB per requested CPU (not 2 GiB).
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=02:00:00
#SBATCH --job-name=pants_h100_calib_prep
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/logs/calib_prep_%j.log

set -euo pipefail
PROJECT_ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
REPO=$PROJECT_ROOT/repo
VENV=$PROJECT_ROOT/venv
CALIB=$PROJECT_ROOT/calibration
RAW=$CALIB/nnUNet_raw
PREPROCESSED=$CALIB/nnUNet_preprocessed
SOURCE=$CALIB/source
ANSWER_KEY=$CALIB/empty_test_answer_key

mkdir -p "$PROJECT_ROOT/logs" "$RAW" "$PREPROCESSED" "$ANSWER_KEY"
echo "=== CPU calibration preparation ==="
hostname
date -u
git -C "$REPO" rev-parse HEAD

if [ ! -f "$CALIB/metadata.xlsx" ]; then
    wget -O "$CALIB/metadata.xlsx" \
      "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/metadata.xlsx?download=true"
fi

if [ ! -f "$SOURCE/selected_cases.txt" ]; then
    "$VENV/bin/python" "$REPO/unetpp_port/bridges2_deployment/prepare_calibration.py" \
      --root "$SOURCE" --metadata "$CALIB/metadata.xlsx" --prefix-cases 200 \
      --selected-cases 9 --tumor-cases 4
fi

if [ ! -f "$RAW/Dataset001_PanTSCalib/dataset.json" ]; then
    "$VENV/bin/python" "$REPO/data_conversion/convert_pants_to_nnunet.py" \
      --pants-root "$SOURCE" \
      --nnunet-dataset-dir "$RAW/Dataset001_PanTSCalib" \
      --test-answer-key-dir "$ANSWER_KEY"
    "$VENV/bin/python" "$REPO/data_conversion/fix_affine_orthonormality.py" \
      "$RAW/Dataset001_PanTSCalib"
fi

export nnUNet_raw=$RAW
export nnUNet_preprocessed=$PREPROCESSED
export nnUNet_results=$CALIB/nnUNet_results_scratch
export nnUNet_n_proc_DA=8

if [ ! -f "$PREPROCESSED/Dataset001_PanTSCalib/nnUNetPlans.json" ]; then
    "$VENV/bin/nnUNetv2_plan_and_preprocess" -d 1 -c 3d_fullres \
      --verify_dataset_integrity -npfp 6 -np 6
fi

"$VENV/bin/python" - <<'PY'
import json
from pathlib import Path

root = Path('/ocean/projects/cis260296p/asanjeev/pants_unetpp/calibration')
dataset = json.loads((root / 'nnUNet_raw/Dataset001_PanTSCalib/dataset.json').read_text())
assert dataset['numTraining'] == 9, dataset['numTraining']
assert dataset['labels']['pancreatic_lesion'] == 28
source = root / 'nnUNet_preprocessed/Dataset001_PanTSCalib/nnUNetPlans.json'
target = source.with_name('nnUNetPlansBS4.json')
plans = json.loads(source.read_text())
plans['plans_name'] = 'nnUNetPlansBS4'
config = plans['configurations']['3d_fullres']
config['batch_size'] = 4
# Match the real full-resolution geometry used for the existing DeltaAI/Delta calibrations. The
# nine-case planner is too small to estimate a representative patch but preprocessing itself is
# still valid; nnU-Net pads sampled cases when needed.
config['patch_size'] = [64, 160, 224]
target.write_text(json.dumps(plans, indent=2))
print('calibration cases=9 batch=4 patch=', config['patch_size'])
PY

test -f "$PREPROCESSED/Dataset001_PanTSCalib/nnUNetPlansBS4.json"
echo BRIDGES2_CALIBRATION_PREP_DONE

#!/bin/bash
# =============================================================================
# Capture the small, durable artefacts of a training campaign into this
# repository, so results survive independently of the server.
#
#   bash ~/pants-unetpp/scripts/capture_server_state.sh
#
# Writes everything into <repo>/docs/results/. Text only, a few hundred KB --
# safe to commit to git. Deliberately does NOT copy checkpoints (~366 MB each);
# back those up separately, see HANDOFF.md.
#
# Safe to re-run: it overwrites its own output and touches nothing else.
# =============================================================================

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/docs/results"
RESULTS="${nnUNet_results:-/Scratch/enl014/nnUNet_results}/Dataset001_PanTS"
PREP="${nnUNet_preprocessed:-/Scratch/enl014/nnUNet_preprocessed}/Dataset001_PanTS"
RAW="${nnUNet_raw:-/Scratch/enl014/nnUNet_raw}/Dataset001_PanTS"
WORK="$HOME/pants_phase7_scoring"
PLANS=nnUNetPlansBS4

TRAINERS="nnUNetTrainer nnUNetTrainerNoDeepSupervision nnUNetTrainerUNetPlusPlus nnUNetTrainerUNetPlusPlusNoDeepSupervision"

mkdir -p "$OUT/metrics" "$OUT/curves" "$OUT/config"

echo "Capturing into $OUT"
echo

# ---- 1. computed metrics -----------------------------------------------------
echo "[1/5] metrics JSONs"
cp "$WORK"/metrics_*.json "$OUT/metrics/" 2>/dev/null && echo "  copied $(ls "$OUT/metrics" | wc -l) file(s)" || echo "  none found in $WORK"

# ---- 2. per-epoch tumour Dice curves ----------------------------------------
echo "[2/5] tumour Dice curves"
for t in $TRAINERS; do
  python3 - "$RESULTS/${t}__${PLANS}__3d_fullres/fold_0" "$OUT/curves/${t}.csv" <<'PY' 2>/dev/null
import glob, re, sys, os
folder, out = sys.argv[1], sys.argv[2]
vals = []
for fn in sorted(glob.glob(os.path.join(folder, "training_log_*.txt"))):
    for line in open(fn):
        if "Pseudo dice" in line:
            nums = re.findall(r"np\.float32\(([0-9.eE+-]+)\)", line)
            if nums:
                vals.append(float(nums[-1]))
if vals:
    with open(out, "w") as f:
        f.write("epoch,tumor_pseudo_dice\n")
        for i, v in enumerate(vals):
            f.write(f"{i},{v}\n")
    print(f"  {os.path.basename(folder)}: {len(vals)} epochs")
PY
done

# ---- 3. configuration --------------------------------------------------------
echo "[3/5] plans and dataset definition"
cp "$PREP/${PLANS}.json"    "$OUT/config/" 2>/dev/null
cp "$PREP/nnUNetPlans.json" "$OUT/config/" 2>/dev/null
cp "$PREP/splits_final.json" "$OUT/config/" 2>/dev/null
cp "$RAW/dataset.json"      "$OUT/config/" 2>/dev/null
ls "$OUT/config" 2>/dev/null | sed 's/^/  /'

# ---- 4. inventory ------------------------------------------------------------
echo "[4/5] inventory"
{
  echo "# Server state, captured $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo
  echo "## Data"
  echo "- imagesTr: $(ls "$RAW/imagesTr"/*.nii.gz 2>/dev/null | wc -l) / 9000"
  echo "- imagesTs: $(ls "$RAW/imagesTs"/*.nii.gz 2>/dev/null | wc -l) / 901"
  echo "- test answer key: $(ls /Scratch/enl014/PanTS_test_answer_key/*.nii.gz 2>/dev/null | wc -l) / 901"
  echo
  echo "## Trained models"
  for t in $TRAINERS; do
    d="$RESULTS/${t}__${PLANS}__3d_fullres/fold_0"
    ck="absent"; [ -f "$d/checkpoint_final.pth" ] && ck="present"
    ep=$(grep -ho "Epoch [0-9][0-9]*" "$d"/training_log_*.txt 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
    val=$(ls "$d/validation"/*.nii.gz 2>/dev/null | wc -l)
    echo "- $t: checkpoint_final=$ck, max epoch=${ep:-n/a}, cross-val predictions=$val/1800"
  done
  echo
  echo "## Test-set predictions (target 901)"
  for r in default_ds default_nods unetpp_ds unetpp_nods; do
    echo "- $r: $(ls /Scratch/enl014/PanTS_te_predictions_${r}_bs4/*.nii.gz 2>/dev/null | wc -l)"
  done
  echo
  echo "## Environment"
  python3 -c "import torch, torchvision; print('- torch', torch.__version__, '/ torchvision', torchvision.__version__)" 2>/dev/null
  echo "- GPUs:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /'
  echo
  echo "## Disk"
  df -h /Scratch 2>/dev/null | tail -1 | sed 's/^/- /'
} > "$OUT/SERVER_STATE.md"
echo "  wrote SERVER_STATE.md"

# ---- 5. probability CSVs -----------------------------------------------------
echo "[5/5] max tumour probability CSVs"
cp "$WORK"/max_tumor_probs_te_*.csv "$OUT/metrics/" 2>/dev/null && echo "  copied" || echo "  none found"

echo
echo "Done. Review $OUT/SERVER_STATE.md, then copy docs/results/ back to your laptop and commit."

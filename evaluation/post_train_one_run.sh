#!/bin/bash
# =============================================================================
# Full post-training pipeline for ONE trained model, on ONE GPU.
#
#   usage:  post_train_one_run.sh <trainer_class_name> <tag> <gpu_id>
#   e.g.:   post_train_one_run.sh nnUNetTrainerUNetPlusPlus unetpp_ds 4
#
# Normally launched by post_train_scheduler.sh, not by hand.
#
# THREE STAGES, in order:
#   1. Cross-validation validation (`--val`) over fold 0's 1,800 held-out cases.
#      Deliberately SINGLE-GPU -- no `-num_gpus`. Under DDP this step has
#      repeatedly died with NCCL collective timeouts, because validation cases
#      take wildly different times and the ranks drift apart past the watchdog
#      limit. One GPU has no ranks to synchronise, so the problem cannot occur.
#
#   2. Official PanTS-te prediction over the 901-case test set, via
#      predict_and_shrink.py. That script predicts in small batches and
#      immediately reduces each case's huge probability file to the single
#      number needed for AUC (max tumour-channel probability) before deleting
#      it, so disk usage stays bounded -- this is what avoids the disk-full
#      crashes we hit when probability files for hundreds of cases piled up.
#
#   3. The five benchmark metrics (DSC, P-Sen, T-Sen, Spe, AUC), computed twice:
#      once on PanTS-te (the reportable numbers) and once on the cross-val fold
#      (useful as a sanity check / larger sample).
#
# IMPORTANT -- GROUND TRUTH LOCATION:
#   `labelsTs` does NOT exist. nnU-Net's convention leaves imagesTs unlabelled,
#   and the conversion script put the official test answer key in a separate
#   directory. PanTS-te must be scored against /Scratch/enl014/PanTS_test_answer_key.
# =============================================================================

set -u

TRAINER="$1"
TAG="$2"
GPU="$3"

RAW=/Scratch/enl014/nnUNet_raw/Dataset001_PanTS
RESULTS=/Scratch/enl014/nnUNet_results/Dataset001_PanTS
TE_GT=/Scratch/enl014/PanTS_test_answer_key
PLANS=nnUNetPlansBS4

cd ~/pants_phase7_scoring || exit 1

export CUDA_VISIBLE_DEVICES="$GPU"
export nnUNet_n_proc_DA=6
export nnUNet_def_n_proc=4

echo "==================================================================="
echo "$(date): post-training pipeline for $TAG ($TRAINER) on GPU $GPU"
echo "==================================================================="

# ---- stage 1: cross-val validation, single GPU ------------------------------
echo "$(date): [1/3] cross-val validation (single-GPU, no DDP)"
nnUNetv2_train 1 3d_fullres 0 -tr "$TRAINER" -p "$PLANS" --val \
    > "val_${TAG}_bs4.log" 2>&1
echo "$(date): [1/3] finished (exit $?) -- see val_${TAG}_bs4.log"

# ---- stage 2: official PanTS-te prediction ----------------------------------
TE_OUT="/Scratch/enl014/PanTS_te_predictions_${TAG}_bs4"
TE_PROBS="max_tumor_probs_te_${TAG}_bs4.csv"

echo "$(date): [2/3] predicting the 901 official PanTS-te cases"
python3 -u predict_and_shrink.py \
    --images-dir "$RAW/imagesTs" \
    --output-dir "$TE_OUT" \
    --dataset 1 --config 3d_fullres \
    -tr "$TRAINER" -p "$PLANS" -f 0 \
    --max-probs-csv "$TE_PROBS" \
    --batch-size 20 \
    > "predict_te_${TAG}_bs4.log" 2>&1
echo "$(date): [2/3] finished (exit $?) -- see predict_te_${TAG}_bs4.log"

# ---- stage 3: the five benchmark metrics ------------------------------------
echo "$(date): [3/3] computing metrics"

# 3a. the reportable numbers: official PanTS-te
python3 compute_tumor_metrics.py \
    --pred-dir "$TE_OUT" \
    --labels-dir "$TE_GT" \
    --tumor-class 28 \
    --probs-csv "$TE_PROBS" \
    --out-json "metrics_PanTSte_${TAG}_bs4.json" \
    > "metrics_PanTSte_${TAG}_bs4.log" 2>&1

# 3b. sanity check on the cross-val fold (no probs CSV -> AUC reported as null)
python3 compute_tumor_metrics.py \
    --pred-dir "$RESULTS/${TRAINER}__${PLANS}__3d_fullres/fold_0/validation" \
    --labels-dir "$RAW/labelsTr" \
    --tumor-class 28 \
    --out-json "metrics_crossval_${TAG}_bs4.json" \
    > "metrics_crossval_${TAG}_bs4.log" 2>&1

echo "$(date): [3/3] done."
echo "$(date): ALL STAGES COMPLETE for $TAG. Reportable metrics:"
cat "metrics_PanTSte_${TAG}_bs4.json" 2>/dev/null
echo

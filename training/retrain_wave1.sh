#!/bin/bash
# =============================================================================
# PanTS 2x2 RETRAIN -- WAVE 1 (the two UNet++ legs)
#
# WHY THIS RETRAIN EXISTS
# -----------------------
# The previous 4-run grid used plans `nnUNetPlansBS2` (batch_size 2) split
# across 2 GPUs via DDP = 1 patch per GPU. All 4 runs failed to learn the tumor
# class (final tumor pseudo-Dice 0.0-0.0004, vs 0.3347 in the earlier working
# run). Diagnosis: rare-class learning here has a DELAYED ONSET -- the working
# run showed 0% tumor recognition for its first ~300 epochs, then phase-
# transitioned and climbed to 0.38 by epoch 1000. The BS2 runs never crossed
# that threshold within the same 1000-epoch budget.
#
# Ruled out with evidence (see PanTS_UNetPlusPlus_PROJECT_LOG.md): truncated
# training, missing tumors in preprocessed data, label misconfiguration, the
# custom UNet++/loss code (stock nnU-Net failed identically), and the scoring
# script.
#
# THIS CONFIG REPRODUCES THE KNOWN-WORKING RUN EXACTLY:
#   nnUNetTrainerUNetPlusPlus + nnUNetPlansBS4 + 4 GPUs (world size 4).
#
# =============================================================================
# TRAINING CONFIGURATION (all values below are what will actually be used)
# =============================================================================
#   Dataset ................. Dataset001_PanTS (id 1), 28 foreground classes
#                             class 28 = pancreatic_lesion (the tumor)
#   Configuration ........... 3d_fullres
#   Fold .................... 0  (7,200 train / 1,800 val, from splits_final.json)
#   Plans ................... nnUNetPlansBS4
#   Batch size (global) ..... 4        <-- THE FIX. Was 2.
#   GPUs per run ............ 4 (DDP, world size 4) = 1 patch per GPU
#   Patch size .............. [64, 160, 224]
#   Spacing ................. [1.25, 0.79296899, 0.8046875]
#   Normalization ........... CTNormalization
#   batch_dice .............. True   (Dice computed across the whole batch --
#                                     this is why global batch size matters so
#                                     much for a rare class)
#   oversample_foreground ... 0.33 (nnU-Net default; with world size 4 this
#                                   makes rank 3 always-foreground, ranks 0-2
#                                   random -- same as the working run)
#   Epochs .................. 1000, 250 iterations each (nnU-Net default;
#                                   also matches PanTS paper Appendix B.3.1)
#   Optimizer ............... SGD, base LR 0.01, poly schedule (nnU-Net default;
#                                   also matches PanTS paper)
#   torch.compile ........... ON (default) -- the working run used it too
#
# MEMORY (measured on the 24GB RTX A5000s):
#   default architecture ~5.7 GB per patch  |  UNet++ ~17 GB per patch
#   => UNet++ can only hold 1 patch per GPU, which is why global batch 4
#      requires 4 GPUs rather than 2.
#
# ONE PATCH IN nnU-Net's CODE WE RELY ON:
#   nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py line ~253 was patched to
#   add `find_unused_parameters=True` to the DDP constructor. This is REQUIRED
#   for the deep-supervision-OFF trainers: the network always builds all
#   deep-supervision heads, but with DS off only the final head contributes to
#   the loss, so DDP would otherwise abort with an "unused parameters" error.
#   It changes bucketing/overhead only, not the training math.
#
# EXPECTED RUNTIME: ~39-50 h per run (the working run took ~39 h). Both run in
#   parallel here, so wave 1 is one ~2-day block. Wave 2 (the two default-
#   architecture legs) follows on the same 8 GPUs.
#
# WHAT "SUCCESS" LOOKS LIKE, AND WHEN TO CHECK:
#   Do NOT judge this before epoch ~400. The working run showed 0% tumor
#   recognition through epoch 300. Check tumor pseudo-Dice around epoch 500;
#   by then the working run was at ~87% of epochs showing a tumor.
#   Use: python3 check_tumor_progress.py   (written alongside this script)
#
# NOTE ON POST-TRAINING VALIDATION:
#   nnU-Net auto-runs validation after training finishes. Under DDP this has
#   repeatedly crashed with NCCL collective timeouts (the ranks drift apart
#   because validation cases take very different times). That is NOT a problem:
#   training checkpoints are already saved, and validation can be re-run
#   afterwards single-GPU with `--val`, which avoids DDP entirely.
# =============================================================================

set -u

cd ~/pants_phase7_scoring || exit 1

# reduced worker pools -- higher values caused CPU contention crashes when
# several jobs shared this node
export nnUNet_n_proc_DA=6
export nnUNet_def_n_proc=4

echo "$(date): launching wave 1 (UNet++ DS-on and DS-off, BS4, 4 GPUs each)"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup nnUNetv2_train 1 3d_fullres 0 \
  -tr nnUNetTrainerUNetPlusPlus \
  -p nnUNetPlansBS4 \
  -num_gpus 4 \
  > retrain_unetpp_ds.log 2>&1 &
echo "  unetpp_ds  -> GPUs 0-3, log: retrain_unetpp_ds.log, pid $!"

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup nnUNetv2_train 1 3d_fullres 0 \
  -tr nnUNetTrainerUNetPlusPlusNoDeepSupervision \
  -p nnUNetPlansBS4 \
  -num_gpus 4 \
  > retrain_unetpp_nods.log 2>&1 &
echo "  unetpp_nods -> GPUs 4-7, log: retrain_unetpp_nods.log, pid $!"

disown -a
echo "$(date): both launched and detached. Safe to close the SSH session."

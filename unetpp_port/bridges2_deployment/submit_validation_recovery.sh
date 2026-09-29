#!/bin/bash
# Only recover plain U-Net DS-off validation; never restart completed training.
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
REPO=$ROOT/repo
TRAINER=nnUNetTrainerBF16NoDeepSupervisionSparseValidation
FOLD=$ROOT/results/Dataset001_PanTS/${TRAINER}__nnUNetPlansBS4__3d_fullres/fold_0
test -s "$FOLD/checkpoint_final.pth"
test -s "$ROOT/progress_20260929/checkpoint_manifest.json"
test ! -s "$FOLD/validation/summary.json" || {
    echo 'validation summary already exists; review before requesting more GPU time' >&2; exit 1;
}
test -z "$(git -C "$REPO" status --porcelain)" || {
    echo 'dirty cluster checkout; refusing an ambiguous recovery snapshot' >&2; exit 1;
}
COMMIT=$(git -C "$REPO" rev-parse HEAD)
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)_${COMMIT:0:12}
echo "validation-only recovery commit=$COMMIT run=$RUN_ID"
sbatch --gres=gpu:h100-80:1 --cpus-per-task=12 --mem=220G \
  --job-name=pants_plain_nods_val \
  --export="ALL,GRID_RETRY_COUNT=0,GRID_COMMIT=$COMMIT,GRID_RUN_ID=$RUN_ID,GRID_RECOVERY_TRAINER=$TRAINER" \
  "$REPO/unetpp_port/bridges2_deployment/grid_coordinated.sbatch"

#!/bin/bash
# Run after the successful final grid job (possibly a bounded successor) exits.
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
REPO=$ROOT/repo
RESULTS=$ROOT/results
TRAIN_COMMIT=b2ca075b078dec56ee340f0bd89bc0d2dc6f5b05
test "$(cat "$ROOT/frozen/$TRAIN_COMMIT/.frozen_commit")" = "$TRAIN_COMMIT"
test -z "$(git -C "$REPO" status --porcelain)" || { echo 'dirty evaluation checkout' >&2; exit 1; }
EVAL_COMMIT=$(git -C "$REPO" rev-parse HEAD)
for trainer in \
  nnUNetTrainerUNetPlusPlusSparseValidation \
  nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation \
  nnUNetTrainerBF16SparseValidation \
  nnUNetTrainerBF16NoDeepSupervisionSparseValidation; do
    fold=$RESULTS/Dataset001_PanTS/${trainer}__nnUNetPlansBS4__3d_fullres/fold_0
    test -s "$fold/checkpoint_final.pth" || { echo "missing final checkpoint: $trainer" >&2; exit 1; }
    test -s "$fold/validation/summary.json" || { echo "missing validation summary: $trainer" >&2; exit 1; }
    n=$(find "$fold/validation" -maxdepth 1 -type f -name 'PanTS_*.nii.gz' | wc -l)
    [ "$n" -eq 1800 ] || { echo "$trainer has $n/1800 validation outputs" >&2; exit 1; }
done
test ! -s "$RESULTS/evaluation_grid/$TRAIN_COMMIT/grid_metrics.json" || {
    echo 'complete evaluation summary already exists; refusing duplicate submission' >&2; exit 1;
}
mkdir -p "$ROOT/grid_logs" "$ROOT/eval_frozen"
FROZEN=$ROOT/eval_frozen/$EVAL_COMMIT
if [ ! -f "$FROZEN/.frozen_commit" ]; then
    TEMP=$ROOT/eval_frozen/.${EVAL_COMMIT}.$$.tmp
    mkdir "$TEMP"
    git -C "$REPO" archive "$EVAL_COMMIT" | tar -x -C "$TEMP"
    printf '%s\n' "$EVAL_COMMIT" > "$TEMP/.frozen_commit"
    mv "$TEMP" "$FROZEN"
fi
test "$(cat "$FROZEN/.frozen_commit")" = "$EVAL_COMMIT"
sbatch --export="ALL,TRAIN_COMMIT=$TRAIN_COMMIT,EVAL_COMMIT=$EVAL_COMMIT" \
  "$FROZEN/unetpp_port/bridges2_deployment/evaluate_grid.sbatch"

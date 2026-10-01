#!/bin/bash
# Run after the successful final grid job (possibly a bounded successor) exits.
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
REPO=$ROOT/repo
RESULTS=$ROOT/results
TRAIN_COMMIT=61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
test "$(cat "$ROOT/frozen/$TRAIN_COMMIT/.frozen_commit")" = "$TRAIN_COMMIT"
test -z "$(git -C "$REPO" status --porcelain)" || { echo 'dirty evaluation checkout' >&2; exit 1; }
EVAL_COMMIT=$(git -C "$REPO" rev-parse HEAD)
"$ROOT/venv/bin/python" "$REPO/evaluation/verify_grid_ready.py" --root "$ROOT" \
  --manifest "$REPO/unetpp_port/bridges2_deployment/progress_20260929/checkpoint_manifest.json"
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
if test -s "$RESULTS/evaluation_grid/$TRAIN_COMMIT/grid_metrics.json"; then
    cd "$REPO"
    "$ROOT/venv/bin/python" -m evaluation.audit_test_artifacts \
      --evaluation-dir "$RESULTS/evaluation_grid/$TRAIN_COMMIT" --verify-grid-report || {
        echo 'stale/unverified summary: investigate before any submission' >&2; exit 1;
      }
    echo 'verified complete evaluation already exists; refusing duplicate submission' >&2
    exit 1
fi
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

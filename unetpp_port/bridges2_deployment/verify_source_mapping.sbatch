#!/bin/bash
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=01:00:00
#SBATCH --job-name=pants_tumor_source_mapping
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/source_mapping_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${MAPPING_SCRIPT:?explicit pinned script required}"
: "${MAPPING_SHA256:?script SHA256 required}"
test "$(sha256sum "$MAPPING_SCRIPT" | cut -d ' ' -f1)" = "$MAPPING_SHA256"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
exec 9<"$RUN/.evaluation.lock"
flock -s -n 9 || exit 1
export PYTHONPATH=$ROOT/eval_frozen/5e99901263f4a9f2092d6674c7c566ba4d9d43cc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout --kill-after=30s 3500s "$ROOT/venv/bin/python" -u "$MAPPING_SCRIPT" \
  --source-dir "$ROOT/source_geometry_recovery_20261005_47451639" \
  --evaluation-dir "$RUN"

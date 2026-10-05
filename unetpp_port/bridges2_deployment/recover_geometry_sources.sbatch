#!/bin/bash
#SBATCH --account=cis260296p
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=04:00:00
#SBATCH --job-name=pants_source_geometry
#SBATCH --output=/ocean/projects/cis260296p/asanjeev/pants_unetpp/grid_logs/source_geometry_%j.log
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
: "${SOURCE_RECOVERY_SCRIPT:?explicit pinned script required}"
: "${SOURCE_RECOVERY_SHA256:?expected script fingerprint required}"
test "$(sha256sum "$SOURCE_RECOVERY_SCRIPT" | cut -d ' ' -f1)" = "$SOURCE_RECOVERY_SHA256"
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
exec 9<"$RUN/.evaluation.lock"
flock -s -n 9 || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# uv's Python may not locate RHEL's trust store automatically. Keep TLS verification.
test -s /etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
timeout --kill-after=30s 13800s "$ROOT/venv/bin/python" -u "$SOURCE_RECOVERY_SCRIPT" \
  --destination "$ROOT/source_geometry_recovery_20261005_$SLURM_JOB_ID" \
  --evaluation-dir "$RUN"

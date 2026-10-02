#!/bin/bash
# Public bounded CPU transfer only. Unique destinations, no retry/substitution.
set -euo pipefail
CODE=/u/asanjeev/compression_probe_smalloutputs/paired_judge_replication_v1
PYTHON=/projects/bdyo/asanjeev/pants_venv/bin/python
export PYTHONPATH=$CODE/deps:$CODE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE"
for study in 100430_00001 100259_00001; do
  timeout --signal=TERM --kill-after=10 360 "$PYTHON" -u fetch_panorama_case.py \
    --eligibility replication_selection_v1.json --study "$study" \
    --max-member-mb 60 --output "$CODE/case_$study"
done
echo REPLICATION_TRANSFER_COMPLETE

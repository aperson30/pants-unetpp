#!/bin/bash
# Read-only status for the protected versioned evaluation chain. No GPU imports.
set -euo pipefail
ROOT=/ocean/projects/cis260296p/asanjeev/pants_unetpp
RUN=$ROOT/results/evaluation_grid/61be5315e8cc5f4ec35637c9b748e5ebc33da1b7
printf 'Recorded chain: source 47497014 -> reference 47497024 -> reuse 47497134 -> inference 47497144 -> scoring 47497147 -> verification 47497968\n'
squeue -j 47497144,47497147,47497968 -o '%.18i %.12T %.12M %.30R' || true
sacct -j 47497014,47497024,47497134,47497144,47497147,47497968 \
  --starttime=2026-10-07 --format=JobID,State,Elapsed,ExitCode -n
squeue --start -j 47497144 || true
python3 - "$RUN" "$ROOT" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path
run, root = map(Path, sys.argv[1:])
for tag in ('unetpp_ds', 'unetpp_nods', 'default_ds', 'default_nods'):
    masks = {p.name[:-7] for p in (run / tag).glob('*.nii.gz')}
    path = run / tag / 'max_tumor_probs.csv'
    scores = []
    if path.exists():
        with path.open(newline='') as handle:
            scores = [row['case_id'] for row in csv.DictReader(handle)]
    print('{}: masks={}, score rows={}, paired IDs={} (counts only, not a fresh integrity audit)'.format(
        tag, len(masks), len(scores), len(masks.intersection(scores))))
report = root / 'versioned_grid_scores_47497147/grid_metrics.json'
log = root / 'grid_logs/final_report_verify_47497968.log'
if not report.exists():
    print('Final versioned table: not published yet; original grid_metrics.json is NOT the new output path.')
else:
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    marker = 'FINAL_VERSIONED_GRID_INDEPENDENTLY_VERIFIED '
    records = []
    if log.exists():
        for line in log.read_text().splitlines():
            if line.startswith(marker):
                records.append(json.loads(line[len(marker):]))
    matched = any(record.get('report_sha256') == digest for record in records)
    print('Final versioned table: {}'.format(report))
    print('Recorded independent verification matches current report fingerprint: {}'.format(matched))
    print('This status command does not rehash all CT/prediction/reference inputs; final delivery uses the full verifier.')
PY

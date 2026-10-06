"""Separate CPU scoring against a certified, versioned class-28 reference.

Runs the ORIGINAL checksum-bound metric script; never substitutes labels in
the old run or changes metric math. No output is published on a failed gate.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from evaluation.audit_test_artifacts import TAGS
from evaluation.audit_versioned_predictions import audit
from evaluation.recover_geometry_sources import sha


def score(root, reference, source_audit, metric_script, destination, *, expected_cases=None, cell_timeout=7200):
    root = root.resolve(); reference = reference.resolve(); destination = destination.resolve()
    if destination.is_relative_to(root) or destination.is_relative_to(reference):
        raise RuntimeError('score outputs must be separate from original evidence/reference')
    manifest = json.loads((root / 'evaluation_manifest.json').read_text())
    metric_hash = sha(metric_script)
    if metric_hash != manifest['source_hashes']['evaluation/compute_tumor_metrics.py']:
        raise RuntimeError('metric script differs from original frozen implementation')
    if isinstance(cell_timeout, bool) or not isinstance(cell_timeout, int) or cell_timeout <= 0:
        raise ValueError('positive per-cell timeout required')
    before = audit(root, reference, source_audit, expected_cases=expected_cases)
    count = before['cases_per_cell']
    destination.mkdir(parents=True, exist_ok=False)
    results = {}; result_hashes = {}; lesion_count = None
    for tag in TAGS:
        out = destination / tag; out.mkdir()
        with (out / 'scoring.log').open('x') as log:
            subprocess.run([sys.executable, str(metric_script), '--pred-dir', str(root / tag),
                        '--labels-dir', str(reference / 'test_ground_truth'), '--tumor-class', '28',
                        '--probs-csv', str(root / tag / 'max_tumor_probs.csv'),
                        '--out-json', str(out / 'metrics.json'), '--per-case-csv', str(out / 'per_case.csv'),
                        '--expected-cases', str(count), '--connectivity', '1'],
                           check=True, timeout=cell_timeout, stdout=log, stderr=subprocess.STDOUT)
        metrics = json.loads((out / 'metrics.json').read_text())
        if metrics['n_cases_evaluated'] != count or metrics['AUC_n_cases'] != count:
            raise RuntimeError(f'{tag}: incomplete scoring')
        if (metrics['DSC_tumor_n_positive_cases'] != before['gt_positive_cases'] or
                metrics['P_Sen_n_positive_cases'] != before['gt_positive_cases'] or
                metrics['Spe_n_negative_cases'] != before['gt_negative_cases']):
            raise RuntimeError(f'{tag}: scored reference denominators differ')
        if metrics['protocol']['tumor_class'] != 28 or metrics['protocol']['T_Sen_connectivity'] != '6-neighbor':
            raise RuntimeError(f'{tag}: metric protocol differs')
        for name in ('DSC_tumor_mean', 'P_Sen', 'T_Sen', 'Spe', 'AUC'):
            value = metrics[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError(f'{tag}: invalid {name}')
        if lesion_count is None:
            lesion_count = metrics['T_Sen_n_true_tumors']
        if not isinstance(lesion_count, int) or lesion_count <= 0 or metrics['T_Sen_n_true_tumors'] != lesion_count:
            raise RuntimeError(f'{tag}: reference lesion count differs')
        results[tag] = metrics
        for path in (out / 'metrics.json', out / 'per_case.csv', out / 'scoring.log'):
            result_hashes[str(path.relative_to(destination))] = sha(path)
    if sha(metric_script) != metric_hash or audit(root, reference, source_audit, expected_cases=expected_cases) != before:
        raise RuntimeError('scoring inputs changed; no grid summary published')
    for path, digest in result_hashes.items():
        if sha(destination / path) != digest:
            raise RuntimeError('metric outputs changed; no grid summary published')
    report = {'schema': 'pants-versioned-grid-metrics-v1', 'project_protocol': True,
              'scope': 'class 28 only; other organs not certified', 'metric_script_sha256': metric_hash,
              'original_evaluation_manifest': manifest, 'artifact_audit': before,
              'metric_artifact_sha256': result_hashes, 'metrics': results}
    temporary = destination / 'grid_metrics.json.tmp'
    with temporary.open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
    temporary.replace(destination / 'grid_metrics.json')
    print('VERSIONED_GRID_SCORING_ALL_DONE', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--metric-script', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    # Hold the existing shared lock; require it to exist rather than creating
    # or truncating anything in the protected original evaluation directory.
    import fcntl
    with (args.evaluation_dir / '.evaluation.lock').open('rb') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        score(args.evaluation_dir, args.reference_dir, args.source_audit,
              args.metric_script, args.destination)

"""Read-only final-report gate; independently aggregate the frozen scorer's CSVs.

Never tunes thresholds, modifies metric conventions or computes new predictions.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from evaluation.audit_test_artifacts import TAGS
from evaluation.audit_versioned_predictions import audit
from evaluation.predict_and_shrink import read_scores
from evaluation.recover_geometry_sources import sha


def verify(root, reference, source_audit, metric_script, scores_dir, *, expected_cases=None):
    report_path = scores_dir / 'grid_metrics.json'
    before = sha(report_path)
    report = json.loads(report_path.read_text())
    if report.get('schema') != 'pants-versioned-grid-metrics-v1' or report.get('project_protocol') is not True:
        raise RuntimeError('unsupported final report/protocol')
    manifest = json.loads((root / 'evaluation_manifest.json').read_text())
    metric_hash = manifest['source_hashes']['evaluation/compute_tumor_metrics.py']
    if (report.get('original_evaluation_manifest') != manifest
            or report.get('metric_script_sha256') != metric_hash or sha(metric_script) != metric_hash):
        raise RuntimeError('final report source identity differs')
    current = audit(root, reference, source_audit, expected_cases=expected_cases)
    if report.get('artifact_audit') != current:
        raise RuntimeError('final report input snapshot differs')
    expected = set(current['reference_certificate']['cases'])
    source = json.loads(source_audit.read_text())['cases']
    paths = {f'{tag}/{name}' for tag in TAGS for name in ('metrics.json', 'per_case.csv', 'scoring.log')}
    hashes = report.get('metric_artifact_sha256', {})
    if set(hashes) != paths or set(report.get('metrics', {})) != set(TAGS):
        raise RuntimeError('final report lacks exact four-cell output set')
    if any(sha(scores_dir / path) != digest for path, digest in hashes.items()):
        raise RuntimeError('metric output fingerprint differs')
    lesion_counts = []
    for tag in TAGS:
        metrics = json.loads((scores_dir / tag / 'metrics.json').read_text())
        if report['metrics'][tag] != metrics:
            raise RuntimeError('final table differs from per-cell metrics')
        if metrics['protocol']['tumor_class'] != 28 or metrics['protocol']['T_Sen_connectivity'] != '6-neighbor':
            raise RuntimeError('metric protocol differs')
        with (scores_dir / tag / 'per_case.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        ids = [row['case_id'] for row in rows]
        if len(ids) != len(expected) or set(ids) != expected:
            raise RuntimeError('per-case CSV has missing/extra/duplicate cases')
        original_scores = read_scores(root / tag / 'max_tumor_probs.csv')
        positive_dice = []; positive_scores = []; negative_scores = []
        patient_hits = negative_hits = true_tumors = detected_tumors = 0
        for row in rows:
            cid = row['case_id']
            gt_voxels = int(row['gt_tumor_voxels']); pred_voxels = int(row['pred_tumor_voxels'])
            gp = int(row['gt_positive']); pp = int(row['pred_positive'])
            nt = int(row['true_tumors']); nd = int(row['detected_tumors'])
            prob = float(row['max_tumor_probability'])
            if (gt_voxels != source[cid]['tumor_voxels'] or pred_voxels < 0
                    or gp != int(gt_voxels > 0) or pp != int(pred_voxels > 0)
                    or nt < 0 or not 0 <= nd <= nt or (nt > 0) != bool(gp)
                    or not math.isfinite(prob) or prob != original_scores[cid]):
                raise RuntimeError('per-case source/count/probability differs')
            if gp:
                value = float(row['dice'])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise RuntimeError('invalid positive-case Dice')
                positive_dice.append(value); positive_scores.append(prob)
                patient_hits += pp; true_tumors += nt; detected_tumors += nd
            else:
                negative_scores.append(prob); negative_hits += 1 - pp
        npos = len(positive_scores); nneg = len(negative_scores)
        if not npos or not nneg or true_tumors <= 0:
            raise RuntimeError('invalid reference denominators')
        # Pairwise rank definition handles ties explicitly; independent of sklearn.
        auc = sum((p > n) + 0.5 * (p == n) for p in positive_scores for n in negative_scores) / (npos * nneg)
        derived = {'DSC_tumor_mean': math.fsum(positive_dice) / npos,
                   'P_Sen': patient_hits / npos, 'T_Sen': detected_tumors / true_tumors,
                   'Spe': negative_hits / nneg, 'AUC': auc}
        denominators = {'n_cases_evaluated': len(expected), 'AUC_n_cases': len(expected),
                        'DSC_tumor_n_positive_cases': npos, 'P_Sen_n_positive_cases': npos,
                        'Spe_n_negative_cases': nneg, 'T_Sen_n_true_tumors': true_tumors}
        if any(metrics[key] != value for key, value in denominators.items()):
            raise RuntimeError('final metric denominators differ')
        for key, value in derived.items():
            recorded = metrics[key]
            if (isinstance(recorded, bool) or not isinstance(recorded, (int, float))
                    or not math.isfinite(recorded) or not math.isclose(recorded, value, rel_tol=0, abs_tol=1e-12)):
                raise RuntimeError('final metric differs from independent CSV aggregation: ' + key)
        lesion_counts.append(true_tumors)
    if len(set(lesion_counts)) != 1:
        raise RuntimeError('reference lesion count differs across cells')
    if sha(report_path) != before or any(sha(scores_dir / path) != digest for path, digest in hashes.items()):
        raise RuntimeError('final scoring outputs changed during verification')
    for path, digest in current['artifact_sha256'].items():
        if sha(Path(path)) != digest:
            raise RuntimeError('final scoring inputs changed during verification')
    return {'schema': 'pants-final-grid-verification-v1', 'report_sha256': before,
            'cases_per_cell': len(expected), 'gt_positive_cases': current['gt_positive_cases'],
            'gt_negative_cases': current['gt_negative_cases'], 'verified_cells': sorted(TAGS)}


if __name__ == '__main__':
    import fcntl
    parser = argparse.ArgumentParser()
    for name in ('evaluation-dir', 'reference-dir', 'source-audit', 'metric-script', 'scores-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    with (args.evaluation_dir / '.evaluation.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        result = verify(args.evaluation_dir, args.reference_dir, args.source_audit, args.metric_script, args.scores_dir)
    print('FINAL_VERSIONED_GRID_INDEPENDENTLY_VERIFIED ' + json.dumps(result), flush=True)

"""Score persistent/downloaded test outputs without allocating a GPU."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
if __package__ in (None, ''):
    # Preserve the previously documented direct-file CLI as well as python -m.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.audit_test_artifacts import audit
from evaluation.safe_artifacts import retire_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation_dir
    retire_report(root / 'grid_metrics.json')
    # Mandatory independent validation, not merely a documented manual step.
    audited = audit(root)
    manifest = json.loads((root / 'evaluation_manifest.json').read_text())
    expected = {f'PanTS_{i:08d}.nii.gz' for i in range(9001, 9902)}
    labels = root / 'test_ground_truth'
    assert {p.name for p in labels.glob('*.nii.gz')} == expected
    checked = json.loads((root / 'checkpoint_validation_provenance.json').read_text())
    identities = {'unetpp_ds': 'unetpp_ds_on', 'unetpp_nods': 'unetpp_ds_off',
                  'default_ds': 'plain_ds_on', 'default_nods': 'plain_ds_off'}
    results = {}
    common_inputs = None
    for tag in ('unetpp_ds', 'unetpp_nods', 'default_ds', 'default_nods'):
        out = root / tag
        assert {p.name for p in out.glob('PanTS_*.nii.gz')} == expected, tag
        provenance = json.loads((out / 'prediction_provenance.json').read_text())
        assert set(provenance['inputs']) == {name[:-7] for name in expected}, tag
        assert provenance['checkpoint_sha256'] == checked[identities[tag]]['checkpoint_sha256'], tag
        assert provenance['tumor_class'] == 28, tag
        if common_inputs is None:
            common_inputs = provenance['inputs']
        assert provenance['inputs'] == common_inputs, f'{tag}: models saw different test inputs'
        subprocess.run([
            sys.executable, str(Path(__file__).with_name('compute_tumor_metrics.py')),
            '--pred-dir', str(out), '--labels-dir', str(labels), '--tumor-class', '28',
            '--probs-csv', str(out / 'max_tumor_probs.csv'), '--out-json', str(out / 'metrics.json'),
            '--per-case-csv', str(out / 'per_case.csv'), '--expected-cases', '901', '--connectivity', '1',
        ], check=True)
        results[tag] = json.loads((out / 'metrics.json').read_text())
        assert results[tag]['n_cases_evaluated'] == 901
    if audit(root) != audited:
        raise RuntimeError('evaluation artifacts changed during scoring; no grid summary published')
    temporary = root / 'grid_metrics.json.tmp'
    temporary.write_text(json.dumps({'provenance': manifest, 'project_protocol': True,
                                     'artifact_audit': audited,
                                     'metrics': results}, indent=2))
    temporary.replace(root / 'grid_metrics.json')
    print('GRID_EVALUATION_ALL_DONE')


if __name__ == '__main__':
    main()

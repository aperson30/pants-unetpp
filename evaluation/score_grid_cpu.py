"""Score persistent/downloaded test outputs without allocating a GPU."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation_dir
    manifest = json.loads((root / 'evaluation_manifest.json').read_text())
    expected = {f'PanTS_{i:08d}.nii.gz' for i in range(9001, 9902)}
    labels = root / 'test_ground_truth'
    assert {p.name for p in labels.glob('*.nii.gz')} == expected
    results = {}
    for tag in ('unetpp_ds', 'unetpp_nods', 'default_ds', 'default_nods'):
        out = root / tag
        assert {p.name for p in out.glob('PanTS_*.nii.gz')} == expected, tag
        provenance = json.loads((out / 'prediction_provenance.json').read_text())
        assert set(provenance['inputs']) == {name[:-7] for name in expected}, tag
        subprocess.run([
            sys.executable, str(Path(__file__).with_name('compute_tumor_metrics.py')),
            '--pred-dir', str(out), '--labels-dir', str(labels), '--tumor-class', '28',
            '--probs-csv', str(out / 'max_tumor_probs.csv'), '--out-json', str(out / 'metrics.json'),
            '--per-case-csv', str(out / 'per_case.csv'), '--expected-cases', '901', '--connectivity', '1',
        ], check=True)
        results[tag] = json.loads((out / 'metrics.json').read_text())
        assert results[tag]['n_cases_evaluated'] == 901
    temporary = root / 'grid_metrics.json.tmp'
    temporary.write_text(json.dumps({'provenance': manifest, 'project_protocol': True,
                                     'metrics': results}, indent=2))
    temporary.replace(root / 'grid_metrics.json')
    print('GRID_EVALUATION_ALL_DONE')


if __name__ == '__main__':
    main()

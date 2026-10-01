"""Read-only identity/provenance gate for the completed four-cell grid."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, manifest_path: Path) -> dict:
    recorded = json.loads(manifest_path.read_text())
    output = {}
    fold_ids = None
    for tag, cell in recorded['cells'].items():
        fold = root / 'results/Dataset001_PanTS' / (
            cell['trainer'] + '__nnUNetPlansBS4__3d_fullres/fold_0')
        checkpoint = fold / 'checkpoint_final.pth'
        assert sha256(checkpoint) == cell['checkpoint']['sha256'], f'{tag}: checkpoint changed'
        summary = json.loads((fold / 'validation/summary.json').read_text())
        rows = summary['metric_per_case']
        ids = [Path(row['prediction_file']).name for row in rows]
        assert len(ids) == 1800 and len(set(ids)) == 1800, f'{tag}: incomplete scoring'
        expected = set(ids)
        actual = {p.name for p in (fold / 'validation').glob('PanTS_*.nii.gz')}
        assert actual == expected, f'{tag}: prediction identities differ from scoring'
        if fold_ids is None:
            fold_ids = expected
        assert expected == fold_ids, f'{tag}: models used different validation cases'
        assert all('28' in row['metrics'] for row in rows), f'{tag}: class 28 missing'
        output[tag] = {'trainer': cell['trainer'], 'checkpoint_sha256': sha256(checkpoint),
                       'validation_summary_sha256': sha256(fold / 'validation/summary.json'),
                       'validation_cases': len(ids)}
    assert len(output) == 4
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.manifest), indent=2))

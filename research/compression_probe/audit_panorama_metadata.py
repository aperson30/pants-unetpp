"""CPU-only public metadata join; never loads weights or changes a workbook.

Run with bundled Python/pandas. Inputs are pinned snapshots; output lists
provisional candidates, not certified independent test subjects.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import pandas as pd


def normalize_folds(folds):
    expected = {f'Fold {k} validation' for k in range(5)}
    if set(folds) != expected:
        raise ValueError('Unexpected fold keys')
    normalized = {}
    seen = set()
    for key, studies in folds.items():
        if not isinstance(studies, list) or len(studies) != len(set(studies)):
            raise ValueError('Duplicate/invalid fold entries')
        members = set(studies)
        if seen & members:
            raise ValueError('Validation folds overlap')
        seen.update(members)
        normalized[key] = members
    return normalized


def fetch(url, limit=2_000_000):
    with urllib.request.urlopen(url, timeout=20) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Metadata exceeds download bound')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    provenance = {}
    snapshots = {}
    for repo in ('panorama_labels', 'PANORAMA_baseline'):
        sha = json.loads(fetch(f'https://api.github.com/repos/DIAGNijmegen/{repo}/commits/main'))['sha']
        provenance[repo] = sha
        snapshots[repo] = f'https://raw.githubusercontent.com/DIAGNijmegen/{repo}/{sha}'
    clinical = fetch(snapshots['panorama_labels'] + '/clinical_information.xlsx')
    (args.output/'clinical_information.xlsx').write_bytes(clinical)  # Unmodified source acquisition.
    provenance['clinical_sha256'] = hashlib.sha256(clinical).hexdigest()
    df = pd.read_excel(io.BytesIO(clinical), sheet_name='Sheet1')
    if df.PANORAMA_study_id.isna().any() or df.PANORAMA_study_id.duplicated().any():
        raise ValueError('Missing/duplicate study ID')
    pdac = json.loads(fetch(snapshots['PANORAMA_baseline'] + '/src/Dataset104_PANORAMA_baseline_PDAC_Detection_folds.json'))
    pancreas = json.loads(fetch(snapshots['PANORAMA_baseline'] + '/src/Dataset103_PANORAMA_baseline_Pancreas_Segmentation_folds.json'))
    if normalize_folds(pdac) != normalize_folds(pancreas):
        raise ValueError('Two-stage fold files differ')
    fold_map = {}
    for k in range(5):
        for study in pdac[f'Fold {k} validation']:
            if study in fold_map:
                raise ValueError('Validation folds overlap')
            fold_map[study] = k
    if set(fold_map) != set(df.PANORAMA_study_id):
        raise ValueError('Clinical/fold ID union mismatch')
    df = df.assign(fold=df.PANORAMA_study_id.map(fold_map))
    patient_folds = df.groupby('PANORAMA_patient_id')['fold'].nunique()
    cross_patients = set(patient_folds[patient_folds > 1].index)
    tree = json.loads(fetch(f"https://api.github.com/repos/DIAGNijmegen/panorama_labels/git/trees/{provenance['panorama_labels']}?recursive=1"))
    if tree['truncated']:
        raise ValueError('Truncated label tree')
    manual = {Path(t['path']).name.removesuffix('.nii.gz') for t in tree['tree']
              if t['path'].startswith('manual_labels/') and t['path'].endswith('.nii.gz')}
    archive_bytes = args.directory.read_bytes()
    archive = json.loads(archive_bytes)
    available = {}
    for entry in archive['entries']:
        name = Path(entry['name']).name
        if name.endswith('_0000.nii.gz'):
            study = name.removesuffix('_0000.nii.gz')
            if study in available:
                raise ValueError('Duplicate archive study')
            available[study] = entry
    clean = df[~df.PANORAMA_patient_id.isin(cross_patients)
               & ~df.level.isin(['MSD_dataset', 'NIH_dataset'])]
    positive = clean[(clean.label == 'PDAC') & clean.PANORAMA_study_id.isin(manual)]
    in_batch = positive[positive.PANORAMA_study_id.isin(available)]
    candidates = [dict(study=r.PANORAMA_study_id, fold=int(r.fold), reference=r.level,
                       archive=available[r.PANORAMA_study_id]) for r in in_batch.itertuples()]
    summary = dict(provenance=provenance, clinical_studies=len(df), patients=int(df.PANORAMA_patient_id.nunique()),
                   cross_fold_patients=len(cross_patients), cross_fold_studies=int(df.PANORAMA_patient_id.isin(cross_patients).sum()),
                   manual_masks=len(manual), clean_manual_pdac=len(positive),
                   clean_manual_pdac_by_fold={str(k): int((positive.fold == k).sum()) for k in range(5)},
                   batch1_manual_pdac=len(candidates),
                   batch1_manual_pdac_by_fold={str(k): sum(c['fold'] == k for c in candidates) for k in range(5)},
                   reference_counts=positive.level.value_counts().to_dict(),
                   archive_directory_sha256=hashlib.sha256(archive_bytes).hexdigest(),
                   caveat='Provisional: release correspondence, checkpoint split identity and validation-selection exposure still require disclosure.')
    (args.output/'eligibility.json').write_text(json.dumps(dict(summary=summary, candidates=candidates), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

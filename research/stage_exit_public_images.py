"""Stage the six fixed public CTs after their label/count audit, CPU/network only.

Immutable source, exclusive output directory, 256MiB total/80MiB per file.
No main-data changes, GPU commands or preprocessing with substitute plans.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from audit_exit_public_labels import BASE, REPO, REVISION, fetch, request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-audit', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    audit_bytes = args.label_audit.read_bytes()
    audit = json.loads(audit_bytes)
    if audit['revision'] != REVISION or len(audit['cases']) != 6:
        raise RuntimeError('Needs completed fixed six-case label audit at pinned revision')
    entries = []
    for case in audit['cases']:
        cid = case['case_id']
        if not (cid.startswith('PanTS_') and len(cid) == 14 and cid[6:].isdigit()):
            raise RuntimeError('Invalid case ID')
        path = f'image_only/{cid}'
        with request(f'{BASE}/api/datasets/{REPO}/tree/{REVISION}/{path}') as r:
            data = r.read(1_000_001)
        if len(data) > 1_000_000:
            raise RuntimeError('Metadata bound exceeded')
        matches = [e for e in json.loads(data) if e['path'] == path+'/ct.nii.gz']
        if len(matches) != 1 or not 0 < matches[0]['size'] <= 80*1024**2:
            raise RuntimeError('CT metadata size/path gate failed')
        entries.append((case, matches[0]))
    total_expected = sum(entry['size'] for _, entry in entries)
    if total_expected > 256*1024**2:
        raise RuntimeError('Total CT staging exceeds256MiB cap')
    args.root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    records, total = [], 0
    for case, entry in entries:
        target = args.root / (case['case_id']+'_0000.nii.gz')
        count, sha = fetch(entry, target, 256*1024**2-total, max_file=80*1024**2)
        total += count
        image = nib.load(str(target))
        if list(image.shape) != case['shape'] or not np.allclose(
                image.affine, case['affine'], rtol=0, atol=1e-6):
            raise RuntimeError('CT/label shape or affine mismatch')
        records.append({'case_id': case['case_id'], 'path': str(target),
                        'source': entry['path'], 'sha256': sha, 'bytes': count,
                        'shape': list(image.shape), 'ct_label_geometry_matches': True})
        print(json.dumps({'case_id': case['case_id'], 'CT_geometry': 'PASS'}), flush=True)
    report = {'dataset': REPO, 'revision': REVISION,
              'label_audit_sha256': hashlib.sha256(audit_bytes).hexdigest(),
              'elapsed_seconds': time.monotonic()-start, 'bytes_downloaded': total,
              'cases': records,
              'scope': 'CT/label header geometry and immutable-file hashes only; not original-archive byte identity or model evidence'}
    (args.root/'image_audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""CPU-only 901-case source audit. Never alters evaluation or computes metrics.

Reuses checksum-pinned source archives, keeping only one CT/mask temporarily.
Imports the original frozen affine correction through PYTHONPATH. Its replay
must reproduce the exact prediction-input bytes, not just similar geometry.
Success certifies tumor index/geometry lineage only, not the 27 other organs.
"""
import argparse
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np

from data_conversion.fix_affine_orthonormality import (
    THRESHOLD, nearest_orthonormal_affine, orthonormality_deviation,
)
from evaluation.recover_geometry_sources import ARCHIVES, sha

CASES = {f'PanTS_{i:08d}' for i in range(9001, 9902)}


def check_ct_identity(cid, actual, expected, diagnostic):
    matches = actual == expected
    if not matches and not diagnostic:
        raise RuntimeError(cid + ': exact frozen CT input replay failed')
    return matches


def publish_report(folder, report, diagnostic):
    # Diagnostic output can NEVER satisfy the reference builder's readiness
    # filename, even if every replay happens to match on this machine.
    filename = 'source_audit_diagnostic.json' if diagnostic else 'all_901_source_audit.json'
    with (folder / filename).open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print('SOURCE_DIAGNOSTIC_DONE_NOT_A_CERTIFICATE' if diagnostic else
          'ALL_901_SOURCE_TUMOR_INPUT_AND_VOXEL_CHECKS_PASSED', flush=True)


def selected_members(archive, basename, scratch):
    """Yield one bounded, regular member at a time; no extractall/path traversal."""
    seen = set()
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise RuntimeError('unsafe archive member')
            ids = CASES.intersection(name.parts)
            if name.name != basename or not ids:
                continue
            if len(ids) != 1 or not member.isfile() or not 0 < member.size <= 1024**3:
                raise RuntimeError('ambiguous/nonregular/oversized selected member')
            cid = ids.pop()
            if cid in seen:
                raise RuntimeError('duplicate selected case: ' + cid)
            seen.add(cid)
            path = scratch / basename
            with stream.extractfile(member) as incoming, path.open('xb') as outgoing:
                remaining = member.size
                while remaining:
                    block = incoming.read(min(8 * 1024**2, remaining))
                    if not block:
                        raise RuntimeError('short source member')
                    outgoing.write(block)
                    remaining -= len(block)
            try:
                yield cid, path, member.name
            finally:
                path.unlink()
    if seen != CASES:
        raise RuntimeError(f'missing selected cases: {sorted(CASES - seen)}')


def check_tumor(source_path, gt_path, ct):
    source = nib.load(source_path)
    gt = nib.load(gt_path)
    if list(source.shape) != ct['shape'] or gt.shape != source.shape:
        raise RuntimeError('source/CT/saved-GT shape mismatch')
    if not np.allclose(source.affine, ct['raw_affine'], rtol=0, atol=1e-4):
        raise RuntimeError('original source tumor geometry differs from original CT')
    source_data = np.asanyarray(source.dataobj)
    saved = np.asanyarray(gt.dataobj)
    if not np.isfinite(source_data).all() or not np.isfinite(saved).all():
        raise RuntimeError('nonfinite source/GT voxels')
    if not np.isin(source_data, (0, 1)).all():
        raise RuntimeError('source tumor is not binary')
    if not np.equal(saved, np.floor(saved)).all() or saved.min() < 0 or saved.max() > 28:
        raise RuntimeError('invalid saved GT labels')
    tumor = source_data > 0
    if not np.array_equal(saved == 28, tumor):
        raise RuntimeError('saved class-28 voxels differ from original source tumor')
    return {'tumor_voxels': int(tumor.sum()), 'source_tumor_sha256': sha(source_path),
            'original_gt_sha256': sha(gt_path), 'tumor_index_equality': True,
            'source_geometry_matches_original_ct': True,
            'saved_gt_matches_prediction_grid': bool(np.allclose(
                gt.affine, ct['prediction_affine'], rtol=0, atol=1e-4))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--diagnose-all', action='store_true',
                        help='continue CT identity mismatches to inspect all source masks; never emit a pass certificate')
    args = parser.parse_args()
    # Fresh diagnostic directory only; failed/partial attempts are not overwritten.
    args.report_dir.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((args.evaluation_dir / 'unetpp_ds/prediction_provenance.json').read_text())
    other = json.loads((args.evaluation_dir / 'unetpp_nods/prediction_provenance.json').read_text())
    if set(provenance['inputs']) != CASES or provenance['inputs'] != other['inputs']:
        raise RuntimeError('prediction cohort/input provenance differs')
    if {p.stem.removesuffix('.nii') for p in (args.evaluation_dir / 'test_ground_truth').glob('*.nii.gz')} != CASES:
        raise RuntimeError('saved GT cohort differs')
    archives = {}
    for name, _, digest, size in ARCHIVES:
        path = args.source_dir / name
        if path.stat().st_size != size or sha(path) != digest:
            raise RuntimeError('original archive checksum/size mismatch: ' + name)
        archives[name] = path
        print('ARCHIVE_VERIFIED ' + name, flush=True)
    rows = {}
    with tempfile.TemporaryDirectory(prefix='source_audit_', dir=args.report_dir) as temporary:
        scratch = Path(temporary)
        for cid, path, member in selected_members(archives[ARCHIVES[0][0]], 'ct.nii.gz', scratch):
            image = nib.load(path)
            if len(image.shape) != 3 or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
                raise RuntimeError('invalid CT geometry')
            raw_digest = sha(path)
            affine = image.affine.copy()
            corrected = orthonormality_deviation(affine) > THRESHOLD
            if corrected:
                fixed_path = scratch / f'{cid}_0000.nii.gz'
                affine = nearest_orthonormal_affine(affine)
                nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine, image.header), fixed_path)
                input_digest = sha(fixed_path)
                affine = nib.load(fixed_path).affine.copy()
                fixed_path.unlink()
            else:
                input_digest = raw_digest
            matches = check_ct_identity(cid, input_digest, provenance['inputs'][cid], args.diagnose_all)
            rows[cid] = {'ct_member': member, 'raw_ct_sha256': raw_digest,
                         'prediction_ct_sha256': input_digest, 'shape': list(image.shape),
                         'raw_affine': image.affine.tolist(), 'prediction_affine': affine.tolist(),
                         'frozen_ct_correction_replayed': corrected,
                         'expected_prediction_ct_sha256': provenance['inputs'][cid],
                         'exact_ct_identity_passed': matches}
            print(('CT_INPUT_VERIFIED ' if matches else 'CT_REPLAY_MISMATCH ') + cid, flush=True)
        with (args.report_dir / 'tumor_cases.jsonl').open('x') as output:
            for cid, path, member in selected_members(archives[ARCHIVES[1][0]], 'pancreatic_lesion.nii.gz', scratch):
                result = check_tumor(path, args.evaluation_dir / 'test_ground_truth' / f'{cid}.nii.gz', rows[cid])
                rows[cid].update(result, source_tumor_member=member)
                output.write(json.dumps({'case': cid, **rows[cid]}, allow_nan=False) + '\n')
                output.flush()
                print(f'TUMOR_SOURCE_VERIFIED {cid} voxels={result["tumor_voxels"]}', flush=True)
    report = {'scope': 'tumor only; no repair/scoring/all-organ certification',
              'diagnostic_only': args.diagnose_all,
              'ct_identity_mismatches': sorted(cid for cid, row in rows.items() if not row['exact_ct_identity_passed']),
              'cases': rows, 'archives': {name: digest for name, _, digest, _ in ARCHIVES},
              'header_mismatches': sorted(cid for cid, row in rows.items() if not row['saved_gt_matches_prediction_grid']),
              'script_sha256': sha(Path(__file__))}
    publish_report(args.report_dir, report, args.diagnose_all)


if __name__ == '__main__':
    main()

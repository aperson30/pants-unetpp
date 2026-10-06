"""Bounded CPU diagnosis for the two CTs changed by frozen preprocessing.

Recover source bytes from the original checksum-pinned archive. Compare the
actual frozen fix_folder call with the audit's manual replay, never modifying
original evidence or relaxing prediction input hashes. Preserve diagnostic CT
copies so future checks do not need another archive scan.
"""
import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np

from data_conversion import fix_affine_orthonormality as frozen
from evaluation.recover_geometry_sources import ARCHIVES, sha

CASES = {'PanTS_00009812', 'PanTS_00009871'}


def compare_replays(raw_path, case_dir, expected_hash):
    actual = case_dir / 'frozen_in_place'; actual.mkdir()
    manual = case_dir / 'manual'; manual.mkdir()
    original_copy = actual / raw_path.name
    manual_copy = manual / raw_path.name
    shutil.copyfile(raw_path, original_copy)
    changed = frozen.fix_folder(actual)
    image = nib.load(raw_path)
    affine = frozen.nearest_orthonormal_affine(image.affine)
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine, image.header), manual_copy)
    fixed = nib.load(original_copy); replay = nib.load(manual_copy)
    raw_values = np.asanyarray(image.dataobj)
    fixed_values = np.asanyarray(fixed.dataobj)
    replay_values = np.asanyarray(replay.dataobj)
    result = {'raw_sha256': sha(raw_path), 'expected_prediction_sha256': expected_hash,
              'frozen_in_place_sha256': sha(original_copy), 'manual_sha256': sha(manual_copy),
              'frozen_files_changed': changed,
              'frozen_matches_prediction_bytes': sha(original_copy) == expected_hash,
              'manual_matches_prediction_bytes': sha(manual_copy) == expected_hash,
              'frozen_preserves_source_values': bool(np.array_equal(raw_values, fixed_values)),
              'manual_preserves_source_values': bool(np.array_equal(raw_values, replay_values)),
              'frozen_manual_affines_equal': bool(np.array_equal(fixed.affine, replay.affine)),
              'raw_affine': image.affine.tolist(), 'frozen_affine': fixed.affine.tolist(),
              'manual_affine': replay.affine.tolist(),
              'frozen_manual_header_equal': fixed.header.binaryblock == replay.header.binaryblock}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    name, _, archive_hash, archive_size = ARCHIVES[0]
    archive = args.source_dir / name
    if archive.stat().st_size != archive_size or sha(archive) != archive_hash:
        raise RuntimeError('original image archive identity differs')
    print('ORIGINAL_IMAGE_ARCHIVE_VERIFIED', flush=True)
    provenance = json.loads((args.evaluation_dir / 'unetpp_ds/prediction_provenance.json').read_text())
    other = json.loads((args.evaluation_dir / 'unetpp_nods/prediction_provenance.json').read_text())
    if provenance['inputs'] != other['inputs']:
        raise RuntimeError('prediction input identities differ')
    found = set(); rows = {}
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise RuntimeError('unsafe archive member')
            ids = CASES.intersection(path.parts)
            if not ids or path.name != 'ct.nii.gz':
                continue
            if len(ids) != 1 or not member.isfile() or not 0 < member.size <= 1024**3:
                raise RuntimeError('invalid selected CT member')
            cid = ids.pop()
            if cid in found:
                raise RuntimeError('duplicate selected CT')
            found.add(cid)
            case_dir = args.destination / cid; case_dir.mkdir()
            raw = case_dir / f'{cid}_0000.nii.gz'
            with stream.extractfile(member) as incoming, raw.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
            if raw.stat().st_size != member.size:
                raise RuntimeError('short CT copy')
            rows[cid] = compare_replays(raw, case_dir, provenance['inputs'][cid])
            rows[cid]['source_member'] = member.name
            print(json.dumps({'case': cid, **rows[cid]}, allow_nan=False), flush=True)
        # Scan to archive end to reject duplicate selected members as well.
    if found != CASES:
        raise RuntimeError('missing selected CT')
    with (args.destination / 'replay_diagnosis.json').open('x') as output:
        json.dump({'cases': rows, 'frozen_correction_module': frozen.__file__,
                   'frozen_correction_sha256': sha(Path(frozen.__file__)),
                   'nibabel_version': nib.__version__, 'numpy_version': np.__version__,
                   'script_sha256': sha(Path(__file__))}, output, indent=2, allow_nan=False)
    print('CORRECTED_CT_REPLAY_DIAGNOSIS_COMPLETE_NOT_A_COHORT_PASS', flush=True)


if __name__ == '__main__':
    main()

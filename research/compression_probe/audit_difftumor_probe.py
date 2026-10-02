"""Independent CPU saved-map audit. Ground truth enters only here, never model."""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np

ARMS = ('native', 'control', 'posterior_mean', 'posterior_seed0', 'posterior_seed1', 'posterior_seed2')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4*1024**2): value.update(block)
    return value.hexdigest()


def decision(rows, drift):
    native = rows['native']['tumor_mean']
    control_error = rows['control']['tumor_max_abs_vs_native']
    if control_error > 1e-4:
        return 'INVALID_CONTROL: no second-judge scientific conclusion'
    if native < .01:
        return 'NO_GO_CURRENT_ROUTE: native response below predeclared adequacy floor'
    deltas = [native - rows[a]['tumor_mean'] for a in ARMS[2:]]
    if all(d >= .2*native and d > 20*drift for d in deltas):
        return 'SUPPORT_ONLY_GO: request PI approval for fixed cohort, not training'
    return 'MIXED_OR_RETAINED: reject blanket lesion-erasure story; no training'


def audit(manifest, folder):
    inputs = json.loads(manifest.read_text())
    result = json.loads((folder/'completion.json').read_text())
    if result.get('completed') is not True or result['manifest_sha256'] != digest(manifest):
        raise ValueError('Incomplete/changed study')
    records = result['results']
    if tuple(r['arm'] for r in records) != ARMS or result['ground_truth_input'] or result['official_postprocessing']:
        raise ValueError('Wrong study scope/arms')
    if result['checkpoint_sha256'] != inputs['weights_sha256']:
        raise ValueError('Checkpoint identity mismatch')
    for record, source in zip(records, inputs['images']):
        if record['arm'] != source['arm'] or record['image_sha256'] != source['sha256']:
            raise ValueError('Arm/input identity mismatch')
    if digest(inputs['label']) != inputs['label_sha256']:
        raise ValueError('Annotation changed')
    label = nib.load(inputs['label'])
    if list(label.shape) != result['native_shape'] or not np.allclose(label.affine, result['affine'], atol=1e-5, rtol=0):
        raise ValueError('Annotation/output grid mismatch')
    tumor = label.get_fdata() == 1  # PANORAMA label1, DiffTumor output2
    if not tumor.any(): raise ValueError('Empty tumor')
    native = np.load(folder/'native_tumor.npy', allow_pickle=False, mmap_mode='r')
    rows = {}; drift = 0.
    for record in records:
        arm = record['arm']
        for suffix, key in (('tumor', 'probability_sha256'), ('labels', 'labels_sha256')):
            if digest(folder/(arm+'_'+suffix+'.npy')) != record[key]:
                raise ValueError('Saved map hash mismatch')
        probability = np.load(folder/(arm+'_tumor.npy'), allow_pickle=False, mmap_mode='r')
        labels = np.load(folder/(arm+'_labels.npy'), allow_pickle=False, mmap_mode='r')
        if probability.shape != tumor.shape or labels.shape != tumor.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)) or not np.isin(labels, [0, 1, 2]).all():
            raise ValueError('Invalid map values/grid')
        if record['all_class_repeat_max_abs'] > 1e-4 or record['labels_different']:
            raise ValueError('Failed repeatability gate')
        drift = max(drift, record['all_class_repeat_max_abs'])
        rows[arm] = dict(tumor_mean=float(probability[tumor].mean()),
                         tumor_max=float(probability[tumor].max()),
                         tumor_voxels_argmax=int(np.count_nonzero((labels == 2) & tumor)),
                         total_argmax_tumor_voxels=int(np.count_nonzero(labels == 2)),
                         global_max=float(probability.max()),
                         tumor_max_abs_vs_native=float(np.abs(probability[tumor]-native[tumor]).max()))
    reference = rows['native']['tumor_mean']
    for row in rows.values(): row['ratio_to_native'] = row['tumor_mean']/reference if reference > 0 else None
    return dict(completed=True, saved_map_hashes_verified=True, tumor_voxels=int(tumor.sum()),
                rows=rows, decision=decision(rows, drift), max_repeat_drift=drift,
                clinical_recall=False, official_postprocessing=False,
                caveat='One feasibility-selected patient; different trained judge, not proven patient independent')


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'folder', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); result = audit(args.manifest, args.folder)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()

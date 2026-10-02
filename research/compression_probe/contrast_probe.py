"""Seven-pass, one-host amplitude/phase engineering probe; NOT a PDAC test.

Reuse the old binary insert, never supersample it mid-comparison. Nonlinearity
does not prove a learned prior; a flat response does not prove clinical safety.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
import torch

from bounded_controls import decode, insert_signal, prepare
from prepare_msd import sha
from prepare_texture_roi import prepare_roi
from run_maisi import load_model, publish


CONTRASTS = (-20., -10., -40., -80., 20.)


def shift_mask(mask, shift=(2, 0, 0)):
    """Translation without wraparound; reject loss at volume edges."""
    positions = np.argwhere(mask)
    moved = positions + np.asarray(shift)
    if np.any(moved < 0) or np.any(moved >= np.asarray(mask.shape)):
        raise ValueError('Shift would clip the insert')
    result = np.zeros_like(mask)
    result[tuple(moved.T)] = True
    if result.sum() != mask.sum():
        raise ValueError('Shift altered insert count')
    return result


def alter(control, mask, roi, contrast):
    if contrast == 0 or not np.isfinite(contrast) or not mask.any() or not roi[mask].all():
        raise ValueError('Invalid contrast/insert or ROI escape')
    output = control.copy()
    output[mask] += contrast
    if np.any(output[mask] < -1000) or np.any(output[mask] > 1000):
        raise ValueError('Clipping would confound amplitude test')
    return output


def response_metrics(response, mask, spacing, contrast):
    distances = ndi.distance_transform_edt(~mask, sampling=spacing)
    ring = (distances >= 2) & (distances <= 5) & ~mask
    if not ring.any() or not np.isfinite(response).all():
        raise ValueError('Invalid response or empty measurement ring')
    raw = float(response[mask].mean())
    background = float(response[ring].mean())
    return dict(contrast_hu=contrast, response_mean_hu=raw,
                ring_response_mean_hu=background, raw_signed_retention=raw/contrast,
                ring_corrected_retention=(raw-background)/contrast,
                mask_voxels=int(mask.sum()), ring_voxels=int(ring.sum()))


def main():
    process_start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if importlib.metadata.version('monai') != '1.5.1' or not torch.__version__.startswith('2.10.0'):
        raise RuntimeError('Frozen MONAI/torch stack mismatch')
    manifest = json.loads((args.root/'small_followup_120/inputs_manifest.json').read_text())
    ct, control, labels, spacing = prepare(manifest)
    reference_path = args.root/'small_results_3290494/pancreas_120/reconstruction.nii.gz'
    cached = nib.load(str(reference_path))
    if cached.shape != ct.shape or not np.allclose(cached.affine, ct.affine, atol=1e-5, rtol=0):
        raise ValueError('Cached reconstruction grid mismatch')
    if sha(reference_path) != 'a2d31659c02f2d2b51eea9d0bf97656ed6cb477a204187977d47131fcd1ac1c1':
        raise ValueError('Cached baseline identity mismatch')
    cached_control = nib.load(str(args.root/'small_results_3290494/pancreas_120/control.nii.gz'))
    if cached_control.shape != ct.shape or not np.allclose(cached_control.affine, ct.affine,
            atol=1e-5, rtol=0) or not np.array_equal(cached_control.get_fdata(dtype=np.float32), control):
        raise ValueError('Preprocessing differs from old control')
    modified, mask, info = insert_signal(control, labels, spacing)
    if info['center_voxel'] != [225, 259, 74] or info['voxels'] != 111:
        raise ValueError('Prespecified binary insert changed')
    roi, _ = prepare_roi(labels, spacing)
    shifted = shift_mask(mask)
    for amplitude in CONTRASTS:
        trial = alter(control, mask, roi, amplitude)
        if amplitude == -20 and not np.array_equal(trial, modified):
            raise ValueError('Old -20HU insert not reproduced exactly')
        del trial
    trial = alter(control, shifted, roi, -20.)
    del trial, modified, roi
    torch.set_num_threads(1 if args.cpu_preflight else 4)
    model = load_model(manifest['model'])
    if args.cpu_preflight:
        with torch.inference_mode():
            mu, _ = model.encode(torch.zeros(1, 1, 8, 64, 8))
            result = model.decode(mu)
        if result.shape != (1, 1, 8, 64, 8) or not torch.isfinite(result).all():
            raise ValueError('Model contract failed')
        print('CONTRAST_CPU_PREFLIGHT_PASS', json.dumps(info), flush=True)
        return
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1 or args.output is None:
        raise RuntimeError('Require one Slurm GPU and unique output directory')
    if not torch.__version__.startswith('2.10.0'):
        raise RuntimeError('Frozen torch stack mismatch')
    if shutil.disk_usage(args.root).free < 1024**3:
        raise RuntimeError('Insufficient metrics headroom')
    args.output.mkdir(exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    publish(args.output/'provenance.json', dict(manifest=manifest, insert=info,
        contrasts=list(CONTRASTS), shift_voxels=[2, 0, 0], shift_mm=(spacing*np.array([2, 0, 0])).tolist(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), job=os.environ['SLURM_JOB_ID'],
        script_sha256=sha(Path(__file__)), cached_baseline_sha256=sha(reference_path),
        scope='one-host local linearity/phase engineering probe; NOT learned-prior proof or clinical safety',
        precision='FP32 mean, TF32 off, whole native volume; no input crops/tiles'))
    model.to('cuda:0')
    start = time.monotonic()
    baseline, timing = decode(model, control, 'cuda:0')
    delta = np.abs(baseline-cached.get_fdata(dtype=np.float32))
    consistency = dict(max_abs_hu=float(delta.max()), mean_abs_hu=float(delta.mean()), timing=timing)
    del delta
    publish(args.output/'baseline_consistency.json', consistency)
    if consistency['max_abs_hu'] > 1e-3:
        raise RuntimeError('Cached baseline mismatch: stop before amplitude sweep')
    projected_seconds = time.monotonic()-process_start + 6*(timing['seconds']+10) + 30
    if projected_seconds > 1000:
        raise RuntimeError('First-pass timing predicts watchdog overrun; stop rather than expand budget')
    outcomes = []
    # Release each full-volume output before the next pass; only scalars accumulate.
    cases = [(a, mask, [0, 0, 0]) for a in CONTRASTS] + [(-20., shifted, [2, 0, 0])]
    for index, (amplitude, region, offset) in enumerate(cases):
        modified = control.copy()
        modified[region] += amplitude  # Bounds/ROI checked on CPU above.
        output, timing = decode(model, modified, 'cuda:0')
        del modified
        metrics = response_metrics(output-baseline, region, spacing, amplitude)
        del output
        metrics.update(shift_voxels=offset, timing=timing)
        publish(args.output/f'pass_{index}.json', metrics)
        outcomes.append(metrics)
        print('CONTRAST_PASS_COMPLETE', index, json.dumps(metrics), flush=True)
    publish(args.output/'metrics.json', dict(outcomes=outcomes, baseline_consistency=consistency,
        note='Thresholds are screening heuristics; one shifted point does not bound all phase effects.'))
    publish(args.output/'completion.json', dict(completed=True, passes=7,
        elapsed_seconds=time.monotonic()-start, volumes_saved=False))
    print('CONTRAST_PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

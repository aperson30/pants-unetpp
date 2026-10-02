"""CPU-only one-host classical control. HF-texture match is NOT noise matching.

No torch/MONAI/model imports. No images saved. Two fixed scikit-image families;
TV crop/iteration checks are empirical, never an exact-crop claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
import skimage
import scipy
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle

from prepare_texture_roi import prepare_roi

CONTRASTS = (-20., -10., -40., -80., 20.)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def insert_mask(labels, spacing):
    roi, _ = prepare_roi(labels, spacing)
    clearance = ndi.distance_transform_edt(np.pad(roi, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    center = np.unravel_index(int(clearance.argmax()), labels.shape)
    if clearance[center] <= 4:
        raise ValueError('Insufficient clearance for frozen insert')
    halo = np.ceil(4/spacing).astype(int)+1
    box = tuple(slice(max(0,c-h), min(n,c+h+1)) for c,h,n in zip(center,halo,labels.shape))
    grid = np.ogrid[tuple(slice(s.start, s.stop) for s in box)]
    mask = np.zeros(labels.shape, dtype=bool)
    mask[box] = sum(((g-c)*sp)**2 for g,c,sp in zip(grid,center,spacing)) <= 16
    return mask, center, roi


def ring(mask, spacing):
    distance = ndi.distance_transform_edt(~mask, sampling=spacing)
    return (distance >= 2) & (distance <= 5) & ~mask


def box_for(region, spacing, halo_mm):
    positions = np.where(region)
    if not len(positions[0]):
        raise ValueError('Empty crop support')
    halo = np.ceil(halo_mm/spacing).astype(int)
    return tuple(slice(max(0,int(p.min())-h), min(n,int(p.max())+h+1))
                 for p,h,n in zip(positions,halo,region.shape))


def high_frequency_sd(array, roi, spacing):
    residual = array - ndi.gaussian_filter(array, sigma=2/spacing, truncate=4, mode='reflect')
    return float(residual[roi].std(dtype=np.float64))


def response(output, baseline, mask, surround, amplitude):
    # Match the frozen VAE probe's float32 regional-mean convention.
    raw = float((output[mask]-baseline[mask]).mean())/amplitude
    background = float((output[surround]-baseline[surround]).mean())/amplitude
    return dict(raw_signed_retention=raw, ring_corrected_retention=raw-background)


def apply_filter(array, family, strength, iterations=200):
    if family == 'tv':
        # Standard native-index-space isotropic TV, NOT spacing-weighted TV.
        result = denoise_tv_chambolle(array, weight=strength, eps=2e-4,
                                     max_num_iter=iterations, channel_axis=None)
    elif family == 'nlm':
        result = denoise_nl_means(array, h=strength, patch_size=3, patch_distance=2,
                                 fast_mode=True, sigma=0, preserve_range=True, channel_axis=None)
    else:
        raise ValueError('Unknown frozen family')
    if result.shape != array.shape or not np.isfinite(result).all():
        raise ValueError('Invalid filter output')
    return result.astype(np.float32, copy=False)


def calibrate(array, roi, spacing, target, family):
    low, high = .1, (512. if family == 'tv' else 128.)
    history = []
    def evaluate(value):
        out = apply_filter(array, family, value)
        score = high_frequency_sd(out, roi, spacing)
        history.append(dict(strength=value, hf_sd_hu=score))
        print('CALIBRATION', family, value, score, flush=True)
        return score
    left, right = evaluate(low), evaluate(high)
    if not right <= target <= left:
        return None, dict(status='target-not-bracketed', history=history)
    for _ in range(8):
        mid = float(np.sqrt(low*high))
        score = evaluate(mid)
        if not right-1e-4 <= score <= left+1e-4:
            return None, dict(status='nonmonotone-calibration', history=history)
        if score > target:
            low, left = mid, score
        else:
            high, right = mid, score
    best = min(history, key=lambda row: abs(row['hf_sd_hu']-target))
    error = abs(best['hf_sd_hu']-target)/target
    return (best['strength'] if error <= .02 else None), dict(
        status='matched' if error <= .02 else 'match-tolerance-failed',
        relative_error=error, selected=best, history=history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if skimage.__version__ != '0.25.2':
        raise ValueError('Require pinned scikit-image 0.25.2')
    if not args.preflight:
        if args.output.exists() or shutil.disk_usage(args.output.parent).free < 16*1024**2:
            raise ValueError('Output must be unique with filesystem headroom')
        with tempfile.TemporaryFile(dir=args.output.parent) as stream:
            stream.write(bytes(1024**2)); stream.flush(); os.fsync(stream.fileno())
    # Hard process bound; no retries or budget extensions.
    timer = threading.Timer(1800, lambda: os._exit(124)); timer.daemon = True; timer.start()
    start = time.monotonic()
    paths = {k: args.inputs/f'{k}.nii.gz' for k in ('image','label','reconstruction')}
    hashes = {'image':'60301cafdd54fe66fb7517eb19eac063c1caa0e5ff106f99fc68fea26ca644c8',
              'label':'60ddf8533b024353ea2b06a4eee66700f07d20c53ec0c5300aba9879714ba34d',
              'reconstruction':'a2d31659c02f2d2b51eea9d0bf97656ed6cb477a204187977d47131fcd1ac1c1'}
    if {k:sha(p) for k,p in paths.items()} != hashes:
        raise ValueError('Cached input identity mismatch')
    imgs = {k:nib.as_closest_canonical(nib.load(p)) for k,p in paths.items()}
    ct = imgs['image']
    for im in imgs.values():
        if im.shape != ct.shape or not np.allclose(im.affine,ct.affine,atol=1e-5,rtol=0):
            raise ValueError('Cached input geometry mismatch')
        if im.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('Require mm spatial units')
    spacing = np.linalg.norm(ct.affine[:3,:3], axis=0)
    if not np.isfinite(ct.affine).all() or np.any(spacing <= 0) or not np.allclose(
            (ct.affine[:3,:3]/spacing).T@(ct.affine[:3,:3]/spacing),np.eye(3),atol=1e-5,rtol=0):
        raise ValueError('Unsupported affine')
    control = np.clip(ct.get_fdata(dtype=np.float32),-1000,1000)
    labels = np.asarray(imgs['label'].dataobj)
    vae = imgs['reconstruction'].get_fdata(dtype=np.float32)
    if not np.isfinite(control).all() or not np.isfinite(vae).all() or control.size > 32_000_000:
        raise ValueError('Invalid/oversized image')
    mask, center, roi = insert_mask(labels,spacing)
    if tuple(center) != (225,259,74) or mask.sum() != 111:
        raise ValueError('Frozen insert changed')
    shifted = np.zeros_like(mask); shifted[2:] = mask[:-2]
    if shifted.sum() != 111 or not roi[shifted].all():
        raise ValueError('Invalid shifted insert')
    surround, shifted_surround = ring(mask,spacing), ring(shifted,spacing)
    calibration = roi & (ndi.distance_transform_edt(~(mask|shifted),sampling=spacing) > 10)
    # Disjoint source-mask-only spatial split; never chosen using output intensity.
    positions = np.where(calibration)
    axis = int(np.argmax([np.ptp(p) for p in positions]))
    cut = int(np.median(positions[axis]))
    grid = np.arange(control.shape[axis]).reshape(tuple(control.shape[axis] if k==axis else 1 for k in range(3)))
    fit, heldout = calibration & (grid <= cut), calibration & (grid > cut)
    if min(fit.sum(),heldout.sum()) < 1000:
        raise ValueError('Insufficient separated calibration/holdout voxels')
    support = calibration | mask | shifted | surround | shifted_surround
    boxes = [box_for(support,spacing,mm) for mm in (24.,40.)]
    target = high_frequency_sd(vae[boxes[0]],fit[boxes[0]],spacing)
    report = dict(scope='one-host engineering control, NOT clinical endpoint or VAE-specific proof',
        hashes=hashes, spacing_mm=spacing.tolist(), fit_voxels=int(fit.sum()),
        holdout_voxels=int(heldout.sum()), target_hf_sd_hu=target,
        calibration_axis=axis, calibration_cut_voxel=cut,
        crop_shapes=[list(control[b].shape) for b in boxes],
        matching='2mm high-pass texture SD, NOT pure noise SD or observer detectability',
        families=['tv','nlm'], numpy=np.__version__, scipy=scipy.__version__,
        nibabel=nib.__version__, skimage=skimage.__version__, runner_sha256=sha(Path(__file__)))
    if not np.isfinite(target) or target <= 0:
        raise ValueError('Invalid matching target')
    if args.preflight:
        print('DENOISER_CPU_PREFLIGHT_PASS',json.dumps(report),flush=True); return
    args.output.mkdir(exist_ok=False)
    publish(args.output/'preflight.json',report)
    small, large = boxes
    outcomes = {}
    for family in ('tv','nlm'):
        strength, match = calibrate(control[small],fit[small],spacing,target,family)
        publish(args.output/f'{family}_calibration.json',match)
        if strength is None:
            outcomes[family] = dict(status=match['status']); continue
        base = apply_filter(control[small],family,strength)
        expanded = apply_filter(control[large],family,strength)
        overlap = tuple(slice(a.start-b.start,a.stop-b.start) for a,b in zip(small,large))
        crop_error = float(np.max(np.abs(base[support[small]]-expanded[overlap][support[small]])))
        iteration_error = 0.
        double_base = None
        if family == 'tv':
            double_base = apply_filter(control[large],family,strength,iterations=400)
            iteration_error = float(np.max(np.abs(double_base[support[large]]-expanded[support[large]])))
        checks = dict(crop_max_abs_hu=crop_error, iteration_max_abs_hu=iteration_error,
            vae_holdout_hf_sd_hu=high_frequency_sd(vae[small],heldout[small],spacing),
            filter_holdout_hf_sd_hu=high_frequency_sd(base,heldout[small],spacing))
        checks['holdout_relative_hf_error'] = abs(checks['filter_holdout_hf_sd_hu']-
            checks['vae_holdout_hf_sd_hu'])/checks['vae_holdout_hf_sd_hu']
        checks['holdout_match_within_10pct'] = checks['holdout_relative_hf_error'] <= .10
        publish(args.output/f'{family}_checks.json',checks)
        if crop_error > .1 or iteration_error > .1:
            outcomes[family] = dict(status='boundary-or-iteration-inconclusive',checks=checks)
            del base,expanded,double_base
            continue
        # Use larger crop for final contrasts; check convergence there again on extremes.
        small_base,base = base,expanded
        del expanded
        cases = [(a,mask,surround,[0,0,0]) for a in CONTRASTS] + [(-20.,shifted,shifted_surround,[2,0,0])]
        records = []
        for index,(amplitude,region,rng,offset) in enumerate(cases):
            trial = control[large].copy(); trial[region[large]] += amplitude
            if trial[region[large]].min() < -1000 or trial[region[large]].max() > 1000:
                raise ValueError('Insert clipping')
            out = apply_filter(trial,family,strength)
            metrics = response(out,base,region[large],rng[large],amplitude)
            boundary = apply_filter(trial[overlap],family,strength)
            small_metrics = response(boundary,small_base,
                                     region[small],rng[small],amplitude)
            metrics['crop_retention_difference'] = abs(metrics['ring_corrected_retention']-small_metrics['ring_corrected_retention'])
            metrics.update(contrast_hu=amplitude,shift_voxels=offset)
            if family == 'tv' and index in (1,4):
                doubled = apply_filter(trial,family,strength,iterations=400)
                m = response(doubled,double_base,region[large],rng[large],amplitude)
                metrics['iteration_retention_difference'] = abs(metrics['ring_corrected_retention']-m['ring_corrected_retention'])
                del doubled
            publish(args.output/f'{family}_pass_{index}.json',metrics)
            records.append(metrics)
            print('DENOISER_PASS_COMPLETE',family,index,json.dumps(metrics),flush=True)
            del out,boundary,trial
        reliable = all(r['crop_retention_difference'] <= .01 and r.get('iteration_retention_difference',0) <= .01 for r in records)
        outcomes[family] = dict(status='valid-engineering-curve' if reliable else 'numerical-sensitivity-inconclusive',
                                strength=strength,checks=checks,records=records)
        del base,small_base,double_base
    publish(args.output/'completion.json',dict(completed=True,outcomes=outcomes,elapsed_seconds=time.monotonic()-start,
        note='No automated scientific GO/STOP; matching and validity limitations must be reviewed.',volumes_saved=False))
    print('DENOISER_CONTROL_COMPLETE',flush=True)


if __name__ == '__main__':
    main()

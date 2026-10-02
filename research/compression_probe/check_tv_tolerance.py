"""Post-run numerical validation, not retuning: freeze TV strength, tighten eps.

No new family, strength search, GPU or image outputs. Distinguishes numerical
stability from a weak 200-vs400 cap check with identical early-stop tolerances.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_tv_chambolle

from denoiser_control import sha, insert_mask, ring, box_for, response, high_frequency_sd, publish


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refuse output overwrite')
    timer = threading.Timer(300,lambda:os._exit(124)); timer.daemon=True; timer.start()
    start=time.monotonic()
    preflight=json.loads((args.results/'preflight.json').read_text())
    completed=json.loads((args.results/'completion.json').read_text())
    if not completed['completed']:
        raise ValueError('Require completed primary run')
    for kind,digest in preflight['hashes'].items():
        if sha(args.inputs/f'{kind}.nii.gz') != digest:
            raise ValueError('Input changed')
    image=nib.as_closest_canonical(nib.load(args.inputs/'image.nii.gz'))
    labels=np.asarray(nib.as_closest_canonical(nib.load(args.inputs/'label.nii.gz')).dataobj)
    spacing=np.asarray(preflight['spacing_mm'])
    control=np.clip(image.get_fdata(dtype=np.float32),-1000,1000)
    mask,center,roi=insert_mask(labels,spacing)
    if tuple(center)!=(225,259,74) or mask.sum()!=111:
        raise ValueError('Frozen insert changed')
    shifted=np.zeros_like(mask); shifted[2:]=mask[:-2]
    surround=ring(mask,spacing); shifted_ring=ring(shifted,spacing)
    calibration=roi & (ndi.distance_transform_edt(~(mask|shifted),sampling=spacing)>10)
    axis=preflight['calibration_axis']; cut=preflight['calibration_cut_voxel']
    grid=np.arange(control.shape[axis]).reshape(tuple(control.shape[axis] if k==axis else 1 for k in range(3)))
    fit=calibration & (grid<=cut); heldout=calibration & (grid>cut)
    support=calibration|mask|shifted|surround|shifted_ring
    box=box_for(support,spacing,40.)
    strength=completed['outcomes']['tv']['strength']
    primary=completed['outcomes']['tv']['records']
    outputs=[]
    # Both tighter settings fixed before their outputs are inspected.
    for eps in (2e-5,2e-6):
        baseline=denoise_tv_chambolle(control[box],weight=strength,eps=eps,
                                      max_num_iter=800,channel_axis=None)
        records=[]
        for amplitude,index in ((-20.,0),(-10.,1),(20.,4)):
            trial=control[box].copy(); trial[mask[box]]+=amplitude
            reconstructed=denoise_tv_chambolle(trial,weight=strength,eps=eps,
                                               max_num_iter=800,channel_axis=None)
            if not np.isfinite(reconstructed).all():
                raise ValueError('Nonfinite solver output')
            metrics=response(reconstructed,baseline,mask[box],surround[box],amplitude)
            metrics.update(contrast_hu=amplitude,
                difference_from_primary=metrics['ring_corrected_retention']-primary[index]['ring_corrected_retention'])
            records.append(metrics)
            print('TV_TOLERANCE_PASS',eps,json.dumps(metrics),flush=True)
        outputs.append(dict(eps=eps,max_num_iter=800,strength=strength,records=records,
            fitting_hf_sd_hu=high_frequency_sd(baseline,fit[box],spacing),
            holdout_hf_sd_hu=high_frequency_sd(baseline,heldout[box],spacing)))
    publish(args.output,dict(completed=True,checks=outputs,elapsed_seconds=time.monotonic()-start,
        runner_sha256=sha(Path(__file__)),note='Numerical validation only; strength never retuned, primary outputs preserved.'))


if __name__=='__main__':
    main()

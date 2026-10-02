"""Independent short VAE controls; no training, detector, or clinical verdict.

Insertion uses a cached tumor-positive host, NOT a healthy subject. This is
engineering signal transport. All arms reconstruct entire native-grid volumes.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import MaisiConvolution

from prepare_msd import MODEL_HASH, sha
from prepare_texture_roi import prepare_roi
from run_maisi import load_model, publish


def insert_signal(control, labels, spacing, diameter_mm=8., contrast_hu=-20.):
    """Prespecified single8mm/-20HU insert, center chosen from source mask only."""
    if control.shape != labels.shape or not np.isfinite(control).all():
        raise ValueError('Invalid CT/grid')
    roi, _ = prepare_roi(labels, spacing)
    clearance = ndi.distance_transform_edt(np.pad(roi, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    center = np.unravel_index(int(np.argmax(clearance)), clearance.shape)
    radius = diameter_mm/2
    if clearance[center] <= radius:
        raise ValueError('No room for prespecified insert; do not shrink or crop')
    halo = np.ceil(radius/np.asarray(spacing)).astype(int)+1
    box = tuple(slice(max(0,c-h), min(n,c+h+1)) for c,h,n in zip(center,halo,labels.shape))
    grids = np.ogrid[tuple(slice(s.start,s.stop) for s in box)]
    squared = sum(((g-c)*sp)**2 for g,c,sp in zip(grids,center,spacing))
    local = squared <= radius**2
    mask = np.zeros(labels.shape, dtype=bool); mask[box] = local
    if not mask.any() or not roi[mask].all():
        raise ValueError('Insert escaped fixed source ROI')
    modified = control.copy(); modified[mask] += contrast_hu
    if np.any(modified[mask] < -1000) or np.any(modified[mask] > 1000):
        raise ValueError('Inserted signal would be clipped')
    return modified, mask, dict(center_voxel=list(map(int,center)), diameter_mm=diameter_mm,
                               contrast_hu=contrast_hu, voxels=int(mask.sum()),
                               host='tumor-positive MSD engineering host; NOT healthy/PDAC simulation')


def prepare(manifest):
    row = manifest['cases'][0]
    if manifest['model_sha256'] != MODEL_HASH or sha(Path(manifest['model'])) != MODEL_HASH:
        raise RuntimeError('Checkpoint mismatch')
    for kind in ('image','label'):
        if sha(Path(row[kind])) != row[kind+'_sha256']:
            raise RuntimeError('Input hash mismatch')
    images = [nib.as_closest_canonical(nib.load(row[k])) for k in ('image','label')]
    ct, label = images
    if ct.shape != label.shape or not np.allclose(ct.affine,label.affine,atol=1e-5,rtol=0):
        raise RuntimeError('Grid mismatch')
    if any(im.header.get_xyzt_units()[0] != 'mm' for im in images):
        raise RuntimeError('Unsupported units')
    matrix=ct.affine[:3,:3]; spacing=np.linalg.norm(matrix,axis=0)
    if not np.isfinite(ct.affine).all() or np.any(spacing <= 0) or not np.allclose(
            (matrix/spacing).T@(matrix/spacing),np.eye(3),atol=1e-5,rtol=0):
        raise RuntimeError('Unsupported affine')
    original=ct.get_fdata(dtype=np.float32); labels=np.asarray(label.dataobj)
    if not np.isfinite(original).all() or original.size > 32_000_000:
        raise RuntimeError('Invalid/oversized input')
    prepare_roi(labels,spacing)  # Strict label/spacing validation.
    return ct, np.clip(original,-1000,1000), labels, spacing


def decode(model, control, device, sample=False):
    pads=[(((-n)%4)//2,(-n)%4-(((-n)%4)//2)) for n in control.shape]
    x=np.pad((control+1000)/2000,pads)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start=time.monotonic()
    with torch.inference_mode():
        tensor=torch.from_numpy(x)[None,None].to(device)
        mu,sigma=model.encode(tensor); del tensor
        torch.manual_seed(20261001)
        latent=model.sampling(mu,sigma) if sample else mu
        output=model.decode(latent)
        if not torch.isfinite(output).all():
            raise RuntimeError('Nonfinite model output')
        array=output[0,0].float().cpu().numpy()
        del output,latent,mu,sigma
    torch.cuda.synchronize()
    report=dict(seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
    array=array[tuple(slice(lo,lo+n) for (lo,_),n in zip(pads,control.shape))]*2000-1000
    torch.cuda.empty_cache()
    return array,report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=('insertion','split4','posterior'),required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cpu-preflight',action='store_true')
    parser.add_argument('--save-volumes',action='store_true',help='Optional; default metrics-only to avoid shared-storage pressure')
    args=parser.parse_args()
    manifest_path=args.root/'small_followup_120/inputs_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    ct,control,labels,spacing=prepare(manifest)
    cached_control=nib.load(str(args.root/'small_results_3290494/pancreas_120/control.nii.gz'))
    if cached_control.shape!=ct.shape or not np.allclose(cached_control.affine,ct.affine,atol=1e-5,rtol=0):
        raise RuntimeError('Cached control grid mismatch')
    if not np.array_equal(cached_control.get_fdata(dtype=np.float32),control):
        raise RuntimeError('Cached control differs from current preprocessing')
    modified,mask,insert_info=insert_signal(control,labels,spacing)
    if importlib.metadata.version('monai') != '1.5.1':
        raise RuntimeError('MONAI version mismatch')
    model=load_model(manifest['model'])
    if args.mode=='split4':
        layers=[m for m in model.modules() if isinstance(m,MaisiConvolution)]
        if len(layers)!=36: raise RuntimeError('Unexpected architecture')
        for layer in layers: layer.num_splits=4
    if args.cpu_preflight:
        torch.set_num_threads(1)
        with torch.inference_mode():
            mu,sigma=model.encode(torch.zeros(1,1,8,64,8))
            latent=model.sampling(mu,sigma) if args.mode=='posterior' else mu
            result=model.decode(latent)
            if result.shape!=(1,1,8,64,8) or not torch.isfinite(result).all():
                raise RuntimeError('CPU contract failed')
        print('CPU_PREFLIGHT_OK',args.mode,json.dumps(insert_info),flush=True)
        return
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1 or args.output is None:
        raise RuntimeError('Require one scheduled GPU and exclusive output directory')
    if shutil.disk_usage(args.root).free < 2*1024**3:
        raise RuntimeError('Need2GiB headroom; do not fill shared quota')
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    publish(args.output/'provenance.json',dict(mode=args.mode,torch=torch.__version__,monai='1.5.1',
        gpu=torch.cuda.get_device_name(),manifest=manifest,precision='FP32',seed=20261001,
        cached_baseline_sha256=sha(args.root/'small_results_3290494/pancreas_120/reconstruction.nii.gz'),
        geometry='native-spacing whole volume; no crops/tiles',insert=insert_info,
        job=os.environ['SLURM_JOB_ID'],scientific_scope='engineering controls; no clinical/detection claim'))
    model.to('cuda:0')
    started=time.monotonic()
    if args.mode=='insertion':
        baseline,time0=decode(model,control,'cuda:0')
        altered,time1=decode(model,modified,'cuda:0')
        response=altered-baseline
        distances=ndi.distance_transform_edt(~mask,sampling=spacing)
        ring=(distances>=2)&(distances<=5)&~mask
        raw=float(response[mask].mean())
        background=float(response[ring].mean()) if ring.any() else None
        metrics=dict(reference_signal_hu=-20.,response_mean_hu=raw,
            raw_signed_retention=raw/-20.,ring_response_mean_hu=background,
            ring_corrected_retention=(raw-background)/-20. if background is not None else None,
            interpretation='paired nonlinear response; NOT matched-filter SNR or real-tumor recall')
        times=[time0,time1]
        save={'baseline':baseline,'inserted':altered}
    else:
        altered,time1=decode(model,control,'cuda:0',sample=args.mode=='posterior')
        baseline_image=nib.load(str(args.root/'small_results_3290494/pancreas_120/reconstruction.nii.gz'))
        if baseline_image.shape!=ct.shape or not np.allclose(baseline_image.affine,ct.affine,atol=1e-5,rtol=0):
            raise RuntimeError('Cached baseline grid mismatch')
        baseline=baseline_image.get_fdata(dtype=np.float32)
        error=np.abs(altered-baseline)
        metrics=dict(whole_image_mean_abs_difference_hu=float(error.mean()),
                     whole_image_max_abs_difference_hu=float(error.max()),
                     tumor_mean_abs_difference_hu=float(error[labels==2].mean()),
                     comparator='cached mean/splits1 result; not a same-allocation repeatability test',
                     interpretation='single-seed/config diagnostic, not quality or clinical claim')
        times=[time1]; save={args.mode:altered}
    if args.save_volumes:
        for name,array in save.items():
            image=nib.Nifti1Image(array,ct.affine,ct.header.copy()); image.set_data_dtype(np.float32)
            nib.save(image,args.output/(name+'.nii.gz'))
    publish(args.output/'metrics.json',dict(metrics=metrics,timings=times,
        rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    publish(args.output/'completion.json',dict(mode=args.mode,elapsed_seconds=time.monotonic()-started,
        completed=True,volumes_saved=args.save_volumes))
    print('CONTROL_COMPLETE',args.mode,json.dumps(metrics),flush=True)


if __name__=='__main__':
    main()

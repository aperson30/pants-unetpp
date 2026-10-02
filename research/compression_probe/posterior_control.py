"""One completed public patient: native/clip/mean/three fixed posterior seeds.

Whole-volume FP32 encode ONCE, decoder unchanged. Not diffusion-generated CT,
clinical sensitivity, irreversible erasure or an independent cohort result.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import data_utils as upstream

from fixed_crop_control import fixed_crop,same_grid
from judge_contracts import crop_slices,expand_map,postprocess_probability,tumor_crop_fraction
from judge_repeatability import backend,compare
from paired_judge_geometry import restore_native,score_tumor_maps
from paired_judge_pilot import detector,predict_image,save_json,sha,log_memory
from run_maisi import load_model

SEEDS=(0,1,2)  # locked before results, report ALL, never choose best


def sample_latent(model,mean,sigma,seed):
    if seed not in SEEDS or mean.shape!=sigma.shape or not torch.isfinite(mean).all() or not torch.isfinite(sigma).all() or (sigma<0).any():
        raise ValueError('Invalid locked seed/posterior parameters')
    torch.manual_seed(seed)
    return model.sampling(mean,sigma)  # pinned MONAI mu+randn_like(sigma)*sigma


def main():
    parser=argparse.ArgumentParser()
    for name in ('case','producer','models','vae','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--vae-sha',required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
        raise RuntimeError('Require one scheduled GPU')
    if not torch.__version__.startswith('2.10.0') or importlib.metadata.version('monai')!='1.5.1' or importlib.metadata.version('report-guided-annotation')!='0.3.4':
        raise ValueError('Pinned runtime mismatch')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8' or sha(args.vae)!=args.vae_sha:
        raise ValueError('VAE identity/deterministic environment mismatch')
    if sha(Path(upstream.__file__))!='5f09cb980619e1f459a0163e6808a1bc20188a466c000218719eff36cd1ad68a':
        raise ValueError('Publisher source mismatch')
    case=json.loads((args.case/'completion.json').read_text())
    producer=json.loads((args.producer/'completion.json').read_text())
    if case.get('completed') is not True or producer.get('completed') is not True or case['case']['fold']!=4:
        raise ValueError('Require existing completed held-out case')
    for name,key in (('image.nii.gz','ct_sha256'),('manual_label.nii.gz','label_sha256')):
        if sha(args.case/name)!=case[key]:
            raise ValueError('Case identity mismatch')
    free,total=torch.cuda.mem_get_info()
    if free<40_000_000_000:
        raise RuntimeError('Require40GB actual free CUDA; no crop fallback')
    args.output.mkdir(mode=0o700,exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    torch.use_deterministic_algorithms(False)
    os.environ['nnUNet_compile']='False'
    summary=dict(completed=False,clinical_recall=False,
        scope='one existing small patient, fixed publisher-native window, posterior-mechanism diagnostic',
        seeds=list(SEEDS),vae_sha256=args.vae_sha,ct_sha256=case['ct_sha256'],
        producer_sha256=sha(args.producer/'completion.json'),
        gpu_free_bytes=free,gpu_total_bytes=total,torch=torch.__version__,monai='1.5.1',
        geometry='native whole volume, RAS only, center padding4, no tiles/crop/interpolation in VAE',
        vae_precision='FP32 posterior mean+sigma, original normalization/decoder, sigma is standard deviation',
        vae_operations=[],results=[])
    try:
        native_nib=nib.load(args.case/'image.nii.gz')
        canonical=nib.as_closest_canonical(native_nib)
        data=canonical.get_fdata(dtype=np.float32)
        if native_nib.header.get_xyzt_units()[0]!='mm' or not np.isfinite(data).all():
            raise ValueError('Invalid CT/units')
        control=np.clip(data,-1000,1000)
        pads=[(((-n)%4)//2,(-n)%4-(((-n)%4)//2)) for n in data.shape]
        normalized=np.pad((control+1000)/2000,pads)
        if normalized.size>16_000_000:
            raise ValueError('Locked small-case padded16M cap; no larger-case rescue')
        model=load_model(str(args.vae)).cuda()
        torch.cuda.reset_peak_memory_stats()
        start=time.monotonic()
        with torch.inference_mode():
            tensor=torch.from_numpy(normalized)[None,None].cuda()
            mean,sigma=model.encode(tensor)
            del tensor
        torch.cuda.synchronize()
        summary['vae_operations'].append(dict(operation='encode_once',seconds=time.monotonic()-start))
        summary['posterior_sigma_mean']=float(sigma.mean())
        summary['posterior_sigma_max']=float(sigma.max())
        log_memory('after_encode')
        unpad=tuple(slice(lo,lo+n) for (lo,_),n in zip(pads,data.shape))
        def save_native(name,array):
            restored=restore_native(array,canonical,native_nib)
            image=nib.Nifti1Image(restored,native_nib.affine,native_nib.header.copy())
            image.set_data_dtype(np.float32)
            nib.save(image,args.output/(name+'.nii.gz'))
        save_native('control',control)
        for name,seed in (('posterior_mean',None),)+tuple(('posterior_seed'+str(s),s) for s in SEEDS):
            start=time.monotonic()
            with torch.inference_mode():
                latent=mean if seed is None else sample_latent(model,mean,sigma,seed)
                output=model.decode(latent)
                if not torch.isfinite(output).all():
                    raise ValueError('Nonfinite decoded volume')
                array=output[0,0].float().cpu().numpy()
                del output,latent
            torch.cuda.synchronize()
            save_native(name,array[unpad]*2000-1000)  # no output clipping
            del array
            summary['vae_operations'].append(dict(operation=name,seed=seed,seconds=time.monotonic()-start))
            log_memory(name)
            gc.collect();torch.cuda.empty_cache()
        summary['vae_peak_bytes']=torch.cuda.max_memory_allocated()
        del model,mean,sigma,data,control,normalized
        gc.collect();torch.cuda.empty_cache()
        log_memory('vae_released_before_detector')
        native=sitk.ReadImage(str(args.case/'image.nii.gz'),sitk.sitkFloat32)
        label=sitk.ReadImage(str(args.case/'manual_label.nii.gz'))
        same_grid(native,label)
        tumor=sitk.GetArrayFromImage(label)==1  # scoring-only
        window=next(r['crop_bounds'] for r in producer['results'] if r['arm']=='native')
        slices=crop_slices(tumor.shape,window)
        predictor=detector(args.models,'Dataset104_PANORAMA_baseline_PDAC_Detection',
            'nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres','checkpoint_best_panorama.pth',7)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=False)
        expected=backend();summary['detector_backend']=expected
        arms=('native','control','posterior_mean')+tuple('posterior_seed'+str(s) for s in SEEDS)
        for arm in arms:
            image=native if arm=='native' else sitk.ReadImage(str(args.output/(arm+'.nii.gz')),sitk.sitkFloat32)
            same_grid(image,native)
            folder=args.output/arm;folder.mkdir()
            started=time.monotonic()
            crop=fixed_crop(image,window)
            segmentation,probabilities=predict_image(predictor,crop)
            repeat_seg,repeat_prob=predict_image(predictor,crop)
            raw=probabilities[1].astype(np.float32)
            replay=compare(raw,repeat_prob[1],tumor[slices])
            replay['all_class_max_abs_probability_drift']=float(np.abs(probabilities-repeat_prob).max())
            replay['segmentation_labels_different']=int(np.count_nonzero(sitk.GetArrayFromImage(segmentation)!=sitk.GetArrayFromImage(repeat_seg)))
            save_json(folder/'repeatability.json',replay)
            if backend()!=expected or replay['all_class_max_abs_probability_drift']>1e-4 or replay['segmentation_labels_different']!=0:
                raise ValueError('Fresh same-process repeat gate failed; no relaxation')
            del repeat_prob,repeat_seg
            raw_full=expand_map(raw,tumor.shape,window)
            seg_path=folder/'stage2_segmentation.nii.gz';sitk.WriteImage(segmentation,str(seg_path))
            filtered=upstream.PostProcessing({'probabilities':probabilities},str(seg_path))
            own,mask=postprocess_probability(raw,sitk.GetArrayFromImage(segmentation))
            if not np.array_equal(filtered,own):
                raise ValueError('Postprocess parity mismatch')
            candidate_image,patient_score=upstream.GetFullSizDetectionMap(filtered,window,image)
            candidates=sitk.GetArrayFromImage(candidate_image)
            masked_full=expand_map(filtered,tumor.shape,window)
            row=score_tumor_maps(tumor,raw_full,masked_full,candidates,tumor_crop_fraction(tumor,window))
            if row['patient_max_candidate_score']!=patient_score:
                raise ValueError('Candidate score mismatch')
            row.update(arm=arm,crop_bounds=window,repeatability=replay,
                tumor_fraction_in_postprocess_mask=float(mask[tumor[slices]].mean()),
                seconds_including_repeat=time.monotonic()-started)
            np.savez_compressed(folder/'crop_maps.npz',raw_pdac=raw,masked_pdac=filtered,candidates=candidates[slices],postprocess_mask=mask)
            save_json(folder/'metrics.json',row)
            summary['results'].append(row)
            print('POSTERIOR_CONTROL_ARM_COMPLETE',json.dumps(row),flush=True)
            del probabilities,segmentation,raw,raw_full,masked_full,candidates,filtered,own,mask
            gc.collect();torch.cuda.empty_cache()
        summary['completed']=True
        save_json(args.output/'completion.json',summary)
    except Exception as error:
        summary['error']=repr(error)
        save_json(args.output/'failure.json',summary)
        raise


if __name__=='__main__':
    main()

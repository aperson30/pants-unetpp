"""NEW same-process image x publisher-window control; NOT historical replay.

Each of four cells has a fresh repeated prediction. Historical replay failure
is retained. No VAE, trained-weight change, GT-driven crop or clinical claim.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import torch
import data_utils as upstream

from fixed_crop_control import fixed_crop, same_grid
from judge_contracts import crop_slices, expand_map, postprocess_probability, tumor_crop_fraction
from judge_repeatability import backend, compare
from paired_judge_geometry import score_tumor_maps
from paired_judge_pilot import detector, predict_image, save_json, sha


def main():
    parser = argparse.ArgumentParser()
    for name in ('case', 'producer', 'models', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('One scheduled GPU required')
    if not torch.__version__.startswith('2.10.0') or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Frozen runtime mismatch')
    if sha(Path(upstream.__file__)) != '5f09cb980619e1f459a0163e6808a1bc20188a466c000218719eff36cd1ad68a':
        raise ValueError('Publisher source mismatch')
    case = json.loads((args.case/'completion.json').read_text())
    producer = json.loads((args.producer/'completion.json').read_text())
    if case.get('completed') is not True or producer.get('completed') is not True or case['case']['fold'] != 4:
        raise ValueError('Incomplete/out-of-fold case')
    for filename, key in (('image.nii.gz','ct_sha256'), ('manual_label.nii.gz','label_sha256')):
        if sha(args.case/filename) != case[key]:
            raise ValueError('Case identity mismatch')
    args.output.mkdir(mode=0o700, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['nnUNet_compile'] = 'False'
    images = dict(native=sitk.ReadImage(str(args.case/'image.nii.gz'),sitk.sitkFloat32),
                  reconstruction=sitk.ReadImage(str(args.producer/'reconstruction.nii.gz'),sitk.sitkFloat32))
    label = sitk.ReadImage(str(args.case/'manual_label.nii.gz'))
    for image in images.values():
        same_grid(image,label)
    tumor = sitk.GetArrayFromImage(label)==1  # scoring-only; never selects bounds
    bounds = {row['arm']:row['crop_bounds'] for row in producer['results'] if row['arm'] in images}
    if set(bounds) != set(images):
        raise ValueError('Missing publisher windows')
    predictor = detector(args.models,'Dataset104_PANORAMA_baseline_PDAC_Detection',
        'nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres','checkpoint_best_panorama.pth',7)
    # Set AFTER predictor construction, which enables benchmarking itself.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True,warn_only=False)
    expected_backend = backend()
    summary = dict(completed=False,clinical_recall=False,historical_replay_gate_passed=False,
        scope='posthoc one-patient same-process factorial diagnostic with fresh repeated controls',
        backend=expected_backend, torch=torch.__version__,cudnn=torch.backends.cudnn.version(),
        original_sha256=case['ct_sha256'],reconstruction_sha256=sha(args.producer/'reconstruction.nii.gz'),
        producer_sha256=sha(args.producer/'completion.json'),results=[])
    try:
        for crop_arm in ('native','reconstruction'):
            window=bounds[crop_arm]
            slices=crop_slices(tumor.shape,window)
            for image_arm in ('native','reconstruction'):
                image=images[image_arm]
                folder=args.output/(image_arm+'_crop_'+crop_arm)
                folder.mkdir()
                crop=fixed_crop(image,window)
                started=time.monotonic()
                segmentation,probabilities=predict_image(predictor,crop)
                raw=probabilities[1].astype(np.float32)
                repeat_segmentation,repeat_probabilities=predict_image(predictor,crop)
                repeat_raw=repeat_probabilities[1].astype(np.float32)
                replay=compare(raw,repeat_raw,tumor[slices])
                replay['all_class_max_abs_probability_drift']=float(np.abs(probabilities-repeat_probabilities).max())
                replay['segmentation_labels_different']=int(np.count_nonzero(
                    sitk.GetArrayFromImage(segmentation)!=sitk.GetArrayFromImage(repeat_segmentation)))
                save_json(folder/'repeatability.json',replay)
                if backend()!=expected_backend or replay['all_class_max_abs_probability_drift']>1e-4 or replay['segmentation_labels_different']!=0:
                    raise ValueError('Fresh same-process repeat gate failed; retain all cells, no relaxation')
                del repeat_segmentation,repeat_probabilities,repeat_raw
                raw_full=expand_map(raw,tumor.shape,window)
                seg_path=folder/'stage2_segmentation.nii.gz'
                sitk.WriteImage(segmentation,str(seg_path))
                filtered=upstream.PostProcessing({'probabilities':probabilities},str(seg_path))
                own,mask=postprocess_probability(raw,sitk.GetArrayFromImage(segmentation))
                if not np.array_equal(filtered,own):
                    raise ValueError('Mask parity mismatch')
                candidate_image,patient_score=upstream.GetFullSizDetectionMap(filtered,window,image)
                candidate_full=sitk.GetArrayFromImage(candidate_image)
                masked_full=expand_map(filtered,tumor.shape,window)
                row=score_tumor_maps(tumor,raw_full,masked_full,candidate_full,tumor_crop_fraction(tumor,window))
                if row['patient_max_candidate_score']!=patient_score:
                    raise ValueError('Candidate score parity mismatch')
                row.update(image_arm=image_arm,crop_arm=crop_arm,crop_bounds=window,
                    seconds_including_repeat=time.monotonic()-started,repeatability=replay,
                    tumor_fraction_in_postprocess_mask=float(mask[tumor[slices]].mean()))
                np.savez_compressed(folder/'crop_maps.npz',raw_pdac=raw,masked_pdac=filtered,
                    candidates=candidate_full[slices],postprocess_mask=mask)
                save_json(folder/'metrics.json',row)
                summary['results'].append(row)
                print('SAME_PROCESS_WINDOW_CELL_COMPLETE',json.dumps(row),flush=True)
                del probabilities,raw,raw_full,masked_full,candidate_full,filtered,own,mask,segmentation
                torch.cuda.empty_cache()
        summary['completed']=True
        save_json(args.output/'completion.json',summary)
    except Exception as error:
        summary['error']=repr(error)
        save_json(args.output/'failure.json',summary)
        raise


if __name__=='__main__':
    main()

"""One public-case three-arm diagnostic. NOT clinical recall or a cohort result.

Requires scheduled GPU, private unique output, frozen source/dependency hashes.
GT is used ONLY after predictions for diagnostic scoring; never for cropping.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

import data_utils as upstream
from judge_contracts import expand_map, postprocess_probability, tumor_crop_fraction
from judge_load_smoke import restricted_checkpoint_load
from paired_judge_geometry import restore_native, score_tumor_maps
from run_maisi import load_model


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(4*1024**2):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, result):
    with path.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


def detector(models, dataset, trainer, checkpoint_name, heads):
    folder = models / dataset / trainer
    path = folder / 'fold_4' / checkpoint_name
    metadata = json.loads((models / 'completion.json').read_text())
    entries = [entry for entry in metadata['results'] if entry['dataset'] == dataset]
    if len(entries) != 1 or sha(path) != entries[0]['sha256']:
        raise ValueError('Detector identity mismatch')
    checkpoint = restricted_checkpoint_load(path)
    plans = json.loads((folder / 'plans.json').read_text())
    dataset_json = json.loads((folder / 'dataset.json').read_text())
    if checkpoint['init_args']['fold'] != 4 or checkpoint['init_args']['plans'] != plans or checkpoint['init_args']['dataset_json'] != dataset_json:
        raise ValueError('Detector fold/metadata mismatch')
    pm = PlansManager(plans)
    cm = pm.get_configuration('3d_fullres')
    if pm.get_label_manager(dataset_json).num_segmentation_heads != heads:
        raise ValueError('Detector class mapping mismatch')
    network = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                    cm.network_arch_init_kwargs_req_import, 1, heads,
                                    allow_init=False, deep_supervision=False)
    network.load_state_dict(checkpoint['network_weights'], strict=True)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                               perform_everything_on_device=True, device=torch.device('cuda:0'),
                               verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    # nnU-Net's exported prediction path iterates and reloads these states,
    # even for one fold. None is accepted at initialization but fails later.
    predictor.manual_initialization(network.eval(), pm, cm, [checkpoint['network_weights']], dataset_json,
                                    checkpoint['trainer_name'], checkpoint['inference_allowed_mirroring_axes'])
    del checkpoint
    return predictor


def predict_image(predictor, image):
    array = sitk.GetArrayFromImage(image).astype(np.float32)[None]
    segmentation, probabilities = predictor.predict_single_npy_array(
        array, {'spacing': list(image.GetSpacing())[::-1]},
        save_or_return_probabilities=True)
    if segmentation.shape != array.shape[1:] or probabilities.shape[1:] != segmentation.shape:
        raise ValueError('Prediction export/native shape mismatch')
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite probabilities')
    mask = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    mask.CopyInformation(image)
    return mask, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--vae', type=Path, required=True)
    parser.add_argument('--vae-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-voxels', type=int, choices=(32_000_000, 33_000_000, 56_000_000),
                        default=32_000_000, help='Explicit full-volume resource cap; no resampling')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('One scheduled GPU required')
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    print('GPU_MEMORY_BYTES', dict(free=free_gpu, total=total_gpu), flush=True)
    required_free = {32_000_000: 0, 33_000_000: 65_000_000_000,
                     56_000_000: 110_000_000_000}[args.max_voxels]
    if free_gpu < required_free:
        raise RuntimeError(f'Whole-volume resource cap requires{required_free} free GPU bytes')
    if args.output.exists():
        raise FileExistsError('Refuse reused output directory')
    if importlib.metadata.version('monai') != '1.5.1' or importlib.metadata.version('report-guided-annotation') != '0.3.4':
        raise RuntimeError('Frozen dependency versions required')
    if not torch.__version__.startswith('2.10.0'):
        raise RuntimeError('Unexpected torch version')
    if sha(Path(upstream.__file__)) != '5f09cb980619e1f459a0163e6808a1bc20188a466c000218719eff36cd1ad68a':
        raise RuntimeError('Published crop/postprocessing source changed')
    case = json.loads((args.case / 'completion.json').read_text())
    if case['completed'] is not True or case['case']['fold'] != 4:
        raise ValueError('Incomplete or non-held-out case')
    image_path, label_path = args.case/'image.nii.gz', args.case/'manual_label.nii.gz'
    if sha(image_path) != case['ct_sha256'] or sha(label_path) != case['label_sha256'] or sha(args.vae) != args.vae_sha:
        raise ValueError('Input or VAE identity mismatch')
    args.output.mkdir(mode=0o700)
    os.environ['nnUNet_compile'] = 'False'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    native = nib.load(image_path)
    canonical = nib.as_closest_canonical(native)
    data = canonical.get_fdata(dtype=np.float32)
    if data.size > args.max_voxels or not np.isfinite(data).all() or native.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Invalid/oversized CT')
    control = np.clip(data, -1000, 1000)
    pads = [(((-n)%4)//2, (-n)%4-(((-n)%4)//2)) for n in data.shape]
    normalized = np.pad((control+1000)/2000, pads)
    if normalized.size > args.max_voxels:
        raise ValueError('Padded full volume exceeds explicit resource cap')
    model = load_model(str(args.vae)).cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        tensor = torch.from_numpy(normalized)[None, None].cuda()
        mean, sigma = model.encode(tensor)
        del tensor, sigma
        output = model.decode(mean)
        if not torch.isfinite(output).all():
            raise ValueError('Nonfinite VAE output')
        reconstruction = output[0, 0].float().cpu().numpy()
        del output, mean
    torch.cuda.synchronize()
    vae_seconds = time.monotonic()-started
    vae_peak = torch.cuda.max_memory_allocated()
    unpad = tuple(slice(lo, lo+n) for (lo, _), n in zip(pads, data.shape))
    reconstruction = reconstruction[unpad]*2000-1000
    del model, normalized
    gc.collect()
    torch.cuda.empty_cache()
    for name, array in [('control', control), ('reconstruction', reconstruction)]:
        restored = restore_native(array, canonical, native)
        restored_image = nib.Nifti1Image(restored, native.affine, native.header.copy())
        restored_image.set_data_dtype(np.float32)
        nib.save(restored_image, args.output/(name+'.nii.gz'))
    del data, control, reconstruction
    label = sitk.ReadImage(str(label_path))
    original = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    if label.GetSize() != original.GetSize() or not np.allclose(label.GetSpacing(), original.GetSpacing(), atol=1e-5, rtol=0) or not np.allclose(label.GetOrigin(), original.GetOrigin(), atol=1e-5, rtol=0) or not np.allclose(label.GetDirection(), original.GetDirection(), atol=1e-5, rtol=0):
        raise ValueError('Label/native physical grid mismatch')
    tumor = sitk.GetArrayFromImage(label) == 1
    pancreas = sitk.GetArrayFromImage(label) == 4
    first = detector(args.models, 'Dataset103_PANORAMA_baseline_Pancreas_Segmentation',
                     'nnUNetTrainer__nnUNetPlans__3d_fullres', 'checkpoint_final.pth', 2)
    second = detector(args.models, 'Dataset104_PANORAMA_baseline_PDAC_Detection',
                      'nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres',
                      'checkpoint_best_panorama.pth', 7)
    reports = []
    for arm, path in [('native', image_path), ('control', args.output/'control.nii.gz'),
                      ('reconstruction', args.output/'reconstruction.nii.gz')]:
        folder = args.output/arm
        folder.mkdir()
        image = sitk.ReadImage(str(path), sitk.sitkFloat32)
        if image.GetSize() != original.GetSize() or not np.allclose(image.GetSpacing(), original.GetSpacing(), atol=1e-5, rtol=0) or not np.allclose(image.GetOrigin(), original.GetOrigin(), atol=1e-5, rtol=0) or not np.allclose(image.GetDirection(), original.GetDirection(), atol=1e-5, rtol=0):
            raise ValueError('Arm geometry mismatch')
        start = time.monotonic()
        low = upstream.resample_img(image, (4.5, 4.5, 9.))
        low_mask, _ = predict_image(first, low)
        cropped, bounds = upstream.CropPancreasROI(image, low_mask, [100, 50, 15])
        stage2_mask, probabilities = predict_image(second, cropped)
        seg_path = folder/'stage2_segmentation.nii.gz'
        sitk.WriteImage(stage2_mask, str(seg_path))
        raw = probabilities[1].astype(np.float32)
        filtered = upstream.PostProcessing({'probabilities': probabilities}, str(seg_path))
        own_filtered, dilated = postprocess_probability(raw, sitk.GetArrayFromImage(stage2_mask))
        if not np.array_equal(filtered, own_filtered):
            raise ValueError('Postprocessing helper parity failed')
        candidate_image, patient_score = upstream.GetFullSizDetectionMap(filtered, bounds, image)
        candidate_map = sitk.GetArrayFromImage(candidate_image)
        raw_full = expand_map(raw, tumor.shape, bounds)
        masked_full = expand_map(filtered, tumor.shape, bounds)
        dilated_full = expand_map(dilated.astype(np.float32), tumor.shape, bounds) > 0
        result = score_tumor_maps(tumor, raw_full, masked_full, candidate_map,
                                  tumor_crop_fraction(tumor, bounds))
        if result['patient_max_candidate_score'] != patient_score:
            raise ValueError('Candidate score parity failed')
        full_mask = upstream.resample_img(low_mask, image.GetSpacing(), is_label=True,
                         out_size=list(image.GetSize()), out_origin=list(image.GetOrigin()),
                         out_direction=list(image.GetDirection()))
        predicted_pancreas = sitk.GetArrayFromImage(full_mask) == 1
        denom = int(predicted_pancreas.sum()+pancreas.sum())
        result.update(arm=arm, crop_bounds=bounds, detector_seconds=time.monotonic()-start,
                      tumor_fraction_in_postprocess_mask=float((dilated_full & tumor).sum()/tumor.sum()),
                      tumor_fraction_mask_excluded_within_crop=float(result['tumor_fraction_in_crop']-(dilated_full & tumor).sum()/tumor.sum()),
                      gt_label4_pancreas_dice=float(2*(predicted_pancreas & pancreas).sum()/denom) if denom else None)
        np.savez_compressed(folder/'tumor_diagnostic_maps.npz', raw_pdac=raw_full,
                            masked_pdac=masked_full, candidates=candidate_map,
                            postprocess_mask=dilated_full)
        sitk.WriteImage(low_mask, str(folder/'stage1_pancreas.nii.gz'))
        save_json(folder/'metrics.json', result)
        reports.append(result)
        print('ARM_COMPLETE', json.dumps(result), flush=True)
        del probabilities, raw_full, masked_full, dilated_full
        torch.cuda.empty_cache()
    save_json(args.output/'completion.json', dict(completed=True, clinical_recall=False,
              scope='one public out-of-fold validation-selected case, not external held-out test',
              vae_seconds=vae_seconds, vae_peak_bytes=vae_peak, detector_fold=4,
              max_voxels=args.max_voxels, native_shape=list(native.shape),
              posterior='mean FP32; native whole volume; no VAE crop/tile', results=reports))
    print('PAIRED_JUDGE_PILOT_COMPLETE_NOT_COHORT', flush=True)


if __name__ == '__main__':
    main()

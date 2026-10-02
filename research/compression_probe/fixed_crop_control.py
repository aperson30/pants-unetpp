"""Posthoc image x crop diagnostic on one already-completed public case.

No VAE rerun, training, GT-driven crop, or clinical recall interpretation.
Both crop windows come from frozen publisher stage-1 predictions. Native and
reconstructed images are each tested in BOTH windows; own-window reruns must
match saved maps before the cross-window scores are interpreted.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import torch

import data_utils as upstream
from judge_contracts import crop_slices, expand_map, postprocess_probability, tumor_crop_fraction
from paired_judge_geometry import score_tumor_maps
from paired_judge_pilot import detector, predict_image, sha, save_json


def same_grid(a, b):
    if a.GetSize() != b.GetSize() or any(not np.allclose(getattr(a, field)(), getattr(b, field)(),
            atol=1e-5, rtol=0) for field in ('GetSpacing', 'GetOrigin', 'GetDirection')):
        raise ValueError('Physical grid mismatch')


def fixed_crop(image, bounds):
    """Exactly the publisher's index slicing, with explicit bounds validation."""
    slices = crop_slices(tuple(reversed(image.GetSize())), bounds)
    return image[slices[2], slices[1], slices[0]]


def main():
    parser = argparse.ArgumentParser()
    for name in ('case', 'producer', 'models', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('Require one scheduled GPU')
    if not torch.__version__.startswith('2.10.0') or importlib.metadata.version('report-guided-annotation') != '0.3.4':
        raise RuntimeError('Frozen runtime mismatch')
    if sha(Path(upstream.__file__)) != '5f09cb980619e1f459a0163e6808a1bc20188a466c000218719eff36cd1ad68a':
        raise ValueError('Publisher implementation changed')
    producer = json.loads((args.producer/'completion.json').read_text())
    case = json.loads((args.case/'completion.json').read_text())
    if producer.get('completed') is not True or case.get('completed') is not True or case['case']['fold'] != 4:
        raise ValueError('Require complete, frozen out-of-fold input')
    for name, key in (('image.nii.gz', 'ct_sha256'), ('manual_label.nii.gz', 'label_sha256')):
        if sha(args.case/name) != case[key]:
            raise ValueError('Input identity mismatch')
    reports = {r['arm']: r for r in producer['results']}
    bounds = {arm: reports[arm]['crop_bounds'] for arm in ('native', 'reconstruction')}
    if bounds['native'] == bounds['reconstruction']:
        raise ValueError('This control requires the already-observed different windows')
    if args.output.exists():
        raise FileExistsError('Refuse reused output')
    args.output.mkdir(mode=0o700)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ['nnUNet_compile'] = 'False'
    label = sitk.ReadImage(str(args.case/'manual_label.nii.gz'))
    native = sitk.ReadImage(str(args.case/'image.nii.gz'), sitk.sitkFloat32)
    reconstruction = sitk.ReadImage(str(args.producer/'reconstruction.nii.gz'), sitk.sitkFloat32)
    same_grid(native, label)
    same_grid(native, reconstruction)
    tumor = sitk.GetArrayFromImage(label) == 1  # scoring ONLY
    predictor = detector(args.models, 'Dataset104_PANORAMA_baseline_PDAC_Detection',
        'nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres', 'checkpoint_best_panorama.pth', 7)
    rows = []
    # Own-window reproduction first: failure stops before any cross-window claim.
    for arm, crop_arm in (('native', 'native'), ('reconstruction', 'reconstruction'),
                          ('native', 'reconstruction'), ('reconstruction', 'native')):
        image = native if arm == 'native' else reconstruction
        window = bounds[crop_arm]
        folder = args.output/(arm+'_crop_'+crop_arm)
        folder.mkdir()
        started = time.monotonic()
        segmentation, probabilities = predict_image(predictor, fixed_crop(image, window))
        raw = probabilities[1].astype(np.float32)
        # Publisher mutates probabilities[1] during masking; preserve raw first.
        raw_full = expand_map(raw, tumor.shape, window)
        seg_path = folder/'stage2_segmentation.nii.gz'
        sitk.WriteImage(segmentation, str(seg_path))
        filtered = upstream.PostProcessing({'probabilities': probabilities}, str(seg_path))
        own_filtered, mask = postprocess_probability(raw, sitk.GetArrayFromImage(segmentation))
        if not np.array_equal(filtered, own_filtered):
            raise ValueError('Mask parity failed')
        candidate_image, patient_score = upstream.GetFullSizDetectionMap(filtered, window, image)
        candidates = sitk.GetArrayFromImage(candidate_image)
        masked_full = expand_map(filtered, tumor.shape, window)
        row = score_tumor_maps(tumor, raw_full, masked_full, candidates, tumor_crop_fraction(tumor, window))
        if row['patient_max_candidate_score'] != patient_score:
            raise ValueError('Candidate score mismatch')
        if arm == crop_arm:
            with np.load(args.producer/arm/'tumor_diagnostic_maps.npz', allow_pickle=False) as saved:
                delta = float(np.max(np.abs(raw_full-saved['raw_pdac'])))
            row['own_window_max_raw_probability_drift'] = delta
            if delta > 1e-4:
                save_json(folder/'replay_failure.json', dict(completed=False,
                    scientific_result=False, reason='own-window replay tolerance exceeded',
                    locked_tolerance=1e-4, diagnostic=row,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic))
                raise ValueError('Own-window replay exceeds prespecified 1e-4 engineering tolerance')
        row.update(image_arm=arm, crop_arm=crop_arm, crop_bounds=window,
                   detector_seconds=time.monotonic()-started,
                   tumor_fraction_in_postprocess_mask=float(expand_map(mask.astype(np.float32), tumor.shape, window)[tumor].mean()))
        save_json(folder/'metrics.json', row)
        rows.append(row)
        print('FIXED_CROP_ARM_COMPLETE', json.dumps(row), flush=True)
        del probabilities, raw, raw_full, masked_full, candidates, filtered, own_filtered, mask
        torch.cuda.empty_cache()
    save_json(args.output/'completion.json', dict(completed=True, clinical_recall=False,
        scope='posthoc one-patient image-by-publisher-window mechanism control; not cohort or clinical recall',
        source_ct_sha256=case['ct_sha256'], reconstruction_sha256=sha(args.producer/'reconstruction.nii.gz'),
        producer_completion_sha256=sha(args.producer/'completion.json'),
        replay_max_probability_tolerance=1e-4, results=rows))


if __name__ == '__main__':
    main()

"""Four same-input calls: publisher defaults twice, deterministic policy twice.

Engineering diagnostic only. No new scientific arms, relaxed historical gate,
VAE, training, threshold changes or GT-selected detector inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import SimpleITK as sitk
import torch

from fixed_crop_control import fixed_crop, same_grid
from paired_judge_pilot import detector, predict_image, sha, save_json


def compare(a, b, tumor):
    if a.shape != b.shape or a.shape != tumor.shape or not tumor.any():
        raise ValueError('Comparison geometry/GT mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite probabilities')
    difference = np.abs(a.astype(np.float32)-b.astype(np.float32))
    return dict(global_max_abs_probability_drift=float(difference.max()),
                global_mean_abs_probability_drift=float(difference.mean()),
                tumor_max_abs_probability_drift=float(difference[tumor].max()),
                tumor_mean_abs_probability_drift=float(difference[tumor].mean()),
                tumor_mean_signed_probability_change=float((a[tumor]-b[tumor]).mean()),
                engineering_gate_max_drift_1e4=bool(difference.max() <= 1e-4))


def backend():
    return dict(benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def main():
    parser = argparse.ArgumentParser()
    for name in ('case', 'producer', 'models', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('Require one scheduled GPU')
    if not torch.__version__.startswith('2.10.0') or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Frozen runtime/determinism setup required')
    if args.output.exists():
        raise FileExistsError('Refuse reused output')
    args.output.mkdir(mode=0o700)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['nnUNet_compile'] = 'False'
    case = json.loads((args.case/'completion.json').read_text())
    producer = json.loads((args.producer/'completion.json').read_text())
    if case.get('completed') is not True or producer.get('completed') is not True or case['case']['fold'] != 4:
        raise ValueError('Incomplete/out-of-fold metadata mismatch')
    for filename, key in (('image.nii.gz', 'ct_sha256'), ('manual_label.nii.gz', 'label_sha256')):
        if sha(args.case/filename) != case[key]:
            raise ValueError('Case identity mismatch')
    native = sitk.ReadImage(str(args.case/'image.nii.gz'), sitk.sitkFloat32)
    label = sitk.ReadImage(str(args.case/'manual_label.nii.gz'))
    same_grid(native, label)
    bounds = next(r['crop_bounds'] for r in producer['results'] if r['arm']=='native')
    crop = fixed_crop(native, bounds)
    from judge_contracts import crop_slices
    tumor = (sitk.GetArrayFromImage(label)==1)[crop_slices(tuple(reversed(native.GetSize())), bounds)]
    with np.load(args.producer/'native/tumor_diagnostic_maps.npz', allow_pickle=False) as maps:
        historical = maps['raw_pdac'][crop_slices(tuple(reversed(native.GetSize())), bounds)].copy()
    predictor = detector(args.models, 'Dataset104_PANORAMA_baseline_PDAC_Detection',
        'nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres', 'checkpoint_best_panorama.pth', 7)
    input_sha = hashlib.sha256(sitk.GetArrayFromImage(crop).tobytes()).hexdigest()
    summary = dict(completed=False, clinical_recall=False, scope='same-input engineering diagnostic only',
        torch=torch.__version__, cudnn_version=torch.backends.cudnn.version(),
        cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(0),
        input_crop_array_sha256=input_sha, producer_sha256=sha(args.producer/'completion.json'),
        historical_gate_unchanged=1e-4, calls=[], comparisons=[])
    previous = None
    default_reference = None
    try:
        for policy in ('publisher_default', 'deterministic_diagnostic'):
            deterministic = policy == 'deterministic_diagnostic'
            torch.backends.cudnn.benchmark = not deterministic
            torch.backends.cudnn.deterministic = deterministic
            torch.use_deterministic_algorithms(deterministic, warn_only=False)
            for repeat in (1, 2):
                flags_before = backend()
                started = time.monotonic()
                segmentation, probabilities = predict_image(predictor, crop)
                raw = probabilities[1].astype(np.float32)
                row = dict(policy=policy, repeat=repeat, seconds=time.monotonic()-started,
                    backend_before=flags_before, backend_after=backend(),
                    tumor_mean_raw_probability=float(raw[tumor].mean()),
                    tumor_max_raw_probability=float(raw[tumor].max()),
                    vs_historical=compare(raw, historical, tumor))
                if flags_before != backend():
                    raise ValueError('Backend policy changed inside prediction')
                if repeat == 2:
                    pair = compare(raw, previous, tumor)
                    summary['comparisons'].append(dict(policy=policy, comparison='same_policy_repeat2_vs1', **pair))
                if policy == 'publisher_default' and repeat == 1:
                    default_reference = raw.copy()
                if deterministic and repeat == 1:
                    summary['comparisons'].append(dict(policy=policy, comparison='deterministic_vs_default',
                        **compare(raw, default_reference, tumor)))
                previous = raw.copy()
                summary['calls'].append(row)
                save_json(args.output/(policy+'_repeat'+str(repeat)+'.json'), row)
                print('REPEATABILITY_CALL', json.dumps(row), flush=True)
                del probabilities, segmentation, raw
                torch.cuda.empty_cache()
        summary['completed'] = True
        save_json(args.output/'completion.json', summary)
    except Exception as error:
        summary['error'] = repr(error)
        save_json(args.output/'failure.json', summary)
        raise


if __name__ == '__main__':
    main()

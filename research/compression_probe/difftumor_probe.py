"""Full-context frozen raw-head diagnostic, NOT official/clinical detection.

Reuses six completed images. No GT fed to the model, mask model, adaptation,
downloads, AMP, crop, re-encode, automatic retry or unbounded inference.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
import monai
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from difftumor_geometry import prepare, invert_logits
from difftumor_cpu_preflight import SHA, SIZE

ARMS = ('native', 'control', 'posterior_mean', 'posterior_seed0',
        'posterior_seed1', 'posterior_seed2')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            value.update(block)
    return value.hexdigest()


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def load_judge(path):
    if Path(path).stat().st_size != SIZE or digest(path) != SHA:
        raise ValueError('Pinned checkpoint integrity mismatch')
    permitted = {'numpy.dtype', 'numpy.core.multiarray.scalar'}
    if not set(torch.serialization.get_unsafe_globals_in_checkpoint(path)).issubset(permitted):
        raise ValueError('Unexpected pickle globals')
    with torch.serialization.safe_globals([
            (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
            np.dtype, type(np.dtype('float32')), type(np.dtype('float64'))]):
        state = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    model = UNet(spatial_dims=3, in_channels=1, out_channels=3,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2).eval()
    if set(state) != set(model.state_dict()) or any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Invalid state keys/tensors')
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model


def freeze_inputs(case, producer, weights, output):
    case_info = json.loads((case / 'completion.json').read_text())
    prior = json.loads((producer / 'completion.json').read_text())
    if case_info.get('completed') is not True or prior.get('completed') is not True or prior.get('seeds') != [0, 1, 2]:
        raise ValueError('Require completed case and locked posterior producer')
    native = nib.load(case / 'image.nii.gz')
    label = nib.load(case / 'manual_label.nii.gz')
    if native.shape != label.shape or not np.allclose(native.affine, label.affine, atol=1e-5, rtol=0):
        raise ValueError('Original annotation geometry mismatch')
    if digest(case / 'image.nii.gz') != case_info['ct_sha256'] or digest(case / 'manual_label.nii.gz') != case_info['label_sha256']:
        raise ValueError('Original case identity mismatch')
    rows = []
    for arm in ARMS:
        path = case / 'image.nii.gz' if arm == 'native' else producer / (arm + '.nii.gz')
        image = nib.load(path)
        if image.shape != native.shape or not np.allclose(image.affine, native.affine, atol=1e-5, rtol=0):
            raise ValueError('Saved image geometry mismatch: ' + arm)
        if image.header.get_xyzt_units()[0] != 'mm' or np.prod(image.shape) > 16_000_000:
            raise ValueError('Invalid units or locked small-case cap exceeded')
        if not np.isfinite(image.get_fdata(dtype=np.float32)).all():
            raise ValueError('Nonfinite saved image')
        rows.append(dict(arm=arm, path=str(path), sha256=digest(path)))
    if digest(weights) != SHA or weights.stat().st_size != SIZE:
        raise ValueError('Weight integrity mismatch')
    save(output, dict(images=rows, weights=str(weights), weights_sha256=SHA,
                      label=str(case / 'manual_label.nii.gz'), label_sha256=case_info['label_sha256'],
                      native_shape=list(native.shape), native_affine=native.affine.tolist(),
                      producer_sha256=digest(producer / 'completion.json'),
                      ground_truth_input=False, purpose='raw tumor-head response only'))
    print('INPUT_MANIFEST_COMPLETE', digest(output), flush=True)


def predict_full(model, tensor, data, transform, shape, affine, sw_device='cuda'):
    with torch.inference_mode():
        logits = sliding_window_inference(tensor, (96, 96, 96), 1,
            model, overlap=.75, mode='gaussian', sw_device=sw_device, device='cpu')
        native = invert_logits(logits[0], data, transform, shape, affine)
        probabilities = torch.softmax(native.as_tensor(), dim=0)
    if not torch.isfinite(probabilities).all():
        raise ValueError('Nonfinite probabilities')
    return probabilities


def run(manifest, expected_sha, output):
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('Require one scheduled GPU; no login inference')
    if not torch.__version__.startswith('2.10.0') or monai.__version__ != '1.5.1':
        raise ValueError('Pinned runtime mismatch')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8' or digest(manifest) != expected_sha:
        raise ValueError('Deterministic environment/manifest mismatch')
    inputs = json.loads(manifest.read_text())
    if tuple(x['arm'] for x in inputs['images']) != ARMS or inputs['ground_truth_input']:
        raise ValueError('Input contract mismatch')
    for row in inputs['images']:
        if digest(row['path']) != row['sha256']:
            raise ValueError('Image changed after preflight')
    torch.set_num_threads(2)
    torch.manual_seed(20261002)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.mem_get_info()[0] < 8_000_000_000:
        raise RuntimeError('Insufficient free GPU; no alternative crop')
    output.mkdir(mode=0o700, exist_ok=False)
    summary = dict(completed=False, clinical_recall=False, official_postprocessing=False,
                   ground_truth_input=False, manifest_sha256=expected_sha,
                   checkpoint_sha256=SHA, torch=torch.__version__, monai=monai.__version__,
                   gpu=torch.cuda.get_device_name(), accumulation_device='cpu', precision='FP32',
                   overlap=.75, roi=[96, 96, 96], batch=1, mode='gaussian',
                   affine=inputs['native_affine'], native_shape=inputs['native_shape'], results=[])
    started = time.monotonic()
    try:
        model = load_judge(inputs['weights']).cuda()
        torch.cuda.reset_peak_memory_stats()
        for row in inputs['images']:
            arm_start = time.monotonic()
            image = nib.load(row['path'])
            array = image.get_fdata(dtype=np.float32)
            data, transform = prepare(array, image.affine)
            del array
            if data['image'].numel() > 48_000_000:
                raise ValueError('Resampled full-volume cap exceeded; no crop fallback')
            image_tensor = data['image'].as_tensor()[None].cpu()
            def predict():
                return predict_full(model, image_tensor, data, transform, image.shape, image.affine)
            probabilities = predict()
            repeat = predict()  # all arms, not just native/control
            drift = float(torch.max(torch.abs(probabilities - repeat)))
            labels = torch.argmax(probabilities, dim=0).to(torch.uint8)
            label_drift = int(torch.count_nonzero(labels != torch.argmax(repeat, dim=0)))
            record = dict(arm=row['arm'], image_sha256=row['sha256'],
                          prepared_shape=list(data['image'].shape),
                          all_class_repeat_max_abs=drift, labels_different=label_drift,
                          seconds_including_repeat=time.monotonic()-arm_start)
            save(output / (row['arm'] + '_repeat.json'), record)
            if drift > 1e-4 or label_drift:
                raise ValueError('Repeat gate failed; no tolerance relaxation')
            np.save(output / (row['arm'] + '_tumor.npy'), probabilities[2].numpy(), allow_pickle=False)
            np.save(output / (row['arm'] + '_labels.npy'), labels.numpy(), allow_pickle=False)
            record.update(probability_sha256=digest(output / (row['arm'] + '_tumor.npy')),
                          labels_sha256=digest(output / (row['arm'] + '_labels.npy')))
            summary['results'].append(record)
            print('RAW_HEAD_ARM_COMPLETE', json.dumps(record), flush=True)
            if row['arm'] == 'native' and 6*record['seconds_including_repeat'] + 60 > 540:
                raise RuntimeError('Real full-volume timing cannot safely fit fixed allocation')
            del probabilities, repeat, labels, image_tensor, data, transform
            gc.collect(); torch.cuda.empty_cache()
        summary.update(completed=True, seconds=time.monotonic()-started,
                       peak_gpu_bytes=torch.cuda.max_memory_allocated())
        save(output / 'completion.json', summary)
    except Exception as error:
        summary.update(error=repr(error), seconds=time.monotonic()-started)
        save(output / 'failure.json', summary)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('freeze', 'run'), required=True)
    for name in ('case', 'producer', 'weights', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--manifest-sha')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze_inputs(args.case, args.producer, args.weights, args.output)
    else:
        run(args.manifest, args.manifest_sha, args.output)


if __name__ == '__main__':
    main()

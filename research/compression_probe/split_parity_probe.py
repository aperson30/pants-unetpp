"""One real-case GPU implementation gate, not clinical quality or diffusion.

Compare split4 with the already-frozen split1 reconstruction. No new patient,
no changed precision/posterior/spacing, no tiles or outcome-based selection.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import nibabel as nib
import numpy as np
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import MaisiConvolution

from paired_judge_geometry import restore_native
from run_maisi import load_model


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4*1024**2):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path('/u/asanjeev/compression_probe_smalloutputs')
    case = root/'paired_judge_pilot_v1/case'
    baseline = root/'paired_judge_pilot_v2/results_3295395'
    output = root/'split_parity_v1'/('result_'+os.environ['SLURM_JOB_ID']+'.json')
    if output.exists() or torch.cuda.device_count() != 1:
        raise RuntimeError('One scheduled GPU and unique output required')
    meta = json.loads((case/'completion.json').read_text())
    prior = json.loads((baseline/'completion.json').read_text())
    if not meta['completed'] or not prior['completed'] or meta['case']['fold'] != 4:
        raise ValueError('Incomplete frozen reference')
    image_path = case/'image.nii.gz'
    weights = Path('/projects/bdyo/asanjeev/compression_probe_20261001/inputs/autoencoder_v1.pt')
    if sha(image_path) != meta['ct_sha256'] or sha(case/'manual_label.nii.gz') != meta['label_sha256'] or sha(weights) != '1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3':
        raise ValueError('Frozen input/model changed')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    native = nib.load(image_path)
    canonical = nib.as_closest_canonical(native)
    data = canonical.get_fdata(dtype=np.float32)
    pads = [(((-n)%4)//2, (-n)%4-(((-n)%4)//2)) for n in data.shape]
    normalized = np.pad((np.clip(data, -1000, 1000)+1000)/2000, pads)
    if normalized.size > 32_000_000 or not np.isfinite(normalized).all():
        raise ValueError('Invalid/oversized fixed case')
    model = load_model(str(weights))
    layers = [m for m in model.modules() if isinstance(m, MaisiConvolution)]
    if len(layers) != 36 or any(m.num_splits != 1 for m in layers):
        raise ValueError('Unexpected baseline convolution topology')
    for layer in layers:
        layer.num_splits = 4
    model.cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        tensor = torch.from_numpy(normalized)[None, None].cuda()
        mean, sigma = model.encode(tensor)
        del tensor, sigma
        decoded = model.decode(mean)
        if not torch.isfinite(decoded).all():
            raise ValueError('Nonfinite split reconstruction')
        reconstruction = decoded[0, 0].float().cpu().numpy()
        del decoded, mean
    torch.cuda.synchronize()
    seconds = time.monotonic()-started
    peak = torch.cuda.max_memory_allocated()
    unpad = tuple(slice(lo, lo+n) for (lo, _), n in zip(pads, data.shape))
    reconstruction = restore_native(reconstruction[unpad]*2000-1000, canonical, native)
    del model, data, normalized
    gc.collect()
    torch.cuda.empty_cache()
    reference_image = nib.load(baseline/'reconstruction.nii.gz')
    if reference_image.shape != native.shape or not np.allclose(reference_image.affine, native.affine, atol=1e-5, rtol=0):
        raise ValueError('Reference grid mismatch')
    reference = reference_image.get_fdata(dtype=np.float32)
    labels = nib.load(case/'manual_label.nii.gz')
    if labels.shape != native.shape or not np.allclose(labels.affine, native.affine, atol=1e-5, rtol=0):
        raise ValueError('Label grid mismatch')
    tumor = np.asarray(labels.dataobj) == 1
    delta = np.abs(reconstruction-reference)
    max_delta = float(delta.max())
    report = dict(completed=True, clinical_quality=False, changed_factor='MaisiConvolution.num_splits1to4 only',
        model_sha256=sha(weights), baseline_reconstruction_sha256=sha(baseline/'reconstruction.nii.gz'),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), split_layers=len(layers),
        precision='whole-volume FP32/posterior mean/native spacing/no tiles',
        numerical_gate_hu=0.1, numerical_gate_pass=max_delta <= 0.1,
        numerical_gate_scope='one-case implementation screen, not detector/clinical parity',
        max_abs_difference_hu=max_delta, mean_abs_difference_hu=float(delta.mean()),
        tumor_max_abs_difference_hu=float(delta[tumor].max()),
        tumor_mean_abs_difference_hu=float(delta[tumor].mean()),
        seconds=seconds, peak_gpu_bytes=peak,
        peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        baseline_seconds=prior['vae_seconds'], baseline_peak_gpu_bytes=prior['vae_peak_bytes'])
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('REAL_SPLIT_PARITY_COMPLETE_NOT_CLINICAL', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

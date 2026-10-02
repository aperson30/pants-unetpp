"""One/two-case whole-volume, posterior-MEAN VAE reconstruction screen.

No segmentation, no diffusion, no training; no sliding-window decoder tiles.
Requires an isolated MONAI 1.5.1 import path and explicit scheduled GPU.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

from prepare_msd import sha
from region_metrics import measure


def load_model(path):
    model = AutoencoderKlMaisi(spatial_dims=3, in_channels=1, out_channels=1,
        latent_channels=4, num_channels=(64,128,256), num_res_blocks=(2,2,2),
        norm_num_groups=32, norm_eps=1e-6, attention_levels=(False,False,False),
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
        use_checkpointing=False, use_convtranspose=False,
        norm_float16=False, num_splits=1, dim_split=1)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'unet_state_dict' in state:
        state = state['unet_state_dict']
    model.load_state_dict(state, strict=True)
    return model.eval()


def publish(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--max-cases', type=int, choices=(1,2), default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if sha(Path(manifest['model'])) != manifest['model_sha256']:
        raise RuntimeError('Weight hash mismatch')
    if importlib.metadata.version('monai') != '1.5.1':
        raise RuntimeError('Unexpected MONAI version')
    model = load_model(manifest['model'])
    if args.cpu_preflight:
        torch.set_num_threads(1)
        with torch.inference_mode():
            mean, sigma = model.encode(torch.zeros(1,1,8,8,8))
            output = model.decode(mean)
            if output.shape != (1,1,8,8,8) or not bool(torch.isfinite(output).all()):
                raise RuntimeError('CPU encode/decode smoke failed')
        print('CPU_STRICT_CHECKPOINT_LOAD_OK', flush=True)
        return
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('Requires scheduled allocation exposing exactly one GPU')
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0'); model.to(device)
    origin = time.monotonic()
    provenance = {'job': os.environ['SLURM_JOB_ID'], 'torch': torch.__version__,
                  'monai': importlib.metadata.version('monai'),
                  'gpu': torch.cuda.get_device_name(), 'precision': 'fp32 including normalization; norm_float16=False',
                  'posterior': 'mean only, sampled posterior not tested',
                  'geometry': 'RAS reorientation only, original spacing, center padding to multiple of 4',
                  'decoder': 'whole volume, num_splits=1, no tiles', 'manifest': manifest}
    publish(args.output / 'provenance.json', provenance)
    def stage(name):
        print('STAGE', name, 'rss_peak_kib', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'cuda_allocated', torch.cuda.memory_allocated(),
              'cuda_reserved', torch.cuda.memory_reserved(), flush=True)
    stage('model_loaded')
    for row in manifest['cases'][:args.max_cases]:
        if time.monotonic() - origin > 240:
            print('STOP_BEFORE_NEXT_CASE_TIME_BUDGET', flush=True); break
        if sha(Path(row['image'])) != row['image_sha256'] or sha(Path(row['label'])) != row['label_sha256']:
            raise RuntimeError('Input hash changed')
        ct = nib.as_closest_canonical(nib.load(row['image']))
        label = nib.as_closest_canonical(nib.load(row['label']))
        if ct.header.get_xyzt_units()[0] != 'mm':
            raise RuntimeError('CT spatial units must be mm')
        if ct.shape != label.shape or not np.allclose(ct.affine, label.affine, rtol=0, atol=1e-5):
            raise RuntimeError('Canonical image/label mismatch')
        matrix = ct.affine[:3,:3]; spacing = np.linalg.norm(matrix, axis=0)
        if not np.allclose((matrix / spacing).T @ (matrix / spacing), np.eye(3), atol=1e-5, rtol=0):
            raise RuntimeError('Sheared affine; no geometry repair permitted')
        original = ct.get_fdata(dtype=np.float32)
        labels = np.asarray(label.dataobj)
        if not np.isfinite(original).all() or not np.any(labels == 2):
            raise RuntimeError('No finite tumor-positive CT')
        control = np.clip(original, -1000, 1000)
        pads = [( ((-n) % 4)//2, ((-n) % 4)-(((-n) % 4)//2) ) for n in ct.shape]
        normalized = np.pad((control + 1000) / 2000, pads)
        # Bound input shape instead of silently using a cropped/tiled substitute.
        if normalized.size > 100_000_000:
            raise RuntimeError('Input too large for bounded first screen')
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        started = time.monotonic()
        stage('before_encode')
        with torch.inference_mode():
            tensor = torch.from_numpy(normalized)[None,None].to(device)
            mean, sigma = model.encode(tensor)
            del sigma, tensor
            stage('after_encode')
            reconstruction = model.decode(mean)
            stage('after_decode')
            if not bool(torch.isfinite(reconstruction).all()):
                raise RuntimeError('Nonfinite reconstruction')
            reconstructed = reconstruction[0,0].float().cpu().numpy()
            del mean, reconstruction
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
        slices = tuple(slice(lo, lo+n) for (lo,_), n in zip(pads, original.shape))
        # No clipping of model output: overshoot is evidence, not hidden.
        reconstructed = reconstructed[slices] * 2000 - 1000
        case_dir = args.output / ('pancreas_' + row['case']); case_dir.mkdir()
        stage('before_save_and_cpu_metrics')
        for name, array in [('control',control),('reconstruction',reconstructed)]:
            image = nib.Nifti1Image(array, ct.affine, ct.header.copy())
            image.set_data_dtype(np.float32); nib.save(image, case_dir / (name+'.nii.gz'))
        report = {'inference_seconds': elapsed, 'peak_allocated_bytes': peak,
                  'peak_reserved_bytes': reserved,
                  'preprocessing_damage': measure(original, control, labels, spacing, tumor_label=2),
                  'compression_damage': measure(control, reconstructed, labels, spacing, tumor_label=2)}
        publish(case_dir / 'metrics.json', report)
        # Fixed grayscale window; same physical grid/slice, GT overlay is location only.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        z = int(np.argmax(np.sum(labels == 2, axis=(0,1))))
        fig, axes = plt.subplots(1,3,figsize=(12,4))
        for ax, name, array in zip(axes, ['native','control','reconstruction'],[original,control,reconstructed]):
            ax.imshow(array[:,:,z].T, cmap='gray', vmin=-100, vmax=200, origin='lower',
                      aspect=float(spacing[1]/spacing[0]))
            ax.contour((labels[:,:,z] == 2).T, levels=[.5], colors='red', linewidths=.5)
            ax.set_title(name); ax.axis('off')
        fig.suptitle('GT outline locates lesion; not proof of reconstruction realism')
        fig.savefig(case_dir / 'comparison.png', dpi=140); plt.close(fig)
        print('CASE_DONE',row['case'],json.dumps(report),flush=True)
        del original,control,labels,normalized,reconstructed
        torch.cuda.empty_cache()
    publish(args.output / 'completion.json', {'elapsed_seconds': time.monotonic()-origin,
            'completed_cases': len(list(args.output.glob('pancreas_*/metrics.json')))})


if __name__ == '__main__':
    main()

"""Pinned public checkpoint integrity and strict CPU compatibility, not inference.

No CUDA, unrestricted pickle, clinical data, or dependency installation.
Downloads exactly one 19 MB checkpoint; no retries or overwrite of bad files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '089ba0f3f94a7858603a55106791d7d977d7bc0b'
SIZE = 19260008
SHA = 'c3d7ca09aa57ce20e3537c5dfae611b4759bbd000f626fcb2d4bb180b1090d5d'
URL = ('https://huggingface.co/MrGiovanni/DiffTumor/resolve/' + REVISION
       + '/SegmentationModel/unet_synt_pancreas_tumors.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    path = args.directory / 'unet_synt_pancreas_tumors.pt'
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=30) as response, path.open('xb') as target:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > SIZE:
                    raise RuntimeError('Download exceeds pinned size')
                target.write(block)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != SIZE or sha != SHA:
        raise RuntimeError('Checkpoint integrity mismatch; no load attempted')
    import torch
    import monai
    from monai.networks.nets import UNet
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    permitted = {'numpy.dtype', 'numpy.core.multiarray.scalar'}
    if not set(unsafe).issubset(permitted):
        raise RuntimeError('Unexpected pickle globals: ' + repr(unsafe))
    # Static inspection found only legacy NumPy scalar/dtype metadata.
    # Narrow allowlist; never allow arbitrary globals or weights_only=False.
    import numpy as np
    with torch.serialization.safe_globals([
            (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
            np.dtype, type(np.dtype('float32')), type(np.dtype('float64'))]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint['state_dict']
    model = UNet(spatial_dims=3, in_channels=1, out_channels=3,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2).cpu().eval()
    if set(state) != set(model.state_dict()):
        raise RuntimeError('Exact state keys differ; no prefix stripping allowed')
    if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError('Non-tensor or nonfinite state')
    model.load_state_dict(state, strict=True)
    # Architecture check only: 32 cube prevents instance-norm singleton at bottleneck.
    with torch.inference_mode():
        output = model(torch.zeros((1, 1, 32, 32, 32), device='cpu'))
    if tuple(output.shape) != (1, 3, 32, 32, 32) or not torch.isfinite(output).all():
        raise RuntimeError('CPU output contract failed')
    result = dict(checkpoint_revision=REVISION, checkpoint_sha256=sha,
                  bytes=SIZE, torch=torch.__version__, monai=monai.__version__,
                  checkpoint_fields=sorted(checkpoint),
                  state_tensors=len(state), parameters=sum(p.numel() for p in model.parameters()),
                  unsafe_globals=unsafe, strict_load=True, device='cpu',
                  tiny_forward_shape=list(output.shape), finite_output=True,
                  clinical_inference=False, gpu_hours=0,
                  end_to_end_ready=False,
                  blockers=['Organ mask producer and checkpoint fold/provenance unresolved',
                            'Full transform/native-grid regression not yet implemented'])
    with (args.directory / 'cpu_preflight.json').open('x') as target:
        json.dump(result, target, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

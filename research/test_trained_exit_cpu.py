"""Strict saved-checkpoint reuse and tiny-input execution contracts, CPU only.

Not patient inference, patch-size parity, GPU timing, or lesion quality evidence.
Run in a separate local environment; never modifies source weights or trainers.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'unetpp_port'))
from unet_plusplus import UNetPlusPlus
from isolated_unetpp_exit import forward_exit

EXPECTED = '81f9598fb86d90e18676c89eb3cc9a91379a7cb932e35cfad116e54bbf52d4a5'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        raise RuntimeError('Refusing report overwrite')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(731)
    start = time.monotonic()
    path = args.model/'checkpoint_final.pth'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != EXPECTED:
        raise RuntimeError('Trusted saved checkpoint hash mismatch')
    # This is our own hash-verified training artifact; pickle is never accepted
    # from a new download. The state includes optimizer metadata.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint['current_epoch'] != 1000 or
            checkpoint['trainer_name'] != 'nnUNetTrainerUNetPlusPlusSparseValidation'):
        raise RuntimeError('Wrong checkpoint provenance')
    plans = json.loads((args.model/'plans.json').read_text())
    dataset = json.loads((args.model/'dataset.json').read_text())
    debug = json.loads((args.model/'debug.json').read_text())
    flags = [debug['enable_deep_supervision'], debug['skip_shallowest_deep_supervision_head']]
    if not all(value is True or value == 'True' for value in flags):
        raise RuntimeError('Wrong supervision provenance')
    kwargs = dict(plans['configurations']['3d_fullres']['architecture']['arch_kwargs'])
    allowed = {'conv_op': ('torch.nn.modules.conv.Conv3d', torch.nn.Conv3d),
               'norm_op': ('torch.nn.modules.instancenorm.InstanceNorm3d', torch.nn.InstanceNorm3d),
               'nonlin': ('torch.nn.LeakyReLU', torch.nn.LeakyReLU)}
    for key, (name, obj) in allowed.items():
        if kwargs[key] != name:
            raise RuntimeError('Unexpected architecture import')
        kwargs[key] = obj
    if kwargs['dropout_op'] is not None:
        raise RuntimeError('Unexpected dropout')
    if dataset['labels'].get('pancreatic_lesion') != 28:
        raise RuntimeError('Tumor label mapping differs')
    network = UNetPlusPlus(input_channels=1, num_classes=29, deep_supervision=True,
                          skip_shallowest_deep_supervision_head=True, **kwargs).eval()
    state = checkpoint['network_weights']
    if state and all(k.startswith('_orig_mod.') for k in state):
        state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
    network.load_state_dict(state, strict=True)
    del checkpoint, state
    x = torch.randn(1, 1, 32, 32, 32)
    rows = []
    with torch.inference_mode():
        full = network(x)
        if len(full) != 4:
            raise RuntimeError('Wrong full-resolution branch count')
        for depth in range(2, 6):
            out = forward_exit(network, x, depth)
            reference = full[5-depth]
            equal = torch.equal(out, reference)
            difference = float((out-reference).abs().max())
            if not equal or not torch.isfinite(out).all():
                raise RuntimeError(f'Depth{depth} contract failed: maxdiff{difference}')
            rows.append({'depth': depth, 'bit_identical': equal, 'max_abs_diff': difference})
        # Default production inference uses only deepest logits. Verify retained
        # DS deepest branch equals that inference output on the same own instance.
        network.decoder.deep_supervision = False
        deepest_only = network(x)
        if not torch.equal(deepest_only, full[0]):
            raise RuntimeError('DS deepest differs from single-output reference')
    report = {'completed': True, 'torch': torch.__version__, 'device': 'cpu',
              'threads': 1, 'checkpoint_sha256': EXPECTED,
              'strict_weights_loaded': True, 'full_architecture': True,
              'synthetic_input_shape': list(x.shape), 'branches': rows,
              'ds_deepest_equals_single_output': True,
              'elapsed_seconds': time.monotonic()-start,
              'scope': 'Saved trained weights, tiny synthetic CPU-input contract only',
              'limits': ['Not full training patch or patient input',
                         'No GPU/BF16/TTA/sliding-window/resampling parity',
                         'No tumor quality or speedup conclusion']}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

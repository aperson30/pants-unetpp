"""Allocated-compute-only load/finite-output check; NOT a clinical experiment.

No unrestricted pickle fallback, no downloads, no environment modification.
Expects completed hash-verified fold4 transfer and explicit unique result path.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(4*1024**2):
            value.update(block)
    return value.hexdigest()


def restricted_checkpoint_load(path):
    # Static audit of these two official files found only old NumPy scalar/dtype
    # metadata. Never allow arbitrary globals or use weights_only=False.
    permitted = {'numpy.core.multiarray.scalar', 'numpy._core.multiarray.scalar', 'numpy.dtype'}
    observed = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if not observed.issubset(permitted):
        raise RuntimeError('Unexpected checkpoint metadata types: ' + repr(sorted(observed)))
    entries = [(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
               (np._core.multiarray.scalar, 'numpy._core.multiarray.scalar'),
               np.dtype, type(np.dtype('float32')), type(np.dtype('float64'))]
    with torch.serialization.safe_globals(entries):
        return torch.load(path, map_location='cpu', weights_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run only in a scheduled compute allocation')
    if args.output.exists():
        raise FileExistsError('Refusing existing smoke-result path')
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Require exactly one allocated CUDA GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261002)
    completion = json.loads((args.models / 'completion.json').read_text())
    if completion.get('completed') is not True or len(completion.get('results', [])) != 2:
        raise RuntimeError('Require completed two-model transfer')
    records = []
    for item in completion['results']:
        dataset = item['dataset']
        expected = {
            'Dataset103_PANORAMA_baseline_Pancreas_Segmentation':
                ('nnUNetTrainer__nnUNetPlans__3d_fullres', 'checkpoint_final.pth', 2),
            'Dataset104_PANORAMA_baseline_PDAC_Detection':
                ('nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres',
                 'checkpoint_best_panorama.pth', 7),
        }
        if dataset not in expected or item['fold'] != 4:
            raise RuntimeError('Unexpected dataset/fold')
        trainer, checkpoint_name, heads = expected[dataset]
        folder = args.models / dataset / trainer
        path = folder / 'fold_4' / checkpoint_name
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise RuntimeError('Checkpoint integrity mismatch')
        checkpoint = restricted_checkpoint_load(path)
        if checkpoint['init_args']['fold'] != 4:
            raise RuntimeError('Checkpoint header fold mismatch')
        plans = json.loads((folder / 'plans.json').read_text())
        dataset_json = json.loads((folder / 'dataset.json').read_text())
        if checkpoint['init_args']['plans'] != plans or checkpoint['init_args']['dataset_json'] != dataset_json:
            raise RuntimeError('Checkpoint/archive metadata mismatch')
        manager = PlansManager(plans)
        configuration = manager.get_configuration('3d_fullres')
        if manager.get_label_manager(dataset_json).num_segmentation_heads != heads:
            raise RuntimeError('Head label mapping mismatch')
        network = get_network_from_plans(
            configuration.network_arch_class_name,
            configuration.network_arch_init_kwargs,
            configuration.network_arch_init_kwargs_req_import,
            input_channels=1, output_channels=heads, allow_init=False,
            deep_supervision=False)
        network.load_state_dict(checkpoint['network_weights'], strict=True)
        del checkpoint
        network.eval().cuda()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.inference_mode():
            output = network(torch.zeros((1, 1, 16, 64, 64), device='cuda'))
            if output.shape != (1, heads, 16, 64, 64) or not torch.isfinite(output).all():
                raise RuntimeError('Invalid synthetic network output')
        torch.cuda.synchronize()
        record = dict(dataset=dataset, fold=4, strict_state_load=True,
                      checkpoint_sha256=item['sha256'], synthetic_output_finite=True,
                      output_shape=list(output.shape), forward_seconds=time.monotonic()-start,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated())
        records.append(record)
        print(json.dumps(record), flush=True)
        del network, output
        gc.collect()
        torch.cuda.empty_cache()
    if len({record['dataset'] for record in records}) != 2:
        raise RuntimeError('Duplicate dataset records')
    with args.output.open('x', encoding='utf8') as target:
        json.dump(dict(completed=True, clinical_inference=False,
                       torch_version=torch.__version__, results=records), target, indent=2)
    print('JUDGE_LOAD_SMOKE_COMPLETE_NOT_CLINICAL', flush=True)


if __name__ == '__main__':
    main()

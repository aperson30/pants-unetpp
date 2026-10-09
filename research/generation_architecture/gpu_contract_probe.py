"""Bounded random-weight compatibility/timing probe; no image-quality evidence."""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from conditional_nested_denoiser import ConditionalNestedDenoiser


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated job cannot access CUDA')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', '.', '-v'],
                   cwd=Path(__file__).parent, check=True, timeout=40)
    device = torch.device('cuda:0')
    model = ConditionalNestedDenoiser().to(device).eval()
    x = torch.randn(1, 8, 64, 64, device=device)
    context = torch.randn(1, 77, 768, device=device)
    original = x.clone()
    with torch.no_grad():
        all_heads = model(x, 500, context, all_heads=True)
        for head, expected in enumerate(all_heads, 1):
            actual = model(x, 500, context, head=head).sample
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
    timings = {}
    with torch.no_grad():
        for label, kwargs in [('shallow', {'head': 1}), ('deepest', {'head': 2}),
                              ('all_heads', {'all_heads': True})]:
            for _ in range(2):
                model(x, 500, context, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            for _ in range(5):
                model(x, 500, context, **kwargs)
            torch.cuda.synchronize()
            timings[label] = {'seconds_per_forward': (time.perf_counter() - tick) / 5,
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    model.train()
    target = torch.randn(1, 4, 64, 64, device=device)
    gradient_results = {}
    for dtype in (torch.float32, torch.bfloat16):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            outputs = model(x, 500, context, all_heads=True)
            loss = sum((output.float() - target).square().mean() for output in outputs) / len(outputs)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Missing or nonfinite parameter gradients')
        gradient_results[str(dtype)] = float(loss.detach())
    torch.cuda.synchronize()
    report = {'status': 'PASS', 'job_id': os.environ.get('SLURM_JOB_ID'),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'cpu_arch': platform.machine(),
              'gpu': torch.cuda.get_device_name(), 'parameters': sum(p.numel() for p in model.parameters()),
              'timings': timings, 'finite_gradient_losses': gradient_results,
              'elapsed_seconds': time.monotonic() - started,
              'scope': 'random-weight prototype only; no SMILE checkpoint, data, generation quality or end-to-end speedup'}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'FAIL', 'type': type(error).__name__, 'message': str(error)}), flush=True)
        raise

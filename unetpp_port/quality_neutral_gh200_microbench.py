"""PyTorch-only microbenchmark for the quality-neutral trainer plumbing at the PanTS shape."""
import argparse
import json
import time

import torch


PATCH = (64, 160, 224)
BATCH = 4
CLASSES = 29


def timed(fn, warmup=3, repeats=10):
    for _ in range(warmup):
        result = fn()
        del result
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(repeats):
        result = fn()
        del result
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / repeats
    return elapsed, torch.cuda.max_memory_allocated() / 1024 ** 3


def dense_counts(prediction, target):
    prediction_onehot = torch.zeros(
        (BATCH, CLASSES, *PATCH), device='cuda', dtype=torch.float16
    )
    prediction_onehot.scatter_(1, prediction[:, None], 1)
    target_onehot = torch.zeros_like(prediction_onehot, dtype=torch.bool)
    target_onehot.scatter_(1, target.long(), 1)
    axes = (0, 2, 3, 4)
    tp = (prediction_onehot * target_onehot).sum(axes)
    fp = (prediction_onehot * ~target_onehot).sum(axes)
    fn = ((1 - prediction_onehot) * target_onehot).sum(axes)
    tn = ((1 - prediction_onehot) * ~target_onehot).sum(axes)
    return tp, fp, fn, tn


def bincount_counts(prediction, target):
    prediction = prediction.reshape(-1).long()
    target = target.reshape(-1).long()
    predicted_count = torch.bincount(prediction, minlength=CLASSES)[:CLASSES]
    target_count = torch.bincount(target, minlength=CLASSES)[:CLASSES]
    tp = torch.bincount(target[prediction == target], minlength=CLASSES)[:CLASSES]
    return tp, predicted_count - tp, target_count - tp


def benchmark_counts(mode):
    generator = torch.Generator(device='cuda').manual_seed(123)
    prediction = torch.randint(CLASSES, (BATCH, *PATCH), device='cuda', generator=generator)
    target = torch.randint(CLASSES, (BATCH, 1, *PATCH), device='cuda', generator=generator)
    fn = (lambda: dense_counts(prediction, target)) if mode == 'dense' else (
        lambda: bincount_counts(prediction, target)
    )
    seconds, peak_gib = timed(fn)
    return {'mode': mode, 'seconds': seconds, 'peak_allocated_gib': peak_gib}


def benchmark_targets(mode):
    source = torch.randint(CLASSES, (BATCH, 1, *PATCH), dtype=torch.int16, pin_memory=True)
    sources = [source.clone() for _ in range(4)]

    def transfer_all():
        return [item.to('cuda', non_blocking=True) for item in sources]

    def transfer_one():
        moved = sources[0].to('cuda', non_blocking=True)
        return [moved] * len(sources)

    seconds, peak_gib = timed(transfer_all if mode == 'target_all' else transfer_one)
    return {'mode': mode, 'seconds': seconds, 'peak_allocated_gib': peak_gib}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('dense', 'bincount', 'target_all', 'target_one'))
    args = parser.parse_args()
    torch.cuda.set_device(0)
    result = benchmark_counts(args.mode) if args.mode in ('dense', 'bincount') else benchmark_targets(args.mode)
    result.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), patch=PATCH, batch=BATCH)
    print(json.dumps(result), flush=True)

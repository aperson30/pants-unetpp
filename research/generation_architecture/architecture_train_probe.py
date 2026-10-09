"""Matched-interface tiny-fixture cost screen; no CT training/quality claim."""
import gc
import json
import os
import statistics
import time

import torch

from conditional_nested_denoiser import ConditionalNestedDenoiser
from conditional_plain_denoiser import ConditionalPlainDenoiser


def probe(label):
    torch.manual_seed(1729)
    factory = ConditionalPlainDenoiser if label == "plain" else ConditionalNestedDenoiser
    model = factory().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    # Model sizes consume different initialization draws: reseed the fixture
    # separately so every architecture sees exactly the same tensors.
    torch.manual_seed(99)
    x = torch.randn(1, 8, 64, 64, device="cuda")
    context = torch.randn(1, 77, 768, device="cuda")
    target = torch.randn(1, 4, 64, 64, device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if label == "nested_all":
                outputs = model(x, 500, context, all_heads=True)
            else:
                outputs = (model(x, 500, context).sample,)
            # Same mean-MSE scale; normalized equal-weight DS is explicit.
            loss = sum((out.float() - target).square().mean() for out in outputs) / len(outputs)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(3):
        loss = step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(8):
        tick = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - tick)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Nonfinite gradient {name}")
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Nonfinite parameter {name}")
    expected_missing = {"heads.1.weight", "heads.1.bias"} if label == "nested_deepest" else set()
    if set(missing) != expected_missing:
        raise RuntimeError(f"Unexpected missing gradients: {missing}")
    report = {"parameters": sum(p.numel() for p in model.parameters()),
              "active_parameters": sum(p.numel() for p in model.parameters() if p.grad is not None),
              "median_train_step_seconds": statistics.median(samples),
              "samples_seconds": samples, "last_random_target_loss": float(loss),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "expected_unused_parameters": missing}
    return report


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation missing")
    results = {}
    failures = {}
    for label in ("plain", "nested_deepest", "nested_all"):
        try:
            results[label] = probe(label)
        except Exception as error:
            failures[label] = f"{type(error).__name__}: {error}"
        finally:
            # No tensor/optimizer accumulation across configs in this own process.
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps({"job_id": os.getenv("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "scope": "tiny random fixture; equal widths NOT equal capacity; not quality/end-to-end speed",
        "precision": "bf16 autocast, FP32 weights", "batch_size": 1,
        "results": results, "failures": failures}, indent=2), flush=True)
    if failures:
        raise RuntimeError("At least one config failed; no automatic retry")


if __name__ == "__main__":
    main()

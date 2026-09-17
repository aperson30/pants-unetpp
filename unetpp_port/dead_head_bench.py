"""Kill-fast GB10 benchmark for the UNet++ dead segmentation-head optimization."""
import gc
import json
import os
import statistics
import time

import torch
from torch import nn

from smoke_test import build


DEVICE = "cuda"
N_STAGES = int(os.environ.get("N_STAGES", "6"))
PATCH_SIZE = tuple(int(i) for i in os.environ.get("PATCH_SIZE", "32,64,64").split(","))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
WARMUP = int(os.environ.get("WARMUP", "5"))
STEPS = int(os.environ.get("STEPS", "20"))


def mem_available_gib():
    with open("/proc/meminfo") as f:
        fields = {line.split(":", 1)[0]: int(line.split()[1]) for line in f}
    return fields["MemAvailable"] / 1024 / 1024


def loss_for(outputs, target, variant, ce):
    if variant.startswith("ds_off"):
        output = outputs[0] if isinstance(outputs, list) else outputs
        return ce(output, target)

    # Stock single-GPU nnU-Net weights for five outputs are [1, .5, .25, .125, 0], normalized.
    output_count = N_STAGES - 1
    weights = [1 / (2 ** i) for i in range(output_count)]
    weights[-1] = 0
    normalizer = sum(weights)
    return sum((weights[i] / normalizer) * ce(output, target)
               for i, output in enumerate(outputs) if weights[i] != 0)


def bench(variant):
    legacy = variant.endswith("legacy")
    deep_supervision = variant.startswith("ds_on") or legacy
    model = build(N_STAGES, deep_supervision).to(DEVICE).train()
    if variant == "ds_on_optimized":
        model.decoder.skip_shallowest_deep_supervision_head = True

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()
    x = torch.randn((BATCH_SIZE, 1, *PATCH_SIZE), device=DEVICE)
    target = torch.randint(0, 28, (BATCH_SIZE, *PATCH_SIZE), device=DEVICE)
    timings = []
    torch.cuda.reset_peak_memory_stats()

    for step in range(WARMUP + STEPS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(x)
            loss = loss_for(outputs, target, variant, ce)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= WARMUP:
            timings.append(elapsed)

    timings.sort()
    result = {
        "variant": variant,
        "n_stages": N_STAGES,
        "patch_size": PATCH_SIZE,
        "physical_batch_size": BATCH_SIZE,
        "warmup": WARMUP,
        "steps": STEPS,
        "mean_s": statistics.mean(timings),
        "median_s": statistics.median(timings),
        "p95_s": timings[min(len(timings) - 1, int(0.95 * len(timings)))],
        "min_s": min(timings),
        "max_s": max(timings),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
        "mem_available_gib": mem_available_gib(),
    }
    del model, optimizer, scaler, ce, x, target, outputs, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    assert torch.cuda.is_available()
    os.environ.setdefault("nnUNet_compile", "false")
    print(json.dumps({
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "cuda_capability": torch.cuda.get_device_capability(0),
        "start_mem_available_gib": mem_available_gib(),
    }), flush=True)
    results = []
    # Interleave legacy/optimized order so one arm does not always get the cold or hot device.
    for variant in ("ds_off_legacy", "ds_off_optimized",
                    "ds_on_optimized", "ds_on_legacy"):
        print(f"START {variant}", flush=True)
        try:
            result = bench(variant)
        except torch.cuda.OutOfMemoryError as error:
            result = {"variant": variant, "ok": False, "error": "OOM", "detail": str(error)}
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            result = {"variant": variant, "ok": False,
                      "error": type(error).__name__, "detail": str(error)}
            gc.collect()
            torch.cuda.empty_cache()
        else:
            result["ok"] = True
        results.append(result)
        print(json.dumps(result), flush=True)

    by_name = {result["variant"]: result for result in results if result.get("ok")}
    speedups = {}
    for mode in ("ds_off", "ds_on"):
        legacy = by_name.get(f"{mode}_legacy")
        optimized = by_name.get(f"{mode}_optimized")
        if legacy and optimized:
            speedups[mode] = {
                "mean_speedup": legacy["mean_s"] / optimized["mean_s"],
                "median_speedup": legacy["median_s"] / optimized["median_s"],
                "peak_allocated_gib_saved": (legacy["peak_allocated_gib"] -
                                              optimized["peak_allocated_gib"]),
            }
    payload = {"results": results, "speedups": speedups}
    with open("dead_head_bench_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"speedups": speedups}), flush=True)
    print("DEAD_HEAD_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()

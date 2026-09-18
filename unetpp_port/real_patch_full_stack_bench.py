"""
Last unverified combination: dead-head-skip + torch.compile(mode="reduce-overhead") together, at
the REAL planner patch size (64,160,224), not a proxy size. Everything so far was verified either
separately, or together-but-at-a-proxy-patch-size. This settles the actual number we'd trust for
planning, for both architectures in the grid.

ARCH env var: "unetpp" or "plainunet". Run on real GB10 hardware, AMP, batch 4.
"""
import gc
import json
import os
import statistics
import time

import torch
from torch import nn

ARCH = os.environ.get("ARCH", "unetpp")
PATCH_SIZE = (64, 160, 224)
BATCH_SIZE = 4
N_STAGES = 6
FEATURES = [32, 64, 128, 256, 320, 320]
STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
KERNELS = [[3, 3, 3]] * N_STAGES
NUM_CLASSES = 29
WARMUP = 5
STEPS = 30


def build_unetpp(skip_shallowest):
    from unet_plusplus import UNetPlusPlus
    model = UNetPlusPlus(
        input_channels=1, n_stages=N_STAGES, features_per_stage=FEATURES,
        conv_op=nn.Conv3d, kernel_sizes=KERNELS, strides=STRIDES,
        n_conv_per_stage=[2] * N_STAGES, num_classes=NUM_CLASSES,
        n_conv_per_stage_decoder=[2] * (N_STAGES - 1),
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True,
        skip_shallowest_deep_supervision_head=skip_shallowest,
    )
    model.initialize()
    return model


def build_plainunet():
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    return PlainConvUNet(
        input_channels=1, n_stages=N_STAGES, features_per_stage=FEATURES,
        conv_op=nn.Conv3d, kernel_sizes=KERNELS, strides=STRIDES,
        n_conv_per_stage=[2] * N_STAGES, num_classes=NUM_CLASSES,
        n_conv_per_stage_decoder=[2] * (N_STAGES - 1),
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True,
    )


def loss_for(outputs, target, ce, skip_shallowest):
    n_outputs_present = len(outputs)
    conceptual_length = n_outputs_present + (1 if skip_shallowest else 0)
    weights = [1 / (2 ** i) for i in range(conceptual_length)]
    weights[-1] = 0
    normalizer = sum(weights)
    if skip_shallowest:
        weights = weights[:-1]
    weights = [w / normalizer for w in weights]
    return sum(w * ce(o, target) for w, o in zip(weights, outputs) if w != 0)


def loss_for_plainunet(outputs, target, ce, device):
    """Plain U-Net's raw DS outputs are genuinely per-level downsampled -- probe shapes, don't
    assume, matching the fix from the gate0b bug earlier."""
    weights = [1 / (2 ** i) for i in range(len(outputs))]
    weights[-1] = 0
    norm = sum(weights)
    total = 0
    for w, o in zip(weights, outputs):
        if w == 0:
            continue
        t = torch.nn.functional.interpolate(
            target.unsqueeze(1).float(), size=o.shape[2:], mode="nearest"
        ).squeeze(1).long()
        total = total + (w / norm) * ce(o, t)
    return total


def bench(skip_shallowest, use_compile):
    label = f"{'skip' if skip_shallowest else 'noskip'}_{'compile' if use_compile else 'eager'}"
    if ARCH == "unetpp":
        model = build_unetpp(skip_shallowest).cuda().train()
    else:
        model = build_plainunet().cuda().train()
    if use_compile:
        model = torch.compile(model, mode="reduce-overhead")
        torch._dynamo.reset()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()
    x = torch.randn((BATCH_SIZE, 1, *PATCH_SIZE), device="cuda")
    target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, *PATCH_SIZE), device="cuda")

    timings = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(WARMUP + STEPS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x)
                if ARCH == "unetpp":
                    loss = loss_for(outputs, target, ce, skip_shallowest)
                else:
                    loss = loss_for_plainunet(outputs, target, ce, "cuda")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= WARMUP:
                timings.append(elapsed)
    except Exception as e:
        return {"label": label, "ok": False, "error": type(e).__name__, "detail": str(e)[:300]}

    timings.sort()
    result = {
        "label": label, "ok": True, "arch": ARCH,
        "mean_s": statistics.mean(timings), "median_s": statistics.median(timings),
        "peak_mem_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    print(f"ARCH={ARCH} torch {torch.__version__} device {torch.cuda.get_device_name(0)}", flush=True)
    results = {}
    skip_options = (False, True) if ARCH == "unetpp" else (False,)
    for skip_shallowest in skip_options:
        for use_compile in (False, True):
            label = f"{'skip' if skip_shallowest else 'noskip'}_{'compile' if use_compile else 'eager'}"
            print(f"=== {label} ===", flush=True)
            r = bench(skip_shallowest, use_compile)
            print(json.dumps(r), flush=True)
            results[label] = r

    baseline_key = "noskip_eager"
    best_key = "skip_compile" if ARCH == "unetpp" else "noskip_compile"
    if results.get(baseline_key, {}).get("ok") and results.get(best_key, {}).get("ok"):
        speedup = results[baseline_key]["mean_s"] / results[best_key]["mean_s"]
        print(f"\nfull-stack speedup at REAL patch size: {speedup:.3f}x", flush=True)
        epoch_s = results[best_key]["mean_s"] * 260  # sparse-validation-adjusted avg iters/epoch
        print(f"optimized epoch time estimate: {epoch_s/60:.2f} min/epoch", flush=True)
        print(f"500 epochs: {epoch_s*500/3600:.1f}h, 1000 epochs: {epoch_s*1000/3600:.1f}h", flush=True)

    with open(f"real_patch_full_stack_{ARCH}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("REAL_PATCH_FULL_STACK_DONE", flush=True)


if __name__ == "__main__":
    main()

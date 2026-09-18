"""
bf16 vs fp16 AMP, at the real patch size, stacked on top of the already-verified dead-head-skip +
reduce-overhead compile config. Flagged early as a free lever (bf16 has fp32's exponent range, so
no GradScaler/loss-scaling needed, and identical raw tensor throughput to fp16 on this hardware --
pure-upside if it works), but never actually tested. Settling it now.

ARCH env var: "unetpp" or "plainunet".
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


def build_unetpp():
    from unet_plusplus import UNetPlusPlus
    model = UNetPlusPlus(
        input_channels=1, n_stages=N_STAGES, features_per_stage=FEATURES,
        conv_op=nn.Conv3d, kernel_sizes=KERNELS, strides=STRIDES,
        n_conv_per_stage=[2] * N_STAGES, num_classes=NUM_CLASSES,
        n_conv_per_stage_decoder=[2] * (N_STAGES - 1),
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True,
        skip_shallowest_deep_supervision_head=True,
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


def loss_for_unetpp(outputs, target, ce):
    n = len(outputs)
    conceptual = n + 1
    weights = [1 / (2 ** i) for i in range(conceptual)]
    weights[-1] = 0
    norm = sum(weights)
    weights = [w / norm for w in weights[:-1]]
    return sum(w * ce(o, target) for w, o in zip(weights, outputs) if w != 0)


def loss_for_plainunet(outputs, target, ce):
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


def bench(dtype):
    label = "bf16" if dtype == torch.bfloat16 else "fp16"
    model = (build_unetpp() if ARCH == "unetpp" else build_plainunet()).cuda().train()
    model = torch.compile(model, mode="reduce-overhead")
    torch._dynamo.reset()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
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
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(x)
                loss = (loss_for_unetpp(outputs, target, ce) if ARCH == "unetpp"
                        else loss_for_plainunet(outputs, target, ce))
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
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
    for dtype in (torch.float16, torch.bfloat16):
        label = "bf16" if dtype == torch.bfloat16 else "fp16"
        print(f"=== {label} ===", flush=True)
        r = bench(dtype)
        print(json.dumps(r), flush=True)
        results[label] = r

    if results.get("fp16", {}).get("ok") and results.get("bf16", {}).get("ok"):
        speedup = results["fp16"]["mean_s"] / results["bf16"]["mean_s"]
        print(f"\nbf16 vs fp16 speedup: {speedup:.3f}x (near 1.0 expected -- same throughput, "
              f"the real win if any is fewer skipped optimizer steps from loss-scale overflow, "
              f"which this synthetic benchmark with random data can't observe)", flush=True)

    with open(f"bf16_vs_fp16_{ARCH}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("BF16_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()

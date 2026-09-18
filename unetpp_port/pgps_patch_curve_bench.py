"""Measure conservative PGPS patch-size scaling on the real PanTS nnU-Net plan.

One invocation measures exactly one patch and exits. The accompanying operator is expected to
launch each patch in a fresh process: on GB10, process exit is a substantially stronger CUDA-memory
reclamation boundary than looping over shapes in one interpreter.

This is a throughput/capacity benchmark only. It does not establish that progressive patch sizing
preserves tumour detection; that requires the pre-agreed 500+ epoch validation gate.
"""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn


BATCH_SIZE = 4
N_STAGES = 6
FEATURES = [32, 64, 128, 256, 320, 320]
STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
KERNELS = [[3, 3, 3]] * N_STAGES
NUM_CLASSES = 29


def parse_patch(value: str) -> tuple[int, int, int]:
    try:
        patch = tuple(int(v) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("patch must contain three comma-separated integers") from exc
    if len(patch) != 3 or any(v <= 0 for v in patch):
        raise argparse.ArgumentTypeError("patch must contain three positive integers")
    # The real plan downsamples z by 16 and the two in-plane axes by 32.
    divisors = (16, 32, 32)
    if any(v % d for v, d in zip(patch, divisors)):
        raise argparse.ArgumentTypeError(
            f"patch {patch} must be divisible by the plan's cumulative strides {divisors}"
        )
    return patch


def build_unetpp():
    from unet_plusplus import UNetPlusPlus

    model = UNetPlusPlus(
        input_channels=1,
        n_stages=N_STAGES,
        features_per_stage=FEATURES,
        conv_op=nn.Conv3d,
        kernel_sizes=KERNELS,
        strides=STRIDES,
        n_conv_per_stage=[2] * N_STAGES,
        num_classes=NUM_CLASSES,
        n_conv_per_stage_decoder=[2] * (N_STAGES - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
        skip_shallowest_deep_supervision_head=True,
    )
    model.initialize()
    return model


def build_plainunet():
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    return PlainConvUNet(
        input_channels=1,
        n_stages=N_STAGES,
        features_per_stage=FEATURES,
        conv_op=nn.Conv3d,
        kernel_sizes=KERNELS,
        strides=STRIDES,
        n_conv_per_stage=[2] * N_STAGES,
        num_classes=NUM_CLASSES,
        n_conv_per_stage_decoder=[2] * (N_STAGES - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    )


def unetpp_loss(outputs, target, ce):
    # The conceptual fifth branch has the original trainer's zero weight and is not computed.
    conceptual_weights = [1 / (2**i) for i in range(len(outputs) + 1)]
    conceptual_weights[-1] = 0
    normalizer = sum(conceptual_weights)
    return sum(
        (weight / normalizer) * ce(output, target)
        for weight, output in zip(conceptual_weights[:-1], outputs)
    )


def plainunet_loss(outputs, target, ce):
    weights = [1 / (2**i) for i in range(len(outputs))]
    weights[-1] = 0
    normalizer = sum(weights)
    loss = 0
    for weight, output in zip(weights, outputs):
        if weight == 0:
            continue
        scaled_target = torch.nn.functional.interpolate(
            target.unsqueeze(1).float(), size=output.shape[2:], mode="nearest"
        ).squeeze(1).long()
        loss = loss + (weight / normalizer) * ce(output, scaled_target)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("unetpp", "plainunet"), required=True)
    parser.add_argument("--patch", type=parse_patch, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model = (build_unetpp() if args.arch == "unetpp" else build_plainunet()).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()
    x = torch.randn((BATCH_SIZE, 1, *args.patch), device="cuda")
    target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, *args.patch), device="cuda")

    timings = []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        for step in range(args.warmup + args.steps):
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x)
                loss = (
                    unetpp_loss(outputs, target, ce)
                    if args.arch == "unetpp"
                    else plainunet_loss(outputs, target, ce)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if step >= args.warmup:
                timings.append(time.perf_counter() - step_started)

        ordered = sorted(timings)
        patch_voxels = args.patch[0] * args.patch[1] * args.patch[2]
        result = {
            "ok": True,
            "arch": args.arch,
            "patch": list(args.patch),
            "batch_size": BATCH_SIZE,
            "warmup": args.warmup,
            "steps": args.steps,
            "mean_s": statistics.mean(timings),
            "median_s": statistics.median(timings),
            "p90_s": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "samples_per_s": BATCH_SIZE / statistics.mean(timings),
            "patch_voxels": patch_voxels,
            "relative_voxels_vs_full": patch_voxels / (64 * 160 * 224),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            "wall_s": time.time() - started,
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        }
    except Exception as exc:
        result = {
            "ok": False,
            "arch": args.arch,
            "patch": list(args.patch),
            "error": type(exc).__name__,
            "detail": str(exc)[:500],
        }
    finally:
        del model, optimizer, scaler, x, target
        gc.collect()
        torch.cuda.empty_cache()

    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

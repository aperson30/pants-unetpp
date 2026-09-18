"""Kill-fast benchmarks for two small training-path optimization candidates.

Each timing invocation measures one arm so CUDA/Inductor state cannot leak between arms. The
``loss-parity`` command is deliberately separate and must pass before the compiled full loss is
trusted. None of these synthetic checks is an accuracy result.
"""
import argparse
import json
import statistics
import time

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


BATCH_SIZE = 4
PATCH_SIZE = (64, 160, 224)
N_STAGES = 6
FEATURES = [32, 64, 128, 256, 320, 320]
STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
KERNELS = [[3, 3, 3]] * N_STAGES
NUM_CLASSES = 29


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


def loss_weights(n_outputs=4, removed_zero_head=True):
    conceptual_outputs = n_outputs + 1 if removed_zero_head else n_outputs
    weights = np.array([1 / (2**i) for i in range(conceptual_outputs)], dtype=np.float64)
    weights[-1] = 0
    weights /= weights.sum()
    return weights[:-1] if removed_zero_head else weights


def build_loss(mode, arch="unetpp"):
    compound = DC_and_CE_loss(
        {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False},
        {},
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    wrapped = DeepSupervisionWrapper(
        compound,
        loss_weights(4, removed_zero_head=True) if arch == "unetpp"
        else loss_weights(5, removed_zero_head=False),
    )
    if mode == "current":
        # Exact current nnU-Net behavior: only Dice is compiled; CE and the wrapper remain eager.
        compound.dc = torch.compile(compound.dc)
        return wrapped
    if mode == "full-compile":
        return torch.compile(wrapped, mode="reduce-overhead")
    if mode == "eager":
        return wrapped
    raise ValueError(mode)


def make_sparse_target(shape, device="cuda"):
    target = torch.zeros((shape[0], 1, *shape[2:]), dtype=torch.long, device=device)
    # Put every semantic class in the target, but reserve only a tiny 2x2x2 focus for class 28.
    flat = target.view(-1)
    stride = max(1, flat.numel() // (NUM_CLASSES * 4))
    for cls in range(1, NUM_CLASSES):
        flat[cls * stride : cls * stride + 4] = cls
    target[:, :, 2:4, 4:6, 6:8] = NUM_CLASSES - 1
    return target


def parity_for_dtype(dtype):
    # Modest tensors keep this a correctness gate rather than a capacity benchmark.
    shape = (2, NUM_CLASSES, 16, 32, 32)
    torch.manual_seed(20260917)
    base = [torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(4)]
    compiled_inputs = [x.detach().clone().requires_grad_(True) for x in base]
    target = make_sparse_target(shape)
    targets = [target] * 4

    eager = build_loss("current")
    compiled = build_loss("full-compile")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        eager_value = eager(base, targets)
    eager_value.backward()
    # This first call exercises compilation and the compiled backward path.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        compiled_value = compiled(compiled_inputs, targets)
    compiled_value.backward()
    torch.cuda.synchronize()

    grad_diffs = [(a.grad - b.grad).abs() for a, b in zip(base, compiled_inputs)]
    max_abs = max(x.max().item() for x in grad_diffs)
    max_ref = max(x.grad.abs().max().item() for x in base)
    tolerance = (5e-3, 2e-5) if dtype == torch.bfloat16 else (2e-4, 2e-6)
    return {
        "ok": bool(torch.allclose(eager_value, compiled_value, rtol=tolerance[0], atol=tolerance[1])
                   and all(torch.allclose(a.grad, b.grad, rtol=tolerance[0], atol=tolerance[1])
                           for a, b in zip(base, compiled_inputs))),
        "dtype": str(dtype),
        "eager_loss": eager_value.item(),
        "compiled_loss": compiled_value.item(),
        "loss_abs_diff": abs(eager_value.item() - compiled_value.item()),
        "grad_max_abs_diff": max_abs,
        "grad_max_reference": max_ref,
        "grad_relative_to_max": max_abs / max_ref if max_ref else 0.0,
    }


def loss_parity():
    result = {
        "fp32": parity_for_dtype(torch.float32),
        "bf16": parity_for_dtype(torch.bfloat16),
    }
    result["ok"] = result["fp32"]["ok"] and result["bf16"]["ok"]
    print(json.dumps(result, indent=2), flush=True)
    if not result["ok"]:
        raise SystemExit(1)


def full_step(arch, network_compile_mode, loss_mode, optimizer_mode, warmup, steps):
    torch.manual_seed(20260917)
    model = (build_unetpp() if arch == "unetpp" else build_plainunet()).cuda().train()
    model = (
        torch.compile(model)
        if network_compile_mode == "default"
        else torch.compile(model, mode="reduce-overhead")
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.99, nesterov=True,
        fused=True if optimizer_mode == "fused" else None,
    )
    loss_fn = build_loss(loss_mode, arch)
    x = torch.randn((BATCH_SIZE, 1, *PATCH_SIZE), device="cuda")
    target = make_sparse_target((BATCH_SIZE, NUM_CLASSES, *PATCH_SIZE))
    if arch == "unetpp":
        targets = [target] * 4
    else:
        plain_shapes = [
            PATCH_SIZE,
            (32, 80, 112),
            (16, 40, 56),
            (8, 20, 28),
            (4, 10, 14),
        ]
        targets = [
            torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").long()
            for shape in plain_shapes
        ]
    timings = []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for step in range(warmup + steps):
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(x)
            loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if step >= warmup:
            timings.append(time.perf_counter() - step_started)

    result = {
        "ok": True,
        "arch": arch,
        "network_compile_mode": network_compile_mode,
        "loss_mode": loss_mode,
        "optimizer_mode": optimizer_mode,
        "warmup": warmup,
        "steps": steps,
        "mean_s": statistics.mean(timings),
        "median_s": statistics.median(timings),
        "stdev_s": statistics.stdev(timings) if len(timings) > 1 else 0,
        "min_s": min(timings),
        "max_s": max(timings),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "wall_s": time.time() - started,
        "final_loss": loss.item(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
    }
    print(json.dumps(result, indent=2), flush=True)


def optimizer_microbench(mode, steps):
    torch.manual_seed(20260917)
    model = build_unetpp().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.99, nesterov=True,
        fused=True if mode == "fused" else None,
    )
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    # Initialize momentum state before timing.
    optimizer.step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.step()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    print(json.dumps({
        "ok": True,
        "optimizer_mode": mode,
        "steps": steps,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "stdev_ms": statistics.stdev(samples),
    }, indent=2), flush=True)


def optimizer_parity():
    """Compare the actual Nesterov update and momentum buffers, not just optimizer timing."""
    torch.manual_seed(20260917)
    default_model = build_unetpp().cuda()
    fused_model = build_unetpp().cuda()
    fused_model.load_state_dict(default_model.state_dict())
    default = torch.optim.SGD(
        default_model.parameters(), lr=1e-2, weight_decay=3e-5,
        momentum=0.99, nesterov=True,
    )
    fused = torch.optim.SGD(
        fused_model.parameters(), lr=1e-2, weight_decay=3e-5,
        momentum=0.99, nesterov=True, fused=True,
    )
    generator = torch.Generator(device="cuda").manual_seed(4242)
    for _ in range(5):
        for left, right in zip(default_model.parameters(), fused_model.parameters()):
            grad = torch.randn(left.shape, dtype=left.dtype, device=left.device, generator=generator)
            left.grad = grad
            right.grad = grad.clone()
        default.step()
        fused.step()
    torch.cuda.synchronize()

    parameter_diffs = [
        (left - right).abs().max().item()
        for left, right in zip(default_model.parameters(), fused_model.parameters())
    ]
    buffer_diffs = [
        (default.state[left]["momentum_buffer"] - fused.state[right]["momentum_buffer"]).abs().max().item()
        for left, right in zip(default_model.parameters(), fused_model.parameters())
    ]
    result = {
        "ok": all(torch.allclose(left, right, rtol=1e-6, atol=1e-7)
                  for left, right in zip(default_model.parameters(), fused_model.parameters())),
        "steps": 5,
        "parameter_max_abs_diff": max(parameter_diffs),
        "momentum_max_abs_diff": max(buffer_diffs),
    }
    print(json.dumps(result, indent=2), flush=True)
    if not result["ok"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("loss-parity")
    subparsers.add_parser("optimizer-parity")
    step = subparsers.add_parser("full-step")
    step.add_argument("--arch", choices=("unetpp", "plainunet"), default="unetpp")
    step.add_argument("--network-compile", choices=("default", "reduce-overhead"),
                      default="reduce-overhead")
    step.add_argument("--loss", choices=("current", "full-compile"), required=True)
    step.add_argument("--optimizer", choices=("default", "fused"), required=True)
    step.add_argument("--warmup", type=int, default=5)
    step.add_argument("--steps", type=int, default=20)
    opt = subparsers.add_parser("optimizer")
    opt.add_argument("--mode", choices=("default", "fused"), required=True)
    opt.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.command == "loss-parity":
        loss_parity()
    elif args.command == "optimizer-parity":
        optimizer_parity()
    elif args.command == "full-step":
        full_step(args.arch, args.network_compile, args.loss, args.optimizer, args.warmup, args.steps)
    else:
        optimizer_microbench(args.mode, args.steps)


if __name__ == "__main__":
    main()

"""Fresh-process GH200 screen for quality-neutral kernel/runtime choices.

This is deliberately a throughput screen, not a tumor-quality experiment. It keeps the PanTS
topology, physical batch four, full patch, BF16 autocast, complete compiled Dice+CE loss, SGD,
gradient clipping, and deep-supervision weights fixed. Run exactly one arm per process so Inductor,
cuDNN autotuning, and CUDA-graph state cannot leak between candidates.
"""
import argparse
import json
import statistics
import time

import torch

from loss_optimizer_bench import (
    BATCH_SIZE,
    NUM_CLASSES,
    PATCH_SIZE,
    build_loss,
    build_plainunet,
    build_unetpp,
    make_sparse_target,
)


COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


class _NetworkAndLoss(torch.nn.Module):
    def __init__(self, network, loss_fn):
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn

    def forward(self, data, targets):
        return self.loss_fn(self.network(data), targets)


def make_targets(arch: str, target: torch.Tensor):
    if arch == "unetpp":
        return [target] * 4
    plain_shapes = [
        PATCH_SIZE,
        (32, 80, 112),
        (16, 40, 56),
        (8, 20, 28),
        (4, 10, 14),
    ]
    return [
        torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").long()
        for shape in plain_shapes
    ]


def parameter_fingerprint(model: torch.nn.Module):
    # Compact drift detector for paired arms. It is not a substitute for the dedicated parity test.
    parameters = list(model.parameters())
    selected = parameters[:2] + parameters[-2:]
    return [
        {
            "shape": list(parameter.shape),
            "sum": parameter.detach().float().sum().item(),
            "l2": parameter.detach().float().norm().item(),
        }
        for parameter in selected
    ]


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    if args.cudnn_benchmark_limit is not None:
        torch.backends.cudnn.benchmark_limit = args.cudnn_benchmark_limit

    model = (build_unetpp() if args.arch == "unetpp" else build_plainunet()).cuda().train()
    x = torch.randn((BATCH_SIZE, 1, *PATCH_SIZE), device="cuda")
    if args.channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)
        x = x.to(memory_format=torch.channels_last_3d)

    compile_started = time.perf_counter()
    loss_fn = build_loss("eager", args.arch)
    if args.compile_boundary == "separate":
        model = torch.compile(model, mode=args.compile_mode)
        loss_fn = torch.compile(loss_fn, mode=args.compile_mode)
        objective = None
    else:
        objective = torch.compile(
            _NetworkAndLoss(model, loss_fn), mode=args.compile_mode, dynamic=False
        )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True
    )

    target = make_sparse_target((BATCH_SIZE, NUM_CLASSES, *PATCH_SIZE))
    targets = make_targets(args.arch, target)
    samples = []
    losses = []
    grad_norms = []
    first_step_finished = None
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if objective is None:
                output = model(x)
                loss = loss_fn(output, targets)
            else:
                loss = objective(x, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if first_step_finished is None:
            first_step_finished = time.perf_counter()
        if step >= args.warmup:
            samples.append(elapsed)
            losses.append(loss.detach().float().item())
            grad_norms.append(grad_norm.detach().float().item())

    result = {
        "ok": True,
        "arch": args.arch,
        "compile_mode": args.compile_mode,
        "compile_boundary": args.compile_boundary,
        "cudnn_benchmark": args.cudnn_benchmark,
        "cudnn_benchmark_limit": torch.backends.cudnn.benchmark_limit,
        "channels_last_3d": args.channels_last_3d,
        "seed": args.seed,
        "warmup": args.warmup,
        "steps": args.steps,
        "first_step_wall_s": first_step_finished - compile_started,
        "mean_s": statistics.mean(samples),
        "median_s": statistics.median(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_s": min(samples),
        "max_s": max(samples),
        "final_loss": losses[-1],
        "final_grad_norm": grad_norms[-1],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "parameter_fingerprint": parameter_fingerprint(model),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
    }
    if args.state_output:
        # This copy/save is deliberately outside the timed region. Loading with map_location='cpu'
        # allows a separate CPU-only comparator to quantify complete parameter, gradient, and
        # momentum drift between otherwise identical arms.
        state_model = getattr(model, "_orig_mod", model)
        torch.save({
            "model": state_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gradients": {
                name: parameter.grad
                for name, parameter in state_model.named_parameters()
                if parameter.grad is not None
            },
            "result": result,
        }, args.state_output)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("unetpp", "plainunet"), required=True)
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, required=True)
    parser.add_argument("--compile-boundary", choices=("separate", "joint"), default="separate")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--cudnn-benchmark-limit", type=int)
    parser.add_argument("--channels-last-3d", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--state-output")
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 2:
        parser.error("use at least one warmup and two measured steps")
    run(args)


if __name__ == "__main__":
    main()

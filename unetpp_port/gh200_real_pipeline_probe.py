"""Measure exposed nnU-Net loader/augmentation wait around the real trainer step.

This consumes real preprocessed cases and fresh online augmentations but is not a training-quality
experiment. It reports loader wait separately from the synchronized CUDA training step so storage
or augmentation work is only optimized if it is actually exposed on GH200.
"""
import argparse
import json
import os
import statistics
import time

import torch


TRAINERS = {
    "unetpp_ds_on": (
        "unetpp_port.nnUNetTrainerUNetPlusPlus",
        "nnUNetTrainerUNetPlusPlus",
    ),
    "unetpp_ds_off": (
        "unetpp_port.nnUNetTrainerUNetPlusPlusNoDeepSupervision",
        "nnUNetTrainerUNetPlusPlusNoDeepSupervision",
    ),
    "plain_ds_on": (
        "unetpp_port.nnUNetTrainerBF16",
        "nnUNetTrainerBF16",
    ),
    "plain_ds_off": (
        "unetpp_port.nnUNetTrainerBF16NoDeepSupervision",
        "nnUNetTrainerBF16NoDeepSupervision",
    ),
}


def load_class(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def summary(values):
    ordered = sorted(values)
    return {
        "mean_s": statistics.mean(values),
        "median_s": statistics.median(values),
        "p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_s": max(values),
    }


def run(args):
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    if args.cudnn_benchmark_limit is not None:
        torch.backends.cudnn.benchmark_limit = args.cudnn_benchmark_limit
    with open(args.dataset_json, encoding="utf-8") as handle:
        dataset_json = json.load(handle)
    with open(args.plans, encoding="utf-8") as handle:
        plans = json.load(handle)
    plans.setdefault("continue_training", False)

    module_name, class_name = TRAINERS[args.trainer]
    trainer_class = load_class(module_name, class_name)
    trainer = trainer_class(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        device=torch.device("cuda"),
    )
    trainer.initialize()

    raw_network = getattr(trainer.network, "_orig_mod", trainer.network)
    torch._dynamo.reset()
    trainer.network = torch.compile(raw_network, mode=args.compile_mode)
    dl_tr, _ = trainer.get_dataloaders()

    loader_samples = []
    train_samples = []
    cycle_samples = []
    torch.cuda.reset_peak_memory_stats()

    # Synchronization gives a clean boundary while background augmentation workers continue to
    # prefetch exactly as they do during normal training.
    torch.cuda.synchronize()
    for step in range(args.warmup + args.steps):
        cycle_started = time.perf_counter()
        batch = next(dl_tr)
        batch_ready = time.perf_counter()
        output = trainer.train_step(batch)
        torch.cuda.synchronize()
        step_finished = time.perf_counter()

        loader_s = batch_ready - cycle_started
        train_s = step_finished - batch_ready
        cycle_s = step_finished - cycle_started
        if step >= args.warmup:
            loader_samples.append(loader_s)
            train_samples.append(train_s)
            cycle_samples.append(cycle_s)
        print(
            f"step={step} loader={loader_s:.6f}s train={train_s:.6f}s "
            f"cycle={cycle_s:.6f}s loss={float(output['loss']):.6f}",
            flush=True,
        )

    result = {
        "ok": True,
        "trainer": args.trainer,
        "configuration": args.configuration,
        "fold": args.fold,
        "compile_mode": args.compile_mode,
        "cudnn_benchmark": args.cudnn_benchmark,
        "cudnn_benchmark_limit": torch.backends.cudnn.benchmark_limit,
        "warmup": args.warmup,
        "steps": args.steps,
        "nnUNet_n_proc_DA": os.environ.get("nnUNet_n_proc_DA"),
        "nnUNet_keep_files_open": os.environ.get("nnUNet_keep_files_open"),
        "loader": summary(loader_samples),
        "train": summary(train_samples),
        "cycle": summary(cycle_samples),
        "exposed_loader_fraction": sum(loader_samples) / sum(cycle_samples),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    print("RESULT_JSON:" + json.dumps(result), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--cudnn-benchmark-limit", type=int)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 2:
        parser.error("use at least one warmup and two measured steps")
    run(args)


if __name__ == "__main__":
    main()

"""Bounded real-data gate for running the PanTS 2x2 grid on Bridges-2 H100s."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch


TRAINERS = {
    "unetpp_ds_on": ("unetpp_port.nnUNetTrainerUNetPlusPlus", "nnUNetTrainerUNetPlusPlus"),
    "unetpp_ds_off": (
        "unetpp_port.nnUNetTrainerUNetPlusPlusNoDeepSupervision",
        "nnUNetTrainerUNetPlusPlusNoDeepSupervision",
    ),
    "plain_ds_on": ("unetpp_port.nnUNetTrainerBF16", "nnUNetTrainerBF16"),
    "plain_ds_off": (
        "unetpp_port.nnUNetTrainerBF16NoDeepSupervision",
        "nnUNetTrainerBF16NoDeepSupervision",
    ),
}


def _head_gradient_report(network: torch.nn.Module) -> dict:
    raw = getattr(network, "_orig_mod", network)
    heads = {}
    for name, parameter in raw.named_parameters():
        if parameter.grad is None or parameter.ndim < 1 or parameter.shape[0] <= 28:
            continue
        if "seg_layers" not in name and "seg_outputs" not in name:
            continue
        row = parameter.grad.detach()[28]
        heads[name] = {
            "finite": bool(torch.isfinite(row).all().item()),
            "norm": float(row.float().norm().item()),
        }
    return heads


def _finite_state(trainer) -> tuple[bool, bool]:
    raw = getattr(trainer.network, "_orig_mod", trainer.network)
    parameters_finite = all(bool(torch.isfinite(p.detach()).all().item()) for p in raw.parameters())
    optimizer_finite = True
    for state in trainer.optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                optimizer_finite = False
    return parameters_finite, optimizer_finite


def _finish_loader(loader) -> None:
    finish = getattr(loader, "_finish", None)
    if callable(finish):
        finish()


def run_one(name: str, plans: dict, dataset_json: dict, warmup: int, steps: int) -> dict:
    torch.manual_seed(20260922)
    torch.cuda.manual_seed_all(20260922)
    module_name, class_name = TRAINERS[name]
    trainer_class = getattr(importlib.import_module(module_name), class_name)
    trainer = trainer_class(
        plans=plans,
        configuration="3d_fullres",
        fold=0,
        dataset_json=dataset_json,
        device=torch.device("cuda"),
    )
    trainer.initialize()
    raw_network = getattr(trainer.network, "_orig_mod", trainer.network)
    raw_loss = getattr(trainer.loss, "_orig_mod", trainer.loss)
    torch._dynamo.reset()
    trainer.network = torch.compile(raw_network, mode="reduce-overhead")
    trainer.loss = torch.compile(raw_loss, mode="reduce-overhead")
    train_loader, validation_loader = trainer.get_dataloaders()

    measured = []
    losses = []
    head_gradients = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for index in range(warmup + steps):
            batch = next(train_loader)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = trainer.train_step(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            loss = float(output["loss"].item())
            if not math.isfinite(loss):
                raise RuntimeError(f"{name} produced non-finite loss at update {index}: {loss}")
            if index >= warmup:
                measured.append(elapsed)
                losses.append(loss)
            head_gradients = _head_gradient_report(trainer.network)
            print(f"{name} update={index} loss={loss:.8f} train_s={elapsed:.6f}", flush=True)

        if not head_gradients:
            raise RuntimeError(f"{name}: no segmentation-head gradients found")
        if not all(item["finite"] for item in head_gradients.values()):
            raise RuntimeError(f"{name}: non-finite class-28 head gradient: {head_gradients}")
        if not any(item["norm"] > 0 for item in head_gradients.values()):
            raise RuntimeError(f"{name}: all class-28 head gradients are zero: {head_gradients}")
        parameters_finite, optimizer_finite = _finite_state(trainer)
        if not parameters_finite or not optimizer_finite:
            raise RuntimeError(
                f"{name}: non-finite state (parameters={parameters_finite}, optimizer={optimizer_finite})"
            )
        return {
            "trainer": name,
            "losses": losses,
            "mean_train_s": statistics.mean(measured),
            "median_train_s": statistics.median(measured),
            "min_train_s": min(measured),
            "max_train_s": max(measured),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "class28_head_gradients": head_gradients,
            "parameters_finite": parameters_finite,
            "optimizer_finite": optimizer_finite,
        }
    finally:
        _finish_loader(train_loader)
        _finish_loader(validation_loader)
        del trainer, train_loader, validation_loader
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    plans = json.loads(args.plans.read_text(encoding="utf-8"))
    dataset_json = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    plans.setdefault("continue_training", False)
    if plans["configurations"]["3d_fullres"]["batch_size"] != 4:
        raise RuntimeError("Calibration plans are not physical batch size 4")
    if dataset_json["labels"].get("pancreatic_lesion") != 28:
        raise RuntimeError("Calibration dataset does not map pancreatic_lesion to class 28")

    started = time.time()
    results = [run_one(name, plans, dataset_json, args.warmup, args.steps) for name in TRAINERS]
    payload = {
        "ok": True,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "elapsed_s": time.time() - started,
        "physical_batch_size": 4,
        "patch_size": plans["configurations"]["3d_fullres"]["patch_size"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("RESULT_JSON:" + json.dumps(payload), flush=True)
    print("BRIDGES2_H100_CALIBRATION_PASS", flush=True)


if __name__ == "__main__":
    main()

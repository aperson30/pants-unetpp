"""
Real-data calibration benchmark on Delta (A100 or H200, whichever partition the job ran on).
Same methodology proven on DeltaAI: real trainer classes, real dataloader, dead-head-skip + bf16 +
explicitly-applied reduce-overhead compile on the network (base nnUNetTrainer.initialize() only
uses default compile mode -- this fixes that gap same as before).
"""
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/projects/bdyo/asanjeev/delta_work/pants-unetpp-fork")

ARCH = os.environ.get("ARCH", "unetpp")
WARMUP = 8
STEPS = 40

nnUNet_preprocessed = os.environ["nnUNet_preprocessed"]
nnUNet_raw = os.environ["nnUNet_raw"]

dataset_json = json.load(open(f"{nnUNet_raw}/Dataset001_PanTSCalib/dataset.json"))
plans = json.load(open(f"{nnUNet_preprocessed}/Dataset001_PanTSCalib/nnUNetPlansBS4.json"))
plans.setdefault("continue_training", False)


def get_trainer_class():
    if ARCH == "unetpp":
        from unetpp_port.nnUNetTrainerUNetPlusPlus import nnUNetTrainerUNetPlusPlus
        return nnUNetTrainerUNetPlusPlus
    else:
        from unetpp_port.nnUNetTrainerBF16 import nnUNetTrainerBF16
        return nnUNetTrainerBF16


def main():
    TrainerClass = get_trainer_class()
    trainer = TrainerClass(
        plans=plans, configuration="3d_fullres", fold=0,
        dataset_json=dataset_json, device=torch.device("cuda"),
    )
    trainer.initialize()

    net = trainer.network
    if hasattr(net, "_orig_mod"):
        net = net._orig_mod
    torch._dynamo.reset()
    trainer.network = torch.compile(net, mode="reduce-overhead")

    dl_tr, dl_val = trainer.get_dataloaders()

    timings = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(WARMUP + STEPS):
        batch = next(dl_tr)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = trainer.train_step(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step >= WARMUP:
            timings.append(elapsed)
        print(f"step {step} loss={out['loss']:.4f} t={elapsed:.3f}s", flush=True)

    timings.sort()
    mean_s = statistics.mean(timings)
    result = {
        "arch": ARCH,
        "n_steps": len(timings),
        "mean_s_per_step": mean_s,
        "median_s_per_step": statistics.median(timings),
        "peak_mem_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    print("RESULT_JSON:" + json.dumps(result), flush=True)
    with open(f"/projects/bdyo/asanjeev/delta_work/real_calib_{ARCH}_{torch.cuda.get_device_name(0).replace(' ', '_')}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("REAL_CALIB_DONE", flush=True)


if __name__ == "__main__":
    main()

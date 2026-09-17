"""
Gate-0b: same methodology as gate0_bench.py, but for the PLAIN U-Net side of the 2x2 grid
(dynamic_network_architectures.architectures.unet.PlainConvUNet) instead of UNet++. We had UNet++
numbers but nothing yet for the control-group architecture, and the two need to be compared on the
same hardware to mean anything (e.g. "does batch 4 fit" and "what does DS cost" differ by
architecture -- UNet++ concatenates far more feature maps per decoder node, so plain U-Net should be
cheaper across the board; this checks that assumption instead of assuming it).

Same caveats as gate0_bench.py: no real plans.json exists yet (needs the actual PanTS planner run),
so patch sizes here are stress-test bounds, not confirmed real values. Kill-fast: each config is
wrapped in try/except; OOM or crash is logged and we move on.
"""
import gc
import json
import time

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet

DEVICE = "cuda"
NUM_CLASSES = 28
WARMUP = 5
STEPS = 20


def build_plain_unet(n_stages, deep_supervision):
    features_per_stage = [min(32 * (2 ** i), 320) for i in range(n_stages)]
    strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
    kernel_sizes = [[3, 3, 3]] * n_stages
    model = PlainConvUNet(
        input_channels=1, n_stages=n_stages, features_per_stage=features_per_stage,
        conv_op=nn.Conv3d, kernel_sizes=kernel_sizes, strides=strides,
        n_conv_per_stage=[2] * n_stages, num_classes=NUM_CLASSES,
        n_conv_per_stage_decoder=[2] * (n_stages - 1),
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=deep_supervision,
    )
    return model


def make_targets(deep_supervision, n_outputs, batch_size, patch_size, device):
    if deep_supervision:
        return [torch.randint(0, NUM_CLASSES, (batch_size, *patch_size), device=device) for _ in range(n_outputs)]
    return torch.randint(0, NUM_CLASSES, (batch_size, *patch_size), device=device)


def run_step(model, x, y, deep_supervision, opt, ce):
    torch.cuda.synchronize()
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    out = model(x)
    if deep_supervision:
        loss = sum(ce(o, yy) for o, yy in zip(out, y)) / len(out)
    else:
        loss = ce(out, y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    return time.time() - t0


def bench_config(n_stages, patch_size, batch_size, deep_supervision, label):
    result = {"label": label, "n_stages": n_stages, "patch_size": patch_size,
              "batch_size": batch_size, "deep_supervision": deep_supervision}
    model = None
    try:
        model = build_plain_unet(n_stages, deep_supervision).to(DEVICE)
        n_outputs = n_stages - 1
        opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
        ce = nn.CrossEntropyLoss()
        x = torch.rand((batch_size, 1, *patch_size), device=DEVICE)
        y = make_targets(deep_supervision, n_outputs, batch_size, patch_size, DEVICE)

        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(WARMUP + STEPS):
            t = run_step(model, x, y, deep_supervision, opt, ce)
            if i >= WARMUP:
                times.append(t)
        times.sort()
        result.update({
            "ok": True,
            "mean_s_per_step": sum(times) / len(times),
            "p50_s_per_step": times[len(times) // 2],
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "params": sum(p.numel() for p in model.parameters()),
        })
    except torch.cuda.OutOfMemoryError as e:
        result.update({"ok": False, "error": "OOM", "detail": str(e)[:200]})
    except Exception as e:
        result.update({"ok": False, "error": type(e).__name__, "detail": str(e)[:300]})
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main():
    print(f"torch {torch.__version__} cuda_ok={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    results = {"batch_sweep": [], "ds_variant_sweep": []}

    print("\n=== [1/2] plain U-Net batch-size sweep, same patch candidates as UNet++ Gate-0 ===", flush=True)
    patch_candidates = [(64, 128, 128), (96, 160, 160)]
    for patch in patch_candidates:
        for bs in (1, 2, 4):
            label = f"plainunet_n6_ds_on_patch{patch}_bs{bs}"
            print(f"--- {label} ---", flush=True)
            r = bench_config(n_stages=6, patch_size=patch, batch_size=bs, deep_supervision=True, label=label)
            print(json.dumps(r), flush=True)
            results["batch_sweep"].append(r)
            if not r["ok"] and r.get("error") == "OOM":
                print(f"OOM at bs={bs}, patch={patch} -- skipping larger batch sizes for this patch", flush=True)
                break

    print("\n=== [2/2] DS-on vs DS-off relative cost, fixed patch/batch ===", flush=True)
    fixed_patch = (64, 128, 128)
    fixed_bs = 2
    for ds in (True, False):
        label = f"plainunet_n6_ds{ds}_patch{fixed_patch}_bs{fixed_bs}"
        print(f"--- {label} ---", flush=True)
        r = bench_config(n_stages=6, patch_size=fixed_patch, batch_size=fixed_bs, deep_supervision=ds, label=label)
        print(json.dumps(r), flush=True)
        results["ds_variant_sweep"].append(r)

    with open("gate0b_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nGATE0B_DONE", flush=True)


if __name__ == "__main__":
    main()

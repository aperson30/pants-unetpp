"""CPU-only 29-channel loss contract, not a trained segmentation experiment.

Own implementation of CE + foreground batch Dice with fp32 reductions.
Tests a directional sufficient-statistic reconstruction against autograd.
No patient data, external code execution, network, GPU or checkpoint changes.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def loss(w, x, y, dice=True):
    logits = x @ w
    ce = F.cross_entropy(logits.reshape(-1, 29), y.reshape(-1))
    if not dice:
        return ce
    p = logits.softmax(-1)[..., 1:]
    target = F.one_hot(y, 29).float()[..., 1:]
    t = (p * target).sum(1, dtype=torch.float32).sum(0, dtype=torch.float32)
    total = p.sum(1, dtype=torch.float32).sum(0, dtype=torch.float32)
    truth = target.sum(1, dtype=torch.float32).sum(0, dtype=torch.float32)
    return ce - ((2*t + 1e-5)/(total + truth + 1e-5).clamp_min(1e-8)).mean()


def gradient(w, x, y, dice=True):
    return torch.autograd.grad(loss(w, x, y, dice), w)[0]


def directional_stats(w, x, y, direction):
    """Per-patch values and directional derivatives; no per-class backprop."""
    logits = x @ w.detach()
    dz = x @ direction
    p = logits.softmax(-1)
    dp = p * (dz - (p * dz).sum(-1, keepdim=True))
    target = F.one_hot(y, 29).float()
    ce_direction = ((p-target)*dz).sum(-1).mean(1)
    p, dp, target = p[..., 1:], dp[..., 1:], target[..., 1:]
    return (ce_direction, (p*target).sum(1), p.sum(1), target.sum(1),
            (dp*target).sum(1), dp.sum(1))


def reconstruct(stats, slots):
    ce, t, p, y, dt, dp = (s[slots] for s in stats)
    t, p, y, dt, dp = (s.sum(0) for s in (t, p, y, dt, dp))
    denominator = (p+y+1e-5).clamp_min(1e-8)
    return ce.mean() + (-2*dt/denominator + (2*t+1e-5)*dp/denominator.square()).mean()


def labels(x):
    # A fixed artificial rule: 28 common categories plus a rare final category.
    ordinary = (x[..., 1].sigmoid()*28).long().clamp_max(27)
    return torch.where(x[..., 0] > 2.3, 28, ordinary)


def rare_dice(w, x, y):
    p = (x @ w).softmax(-1)[..., 28]
    target = (y == 28).to(p.dtype)
    return -(2*(p*target).sum()+1e-5)/(p.sum()+target.sum()+1e-5)


def run(trials=256):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rows = []
    for seed in range(trials):
        rng = torch.Generator().manual_seed(seed)
        w = (torch.randn(4, 29, generator=rng)*0.3).requires_grad_()
        x = torch.randn(5, 128, 4, generator=rng)
        x[..., -1] = 1
        y = labels(x)
        xv = torch.randn(4, 128, 4, generator=rng)
        xv[..., -1] = 1
        yv = labels(xv)
        old, new = [0, 1, 2, 3], [4, 1, 2, 3]
        row = {"seed": seed, "class28_counts": (y == 28).sum(1).tolist()}
        for dice in (False, True):
            gv = gradient(w, xv, yv, dice)
            g_old, g_new = gradient(w, x[old], y[old], dice), gradient(w, x[new], y[new], dice)
            exact = (gv*(g_new-g_old)).sum()
            isolated = (gv*(gradient(w, x[4:5], y[4:5], dice)-gradient(w, x[:1], y[:1], dice))).sum()/4
            result = {"exact": float(exact), "isolated": float(isolated),
                      "sign_disagreement": bool(exact*isolated < 0)}
            if dice:
                stats = directional_stats(w, x, y, gv)
                corrected = reconstruct(stats, new)-reconstruct(stats, old)
                torch.testing.assert_close(corrected, exact, rtol=2e-4, atol=2e-7)
                result.update(corrected=float(corrected), absolute_correction_error=float(abs(corrected-exact)))
            else:
                torch.testing.assert_close(isolated, exact, rtol=2e-4, atol=2e-7)
            gains = {}
            for eta in (1e-4, 1e-3, 1e-2):
                gains[str(eta)] = float(loss(w.detach()-eta*g_old, xv, yv, dice)-
                                        loss(w.detach()-eta*g_new, xv, yv, dice))
            result["finite_step_validation_gains"] = gains
            row["ce_plus_batch_dice" if dice else "ce_control"] = result
        # Prespecified follow-up on the same complete seed set: target the rare
        # category's soft Dice, not the foreground average. This is NOT recall.
        gv_rare = torch.autograd.grad(rare_dice(w, xv, yv), w)[0]
        g_old, g_new = gradient(w, x[old], y[old]), gradient(w, x[new], y[new])
        exact = (gv_rare*(g_new-g_old)).sum()
        isolated = (gv_rare*(gradient(w, x[4:5], y[4:5])-gradient(w, x[:1], y[:1]))).sum()/4
        stats = directional_stats(w, x, y, gv_rare)
        corrected = reconstruct(stats, new)-reconstruct(stats, old)
        torch.testing.assert_close(corrected, exact, rtol=2e-4, atol=2e-7)
        # Double-precision evaluation resolves tiny finite differences; training
        # gradients above are still fp32. Distinct from deployed loss parity.
        rare_gains = {}
        for eta in (1e-4, 1e-3, 1e-2):
            rare_gains[str(eta)] = float(rare_dice(w.detach().double()-eta*g_old.double(), xv.double(), yv)-
                                        rare_dice(w.detach().double()-eta*g_new.double(), xv.double(), yv))
        row["rare_class_dice_target"] = {"exact": float(exact), "isolated": float(isolated),
            "corrected": float(corrected), "sign_disagreement": bool(exact*isolated < 0),
            "absolute_correction_error": float(abs(corrected-exact)),
            "finite_step_validation_gains_float64_evaluation": rare_gains}
        rows.append(row)
    return {"scope": "Artificial linear softmax model, 29 channels; not deployed nnU-Net parity or tumor recall",
            "seed_count": trials, "batch_size": 4, "voxels_per_patch": 128,
            "gpu_hours": 0, "dtype": "float32", "cpu_seconds_excluding_import": time.perf_counter()-started,
            "summary": {key: {"sign_disagreements": sum(r[key]["sign_disagreement"] for r in rows)}
                        for key in ("ce_control", "ce_plus_batch_dice", "rare_class_dice_target")},
            "max_directional_correction_absolute_error": max(r["ce_plus_batch_dice"]["absolute_correction_error"] for r in rows),
            "all_trials": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "all_trials"}, indent=2))

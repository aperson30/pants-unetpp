"""Bounded CPU mechanism screen; artificial binary patches, not PanTS outcomes.

Compare isolated-example replacement utility with the exact BS4 CE+batch-Dice
gradient change. All seeded trials retained. No patient data, network access,
pretrained weights, GPU, optimizer tuning, author implementation or training.
Non-decomposable data attribution is known prior art; this is not a novel theorem.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch


def objective(theta, features, labels, dice=True):
    logits = features @ theta
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    if not dice:
        return ce
    p = logits.sigmoid()
    return ce - (2.0 * (p * labels).sum() + 1e-5) / (p.sum() + labels.sum() + 1e-5)


def grad(theta, x, y, dice=True):
    return torch.autograd.grad(objective(theta, x, y, dice), theta)[0]


def run(trials=256):
    torch.set_num_threads(1)
    start = time.perf_counter()
    rows = []
    for seed in range(trials):
        rng = torch.Generator().manual_seed(seed)
        theta = (torch.randn(3, generator=rng, dtype=torch.float64) * 0.8).requires_grad_()
        # Features include a constant intercept; fixed feature-derived labels.
        x = torch.randn(5, 64, 3, generator=rng, dtype=torch.float64)
        x[..., 2] = 1.0
        truth = x[..., 0] + 0.4 * x[..., 1]
        y = (truth > 1.5).to(torch.float64)
        # Replacement has the same ground-truth rule, never label flipping.
        xv = torch.randn(4, 64, 3, generator=rng, dtype=torch.float64)
        xv[..., 2] = 1.0
        yv = (xv[..., 0] + 0.4 * xv[..., 1] > 1.5).to(torch.float64)
        old_x, old_y = x[:4], y[:4]
        new_x, new_y = torch.cat((x[4:5], x[1:4])), torch.cat((y[4:5], y[1:4]))
        row = {"seed": seed, "old_patch_positive_counts": y[:4].sum(1).tolist(),
               "replacement_positive_count": int(y[4].sum())}
        for use_dice in (False, True):
            key = "ce_plus_batch_dice" if use_dice else "ce_control"
            gv = grad(theta, xv, yv, use_dice)
            exact = grad(theta, new_x, new_y, use_dice) - grad(theta, old_x, old_y, use_dice)
            isolated = (grad(theta, x[4:5], y[4:5], use_dice) -
                        grad(theta, x[:1], y[:1], use_dice)) / 4.0
            exact_score, proxy_score = float(gv @ exact), float(gv @ isolated)
            eta = 1e-4
            # Difference between two full-batch one-step validation losses.
            old_update = theta.detach() - eta * grad(theta, old_x, old_y, use_dice)
            new_update = theta.detach() - eta * grad(theta, new_x, new_y, use_dice)
            actual_gain = float(objective(old_update, xv, yv, use_dice) -
                                objective(new_update, xv, yv, use_dice))
            row[key] = {
                "exact_replacement_alignment": exact_score,
                "isolated_replacement_alignment": proxy_score,
                "sign_disagreement": exact_score * proxy_score < 0,
                "actual_validation_gain_eta_1e_4": actual_gain,
                "relative_gradient_discrepancy": float((exact - isolated).norm() /
                                                       exact.norm().clamp_min(1e-12)),
            }
            if not use_dice:
                torch.testing.assert_close(exact, isolated, rtol=1e-9, atol=1e-12)
            if abs(exact_score) > 1e-5:
                assert exact_score * actual_gain > 0, "Finite-step direction does not match local prediction"
        rows.append(row)
    summary = {}
    for key in ("ce_control", "ce_plus_batch_dice"):
        discrepancies = sorted(r[key]["relative_gradient_discrepancy"] for r in rows)
        summary[key] = {
            "sign_disagreements": sum(r[key]["sign_disagreement"] for r in rows),
            "trials": trials,
            "median_relative_gradient_discrepancy": discrepancies[len(discrepancies)//2],
        }
    return {
        "scope": "Artificial binary logistic patches, not actual 28-class trainer or medical quality",
        "batch_size": 4, "voxels_per_patch": 64, "smooth": 1e-5,
        "label_rule": "x0 + 0.4*x1 > 1.5 for every candidate and validation patch",
        "utility": "Extra validation-loss improvement from replacing slot0; not absolute update benefit",
        "cpu_seconds_excluding_import": time.perf_counter() - start,
        "gpu_hours": 0, "summary": summary, "all_trials": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(json.dumps({k: v for k, v in result.items() if k != "all_trials"}, indent=2))

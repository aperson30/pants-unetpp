"""CPU-only algebra check, NOT a reproduction of C2I's author implementation.

Checks the sign ambiguity of the squared class-influence gap printed in
arXiv:2607.12464v1 equations 5/9 versus the signed softmax in equation 7.
Uses a deliberately mislabeled synthetic point and disjoint toy validation
classes. This is not medical evidence, a generator experiment, or novelty.
Standard library only; no network, model weights, patient data or GPU.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


WEIGHTS = (0.4, -0.2, 0.1)  # Two weights plus bias.
VALIDATION = {
    0: [((1.0, 0.2), 0), ((0.8, 0.6), 0), ((1.2, 0.4), 0)],
    1: [((-1.0, -0.2), 1), ((-0.8, -0.6), 1), ((-1.2, -0.4), 1)],
}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def gradient(x, y, weights=WEIGHTS):
    z = (*x, 1.0)
    p = 1.0 / (1.0 + math.exp(-dot(weights, z)))
    return tuple((p - y) * value for value in z)


def cosine(a, b):
    return dot(a, b) / math.sqrt(dot(a, a) * dot(b, b))


def loss(x, y, weights):
    logit = dot(weights, (*x, 1.0))
    return max(logit, 0.0) + math.log1p(math.exp(-abs(logit))) - y * logit


def summarize(g):
    values = {
        c: [cosine(g, gradient(x, y)) for x, y in examples]
        for c, examples in VALIDATION.items()
    }
    means = {c: statistics.mean(v) for c, v in values.items()}
    variances = {c: statistics.variance(v) for c, v in values.items()}
    gap = means[0] - means[1]
    return {
        "means": means,
        "variances": variances,
        "squared_gap_score_eq5": gap**2 / math.sqrt(sum(variances.values())),
        "signed_softmax_eq7_fixed_condition0": 1.0 / (1.0 + math.exp(-gap)),
    }


def run():
    rows = []
    for label in (0, 1):
        g = gradient((1.0, 0.4), label)
        row = {"candidate_training_label": label, **summarize(g)}
        gv = tuple(
            statistics.mean(gradient(x, y)[j] for x, y in VALIDATION[0])
            for j in range(3)
        )
        row["class0_loss_first_order_coefficient"] = -dot(g, gv)
        eta = 1e-4
        updated = tuple(w - eta * value for w, value in zip(WEIGHTS, g))
        row["class0_loss_actual_change_eta_1e_4"] = statistics.mean(
            loss(x, y, updated) - loss(x, y, WEIGHTS)
            for x, y in VALIDATION[0]
        )
        rows.append(row)

    good, flipped = rows
    assert math.isclose(
        good["squared_gap_score_eq5"], flipped["squared_gap_score_eq5"],
        rel_tol=1e-10,
    )
    for key in ("class0_loss_first_order_coefficient", "class0_loss_actual_change_eta_1e_4"):
        assert good[key] < 0 < flipped[key]
    for scale in (0.01, 0.5, 2.0, 100.0):
        reversed_g = tuple(-scale * value for value in gradient((1.0, 0.4), 0))
        assert math.isclose(
            good["squared_gap_score_eq5"],
            summarize(reversed_g)["squared_gap_score_eq5"], rel_tol=1e-10,
        )
    return {
        "scope": "Algebra illustration, not author-code reproduction or real medical data",
        "source": "https://arxiv.org/html/2607.12464v1",
        "intended_condition_class": 0,
        "control": "Same toy point, deliberately flipped candidate training label",
        "validation": "Disjoint hand-constructed toy class features, no patient data",
        "gpu_hours": 0,
        "checks": "Equal unsigned scores; opposite first-order and finite-step class0 effects; anti-scaling invariance",
        "results": rows,
        "caveat": "Signed reward is already in the paper. No claim about implemented reward, generated-label errors or published empirical results.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload)
    print(payload, end="")

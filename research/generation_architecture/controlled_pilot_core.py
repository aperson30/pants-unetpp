"""Pure controlled-pilot contracts: engineering only, no clinical adequacy claim."""
import torch

ARMS = {"plain_conditioned": (34, 68, 136),
        "nested_conditioned": (32, 64, 128),
        "plain_phase_only": (34, 68, 136)}
SEEDS = (1729, 2718)
UPDATES = 1000


def intervene(sample, condition, replacement=None):
    if sample.ndim != 4 or sample.shape[1] != 8:
        raise ValueError("Expected concatenated four-channel latents")
    result = sample.clone()
    if condition == "correct":
        return result
    if condition == "absent":
        result[:, 4:] = 0
    elif condition == "mismatched":
        if replacement is None or replacement.shape != sample[:, 4:].shape:
            raise ValueError("Require matched replacement latent shape")
        result[:, 4:] = replacement
    else:
        raise ValueError("Unknown intervention")
    return result


def validate_arrays(arrays):
    import numpy as np
    for split, count in (("train", 8), ("development", 4)):
        for role in ("source", "target"):
            a = arrays[f"{split}_{role}"]
            if a.shape != (count, 3, 512, 512) or a.dtype != np.float32 or not np.isfinite(a).all() or np.abs(a).max() > 1:
                raise ValueError("Invalid normalized triplets")
    # Only first four training triplets are abdominal/venous candidates.
    if not np.array_equal(arrays["train_phase_index"], [0]*4+[1]*4) or not np.array_equal(arrays["development_phase_index"], [0]*4):
        raise ValueError("Phase ordering differs")


def predicted_clean(noisy, epsilon, alpha):
    return (noisy - (1-alpha).sqrt()*epsilon) / alpha.sqrt()

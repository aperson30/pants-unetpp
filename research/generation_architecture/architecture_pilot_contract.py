"""Budget/data/draw contracts for the authorized architecture-only pilot."""
import hashlib
import math

import torch


def validate_engineering_manifest(manifest, charged_cap):
    if type(charged_cap) not in (int, float) or not math.isfinite(charged_cap) or not 1/3 <= charged_cap <= .5:
        raise ValueError("Ten-minute factor-two allocation must fit explicit <=0.5h authorization")
    if manifest.get("scope") != "candidate-paired architecture-only exploratory feasibility":
        raise ValueError("Wrong dataset scope")
    if manifest.get("medical_quality_tested") is not False or manifest.get("clinical_training_eligible") is not False:
        raise ValueError("Do not mislabel exploratory inputs as clinical evidence")
    if manifest.get("phase_prompts") != ["An venous phase CT slice.", "An arterial phase CT slice."]:
        raise ValueError("Phase ordering must match the prepared arrays")
    train, dev = manifest.get("training_case_ids", []), manifest.get("development_case_ids", [])
    if len(train) != 4 or len(dev) != 2 or len(set(train + dev)) != 6:
        raise ValueError("Require six distinct public scans in provisional splits")
    digest = manifest.get("triplets_sha256", "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Require prepared-data content hash")


def draw_latents(stats, index, seed, phase_context, scheduler):
    """Fresh, explicit shared draws across arms; no fixed posterior sample cache."""
    mean_s, std_s, mean_t, std_t = stats
    g = torch.Generator(device=mean_s.device).manual_seed(seed)
    rand = lambda like: torch.randn(like.shape, dtype=like.dtype, device=like.device, generator=g)
    source = mean_s[index:index+1] + std_s[index:index+1] * rand(mean_s[index:index+1])
    target = mean_t[index:index+1] + std_t[index:index+1] * rand(mean_t[index:index+1])
    noise = rand(target)
    timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=source.device, generator=g)
    noisy = scheduler.add_noise(target, noise, timestep)
    return torch.cat((noisy, source), dim=1), timestep, phase_context, noise


def require_data_hash(path, expected):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("Pilot arrays differ from manifest")

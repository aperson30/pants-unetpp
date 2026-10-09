"""Small epsilon/DDIM mathematical tests, not the production SMILE scheduler.

Betas, timestep sequence and exit sequence must be supplied explicitly. No
learned schedule, data selection, guidance or clinical claim is encoded here.
Production experiments must use the baseline's pinned scheduler/configuration.
"""

import torch


def cumulative_alphas(betas):
    if (betas.ndim != 1 or not betas.is_floating_point() or len(betas) == 0
            or not torch.isfinite(betas).all() or (betas <= 0).any() or (betas >= 1).any()):
        raise ValueError("Supply finite 1D beta values strictly between zero and one")
    alphas = torch.cumprod(1 - betas, dim=0)
    if (alphas <= 0).any() or (alphas >= 1).any():
        raise ValueError("Schedule loses numerical precision; use higher precision")
    return alphas


def noisy_sample(clean, noise, alphas, timesteps):
    if clean.shape != noise.shape or clean.device != noise.device or clean.dtype != noise.dtype:
        raise ValueError("Clean/noise tensor contracts must match")
    if (timesteps.shape != (clean.shape[0],) or timesteps.dtype != torch.long
            or (timesteps < 0).any() or (timesteps >= len(alphas)).any()):
        raise ValueError("Expected valid integer timestep per example")
    a = alphas.to(clean.device)[timesteps.to(clean.device)].to(clean.dtype)
    a = a.reshape((-1,) + (1,) * (clean.ndim - 1))
    return a.sqrt() * clean + (1 - a).sqrt() * noise


def epsilon_ddim_step(sample, epsilon, alpha_now, alpha_next):
    if sample.shape != epsilon.shape:
        raise ValueError("Denoiser residual shape must match noisy latent, not concatenated input")
    if not (0 < alpha_now <= alpha_next <= 1):
        raise ValueError("DDIM reverse step needs increasing cumulative alpha")
    clean = (sample - (1 - alpha_now) ** 0.5 * epsilon) / alpha_now ** 0.5
    return alpha_next ** 0.5 * clean + (1 - alpha_next) ** 0.5 * epsilon


@torch.no_grad()
def contract_trajectory(model, initial_noise, source_latents, context, alphas, timesteps, heads):
    """Deterministic eta=0 reference; caller owns all RNG and schedule decisions."""
    if initial_noise.shape != source_latents.shape:
        raise ValueError("Source latents must match noisy latents")
    timesteps, heads = tuple(timesteps), tuple(heads)
    if not timesteps or len(timesteps) != len(heads):
        raise ValueError("One explicit head per diffusion step is required")
    if any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < len(alphas) for t in timesteps):
        raise ValueError("Invalid diffusion timestep")
    if any(a <= b for a, b in zip(timesteps, timesteps[1:])):
        raise ValueError("Reverse timesteps must strictly decrease")
    if any(isinstance(h, bool) or not isinstance(h, int) or not 1 <= h <= model.depth for h in heads):
        raise ValueError("Invalid denoiser head")
    sample = initial_noise.clone()
    for index, (t, head) in enumerate(zip(timesteps, heads)):
        epsilon = model(torch.cat((sample, source_latents), dim=1), t, context,
                        head=head, return_dict=False)[0]
        if not torch.isfinite(epsilon).all():
            raise FloatingPointError("Nonfinite denoiser output")
        alpha_next = float(alphas[timesteps[index + 1]]) if index + 1 < len(timesteps) else 1.0
        sample = epsilon_ddim_step(sample, epsilon, float(alphas[t]), alpha_next)
        if not torch.isfinite(sample).all():
            raise FloatingPointError("Nonfinite sampled latent")
    return sample

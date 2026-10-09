"""Original experimental 2D denoiser; NOT a trained SMILE replacement.

Input is concatenated noisy and source latents; output is a continuous residual,
not segmentation logits. Token cross-attention preserves phase conditioning.
Head j retains exactly the triangle i+k<=j. This preserves that head's result,
NOT the deepest head's result. No SMILE implementation is copied here.

This small contract prototype is not a complete diffusers ModelMixin: production
integration, checkpoint/pretraining strategy, losses and capacity are unverified.
"""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class ConditionedBlock(nn.Module):
    def __init__(self, in_channels, channels, time_dim, context_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(math.gcd(channels, 8), channels)
        self.time_affine = nn.Linear(time_dim, 2 * channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(math.gcd(channels, 8), channels)
        self.skip = nn.Conv2d(in_channels, channels, 1)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(context_dim, channels)
        self.value = nn.Linear(context_dim, channels)
        self.attn_out = nn.Linear(channels, channels)

    def forward(self, x, time, context):
        h = self.norm1(self.conv1(x))
        scale, shift = self.time_affine(time).chunk(2, dim=1)
        h = F.silu(h * (1 + scale[:, :, None, None]) + shift[:, :, None, None])
        h = self.norm2(self.conv2(h)) + self.skip(x)
        b, c, height, width = h.shape
        tokens = h.flatten(2).transpose(1, 2)
        # Cross-attention: O(spatial tokens * text tokens), not spatial self-attention.
        attended = F.scaled_dot_product_attention(
            self.query(tokens)[:, None], self.key(context)[:, None],
            self.value(context)[:, None], dropout_p=0.0,
        )[:, 0]
        h = h + self.attn_out(attended).transpose(1, 2).reshape(b, c, height, width)
        return F.silu(h)


class ConditionalNestedDenoiser(nn.Module):
    def __init__(self, channels=(32, 64, 128), latent_channels=4,
                 context_dim=768, time_dim=64, dense=True):
        super().__init__()
        if len(channels) < 2 or any(c < 2 for c in channels):
            raise ValueError("Need at least two resolution levels with >=2 channels")
        if time_dim < 4 or time_dim % 2:
            raise ValueError("time_dim must be even and >=4")
        if latent_channels < 1 or context_dim < 1:
            raise ValueError("Channel dimensions must be positive")
        self.channels = tuple(channels)
        self.depth = len(channels) - 1
        self.dense = dense
        self.time_dim = time_dim
        self.config = SimpleNamespace(in_channels=2 * latent_channels,
                                      out_channels=latent_channels,
                                      cross_attention_dim=context_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        self.nodes = nn.ModuleDict()
        for i, c in enumerate(channels):
            inputs = self.config.in_channels if i == 0 else channels[i - 1]
            self.nodes[f"{i}_0"] = ConditionedBlock(inputs, c, time_dim, context_dim)
        for j in range(1, self.depth + 1):
            for i in range(len(channels) - j):
                inputs = channels[i] * (j if dense else 1) + channels[i + 1]
                self.nodes[f"{i}_{j}"] = ConditionedBlock(inputs, channels[i], time_dim, context_dim)
        self.heads = nn.ModuleDict({str(j): nn.Conv2d(channels[0], latent_channels, 1)
                                   for j in range(1, self.depth + 1)})

    def _time(self, timestep, sample):
        t = torch.as_tensor(timestep, device=sample.device)
        if t.ndim == 0:
            t = t.expand(sample.shape[0])
        if t.shape != (sample.shape[0],) or not torch.isfinite(t).all() or (t < 0).any():
            raise ValueError("Expected finite nonnegative scalar or batch timesteps")
        frequencies = torch.exp(-math.log(10000) * torch.arange(
            self.time_dim // 2, device=sample.device, dtype=torch.float32
        ) / (self.time_dim // 2 - 1))
        phase = t.float()[:, None] * frequencies[None]
        return self.time_mlp(torch.cat([phase.cos(), phase.sin()], 1).to(sample.dtype))

    def forward(self, sample, timestep, encoder_hidden_states, *, head=None,
                all_heads=False, return_dict=True, **kwargs):
        # Do not silently discard a diffusers conditioning feature we haven't implemented.
        if any(v is not None for v in kwargs.values()):
            raise ValueError("Nonempty extra conditioning is not implemented in this prototype")
        if sample.ndim != 4 or sample.shape[1] != self.config.in_channels:
            raise ValueError("Expected B x (noisy+source latent channels) x H x W")
        if not sample.is_floating_point():
            raise ValueError("Latents must be floating point")
        limit = self.depth if head is None else head
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.depth:
            raise ValueError("Invalid exit head")
        if all_heads and head is not None:
            raise ValueError("Choose all heads OR a specific exit")
        if min(sample.shape[-2:]) < 2 ** limit:
            raise ValueError("Input too small for requested depth")
        context = encoder_hidden_states
        if (context.ndim != 3 or context.shape[0] != sample.shape[0]
                or context.shape[1] < 1 or context.shape[2] != self.config.cross_attention_dim
                or context.device != sample.device or context.dtype != sample.dtype):
            raise ValueError("Context must match latent batch/device/dtype and token width")
        time = self._time(timestep, sample)
        grid = {}
        for i in range(limit + 1):
            x = sample if i == 0 else F.avg_pool2d(grid[i - 1, 0], 2)
            grid[i, 0] = self.nodes[f"{i}_0"](x, time, context)
        for j in range(1, limit + 1):
            for i in range(limit + 1 - j):
                up = F.interpolate(grid[i + 1, j - 1], size=grid[i, 0].shape[-2:],
                                   mode="bilinear", align_corners=False)
                skips = [grid[i, k] for k in range(j)] if self.dense else [grid[i, j - 1]]
                grid[i, j] = self.nodes[f"{i}_{j}"](torch.cat(skips + [up], 1), time, context)
        if all_heads:
            return tuple(self.heads[str(j)](grid[0, j]) for j in range(1, limit + 1))
        result = self.heads[str(limit)](grid[0, limit])
        return SimpleNamespace(sample=result) if return_dict else (result,)


def supervised_denoising_loss(outputs, target, weights):
    """Explicit weights: no hidden normalization or segmentation loss reuse."""
    weights = tuple(weights)
    if not outputs or len(outputs) != len(weights):
        raise ValueError("Provide one explicit weight per output")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be finite/nonnegative with positive sum")
    if any(output.shape != target.shape for output in outputs):
        raise ValueError("Every head must predict the same denoising target shape")
    return sum(w * F.mse_loss(output.float(), target.float())
               for output, w in zip(outputs, weights) if w > 0)

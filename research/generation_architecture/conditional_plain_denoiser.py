"""Original conventional encoder-decoder control; not pretrained SMILE.

Uses the nested fixture's conditioning blocks, but a single U-shaped decoder
and one skip per resolution. Equal widths do NOT imply equal parameter count.
This is a small controlled architecture fixture, not a capacity-matched SD1.5.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from conditional_nested_denoiser import ConditionedBlock, ConditionalNestedDenoiser


class ConditionalPlainDenoiser(nn.Module):
    def __init__(self, channels=(32, 64, 128), latent_channels=4,
                 context_dim=768, time_dim=64):
        super().__init__()
        if len(channels) < 2 or any(c < 2 for c in channels):
            raise ValueError("Need at least two resolution levels with >=2 channels")
        if time_dim < 4 or time_dim % 2 or latent_channels < 1 or context_dim < 1:
            raise ValueError("Invalid conditioning dimensions")
        self.channels = tuple(channels)
        self.depth = len(channels) - 1
        self.time_dim = time_dim
        self.config = SimpleNamespace(in_channels=2 * latent_channels,
                                      out_channels=latent_channels,
                                      cross_attention_dim=context_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        self.encoder = nn.ModuleList([
            ConditionedBlock(self.config.in_channels if i == 0 else channels[i - 1],
                             c, time_dim, context_dim) for i, c in enumerate(channels)])
        self.decoder = nn.ModuleDict({str(i): ConditionedBlock(
            channels[i] + channels[i + 1], channels[i], time_dim, context_dim)
            for i in range(self.depth)})
        self.head = nn.Conv2d(channels[0], latent_channels, 1)

    def forward(self, sample, timestep, encoder_hidden_states, *, return_dict=True, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError("Extra conditioning/exit features not implemented")
        if (sample.ndim != 4 or sample.shape[1] != self.config.in_channels
                or not sample.is_floating_point() or sample.shape[0] < 1
                or min(sample.shape[-2:]) < 2 ** self.depth):
            raise ValueError("Invalid concatenated latent shape/dtype")
        context = encoder_hidden_states
        if (context.ndim != 3 or context.shape[0] != sample.shape[0]
                or context.shape[1] < 1 or context.shape[2] != self.config.cross_attention_dim
                or context.device != sample.device or context.dtype != sample.dtype):
            raise ValueError("Context must match latent batch/device/dtype/token width")
        # Reuse the exact tested timestep embedding, without constructing nested nodes.
        time = ConditionalNestedDenoiser._time(self, timestep, sample)
        skips = []
        x = sample
        for i, block in enumerate(self.encoder):
            if i:
                x = F.avg_pool2d(x, 2)
            x = block(x, time, context)
            skips.append(x)
        for i in reversed(range(self.depth)):
            x = F.interpolate(x, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.decoder[str(i)](torch.cat((skips[i], x), dim=1), time, context)
        result = self.head(x)
        return SimpleNamespace(sample=result) if return_dict else (result,)

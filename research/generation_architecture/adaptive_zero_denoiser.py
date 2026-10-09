"""Original convolutional AdaLN-zero adapter for PI-direction preparation.

Channel LayerNorm, timestep-derived shift/scale/residual gates, zero modulation
and zero prediction heads. NOT a copied DiT implementation, pretrained SMILE
replacement, or validated production architecture. Pruning retains base graph.
"""
import torch
from torch import nn
from torch.nn import functional as F

from conditional_nested_denoiser import ConditionalNestedDenoiser
from conditional_plain_denoiser import ConditionalPlainDenoiser


class AdaptiveZeroBlock(nn.Module):
    def __init__(self, in_channels, channels, time_dim, context_dim):
        super().__init__()
        self.channels = channels
        self.project = nn.Conv2d(in_channels, channels, 1)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 6*channels))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(context_dim, channels)
        self.value = nn.Linear(context_dim, channels)
        self.attn_out = nn.Linear(channels, channels)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1))

    def norm(self, x):
        return F.layer_norm(x.permute(0,2,3,1), (self.channels,)).permute(0,3,1,2)

    def forward(self, x, time, context):
        h = self.project(x)
        s1, a1, g1, s2, a2, g2 = [v[:, :, None, None] for v in self.modulation(time).chunk(6,1)]
        q = (self.norm(h)*(1+a1)+s1).flatten(2).transpose(1,2)
        attended = F.scaled_dot_product_attention(self.query(q)[:, None], self.key(context)[:, None],
                                                  self.value(context)[:, None], dropout_p=0)[:,0]
        attended = self.attn_out(attended).transpose(1,2).reshape_as(h)
        h = h + g1*attended
        return h + g2*self.conv(self.norm(h)*(1+a2)+s2)


def replacement(block, time_dim, context_dim):
    return AdaptiveZeroBlock(block.conv1.in_channels, block.conv1.out_channels, time_dim, context_dim)


class AdaptiveZeroNestedDenoiser(ConditionalNestedDenoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, block in self.nodes.items():
            self.nodes[key] = replacement(block, self.time_dim, self.config.cross_attention_dim)
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)


class AdaptiveZeroPlainDenoiser(ConditionalPlainDenoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i, block in enumerate(self.encoder):
            self.encoder[i] = replacement(block, self.time_dim, self.config.cross_attention_dim)
        for key, block in self.decoder.items():
            self.decoder[key] = replacement(block, self.time_dim, self.config.cross_attention_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

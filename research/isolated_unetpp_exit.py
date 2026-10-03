"""Research-only truncated inference. Never monkeypatches production models.

Caller must verify checkpoint provenance and whether each head was supervised.
Flag checks here reject obvious unsupported configurations, not prove training.
"""
import torch


@torch.inference_mode()
def forward_exit(network, x, depth):
    decoder = network.decoder
    if network.training:
        raise ValueError('Inference only: model must be in eval mode')
    if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= decoder.L:
        raise ValueError('Invalid nesting depth')
    if not decoder.deep_supervision or decoder.average_outputs_at_inference:
        raise ValueError('Require audited DS-on individual-branch configuration')
    if decoder.skip_shallowest_deep_supervision_head and depth == 1:
        raise ValueError('Shallowest head was omitted from supervision')
    stages = network.encoder.stages
    if len(stages) != decoder.L+1:
        raise ValueError('Unexpected encoder stage topology')
    nodes = {}
    for i in range(depth+1):
        x = stages[i](x)
        nodes[i, 0] = x
    for j in range(1, depth+1):
        for i in range(depth-j, -1, -1):
            key = f'{i}_{j}'
            upsampled = decoder.transpconvs[key](nodes[i+1, j-1])
            cat = torch.cat([nodes[i, k] for k in range(j)]+[upsampled], dim=1)
            nodes[i, j] = decoder.convs[key](cat)
    return decoder.seg_layers[f'0_{depth}'](nodes[0, depth])

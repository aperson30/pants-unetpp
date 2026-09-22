"""
UNet++ ported to nnU-Net v2 conventions.

Why this file exists (read this before touching the code):
The official UNet++ implementation (MrGiovanni/UNetPlusPlus/pytorch/nnunet/network_architecture/
generic_UNetPlusPlus.py) is hardcoded for exactly 5 downsampling stages (it builds precisely
`loc0`..`loc4` and its forward() indexes `conv_blocks_context[0..5]`). nnU-Net v1 always happened to
pass in a network with 5 pooling stages for the datasets it was built for, so nobody noticed this was
hardcoded rather than general. nnU-Net v2's automatic experiment planner does NOT always pick 5 stages
-- the number of stages depends on the dataset's median image size and spacing, so for PanTS it could
be 5, 6, or 7. Blindly copying the official v1 forward() into v2 would work by luck on some datasets
and silently produce a wrong/broken network graph (or crash) on others.

This module re-derives the same nested, densely-skip-connected UNet++ architecture but builds the
nesting grid with loops over an arbitrary number of encoder stages (n_stages), the way every other
nnU-Net v2 architecture (PlainConvUNet, ResidualEncoderUNet) is written. The math/architecture is
identical to the original paper -- only the "hardcoded 5" assumption has been removed.

Second important difference from the official v1 code, and the reason a custom trainer
(nnUNetTrainerUNetPlusPlus.py, next to this file) is required, not just this network file:
UNet++'s own deep supervision outputs (x0_1, x0_2, ... in the original paper) are all produced at
the SAME (full input) resolution -- they differ in how many nested refinement steps feed into them,
not in spatial size. This is different from nnU-Net's default deep supervision scheme, where each
auxiliary output is at a progressively downsampled resolution and the ground truth is downsampled to
match. The custom trainer handles this by telling nnU-Net "don't downsample the targets for any of
these outputs" -- see the comment in nnUNetTrainerUNetPlusPlus.py's _get_deep_supervision_scales
override for details.
"""
import math
from typing import Union, Type, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0


class UNetPlusPlusDecoder(nn.Module):
    """
    Builds the UNet++ nested skip-connection grid on top of a PlainConvEncoder.

    Notation (matches the UNet++ paper): X[i][j] is the feature map at encoder resolution level i
    (0 = full input resolution, L = bottleneck, where L = n_stages - 1) and nesting depth j.
    - X[i][0] is just the plain encoder's output at level i (no decoder work needed).
    - X[i][j] for j >= 1 is built from: all shallower same-level nodes X[i][0..j-1], concatenated
      with an upsampled X[i+1][j-1] from one level deeper. This is what creates the "nested" dense
      skip connections that distinguish UNet++ from a plain U-Net.
    - The row i=0 nodes X[0][1..L] are the deep-supervision outputs. X[0][L] (the last, most-nested
      one) is the main/final prediction.
    """

    def __init__(self,
                 encoder: PlainConvEncoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 deep_supervision: bool,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 average_outputs_at_inference: bool = False,
                 skip_shallowest_deep_supervision_head: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        # When deep supervision is switched off (which is what nnU-Net does for validation and
        # inference), the default is to return the single most-nested output X[0][L]. Setting this
        # to True instead returns the MEAN of all nested outputs, which is what the UNet++ paper
        # specifies ("the segmentation results from all segmentation branches are collected and then
        # averaged"). Only nnUNetTrainerUNetPlusPlusPaper turns this on -- see that file for why the
        # two settings have to travel together.
        self.average_outputs_at_inference = average_outputs_at_inference
        # nnU-Net's default single-GPU deep-supervision loss assigns exactly zero weight to the
        # final returned output. Because we return the branches deepest-first, that is X[0][1].
        # A trainer may opt out of evaluating that segmentation head while keeping the parameter in
        # the state dict for checkpoint compatibility. The paper trainer must leave this False.
        self.skip_shallowest_deep_supervision_head = skip_shallowest_deep_supervision_head
        self.encoder = encoder
        self.num_classes = num_classes

        n_stages_encoder = len(encoder.output_channels)  # L + 1 resolution levels, incl. bottleneck
        self.L = n_stages_encoder - 1
        if self.L < 1:
            raise ValueError("UNet++ needs at least 2 encoder stages (1 downsampling step)")

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * self.L
        assert len(n_conv_per_stage) == self.L, \
            f"n_conv_per_stage must have {self.L} entries (one per non-bottleneck resolution level), got {len(n_conv_per_stage)}"
        self.n_conv_per_stage = n_conv_per_stage

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        # every node in row i (any j) keeps the same channel count as the encoder's level-i output
        self.transpconvs = nn.ModuleDict()
        self.convs = nn.ModuleDict()
        self.seg_layers = nn.ModuleDict()  # only row 0 (i=0) nodes are used as outputs

        for i in range(self.L):  # i = 0 .. L-1 (row L is the bottleneck itself, no nested nodes there)
            ch_i = encoder.output_channels[i]
            ch_below = encoder.output_channels[i + 1]
            stride_for_transpconv = encoder.strides[i + 1]
            for j in range(1, self.L - i + 1):
                key = f"{i}_{j}"
                self.transpconvs[key] = transpconv_op(
                    ch_below, ch_i, stride_for_transpconv, stride_for_transpconv, bias=conv_bias
                )
                in_channels = (j + 1) * ch_i  # j same-row predecessors + 1 upsampled feature
                self.convs[key] = StackedConvBlocks(
                    n_conv_per_stage[i], encoder.conv_op, in_channels, ch_i,
                    encoder.kernel_sizes[i], 1, conv_bias, norm_op, norm_op_kwargs,
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                if i == 0:
                    self.seg_layers[key] = encoder.conv_op(ch_i, num_classes, 1, 1, 0, bias=True)

    def forward(self, skips: List[torch.Tensor]):
        """skips: encoder output list, skips[i] = X[i][0], skips[-1] = bottleneck (level L)."""
        assert len(skips) == self.L + 1
        nodes = {(i, 0): skips[i] for i in range(self.L + 1)}

        # fill the grid one nesting-depth column at a time; each X[i][j] needs X[i+1][j-1], which is
        # always already computed at this point because it belongs to an earlier or equal column
        for j in range(1, self.L + 1):
            for i in range(self.L - j, -1, -1):
                key = f"{i}_{j}"
                upsampled = self.transpconvs[key](nodes[(i + 1, j - 1)])
                cat_in = torch.cat([nodes[(i, k)] for k in range(j)] + [upsampled], dim=1)
                nodes[(i, j)] = self.convs[key](cat_in)

        # row-0 outputs, in increasing nesting depth: seg_outputs[0] = shallowest (weakest), ...,
        # seg_outputs[-1] = X[0][L] = the main/final prediction (most nested, highest quality)
        if self.deep_supervision:
            first_j = 2 if self.skip_shallowest_deep_supervision_head else 1
            seg_outputs = [self.seg_layers[f"0_{j}"](nodes[(0, j)])
                           for j in range(first_j, self.L + 1)]
            # nnU-Net v2 convention: index 0 of the returned list/tuple is the primary output
            return seg_outputs[::-1]
        elif self.average_outputs_at_inference:
            seg_outputs = [self.seg_layers[f"0_{j}"](nodes[(0, j)])
                           for j in range(1, self.L + 1)]
            # The paper averages branch PREDICTIONS after the output nonlinearity, not raw logits.
            # PanTS is an exclusive 29-class nnU-Net task, so that nonlinearity is softmax. Return
            # log(mean(softmax(branch_logits))) as an equivalent logit representation: nnU-Net's
            # downstream softmax then recovers the arithmetic mean of branch probabilities exactly.
            # log_softmax/logsumexp is stable even when a rare class probability is extremely small.
            #
            # nnU-Net subsequently averages these returned values across mirror augmentations and
            # overlapping windows before its final softmax. Standard nnU-Net does the same with raw
            # logits (a normalized geometric aggregation of per-window distributions), so this keeps
            # the framework's TTA/window semantics while making the branch ensemble paper-faithful.
            log_probabilities = torch.stack(
                [torch.log_softmax(output.float(), dim=1) for output in seg_outputs], dim=0
            )
            return torch.logsumexp(log_probabilities, dim=0) - math.log(len(seg_outputs))
        else:
            # Validation/inference without branch averaging needs only the deepest head. Computing
            # every other 1x1x1 head produced large full-resolution tensors that were discarded.
            return self.seg_layers[f"0_{self.L}"](nodes[(0, self.L)])

    def compute_conv_feature_map_size(self, input_size):
        """Rough activation-memory estimate, used by nnU-Net's planner for VRAM budgeting."""
        sizes = [input_size]
        for s in range(1, self.L + 1):
            sizes.append([i // j for i, j in zip(sizes[-1], self.encoder.strides[s])])
        # sizes[i] = spatial size at encoder level i

        output = np.int64(0)
        for i in range(self.L):
            for j in range(1, self.L - i + 1):
                key = f"{i}_{j}"
                output += self.convs[key].compute_conv_feature_map_size(sizes[i])
                output += np.prod([self.encoder.output_channels[i], *sizes[i]], dtype=np.int64)  # transpconv out
                if i == 0 and (j == self.L or
                               (self.deep_supervision and
                                not (self.skip_shallowest_deep_supervision_head and j == 1)) or
                               self.average_outputs_at_inference):
                    output += np.prod([self.num_classes, *sizes[0]], dtype=np.int64)
        return output


class UNetPlusPlus(nn.Module):
    """Full UNet++: PlainConvEncoder + UNetPlusPlusDecoder. Drop-in replacement network class for
    nnU-Net v2's `arch_class_name` mechanism (see nnUNetTrainerUNetPlusPlus.py for how it's selected).
    """

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 average_outputs_at_inference: bool = False,
                 skip_shallowest_deep_supervision_head: bool = False):
        super().__init__()
        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first
        )
        self.decoder = UNetPlusPlusDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
            average_outputs_at_inference=average_outputs_at_inference,
            skip_shallowest_deep_supervision_head=skip_shallowest_deep_supervision_head,
            nonlin_first=nonlin_first
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        return self.encoder.compute_conv_feature_map_size(input_size) + \
            self.decoder.compute_conv_feature_map_size(input_size)

    def initialize(self):
        InitWeights_He(1e-2)(self)
        init_last_bn_before_add_to_0(self)

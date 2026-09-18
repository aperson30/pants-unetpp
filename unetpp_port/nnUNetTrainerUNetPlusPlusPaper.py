"""
UNet++ configured to follow the original paper as closely as nnU-Net v2 allows.

Differs from nnUNetTrainerUNetPlusPlus in exactly two ways, and BOTH matter -- see the warning below
before using either one on its own:

  1. LOSS WEIGHTING. The paper defines the overall loss as a weighted sum over the nested decoders,
     L = sum_i eta_i * L(Y, P_i), and states "we give same balanced weights eta_i to each loss,
     i.e. eta_i == 1". This class overrides _build_loss() to weight every deep-supervision output
     equally, replacing nnU-Net's exponential decay (1, 1/2, 1/4, ...) with its last entry zeroed.

  2. INFERENCE. The paper collects "the segmentation results from all segmentation branches" and
     averages them. nnU-Net switches deep supervision off for validation and inference, at which
     point the default port returns only the most-nested output X[0][L]. This class builds the
     network with average_outputs_at_inference=True so that path returns the mean of all branches
     instead.

=== WARNING: DO NOT APPLY THE LOSS CHANGE WITHOUT THE INFERENCE CHANGE ===

Equal weighting was tried on its own during the September 2026 campaign, keeping nnU-Net's
final-branch-only inference, and it was measurably much worse: 0% of epochs detected any tumour
through epoch 500, against 87% for the otherwise identical run using nnU-Net's decaying weights.
Same architecture, same deep supervision setting, same batch size, same 4-GPU configuration -- the
loss weighting was the only variable.

The interpretation is that the paper's two rules are a matched pair. If every branch is averaged into
the final prediction, training every branch equally is coherent. If only the deepest branch is used,
equal weighting spends roughly three quarters of the gradient on outputs that are then discarded,
starving the branch that actually produces the prediction. That matters little for large common
organs and a great deal for the pancreatic tumour, which appears in only ~10% of cases.

So this class exists to test the paper's COMPLETE design, not to re-run the half that already failed.

=== KNOWN APPROXIMATION ===

The paper applies its final nonlinearity per branch and averages the resulting probability maps.
This implementation averages raw LOGITS, because nnU-Net v2 requires the network to return logits and
applies softmax itself downstream. Averaging logits is not mathematically identical to averaging
probabilities. It is the closest faithful option that remains compatible with nnU-Net's inference
path; if results come out close to the default configuration, this is the first approximation to
revisit.

=== HOW TO EVALUATE THIS FAIRLY ===

Run it against nnUNetTrainerUNetPlusPlus on the same fold and plans, and compare tumour-class metrics
on the official test set -- not aggregate Dice, which is dominated by the 27 other structures and
looks healthy (~0.67) even when tumour detection has failed completely.

Judge nothing before epoch ~500. Tumour learning here has a delayed onset: a healthy run shows
exactly 0% tumour detection for its first ~300 epochs and is indistinguishable from total failure.

    nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerUNetPlusPlusPaper -p nnUNetPlansBS4 -num_gpus 4
"""
from typing import Union

import numpy as np
import torch
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

from .nnUNetTrainerUNetPlusPlus import nnUNetTrainerUNetPlusPlus
from .unet_plusplus import UNetPlusPlus


class nnUNetTrainerUNetPlusPlusPaper(nnUNetTrainerUNetPlusPlus):

    # MUST override this back to False: the base trainer skips computing the shallowest,
    # zero-weighted deep-supervision head as a speed optimization (see nnUNetTrainerUNetPlusPlus's
    # class attribute docstring), but this trainer weights every branch equally and averages every
    # branch at inference -- there is no zero-weighted branch here to skip. Inheriting True from the
    # base class would silently drop a branch this trainer actually needs.
    skip_shallowest_deep_supervision_head = False

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                    configuration_manager: ConfigurationManager,
                                    num_input_channels: int,
                                    num_output_channels: int,
                                    enable_deep_supervision: bool = True) -> nn.Module:
        import pydoc
        from copy import deepcopy

        arch_kwargs = deepcopy(configuration_manager.network_arch_init_kwargs)
        for req_import_key in configuration_manager.network_arch_init_kwargs_req_import:
            if arch_kwargs[req_import_key] is not None:
                arch_kwargs[req_import_key] = pydoc.locate(arch_kwargs[req_import_key])

        return UNetPlusPlus(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            average_outputs_at_inference=True,   # <-- the paper's inference rule
            skip_shallowest_deep_supervision_head=False,   # every branch is needed, see above
            **arch_kwargs
        )

    def _build_loss(self):
        """Identical to nnU-Net's default except for the weights: every deep-supervision output is
        weighted equally, per the paper's eta_i == 1."""
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                    {'batch_dice': self.configuration_manager.batch_dice,
                                     'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                   weight_ce=1, weight_dice=1,
                                   ignore_label=self.label_manager.ignore_label,
                                   dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # eta_i == 1 for every output, normalised so the total loss stays on a comparable scale
            # to the default trainer's. Note no entry is zeroed: under this scheme the shallowest
            # branch IS supervised, unlike nnU-Net's default where it is dropped.
            weights = np.ones(len(deep_supervision_scales), dtype=float)
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        if self._do_i_compile():
            loss = torch.compile(loss, mode="reduce-overhead")
        return loss

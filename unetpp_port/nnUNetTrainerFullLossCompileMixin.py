"""Compile nnU-Net's complete loss instead of only its Dice submodule.

nnU-Net leaves CE and DeepSupervisionWrapper eager because PyTorch 2.2.2 used to crash when CE was
compiled. The project environment uses PyTorch 2.10.0, where the complete loss passes forward and
gradient parity tests and is faster on the target GB10. Keep this mixin on the plain-U-Net side so
the optimization is applied symmetrically across the architecture comparison.
"""
import numpy as np
import torch
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class nnUNetTrainerFullLossCompileMixin:
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        if self.enable_deep_supervision:
            weights = np.array([1 / (2**i) for i in range(len(self._get_deep_supervision_scales()))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        if self._do_i_compile():
            loss = torch.compile(loss, mode="reduce-overhead")
        return loss

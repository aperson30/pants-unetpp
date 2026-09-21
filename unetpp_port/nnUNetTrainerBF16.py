"""
Otherwise-stock nnU-Net trainer (the plain-U-Net "control group" per the project README) with only
the bf16 change applied -- see nnUNetTrainerBF16Mixin.py for why, and why this exists as its own
file rather than leaving the plain-U-Net side on fp16: applying bf16 to only the UNet++ side would
make precision an unintended second variable in the architecture comparison.

    nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerBF16 -p nnUNetPlansBS4
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from .nnUNetTrainerBF16Mixin import nnUNetTrainerBF16Mixin
from .nnUNetTrainerFullLossCompileMixin import nnUNetTrainerFullLossCompileMixin
from .nnUNetTrainerQualityNeutralOptimizationMixin import nnUNetTrainerQualityNeutralOptimizationMixin


class nnUNetTrainerBF16(nnUNetTrainerQualityNeutralOptimizationMixin,
                       nnUNetTrainerFullLossCompileMixin,
                       nnUNetTrainerBF16Mixin,
                       nnUNetTrainer):
    pass

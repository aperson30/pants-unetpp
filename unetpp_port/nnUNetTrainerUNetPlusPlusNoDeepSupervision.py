"""
Same UNet++ architecture as nnUNetTrainerUNetPlusPlus.py, but with deep supervision turned off --
for the 2x2 comparison (architecture x deep supervision) requested alongside the main run.

Mirrors how nnU-Net's own built-in nnUNetTrainerNoDeepSupervision works: just flips one flag after
the normal setup. Since UNetPlusPlus's own forward() already returns a single tensor (not a list)
when deep_supervision=False, and our _get_deep_supervision_scales() override (inherited from
nnUNetTrainerUNetPlusPlus) already returns None when deep supervision is off, this Just Works without
any further changes -- both pieces were already written to handle this case correctly.
"""
import torch

from .nnUNetTrainerUNetPlusPlus import nnUNetTrainerUNetPlusPlus


class nnUNetTrainerUNetPlusPlusNoDeepSupervision(nnUNetTrainerUNetPlusPlus):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

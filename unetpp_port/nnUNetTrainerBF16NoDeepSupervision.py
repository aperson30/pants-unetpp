"""
The plain-U-Net, deep-supervision-off leg of the grid, with bf16 applied -- mirrors how
nnUNetTrainerUNetPlusPlusNoDeepSupervision mirrors nnUNetTrainerUNetPlusPlus.

    nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerBF16NoDeepSupervision -p nnUNetPlansBS4
"""
import torch

from .nnUNetTrainerBF16 import nnUNetTrainerBF16


class nnUNetTrainerBF16NoDeepSupervision(nnUNetTrainerBF16):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

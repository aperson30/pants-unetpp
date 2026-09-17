"""
Not for real training. Runs 2 epochs of 3 iterations each (no validation) purely to exercise the
real nnUNetv2_train CLI end-to-end -- dataset.json, plans file, checkpoint saving, the real
dataloader/augmentation pipeline, the real trainer class -- against real (if tiny) PanTS data,
before committing to a multi-day run on the full dataset. If this doesn't finish cleanly, don't
launch the real grid yet.
"""
import torch

from .nnUNetTrainerUNetPlusPlus import nnUNetTrainerUNetPlusPlus


class nnUNetTrainerUNetPlusPlusQuickSmoke(nnUNetTrainerUNetPlusPlus):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 3
        self.num_val_iterations_per_epoch = 2
        self.save_every = 1

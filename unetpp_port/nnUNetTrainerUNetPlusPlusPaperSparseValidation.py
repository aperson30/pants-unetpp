"""Paper-faithful UNet++ configuration with validation every five epochs."""
from .nnUNetTrainerSparseValidationMixin import nnUNetTrainerSparseValidationMixin
from .nnUNetTrainerUNetPlusPlusPaper import nnUNetTrainerUNetPlusPlusPaper


class nnUNetTrainerUNetPlusPlusPaperSparseValidation(nnUNetTrainerSparseValidationMixin,
                                                     nnUNetTrainerUNetPlusPlusPaper):
    pass

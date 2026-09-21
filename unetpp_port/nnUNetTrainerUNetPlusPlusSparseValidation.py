"""UNet++ with validation every five epochs; optimizer updates are unchanged."""
from .nnUNetTrainerSparseValidationMixin import nnUNetTrainerSparseValidationMixin
from .nnUNetTrainerUNetPlusPlus import nnUNetTrainerUNetPlusPlus


class nnUNetTrainerUNetPlusPlusSparseValidation(nnUNetTrainerSparseValidationMixin,
                                                nnUNetTrainerUNetPlusPlus):
    pass

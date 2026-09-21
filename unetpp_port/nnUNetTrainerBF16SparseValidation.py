"""Plain U-Net with BF16 and validation every five epochs."""
from .nnUNetTrainerBF16 import nnUNetTrainerBF16
from .nnUNetTrainerSparseValidationMixin import nnUNetTrainerSparseValidationMixin


class nnUNetTrainerBF16SparseValidation(nnUNetTrainerSparseValidationMixin, nnUNetTrainerBF16):
    pass

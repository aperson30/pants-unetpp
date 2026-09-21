"""Plain U-Net without deep supervision, with validation every five epochs."""
from .nnUNetTrainerBF16NoDeepSupervision import nnUNetTrainerBF16NoDeepSupervision
from .nnUNetTrainerSparseValidationMixin import nnUNetTrainerSparseValidationMixin


class nnUNetTrainerBF16NoDeepSupervisionSparseValidation(
        nnUNetTrainerSparseValidationMixin, nnUNetTrainerBF16NoDeepSupervision):
    pass

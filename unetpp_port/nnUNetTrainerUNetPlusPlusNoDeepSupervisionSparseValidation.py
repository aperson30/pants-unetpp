"""UNet++ without deep supervision, with validation every five epochs."""
from .nnUNetTrainerSparseValidationMixin import nnUNetTrainerSparseValidationMixin
from .nnUNetTrainerUNetPlusPlusNoDeepSupervision import nnUNetTrainerUNetPlusPlusNoDeepSupervision


class nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation(
        nnUNetTrainerSparseValidationMixin, nnUNetTrainerUNetPlusPlusNoDeepSupervision):
    pass

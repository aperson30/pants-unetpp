"""CPU-runnable contract test for the shared BF16/cuDNN trainer mixin."""
from types import SimpleNamespace

import torch

from nnUNetTrainerBF16Mixin import nnUNetTrainerBF16Mixin


class _Base:
    def __init__(self, plans, configuration, fold, dataset_json, device):
        self.device = device
        self.grad_scaler = object()


class _Subject(nnUNetTrainerBF16Mixin, _Base):
    pass


def main():
    original_benchmark = torch.backends.cudnn.benchmark
    original_dtype = torch.get_autocast_dtype("cuda")
    try:
        torch.backends.cudnn.benchmark = False
        cuda_subject = _Subject({}, "3d_fullres", 0, {}, SimpleNamespace(type="cuda"))
        assert torch.backends.cudnn.benchmark is True
        assert torch.get_autocast_dtype("cuda") == torch.bfloat16
        assert cuda_subject.grad_scaler is None

        torch.backends.cudnn.benchmark = False
        cpu_subject = _Subject({}, "3d_fullres", 0, {}, SimpleNamespace(type="cpu"))
        assert torch.backends.cudnn.benchmark is False
        assert cpu_subject.grad_scaler is not None
    finally:
        torch.backends.cudnn.benchmark = original_benchmark
        torch.set_autocast_dtype("cuda", original_dtype)
    print("BF16/cuDNN mixin contract passed.")


if __name__ == "__main__":
    main()

"""
Mixin: switches CUDA autocast from nnU-Net's implicit fp16 default to bf16, and disables
GradScaler accordingly (nnU-Net's own train_step already has a clean no-scaler branch when
self.grad_scaler is None -- see nnUNetTrainer.train_step -- so this needs no other override).

Measured identical throughput to fp16 on this exact GB10 hardware (both architectures, real patch
size, on top of dead-head-skip + reduce-overhead compile): UNet++ 3.639s vs 3.621s (1.005x),
plain U-Net 0.759s vs 0.761s (0.998x) -- confirmed, not assumed. bf16 has fp32's full exponent
range, so it never needs loss scaling; fp16's GradScaler silently SKIPS an optimizer step whenever
it detects overflow, which is a real (if easy to miss) tax specifically on a class that's already
starved for gradient signal (~10% prevalence). Pure upside at this measured speed parity.

MUST be applied to BOTH arms of the UNet++-vs-plain-U-Net comparison symmetrically, never to only
one -- otherwise a precision difference becomes an unintended second variable confounding the
architecture comparison. nnUNetTrainerUNetPlusPlus uses this mixin directly (covering all of its
subclasses); nnUNetTrainerBF16 gives the plain-U-Net side of the grid the identical change on top
of otherwise-stock nnU-Net, keeping it the "control group" the project's README describes it as.
"""
import torch


class nnUNetTrainerBF16Mixin:
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.device.type == "cuda":
            torch.set_autocast_dtype("cuda", torch.bfloat16)
            self.grad_scaler = None

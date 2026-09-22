"""Contract checks that require the UNet++ trainer files to be installed in editable nnU-Net.

Verifies the trainer-level wiring for the dead-head-skip optimization: the default trainer must
skip exactly the shallowest (zero-weight) branch and shrink its loss/scales accordingly, while the
paper trainer must retain every branch -- proving the inheritance hazard flagged in
nnUNetTrainerUNetPlusPlus's skip_shallowest_deep_supervision_head docstring is actually fixed, not
just described.
"""
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBF16 import nnUNetTrainerBF16
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlus import nnUNetTrainerUNetPlusPlus
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlusPaper import (
    nnUNetTrainerUNetPlusPlusPaper,
)


def configuration():
    return SimpleNamespace(
        patch_size=(16, 32, 32),
        pool_op_kernel_sizes=[[1, 1, 1]] + [[2, 2, 2]] * 3,
        batch_dice=True,
        network_arch_init_kwargs={
            "n_stages": 4,
            "features_per_stage": [8, 16, 32, 64],
            "conv_op": nn.Conv3d,
            "kernel_sizes": [[3, 3, 3]] * 4,
            "strides": [[1, 1, 1]] + [[2, 2, 2]] * 3,
            "n_conv_per_stage": [1] * 4,
            "n_conv_per_stage_decoder": [1] * 3,
            "conv_bias": True,
            "norm_op": nn.InstanceNorm3d,
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "nonlin": nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
        },
        network_arch_init_kwargs_req_import=[],
    )


def trainer_shell(cls):
    trainer = object.__new__(cls)
    trainer.enable_deep_supervision = True
    trainer.is_ddp = False
    trainer.device = torch.device(os.environ.get("DEVICE", "cpu"))
    trainer.configuration_manager = configuration()
    trainer.label_manager = SimpleNamespace(has_regions=False, ignore_label=None)
    return trainer


if __name__ == "__main__":
    os.environ["nnUNet_compile"] = "false"
    default = trainer_shell(nnUNetTrainerUNetPlusPlus)
    paper = trainer_shell(nnUNetTrainerUNetPlusPlusPaper)

    assert len(default._get_deep_supervision_scales()) == 2
    assert len(paper._get_deep_supervision_scales()) == 3
    default_loss = default._build_loss()
    paper_loss = paper._build_loss()
    assert isinstance(default_loss, DeepSupervisionWrapper)
    assert isinstance(paper_loss, DeepSupervisionWrapper)
    np.testing.assert_allclose(default_loss.weight_factors, [2 / 3, 1 / 3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(paper_loss.weight_factors, [1.0] * 3, rtol=0, atol=0)
    assert sum(paper_loss.weight_factors) == 3.0, \
        "paper weights must remain a literal unnormalized sum, not an average"

    default_network = default.build_network_architecture(None, default.configuration_manager, 1, 5, True)
    paper_network = paper.build_network_architecture(None, paper.configuration_manager, 1, 5, True)
    assert default_network.decoder.skip_shallowest_deep_supervision_head is True
    assert default_network.decoder.average_outputs_at_inference is False
    assert paper_network.decoder.skip_shallowest_deep_supervision_head is False
    assert paper_network.decoder.average_outputs_at_inference is True

    # Regression guard for the PyTorch-2.10 full-loss optimization: torch.compile must receive the
    # final wrapper (CE + Dice + branch weighting), rather than only the Dice child. Monkeypatching
    # compile keeps this contract check CPU-only and independent of Triton availability.
    compile_calls = []
    original_compile = torch.compile
    original_device = os.environ.get("DEVICE")
    try:
        torch.compile = lambda module, **kwargs: compile_calls.append((module, kwargs)) or module
        os.environ["nnUNet_compile"] = "true"
        os.environ["DEVICE"] = "cuda"
        compiled_default = trainer_shell(nnUNetTrainerUNetPlusPlus)._build_loss()
        compiled_paper = trainer_shell(nnUNetTrainerUNetPlusPlusPaper)._build_loss()
        compiled_plain = trainer_shell(nnUNetTrainerBF16)._build_loss()
    finally:
        torch.compile = original_compile
        os.environ["nnUNet_compile"] = "false"
        if original_device is None:
            os.environ.pop("DEVICE", None)
        else:
            os.environ["DEVICE"] = original_device

    assert all(isinstance(loss, DeepSupervisionWrapper)
               for loss in (compiled_default, compiled_paper, compiled_plain))
    assert len(compile_calls) == 3
    assert all(isinstance(module, DeepSupervisionWrapper) for module, _ in compile_calls)
    assert all(kwargs == {"mode": "reduce-overhead"} for _, kwargs in compile_calls)
    print("Trainer contracts passed: default drops only the zero-weight head; paper retains all "
          "heads with literal unnormalized eta_i=1 weights.")

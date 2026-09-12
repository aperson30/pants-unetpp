"""
Custom nnU-Net v2 trainer that swaps in the ported UNet++ architecture (unet_plusplus.py, next to
this file) instead of the framework's default network.

How to use this file: copy both this file and unet_plusplus.py into nnU-Net v2's
`nnunetv2/training/nnUNetTrainer/` folder on the GPU server (inside your installed/cloned nnU-Net v2
checkout), so nnU-Net's class lookup (`recursive_find_python_class`) can discover
`nnUNetTrainerUNetPlusPlus` by name. Then train with:

    nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerUNetPlusPlus

Two design decisions worth understanding, not just copying:

1. build_network_architecture() reuses `configuration_manager.network_arch_init_kwargs` -- the exact
   same n_stages / features_per_stage / kernel_sizes / strides that nnU-Net's automatic planner chose
   for the DEFAULT architecture (PlainConvUNet) on this dataset -- and just constructs our UNetPlusPlus
   with them instead. This works because UNetPlusPlus's constructor signature was deliberately written
   to match PlainConvUNet's exactly (see unet_plusplus.py). This is the "quick approach" nnU-Net's own
   docs describe for adding a custom architecture, as opposed to also writing a custom
   ExperimentPlanner (the "proper/complex approach").

   IMPORTANT CONSEQUENCE: UNet++'s decoder does a lot more feature-map concatenation than a plain
   U-Net decoder does (that's the whole point of the architecture), so it uses noticeably more GPU
   memory at the same batch size and patch size. Reusing PlainConvUNet's plan values means you are
   NOT guaranteed to avoid a CUDA out-of-memory error -- if training crashes with OOM on your first
   attempt, that is the expected failure mode of the "quick approach", not a bug in this port. The fix
   is to manually lower `batch_size` (and/or patch size) in a copy of the dataset's plans.json for this
   configuration, or invest in a custom ExperimentPlanner that accounts for UNetPlusPlus's real memory
   footprint via its compute_conv_feature_map_size method.

2. _get_deep_supervision_scales() is overridden to return "no downsampling" scales for every UNet++
   deep-supervision output. Regular nnU-Net architectures produce auxiliary outputs at progressively
   downsampled resolutions, so nnU-Net downsamples the ground truth to match each one. UNet++'s own
   deep-supervision outputs are all at FULL input resolution (they differ in nesting depth, not
   spatial size -- see unet_plusplus.py's docstring). Telling nnU-Net "don't downsample the target for
   any of these outputs" is what makes the existing DeepSupervisionWrapper loss and data-augmentation
   pipeline produce the loss UNet++'s original paper actually uses, without writing a new loss class.

3. LOSS WEIGHTING -- AN OVERRIDE THAT WAS TRIED AND REVERTED. READ BEFORE RE-ADDING IT.
   nnU-Net's default _build_loss() weights deep-supervision outputs by exponential decay
   (1, 1/2, 1/4, ...) and zeroes the weight of the LAST output. That rule is designed for nnU-Net's
   own architectures, where the last output is the coarsest downsampled prediction. Our decoder
   returns seg_outputs[::-1], so under that rule the zero-weighted output is X[0][1] -- UNet++'s
   SHALLOWEST branch, not a low-resolution one. On paper that looks wrong, and the UNet++ paper
   weights its deep-supervision outputs equally, so an override was added here that gave all outputs
   equal weight.

   EMPIRICALLY THAT MADE THINGS MUCH WORSE and it has been removed. Evidence (2026-08-30): with the
   equal-weighting override, UNet++ DS-on showed 0% of epochs detecting any tumor at epoch 400-500,
   versus 87% at the same point for the identical run WITHOUT the override (same architecture, same
   deep supervision setting, same nnUNetPlansBS4 batch size 4, same 4-GPU DDP setup -- the loss
   weighting was the only difference). The plausible reason: giving the shallowest, least-refined
   nested branch equal say in the loss dilutes the gradient signal reaching the fully-nested output
   that actually produces the final prediction. That matters disproportionately for a rare class
   like the pancreatic tumor, which needs every bit of signal it can get.

   So this class now inherits nnU-Net's default _build_loss() unchanged. If you revisit this, treat
   "the paper says equal weights" as a hypothesis to test, not a correctness fix -- and note that any
   such test needs to run at least ~500 epochs, because tumor learning here has a delayed onset and
   looks identical to failure before epoch ~300.
"""
from typing import Union

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

from .unet_plusplus import UNetPlusPlus


class nnUNetTrainerUNetPlusPlus(nnUNetTrainer):

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                    configuration_manager: ConfigurationManager,
                                    num_input_channels: int,
                                    num_output_channels: int,
                                    enable_deep_supervision: bool = True) -> nn.Module:
        import pydoc
        from copy import deepcopy

        arch_kwargs = deepcopy(configuration_manager.network_arch_init_kwargs)
        for req_import_key in configuration_manager.network_arch_init_kwargs_req_import:
            if arch_kwargs[req_import_key] is not None:
                arch_kwargs[req_import_key] = pydoc.locate(arch_kwargs[req_import_key])

        return UNetPlusPlus(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **arch_kwargs
        )

    def _get_deep_supervision_scales(self) -> Union[list, None]:
        if not self.enable_deep_supervision:
            return None
        ndim = len(self.configuration_manager.patch_size)
        n_outputs = len(self.configuration_manager.pool_op_kernel_sizes) - 1
        return [[1] * ndim for _ in range(n_outputs)]

    # NOTE: _build_loss() is deliberately NOT overridden -- we inherit nnU-Net's default weighting.
    # An equal-weighting override was tried and measurably broke tumor learning; see point 3 of the
    # module docstring for the evidence before considering re-adding it.

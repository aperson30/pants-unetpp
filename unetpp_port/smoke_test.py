"""
Quick sanity check for unet_plusplus.py -- NOT a substitute for real training, just a fast way to
catch shape/wiring bugs before committing a multi-day GPU job to this architecture.

Run this on your GPU server (or anywhere with torch installed) right after copying the two files
over, before you touch nnU-Net or PanTS at all:

    python smoke_test.py

It builds small dummy UNet++ networks with different numbers of downsampling stages and runs one
forward pass through each, on random data. If this doesn't finish cleanly, don't bother starting a
real preprocessing/training run yet -- something in the port broke and needs fixing first.
"""
import torch
from torch import nn

from unet_plusplus import UNetPlusPlus


def build(n_stages, deep_supervision, average_outputs_at_inference=False):
    features_per_stage = [min(32 * (2 ** i), 320) for i in range(n_stages)]
    strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
    kernel_sizes = [[3, 3, 3]] * n_stages

    model = UNetPlusPlus(
        input_channels=1, n_stages=n_stages, features_per_stage=features_per_stage,
        conv_op=nn.Conv3d, kernel_sizes=kernel_sizes, strides=strides,
        n_conv_per_stage=[2] * n_stages, num_classes=29,
        n_conv_per_stage_decoder=[2] * (n_stages - 1),
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=deep_supervision,
        average_outputs_at_inference=average_outputs_at_inference,
    )
    model.initialize()
    model.eval()
    return model


def check(n_stages, patch_size, deep_supervision):
    model = build(n_stages, deep_supervision)

    x = torch.rand((1, 1, *patch_size))
    with torch.no_grad():
        out = model(x)

    if deep_supervision:
        assert isinstance(out, list) and len(out) == n_stages - 1
        for o in out:
            assert list(o.shape) == [1, 29, *patch_size]
    else:
        assert list(out.shape) == [1, 29, *patch_size]

    print(f"n_stages={n_stages} patch={patch_size} deep_supervision={deep_supervision}: OK "
          f"({sum(p.numel() for p in model.parameters()):,} params)")


def check_branch_averaging(n_stages, patch_size):
    """Covers the paper-faithful inference path used by nnUNetTrainerUNetPlusPlusPaper.

    Two things have to hold. First, with deep supervision off and averaging on, the network must
    still return a single tensor of the right shape -- nnU-Net's inference path cannot cope with a
    list. Second, applying nnU-Net's downstream softmax must produce the arithmetic mean of the
    branches' softmax probabilities, not softmax(mean logits).
    """
    model = build(n_stages, deep_supervision=True, average_outputs_at_inference=True)
    x = torch.rand((1, 1, *patch_size))

    with torch.no_grad():
        branches = model(x)                      # deep supervision on -> every branch
        model.decoder.deep_supervision = False   # exactly how nnU-Net flips it for inference
        averaged = model(x)

    assert isinstance(branches, list) and len(branches) == n_stages - 1
    assert list(averaged.shape) == [1, 29, *patch_size], "averaged output has the wrong shape"

    expected_probabilities = torch.stack(
        [torch.softmax(branch.float(), dim=1) for branch in branches], dim=0
    ).mean(dim=0)
    actual_probabilities = torch.softmax(averaged, dim=1)
    assert torch.allclose(actual_probabilities, expected_probabilities, rtol=1e-5, atol=1e-6), \
        "downstream softmax does not recover the mean branch probability"

    deepest_probability = torch.softmax(branches[0].float(), dim=1)
    assert not torch.allclose(actual_probabilities, deepest_probability, atol=1e-5), \
        "averaging produced the deepest branch alone -- the flag is not taking effect"

    print(f"n_stages={n_stages} patch={patch_size} branch-averaging inference: OK "
          f"(probability mean of {len(branches)} branches, distinct from deepest alone)")


if __name__ == "__main__":
    check(n_stages=5, patch_size=(32, 64, 64), deep_supervision=True)
    check(n_stages=6, patch_size=(32, 64, 64), deep_supervision=True)
    check(n_stages=6, patch_size=(32, 64, 64), deep_supervision=False)
    check_branch_averaging(n_stages=5, patch_size=(32, 64, 64))
    check_branch_averaging(n_stages=6, patch_size=(32, 64, 64))
    print("\nAll smoke tests passed.")

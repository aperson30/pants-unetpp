"""CPU-sized contract test for paper-faithful probability branch averaging."""
import torch
from torch import nn

from unet_plusplus import UNetPlusPlus


def main() -> None:
    torch.manual_seed(20260921)
    model = UNetPlusPlus(
        input_channels=1,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 4,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 3,
        n_conv_per_stage=[1] * 4,
        num_classes=29,
        n_conv_per_stage_decoder=[1] * 3,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
        average_outputs_at_inference=True,
    ).eval()
    inputs = torch.randn(1, 1, 8, 16, 16)
    with torch.no_grad():
        branches = model(inputs)
        model.decoder.deep_supervision = False
        ensemble_logits = model(inputs)

    expected = torch.stack(
        [torch.softmax(branch.float(), dim=1) for branch in branches], dim=0
    ).mean(dim=0)
    actual = torch.softmax(ensemble_logits, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(ensemble_logits).all()
    torch.testing.assert_close(actual.sum(dim=1), torch.ones_like(actual[:, 0]))

    old_behavior = torch.softmax(torch.stack(branches, dim=0).mean(dim=0), dim=1)
    assert not torch.allclose(actual, old_behavior, rtol=1e-7, atol=1e-8), \
        "test failed to distinguish probability averaging from the old logit averaging"
    print("Paper inference contract passed: downstream softmax equals arithmetic branch-probability mean.")


if __name__ == "__main__":
    main()

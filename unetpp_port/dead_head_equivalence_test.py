"""Prove that skipping unused UNet++ segmentation heads preserves live logits and gradients."""
import os

import torch
from torch import nn

from unet_plusplus import UNetPlusPlus


def build(deep_supervision: bool, skip_shallowest: bool, device: torch.device):
    model = UNetPlusPlus(
        input_channels=1,
        n_stages=4,
        features_per_stage=[8, 16, 32, 64],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 4,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 3,
        n_conv_per_stage=[1] * 4,
        num_classes=5,
        n_conv_per_stage_decoder=[1] * 3,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
        skip_shallowest_deep_supervision_head=skip_shallowest,
    ).to(device)
    model.initialize()
    model.train()
    return model


def head_counter(model):
    calls = []
    handles = [layer.register_forward_hook(lambda _m, _i, _o, key=key: calls.append(key))
               for key, layer in model.decoder.seg_layers.items()]
    return calls, handles


def assert_shared_gradients_equal(reference, optimized):
    ref_parameters = dict(reference.named_parameters())
    opt_parameters = dict(optimized.named_parameters())
    assert ref_parameters.keys() == opt_parameters.keys()
    for name in ref_parameters:
        ref_grad = ref_parameters[name].grad
        opt_grad = opt_parameters[name].grad
        if ref_grad is None or opt_grad is None:
            assert ref_grad is None and opt_grad is None, f"gradient presence differs: {name}"
            continue
        assert torch.equal(ref_grad, opt_grad), f"gradient differs: {name}"


def check_no_deep_supervision(device):
    # A DS-on reference evaluates every head. Taking only its primary output reproduces the old
    # DS-off implementation, which evaluated all heads and returned the deepest one.
    reference = build(deep_supervision=True, skip_shallowest=False, device=device)
    optimized = build(deep_supervision=False, skip_shallowest=False, device=device)
    optimized.load_state_dict(reference.state_dict())
    x_ref = torch.randn((1, 1, 16, 32, 32), device=device, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    ref_calls, ref_handles = head_counter(reference)
    opt_calls, opt_handles = head_counter(optimized)

    ref_output = reference(x_ref)[0]
    opt_output = optimized(x_opt)
    assert torch.equal(ref_output, opt_output), "DS-off live logits changed"
    ref_loss = ref_output.square().mean()
    opt_loss = opt_output.square().mean()
    assert torch.equal(ref_loss, opt_loss), "DS-off loss changed"
    ref_loss.backward()
    opt_loss.backward()
    assert torch.equal(x_ref.grad, x_opt.grad), "DS-off input gradient changed"
    assert_shared_gradients_equal(reference, optimized)
    assert len(ref_calls) == reference.decoder.L
    assert opt_calls == [f"0_{optimized.decoder.L}"]
    for handle in ref_handles + opt_handles:
        handle.remove()


def check_default_deep_supervision(device):
    reference = build(deep_supervision=True, skip_shallowest=False, device=device)
    optimized = build(deep_supervision=True, skip_shallowest=True, device=device)
    optimized.load_state_dict(reference.state_dict())
    x_ref = torch.randn((1, 1, 16, 32, 32), device=device, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    ref_calls, ref_handles = head_counter(reference)
    opt_calls, opt_handles = head_counter(optimized)

    ref_outputs = reference(x_ref)
    opt_outputs = optimized(x_opt)
    assert len(ref_outputs) == reference.decoder.L
    assert len(opt_outputs) == optimized.decoder.L - 1
    for ref_output, opt_output in zip(ref_outputs[:-1], opt_outputs):
        assert torch.equal(ref_output, opt_output), "live deep-supervision logits changed"

    # Exact nnU-Net single-GPU weighting: exponential decay, shallowest entry zero, normalize.
    weights = torch.tensor([1 / (2 ** i) for i in range(len(ref_outputs))], device=device)
    weights[-1] = 0
    weights /= weights.sum()
    ref_loss = sum(weights[i] * output.square().mean()
                   for i, output in enumerate(ref_outputs) if weights[i] != 0)
    opt_loss = sum(weights[i] * output.square().mean()
                   for i, output in enumerate(opt_outputs))
    assert torch.equal(ref_loss, opt_loss), "default-DS loss changed"
    ref_loss.backward()
    opt_loss.backward()
    assert torch.equal(x_ref.grad, x_opt.grad), "default-DS input gradient changed"
    assert_shared_gradients_equal(reference, optimized)
    assert dict(reference.named_parameters())["decoder.seg_layers.0_1.weight"].grad is None
    assert dict(optimized.named_parameters())["decoder.seg_layers.0_1.weight"].grad is None
    assert len(ref_calls) == reference.decoder.L
    assert len(opt_calls) == optimized.decoder.L - 1 and "0_1" not in opt_calls
    for handle in ref_handles + opt_handles:
        handle.remove()


if __name__ == "__main__":
    device = torch.device(os.environ.get("DEVICE", "cpu"))
    torch.manual_seed(20260917)
    torch.use_deterministic_algorithms(True)
    check_no_deep_supervision(device)
    check_default_deep_supervision(device)
    print(f"Dead-head equivalence passed on {device}: live logits/loss/input+parameter gradients are bit-identical.")

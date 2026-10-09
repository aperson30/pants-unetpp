import unittest
import torch
from adaptive_zero_denoiser import AdaptiveZeroBlock, AdaptiveZeroNestedDenoiser, AdaptiveZeroPlainDenoiser
from conditional_nested_denoiser import supervised_denoising_loss


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)

    def test_zero_gates_equal_projected_skip(self):
        block = AdaptiveZeroBlock(8,4,16,6)
        x, time, context = torch.randn(1,8,8,8), torch.randn(1,16), torch.randn(1,3,6)
        torch.testing.assert_close(block(x,time,context), block.project(x), rtol=0, atol=0)
        block(x,time,context).square().mean().backward()
        self.assertIsNotNone(block.modulation[-1].weight.grad)
        self.assertTrue(torch.isfinite(block.modulation[-1].weight.grad).all())

    def test_zero_heads_and_supervision_gradients(self):
        x, t, ctx = torch.randn(1,8,16,16), torch.tensor([300]), torch.randn(1,3,6)
        nested = AdaptiveZeroNestedDenoiser(channels=(4,8,16), context_dim=6, time_dim=16)
        outputs = nested(x,t,ctx,all_heads=True)
        self.assertTrue(all(torch.count_nonzero(y) == 0 for y in outputs))
        supervised_denoising_loss(outputs,torch.randn_like(outputs[0]),(1,1)).backward()
        for head in nested.heads.values():
            self.assertGreater(float(head.weight.grad.abs().sum()),0)
        plain = AdaptiveZeroPlainDenoiser(channels=(4,8,16), context_dim=6, time_dim=16)
        self.assertEqual(int(torch.count_nonzero(plain(x,t,ctx).sample)),0)

    def test_nonzero_pruning_parity_and_omitted_nodes(self):
        model = AdaptiveZeroNestedDenoiser(channels=(4,8,16), context_dim=6, time_dim=16)
        for block in model.nodes.values():
            torch.nn.init.normal_(block.modulation[-1].weight, std=.05)
        for head in model.heads.values():
            torch.nn.init.normal_(head.weight, std=.05)
        x,t,ctx = torch.randn(1,8,16,16),torch.tensor([500]),torch.randn(1,3,6)
        outputs = model(x,t,ctx,all_heads=True)
        visited = []
        hooks = [block.register_forward_hook(lambda module,args,result,k=key:visited.append(k))
                 for key,block in model.nodes.items()]
        shallow = model(x,t,ctx,head=1).sample
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(shallow,outputs[0],rtol=0,atol=0)
        self.assertEqual(set(visited), {"0_0","1_0","0_1"})


if __name__ == "__main__":
    unittest.main()

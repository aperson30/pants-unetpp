"""Tiny CPU path/parity tests. No trained lesion quality or GPU speed claim."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from isolated_unetpp_exit import forward_exit

port = Path(sys.argv.pop(1)) if len(sys.argv)>1 and sys.argv[1].endswith('.py') else (
    Path(__file__).parent.parent/'unetpp_port'/'unet_plusplus.py')
spec = importlib.util.spec_from_file_location('audited_unetpp', port)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def model(ndim, stages=4):
    conv = nn.Conv2d if ndim==2 else nn.Conv3d
    norm = nn.InstanceNorm2d if ndim==2 else nn.InstanceNorm3d
    return module.UNetPlusPlus(
        input_channels=1, n_stages=stages, features_per_stage=list(range(2,2+stages)),
        conv_op=conv, kernel_sizes=[[3]*ndim]*stages,
        strides=([[1]*ndim]+[[2]*ndim]*(stages-1) if stages!=6 else
                 [[1,1,1]]+[[2,2,2]]*4+[[1,2,2]]), n_conv_per_stage=[1]*stages,
        num_classes=3, n_conv_per_stage_decoder=[1]*(stages-1),
        norm_op=norm, norm_op_kwargs={'affine':True}, nonlin=nn.LeakyReLU,
        deep_supervision=True, skip_shallowest_deep_supervision_head=True).eval()


class Contracts(unittest.TestCase):
    def test_exact_cpu_logits_and_skipped_work(self):
        torch.manual_seed(731)
        for ndim, stages in [(2,4),(3,4),(3,6)]:
            net = model(ndim, stages)
            x = torch.randn(1,1,*([32 if stages==6 else 16]*ndim))
            with torch.inference_mode():
                full = net(x)
            for depth in range(2,net.decoder.L+1):
                executed = []
                handles = []
                for i, stage in enumerate(net.encoder.stages):
                    handles.append(stage.register_forward_hook(
                        lambda m, a, o, i=i: executed.append(('encoder',i))))
                for key, block in net.decoder.convs.items():
                    handles.append(block.register_forward_hook(
                        lambda m,a,o,key=key: executed.append(('decoder',key))))
                try:
                    out = forward_exit(net, x, depth)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertTrue(torch.equal(out, full[net.decoder.L-depth]))
                self.assertFalse(out.requires_grad)
                self.assertEqual([i for kind,i in executed if kind=='encoder'], list(range(depth+1)))
                self.assertEqual(sum(kind=='decoder' for kind,i in executed), depth*(depth+1)//2)
                self.assertTrue(all(sum(map(int,key.split('_')))<=depth
                                    for kind,key in executed if kind=='decoder'))

    def test_reject_unsupported_heads_and_training(self):
        net = model(2)
        x = torch.randn(1,1,16,16)
        for depth in [0,1,4,True]:
            with self.assertRaises(ValueError):
                forward_exit(net,x,depth)
        net.train()
        with self.assertRaisesRegex(ValueError,'eval'):
            forward_exit(net,x,2)
        net.eval()
        net.decoder.deep_supervision=False
        with self.assertRaisesRegex(ValueError,'DS-on'):
            forward_exit(net,x,2)


if __name__=='__main__':
    torch.set_num_threads(1)
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(dict(scope='Tiny CPU random-weight execution contracts only',
                         torch=torch.__version__, gpu=False, tests=result.testsRun)))

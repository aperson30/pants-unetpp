"""Pinned sampling identity and CPU orchestration; neural compute mocked."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch
from monai.networks.nets.autoencoderkl import AutoencoderKL
import posterior_control as control


class FakeModel:
    sampling=AutoencoderKL.sampling
    def __init__(self):self.encodes=0;self.decodes=0
    def cuda(self):return self
    def encode(self,tensor):
        self.encodes+=1
        return tensor.clone(),torch.full_like(tensor,.03)
    def decode(self,latent):
        self.decodes+=1
        return latent.clone()


class PosteriorTests(unittest.TestCase):
    def test_sampling_matches_pinned_std_formula_and_locked_seeds(self):
        mu=torch.ones(2,3)
        sigma=torch.full_like(mu,.2)
        model=FakeModel()
        torch.manual_seed(1)
        expected=mu+torch.randn_like(sigma)*sigma
        torch.testing.assert_close(control.sample_latent(model,mu,sigma,1),expected,rtol=0,atol=0)
        torch.testing.assert_close(control.sample_latent(model,mu,torch.zeros_like(sigma),0),mu,rtol=0,atol=0)
        first=control.sample_latent(model,mu,sigma,2)
        torch.testing.assert_close(control.sample_latent(model,mu,sigma,2),first,rtol=0,atol=0)
        with self.assertRaises(ValueError):control.sample_latent(model,mu,sigma,99)
        with self.assertRaises(ValueError):control.sample_latent(model,mu,-sigma,0)

    def test_one_encode_four_decodes_all_six_arms_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=root/'case';producer=root/'producer'
            case.mkdir();producer.mkdir()
            image=sitk.GetImageFromArray(np.zeros((8,8,8),dtype=np.float32))
            label=sitk.GetImageFromArray(np.zeros((8,8,8),dtype=np.uint8));label[3,3,3]=1
            sitk.WriteImage(image,str(case/'image.nii.gz'))
            sitk.WriteImage(label,str(case/'manual_label.nii.gz'))
            import nibabel as nib
            ct=nib.load(case/'image.nii.gz');ct.header.set_xyzt_units('mm');nib.save(ct,case/'image.nii.gz')
            (case/'completion.json').write_text(json.dumps(dict(completed=True,case=dict(fold=4),
                ct_sha256=control.sha(case/'image.nii.gz'),label_sha256=control.sha(case/'manual_label.nii.gz'))))
            window=dict(x_start=1,x_finish=7,y_start=1,y_finish=7,z_start=1,z_finish=7)
            (producer/'completion.json').write_text(json.dumps(dict(completed=True,results=[dict(arm='native',crop_bounds=window)])))
            vae=root/'fake.pt';vae.write_bytes(b'fixture')
            model=FakeModel();calls=[]
            def predict(predictor,crop):
                calls.append(crop.GetSize())
                array=sitk.GetArrayFromImage(crop)
                segmentation=sitk.GetImageFromArray(np.full(array.shape,4,dtype=np.uint8));segmentation.CopyInformation(crop)
                probabilities=np.zeros((7,*array.shape),dtype=np.float32);probabilities[1]=.2
                return segmentation,probabilities
            def candidates(raw,bounds,image):
                values=control.expand_map(raw,tuple(reversed(image.GetSize())),bounds)
                output=sitk.GetImageFromArray(values);output.CopyInformation(image)
                return output,float(values.max())
            argv=['probe','--case',str(case),'--producer',str(producer),'--models',str(root),
                '--vae',str(vae),'--vae-sha',control.sha(vae),'--output',str(root/'output')]
            old=control.backend()
            try:
                with patch.dict(control.os.environ,SLURM_JOB_ID='CPU_TEST',CUBLAS_WORKSPACE_CONFIG=':4096:8'), \
                     patch('sys.argv',argv),patch.object(torch.cuda,'device_count',return_value=1), \
                     patch.object(torch.cuda,'mem_get_info',return_value=(100_000_000_000,102_000_000_000)), \
                     patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'synchronize'), \
                     patch.object(torch.cuda,'empty_cache'),patch.object(torch.cuda,'max_memory_allocated',return_value=0), \
                     patch.object(torch.Tensor,'cuda',lambda self:self),patch.object(control,'load_model',return_value=model), \
                     patch.object(control,'log_memory'),patch.object(control,'detector',return_value=object()), \
                     patch.object(control,'predict_image',side_effect=predict), \
                     patch.object(control.upstream,'GetFullSizDetectionMap',side_effect=candidates), \
                     contextlib.redirect_stdout(io.StringIO()):
                    control.main()
            finally:
                torch.backends.cudnn.benchmark=old['benchmark'];torch.backends.cudnn.deterministic=old['cudnn_deterministic']
                torch.backends.cuda.matmul.allow_tf32=old['matmul_tf32'];torch.backends.cudnn.allow_tf32=old['cudnn_tf32']
                torch.use_deterministic_algorithms(old['deterministic_algorithms'])
            self.assertEqual(model.encodes,1);self.assertEqual(model.decodes,4);self.assertEqual(len(calls),12)
            result=json.loads((root/'output/completion.json').read_text())
            self.assertEqual(result['seeds'],[0,1,2]);self.assertEqual(len(result['results']),6)
            self.assertTrue(result['completed']);self.assertFalse(result['clinical_recall'])


if __name__=='__main__':unittest.main()

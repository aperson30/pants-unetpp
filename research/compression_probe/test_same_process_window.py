"""CPU orchestration regression; neural predictions mocked, not accuracy proof."""
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
import same_process_window_control as control


class WindowControlTests(unittest.TestCase):
    def test_all_four_cells_twice_and_failed_repeat_stops(self):
        for mismatch in (False,True):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                case,producer=root/'case',root/'producer'
                case.mkdir(); producer.mkdir()
                image=sitk.GetImageFromArray(np.zeros((6,7,8),dtype=np.float32))
                label=sitk.GetImageFromArray(np.zeros((6,7,8),dtype=np.uint8))
                label[3,3,3]=1
                sitk.WriteImage(image,str(case/'image.nii.gz'))
                sitk.WriteImage(label,str(case/'manual_label.nii.gz'))
                sitk.WriteImage(image+1,str(producer/'reconstruction.nii.gz'))
                (case/'completion.json').write_text(json.dumps(dict(completed=True,case=dict(fold=4),
                    ct_sha256=control.sha(case/'image.nii.gz'),label_sha256=control.sha(case/'manual_label.nii.gz'))))
                windows=[dict(x_start=1,x_finish=7,y_start=1,y_finish=6,z_start=z,z_finish=5) for z in (1,2)]
                (producer/'completion.json').write_text(json.dumps(dict(completed=True,results=[
                    dict(arm=arm,crop_bounds=window) for arm,window in zip(('native','reconstruction'),windows)])))
                calls=[]
                def predict(predictor,crop):
                    calls.append(crop.GetSize())
                    array=sitk.GetArrayFromImage(crop)
                    segmentation=sitk.GetImageFromArray(np.full(array.shape,4,dtype=np.uint8))
                    segmentation.CopyInformation(crop)
                    probabilities=np.zeros((7,*array.shape),dtype=np.float32)
                    probabilities[1]=.1+float(array.mean())*.1
                    if mismatch and len(calls)==2:
                        probabilities[1]+=.01
                    return segmentation,probabilities
                def candidates(raw,bounds,image):
                    array=control.expand_map(raw,tuple(reversed(image.GetSize())),bounds)
                    result=sitk.GetImageFromArray(array); result.CopyInformation(image)
                    return result,float(array.max())
                argv=['probe','--case',str(case),'--producer',str(producer),'--models',str(root),
                      '--output',str(root/'output')]
                old=control.backend()
                try:
                    with patch.dict(control.os.environ,SLURM_JOB_ID='CPU_TEST',CUBLAS_WORKSPACE_CONFIG=':4096:8'), \
                         patch('sys.argv',argv), patch.object(torch.cuda,'device_count',return_value=1), \
                         patch.object(torch.cuda,'empty_cache'), patch.object(control,'detector',return_value=object()), \
                         patch.object(control,'predict_image',side_effect=predict), \
                         patch.object(control.upstream,'GetFullSizDetectionMap',side_effect=candidates), \
                         contextlib.redirect_stdout(io.StringIO()):
                        if mismatch:
                            with self.assertRaises(ValueError): control.main()
                        else: control.main()
                finally:
                    torch.backends.cudnn.benchmark=old['benchmark']
                    torch.backends.cudnn.deterministic=old['cudnn_deterministic']
                    torch.backends.cuda.matmul.allow_tf32=old['matmul_tf32']
                    torch.backends.cudnn.allow_tf32=old['cudnn_tf32']
                    torch.use_deterministic_algorithms(old['deterministic_algorithms'])
                if mismatch:
                    self.assertEqual(len(calls),2)
                    self.assertFalse((root/'output/completion.json').exists())
                    self.assertTrue((root/'output/failure.json').exists())
                else:
                    self.assertEqual(len(calls),8)
                    result=json.loads((root/'output/completion.json').read_text())
                    self.assertEqual(len(result['results']),4)
                    self.assertFalse(result['historical_replay_gate_passed'])


if __name__=='__main__':
    unittest.main()

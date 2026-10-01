"""Bounded CPU-only installed-stack checks; run with OMP/MKL threads limited."""
from pathlib import Path
import importlib.metadata
import subprocess
import sys
import os
os.environ['nnUNet_compile'] = 'false'

print('PREFLIGHT_BEGIN', flush=True)
import torch
import torchvision
import nnunetv2
print('torch', torch.__version__, 'torchvision', torchvision.__version__, flush=True)
assert torch.__version__ == '2.10.0+cu126'
assert torch.backends.cudnn.version() == 91002
assert importlib.metadata.version('nnunetv2') == '2.8.1'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlusSparseValidation import nnUNetTrainerUNetPlusPlusSparseValidation
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation import nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBF16SparseValidation import nnUNetTrainerBF16SparseValidation
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBF16NoDeepSupervisionSparseValidation import nnUNetTrainerBF16NoDeepSupervisionSparseValidation
print('FOUR_TRAINER_IMPORTS_PASS', flush=True)
for trainer in (nnUNetTrainerUNetPlusPlusSparseValidation, nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation,
                nnUNetTrainerBF16SparseValidation, nnUNetTrainerBF16NoDeepSupervisionSparseValidation):
    folder = Path('/ocean/projects/cis260296p/asanjeev/pants_unetpp/results/Dataset001_PanTS') / (
        trainer.__name__ + '__nnUNetPlansBS4__3d_fullres')
    checkpoint = torch.load(folder / 'fold_0/checkpoint_final.pth', map_location='cpu', weights_only=False)
    assert checkpoint['current_epoch'] == 1000
    assert all(torch.isfinite(value).all() for value in checkpoint['network_weights'].values()
               if isinstance(value, torch.Tensor) and value.is_floating_point())
    del checkpoint
    predictor = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False)
    predictor.initialize_from_trained_model_folder(str(folder), use_folds=(0,),
                                                 checkpoint_name='checkpoint_final.pth')
    del predictor
    print('CHECKPOINT_EPOCH_PASS', trainer.__name__, flush=True)
help_result = subprocess.run([str(Path(sys.executable).with_name('nnUNetv2_predict')), '--help'],
                             capture_output=True, text=True, timeout=60, check=True)
for flag in ('-npp', '-nps', '-chk', '--save_probabilities', '--continue_prediction'):
    assert flag in help_result.stdout, flag
print('PREFLIGHT_PASS', flush=True)

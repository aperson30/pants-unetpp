"""CPU regression: real detector initialization + installed weight-loop API.

Sliding-window neural computation is mocked; this is an integration-contract
test, not full-image inference or a tumor-quality test. No GPU needed.
"""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from paired_judge_pilot import detector


class PredictorWeightContractTests(unittest.TestCase):
    def test_both_real_checkpoint_states_are_iterable_and_used_once(self):
        os.environ['nnUNet_compile'] = 'False'
        torch.set_num_threads(2)
        models = Path(os.environ['PAIRED_JUDGE_MODELS'])
        configurations = [
            ('Dataset103_PANORAMA_baseline_Pancreas_Segmentation',
             'nnUNetTrainer__nnUNetPlans__3d_fullres', 'checkpoint_final.pth', 2),
            ('Dataset104_PANORAMA_baseline_PDAC_Detection',
             'nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres',
             'checkpoint_best_panorama.pth', 7),
        ]
        for dataset, trainer, filename, heads in configurations:
            with self.subTest(dataset=dataset):
                predictor = detector(models, dataset, trainer, filename, heads)
                self.assertIsInstance(predictor.list_of_parameters, list)
                self.assertEqual(len(predictor.list_of_parameters), 1)
                expected = torch.zeros((heads, 2, 3, 4))
                with patch.object(predictor, 'predict_sliding_window_return_logits',
                                  return_value=expected) as sliding:
                    actual = predictor.predict_logits_from_preprocessed_data(
                        torch.zeros((1, 2, 3, 4)))
                sliding.assert_called_once()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name, value in predictor.network.state_dict().items():
                    torch.testing.assert_close(value,
                        predictor.list_of_parameters[0][name], rtol=0, atol=0)
                del predictor


if __name__ == '__main__':
    unittest.main()

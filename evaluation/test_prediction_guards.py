import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import nibabel as nib
import numpy as np
from evaluation.predict_and_shrink import valid_segmentation, verify_provenance, main, read_scores


class PredictionGuardTest(unittest.TestCase):
    def test_failed_prediction_never_commits_success_and_retry_recovers(self):
        for failure in ('crash', 'timeout', 'missing_archive', 'nan', 'save_failure'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                images, output = root / 'images', root / 'out'
                images.mkdir()
                cid = 'PanTS_00009001'
                nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), np.eye(4)),
                         images / f'{cid}_0000.nii.gz')
                args = SimpleNamespace(batch_size=5, num_parts=1, part_id=0, preprocess_workers=1,
                                       export_workers=1, batch_timeout=7200, images_dir=images,
                                       output_dir=output, max_probs_csv=output / 'scores.csv',
                                       expected_cases=1, dataset='1', config='C', tr='T', p='P', f='0',
                                       tumor_class=28, shrink_segmentations_to_tumor=True)
                broken = True
                def fake_cli(command, **kwargs):
                    if broken and failure == 'crash':
                        raise subprocess.CalledProcessError(1, command)
                    if broken and failure == 'timeout':
                        raise subprocess.TimeoutExpired(command, 7200)
                    out = Path(command[command.index('-o') + 1])
                    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.uint8), np.eye(4)),
                             out / f'{cid}.nii.gz')
                    if broken and failure == 'missing_archive':
                        return
                    probabilities = np.zeros((29, 3, 4, 5), dtype=np.float32)
                    probabilities[28, 0, 0, 0] = np.nan if broken and failure == 'nan' else 0.875
                    np.savez_compressed(out / f'{cid}.npz', probabilities=probabilities)
                from evaluation.predict_and_shrink import write_scores
                def save_scores(path, scores):
                    if broken and failure == 'save_failure':
                        raise OSError('injected disk failure')
                    write_scores(path, scores)
                with patch('evaluation.predict_and_shrink.parse_args', return_value=args), \
                     patch('evaluation.predict_and_shrink.verify_provenance'), \
                     patch.object(Path, 'symlink_to', autospec=True,
                                  side_effect=lambda path, target: shutil.copyfile(target, path)), \
                     patch('evaluation.predict_and_shrink.subprocess.run', side_effect=fake_cli) as cli, \
                     patch('evaluation.predict_and_shrink.write_scores', side_effect=save_scores):
                    with self.assertRaises((RuntimeError, OSError, subprocess.SubprocessError)):
                        main()
                    self.assertEqual(read_scores(args.max_probs_csv), {})
                    broken = False
                    main()
                    self.assertEqual(cli.call_count, 2)
                    self.assertEqual(read_scores(args.max_probs_csv)[cid], 0.875)

    def test_local_probabilities_persistent_mask_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, output = root / 'images', root / 'out'
            images.mkdir()
            cid = 'PanTS_00009001'
            image = images / f'{cid}_0000.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), np.eye(4)), image)
            args = SimpleNamespace(batch_size=5, num_parts=1, part_id=0, preprocess_workers=1,
                                   export_workers=1, batch_timeout=7200, images_dir=images,
                                   output_dir=output, max_probs_csv=output / 'scores.csv',
                                   expected_cases=1, dataset='1', config='C', tr='T', p='P', f='0',
                                   tumor_class=28, shrink_segmentations_to_tumor=True)
            def fake_cli(command, **kwargs):
                out = Path(command[command.index('-o') + 1])
                self.assertNotEqual(out, output)
                data = np.zeros((3, 4, 5), dtype=np.uint8)
                data[0, 0, 0] = 28; data[1, 1, 1] = 17
                nib.save(nib.Nifti1Image(data, np.eye(4)), out / f'{cid}.nii.gz')
                probabilities = np.zeros((29, 3, 4, 5), dtype=np.float32)
                probabilities[28, 0, 0, 0] = 0.875
                np.savez_compressed(out / f'{cid}.npz', probabilities=probabilities)
            with patch('evaluation.predict_and_shrink.parse_args', return_value=args), \
                 patch('evaluation.predict_and_shrink.verify_provenance'), \
                 patch.object(Path, 'symlink_to', autospec=True,
                              side_effect=lambda path, target: shutil.copyfile(target, path)), \
                 patch('evaluation.predict_and_shrink.subprocess.run', side_effect=fake_cli) as cli:
                main()
                main()
                self.assertEqual(cli.call_count, 1)
            self.assertEqual(read_scores(output / 'scores.csv')[cid], 0.875)
            saved = np.asanyarray(nib.load(output / f'{cid}.nii.gz').dataobj)
            self.assertEqual(set(np.unique(saved)), {0, 28})
            self.assertFalse(list(output.glob('*.npz')))

    def test_geometry_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref, pred = root / 'ref.nii.gz', root / 'pred.nii.gz'
            data = np.zeros((3, 4, 5), dtype=np.float32)
            nib.save(nib.Nifti1Image(data, np.eye(4)), ref)
            nib.save(nib.Nifti1Image(data, np.eye(4)), pred)
            self.assertTrue(valid_segmentation(pred, ref))
            shifted = np.eye(4); shifted[0, 3] = 1
            nib.save(nib.Nifti1Image(data, shifted), pred)
            self.assertFalse(valid_segmentation(pred, ref))
            for value in (29, -1, 0.5, np.nan):
                data[0, 0, 0] = value
                nib.save(nib.Nifti1Image(data, np.eye(4)), pred)
                self.assertFalse(valid_segmentation(pred, ref))

    def test_resume_rejects_changed_checkpoint_or_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'Dataset001_PanTS/T__P__C'
            (model / 'fold_0').mkdir(parents=True)
            checkpoint = model / 'fold_0/checkpoint_final.pth'
            checkpoint.write_bytes(b'checkpoint')
            for name in ('plans.json', 'dataset.json'):
                (model / name).write_text('{}')
            images = root / 'images'; images.mkdir()
            image = images / 'PanTS_00009001_0000.nii.gz'; image.write_bytes(b'input')
            output = root / 'out'; output.mkdir()
            args = SimpleNamespace(dataset='1', tr='T', p='P', config='C', f='0',
                                   output_dir=output, images_dir=images, tumor_class=28,
                                   max_probs_csv=output / 'scores.csv')
            with patch.dict(os.environ, {'nnUNet_results': str(root)}):
                verify_provenance(args, ['PanTS_00009001'])
                verify_provenance(args, ['PanTS_00009001'])
                checkpoint.write_bytes(b'changed')
                with self.assertRaises(RuntimeError):
                    verify_provenance(args, ['PanTS_00009001'])
                checkpoint.write_bytes(b'checkpoint')
                image.write_bytes(b'changed input')
                with self.assertRaises(RuntimeError):
                    verify_provenance(args, ['PanTS_00009001'])


if __name__ == '__main__':
    unittest.main()

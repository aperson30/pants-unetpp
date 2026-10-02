"""CPU decision/audit tests; no clinical validation or GPU inference."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import nibabel as nib
import torch
from audit_difftumor_probe import ARMS, audit, decision, digest
from difftumor_geometry import prepare
from difftumor_probe import predict_full


class DecisionTests(unittest.TestCase):
    def test_actual_sliding_inverse_softmax_path(self):
        shape = (16, 20, 24); affine = np.diag([-2., -1.5, 2.5, 1.])
        data, transform = prepare(np.zeros(shape, np.float32), affine)
        def constant_logits(x):
            return torch.cat([torch.full_like(x, float(c)) for c in range(3)], dim=1)
        tensor = data['image'].as_tensor()[None]
        first = predict_full(constant_logits, tensor, data, transform, shape, affine, 'cpu')
        repeat = predict_full(constant_logits, tensor, data, transform, shape, affine, 'cpu')
        expected = torch.softmax(torch.arange(3, dtype=torch.float32), dim=0)
        self.assertEqual(tuple(first.shape), (3,) + shape)
        self.assertLess(float(torch.abs(first-repeat).max()), 1e-7)
        for c in range(3): self.assertTrue(torch.allclose(first[c], torch.full(shape, expected[c]), atol=1e-6))

    def rows(self):
        return {a: dict(tumor_mean=.1 if a in ARMS[:2] else .05,
                        tumor_max_abs_vs_native=0.) for a in ARMS}

    def test_all_samples_required(self):
        rows = self.rows()
        self.assertTrue(decision(rows, 0).startswith('SUPPORT_ONLY_GO'))
        rows['posterior_seed1']['tumor_mean'] = .099
        self.assertTrue(decision(rows, 0).startswith('MIXED_OR_RETAINED'))

    def test_control_adequacy_and_noise(self):
        rows = self.rows(); rows['control']['tumor_max_abs_vs_native'] = .001
        self.assertTrue(decision(rows, 0).startswith('INVALID_CONTROL'))
        rows = self.rows(); rows['native']['tumor_mean'] = .001
        self.assertTrue(decision(rows, 0).startswith('NO_GO_CURRENT_ROUTE'))
        self.assertTrue(decision(self.rows(), .01).startswith('MIXED_OR_RETAINED'))

    def test_saved_map_audit_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); shape = (4, 5, 6); affine = np.eye(4)
            labels = np.zeros(shape, np.uint8); labels[1:3, 2:4, 2:4] = 1
            label_path = root/'gt.nii.gz'; nib.save(nib.Nifti1Image(labels, affine), label_path)
            images = [dict(arm=a, sha256='image-'+a) for a in ARMS]
            manifest = root/'inputs.json'
            manifest.write_text(json.dumps(dict(label=str(label_path), label_sha256=digest(label_path),
                                                images=images, weights_sha256='pinned-weight')))
            records = []
            for a in ARMS:
                probability = np.full(shape, .1 if a in ARMS[:2] else .05, np.float32)
                segmentation = np.zeros(shape, np.uint8); segmentation[labels == 1] = 2
                np.save(root/(a+'_tumor.npy'), probability)
                np.save(root/(a+'_labels.npy'), segmentation)
                records.append(dict(arm=a, image_sha256='image-'+a,
                    probability_sha256=digest(root/(a+'_tumor.npy')),
                    labels_sha256=digest(root/(a+'_labels.npy')),
                    all_class_repeat_max_abs=0., labels_different=0))
            result = dict(completed=True, manifest_sha256=digest(manifest), results=records,
                          checkpoint_sha256='pinned-weight', ground_truth_input=False,
                          official_postprocessing=False, native_shape=list(shape), affine=affine.tolist())
            (root/'completion.json').write_text(json.dumps(result))
            measured = audit(manifest, root)
            self.assertEqual(measured['tumor_voxels'], 8)
            self.assertEqual(measured['rows']['native']['tumor_voxels_argmax'], 8)
            self.assertAlmostEqual(measured['rows']['posterior_mean']['ratio_to_native'], .5)
            self.assertTrue(measured['decision'].startswith('SUPPORT_ONLY_GO'))
            np.save(root/'posterior_seed0_tumor.npy', np.ones(shape, np.float32))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                audit(manifest, root)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

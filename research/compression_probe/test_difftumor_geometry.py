"""Synthetic CPU geometry checks, not clinical or legacy-version parity tests."""
import unittest
import numpy as np
import torch
from difftumor_geometry import prepare, invert_logits


class GeometryTests(unittest.TestCase):
    def test_native_restore_three_orientations(self):
        shape = (16, 20, 24)
        for matrix in (np.diag([2., 1.5, 2.5]), np.diag([-2., -1.5, 2.5]),
                       np.array([[0., -1.5, 0.], [2., 0., 0.], [0., 0., 2.5]])):
            affine = np.eye(4); affine[:3, :3] = matrix
            affine[:3, 3] = [70., -40., 12.]
            data, transform = prepare(np.full(shape, 37.5, np.float32), affine)
            self.assertEqual(tuple(data['image'].shape), (1, 96, 96, 96))
            self.assertTrue(torch.allclose(data['image'].as_tensor(), torch.full_like(data['image'].as_tensor(), .5)))
            logits = torch.stack([torch.full((96, 96, 96), c) for c in (0., 1., 2.)])
            native = invert_logits(logits, data, transform, shape, affine)
            for c in range(3):
                self.assertTrue(torch.allclose(native[c].as_tensor(), torch.full(shape, float(c)), atol=1e-5))

    def test_spatial_landmark_restore(self):
        # Coordinate ramp checks that constant-channel restoration does not hide
        # swaps or flips. Convert processed voxel coords to native coordinates.
        shape = (16, 20, 24)
        affine = np.diag([-2., -1.5, 2.5, 1.]); affine[:3, 3] = [80., 60., -20.]
        data, transform = prepare(np.zeros(shape, np.float32), affine)
        coords = np.indices(data['image'].shape[1:], dtype=np.float64).reshape(3, -1)
        homogeneous = np.vstack([coords, np.ones(coords.shape[1])])
        native_coords = np.linalg.inv(affine) @ data['image'].affine.numpy() @ homogeneous
        logits = torch.from_numpy(native_coords[:3].reshape((3,) + tuple(data['image'].shape[1:])).astype(np.float32))
        restored = invert_logits(logits, data, transform, shape, affine).as_tensor().numpy()
        expected = np.indices(shape, dtype=np.float32)
        # Spacingd rounds the new extent: 24 slices at 2.5 mm span57.5mm,
        # represented by 58 integer-mm samples ending at57mm. Its inverse
        # border-clamps the final native slice at22.8 rather than23. This
        # is an expected interpolation boundary, NOT an exact inverse.
        interior = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
        self.assertLess(float(np.abs(restored[interior] - expected[interior]).max()), 1e-4)
        self.assertTrue(np.allclose(restored[2, :, :, -1], 22.8, atol=1e-4))
        self.assertLess(float(np.abs(restored[:2] - expected[:2]).max()), 1e-4)
        self.assertLess(float(np.abs(restored - expected).max()), .20001)

    def test_reject_shear_and_bad_logit_shape(self):
        affine = np.eye(4); affine[0, 1] = .2
        with self.assertRaises(ValueError):
            prepare(np.zeros((16, 16, 16), np.float32), affine)
        data, transform = prepare(np.zeros((16, 16, 16), np.float32), np.eye(4))
        with self.assertRaises(ValueError):
            invert_logits(torch.zeros((2, 96, 96, 96)), data, transform, (16, 16, 16), np.eye(4))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

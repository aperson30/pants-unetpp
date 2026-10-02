import unittest
import numpy as np
import SimpleITK as sitk
from fixed_crop_control import fixed_crop, same_grid


class FixedCropTests(unittest.TestCase):
    def test_exact_values_spacing_direction_and_physical_origin(self):
        array = np.arange(6*7*8, dtype=np.float32).reshape(6, 7, 8)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((.7, .8, 3.))
        image.SetOrigin((13., -9., 27.))
        image.SetDirection((-1., 0., 0., 0., 1., 0., 0., 0., -1.))
        bounds = dict(x_start=1, x_finish=7, y_start=2, y_finish=6, z_start=3, z_finish=5)
        crop = fixed_crop(image, bounds)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(crop), array[3:5, 2:6, 1:7])
        self.assertEqual(crop.GetOrigin(), image.TransformIndexToPhysicalPoint((1, 2, 3)))
        self.assertEqual(crop.GetDirection(), image.GetDirection())
        self.assertEqual(crop.GetSpacing(), image.GetSpacing())

    def test_bad_bounds_and_shifted_grid_rejected(self):
        image = sitk.GetImageFromArray(np.zeros((6, 7, 8), dtype=np.float32))
        bounds = dict(x_start=0, x_finish=9, y_start=0, y_finish=7, z_start=0, z_finish=6)
        with self.assertRaises(ValueError):
            fixed_crop(image, bounds)
        shifted = sitk.Image(image)
        shifted.SetOrigin((0., 0., 1.))
        with self.assertRaises(ValueError):
            same_grid(image, shifted)


if __name__ == '__main__':
    unittest.main()

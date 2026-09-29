"""Serial/parallel test conversion parity without any training data or GPU."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from data_conversion.convert_pants_to_nnunet import convert_split


class TestConversionParity(unittest.TestCase):
    def test_parallel_preserves_images_labels_geometry_and_tumor_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.diag([1.1, 1.2, 2.5, 1.])
            for cid in ("PanTS_00009001", "PanTS_00009002"):
                image_dir = root / "ImageTe" / cid
                seg_dir = root / "LabelTe" / cid / "segmentations"
                image_dir.mkdir(parents=True)
                seg_dir.mkdir(parents=True)
                image = np.arange(120, dtype=np.int16).reshape(4, 5, 6)
                nib.save(nib.Nifti1Image(image, affine), image_dir / "ct.nii.gz")
                pancreas = np.zeros(image.shape, np.uint8)
                pancreas[1:3, 1:4, 1:5] = 1
                tumor = np.zeros_like(pancreas)
                tumor[1, 2, 3] = 1
                for organ, data in (("pancreas", pancreas), ("pancreatic_lesion", tumor)):
                    nib.save(nib.Nifti1Image(data, affine), seg_dir / f"{organ}.nii.gz")
            serial = convert_split(root, "Te", root / "serial/images", root / "serial/labels", workers=1)
            parallel = convert_split(root, "Te", root / "parallel/images", root / "parallel/labels", workers=2)
            self.assertEqual(serial, parallel)
            for cid in serial:
                for kind, suffix in (("images", "_0000.nii.gz"), ("labels", ".nii.gz")):
                    left = nib.load(root / "serial" / kind / (cid + suffix))
                    right = nib.load(root / "parallel" / kind / (cid + suffix))
                    np.testing.assert_array_equal(np.asanyarray(left.dataobj), np.asanyarray(right.dataobj))
                    np.testing.assert_array_equal(left.affine, right.affine)
                labels = np.asanyarray(nib.load(root / "parallel/labels" / f"{cid}.nii.gz").dataobj)
                self.assertEqual(labels[1, 2, 3], 28)
                self.assertEqual(labels[2, 2, 3], 17)


if __name__ == "__main__":
    unittest.main()

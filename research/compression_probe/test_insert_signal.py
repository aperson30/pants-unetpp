import unittest
import numpy as np
from bounded_controls import insert_signal


class InsertionTests(unittest.TestCase):
    def test_fixed_signal_texture_and_source_unchanged(self):
        labels=np.ones((41,41,41),dtype=np.uint8)
        labels[4,4,4]=2
        image=np.arange(labels.size,dtype=np.float32).reshape(labels.shape)%100
        before=image.copy()
        modified,mask,info=insert_signal(image,labels,(1,1,1))
        self.assertTrue(mask.any())
        np.testing.assert_array_equal(image,before)
        np.testing.assert_array_equal(modified[mask]-image[mask],np.full(int(mask.sum()),-20.))
        np.testing.assert_array_equal(modified[~mask],image[~mask])
        self.assertTrue(np.all(labels[mask]==1))
        self.assertEqual(info['diameter_mm'],8.)

    def test_no_room_and_clipping_fail(self):
        small=np.ones((5,5,5),dtype=np.uint8)
        with self.assertRaises(ValueError):
            insert_signal(np.zeros(small.shape,dtype=np.float32),small,(1,1,1))
        labels=np.ones((41,41,41),dtype=np.uint8)
        with self.assertRaises(ValueError):
            insert_signal(np.full(labels.shape,-995.,dtype=np.float32),labels,(1,1,1))


if __name__=='__main__':
    unittest.main()

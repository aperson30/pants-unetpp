import unittest
import numpy as np
from denoiser_control import apply_filter, response, ring, box_for


class DenoiserContracts(unittest.TestCase):
    def test_constant_volume_preserved_by_both_filters(self):
        image = np.full((12,12,12),73.,dtype=np.float32)
        for family in ('tv','nlm'):
            out = apply_filter(image,family,10.)
            np.testing.assert_allclose(out,image,atol=1e-4,rtol=0)

    def test_metrics_identify_linear_retention_for_both_signs(self):
        mask = np.zeros((24,24,24),dtype=bool); mask[11:14,11:14,11:14] = True
        background = np.zeros(mask.shape,dtype=np.float32)
        surrounding = ring(mask,np.ones(3))
        for amplitude in (-80.,-10.,20.):
            out = background.copy(); out[mask] = .4*amplitude
            result = response(out,background,mask,surrounding,amplitude)
            self.assertAlmostEqual(result['ring_corrected_retention'],.4,places=6)

    def test_nlm_crop_interior_agrees_with_larger_volume(self):
        image = np.random.default_rng(20261002).normal(50,5,(32,32,32)).astype(np.float32)
        region = np.zeros(image.shape,dtype=bool); region[14:18,14:18,14:18] = True
        box = box_for(region,np.ones(3),6.)
        whole = apply_filter(image,'nlm',5.)
        local = apply_filter(image[box],'nlm',5.)
        np.testing.assert_allclose(whole[box][region[box]],local[region[box]],atol=1e-4,rtol=0)

    def test_global_sign_inversion_is_not_a_learned_model_property(self):
        image = np.random.default_rng(4).normal(0,10,(12,12,12)).astype(np.float32)
        for family in ('tv','nlm'):
            np.testing.assert_allclose(apply_filter(-image,family,4.),
                                       -apply_filter(image,family,4.),atol=1e-4,rtol=0)


if __name__ == '__main__':
    unittest.main()

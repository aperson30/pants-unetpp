import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from baseline_contracts import (TripletAssembler, hu_to_unit, unit_to_hu,
                                validate_geometry, validate_latents, inspect_safetensors)


class BaselineContracts(unittest.TestCase):
    def test_hu_roundtrip_and_explicit_clipping(self):
        hu = torch.tensor([-1500., -1000., -200., 0., 200., 1000., 1500.])
        original = hu.clone()
        torch.testing.assert_close(unit_to_hu(hu_to_unit(hu)), hu.clamp(-1000, 1000))
        self.assertTrue(torch.equal(hu, original))
        for invalid in (torch.tensor([float("nan")]), torch.tensor([1.01]), torch.tensor([-0.01])):
            with self.assertRaises(ValueError):
                unit_to_hu(invalid)

    def test_latent_four_eight_four_contract(self):
        noisy = torch.zeros(2, 4, 8, 8)
        validate_latents(noisy, noisy.clone(), noisy.clone())
        for other in (torch.zeros(2, 8, 8, 8), torch.zeros(1, 4, 8, 8), noisy.double()):
            with self.assertRaises(ValueError):
                validate_latents(noisy, other)
        with self.assertRaises(ValueError):
            validate_latents(noisy, noisy, torch.zeros(2, 8, 8, 8))

    def test_every_slice_and_chunk_order(self):
        for depth in (3, 4, 5, 17):
            ct = torch.arange(2 * 4 * depth).reshape(2, 4, depth).float()
            assembler = TripletAssembler(ct.shape)
            for start in reversed(range(depth - 2)):
                assembler.add(start, ct[:, :, start:start + 3])
            torch.testing.assert_close(assembler.finish(), ct, rtol=0, atol=0)
            self.assertTrue((assembler.counts > 0).all())

    def test_missing_duplicate_invalid_triplets_fail_closed(self):
        with self.assertRaises(ValueError):
            TripletAssembler((2, 4, 2))
        assembler = TripletAssembler((2, 4, 5))
        assembler.add(0, torch.ones(2, 4, 3))
        with self.assertRaises(ValueError):
            assembler.finish()
        snapshot = assembler.total.clone()
        for start, image in ((0, torch.ones(2, 4, 3)), (3, torch.ones(2, 4, 3)),
                             (1, torch.ones(2, 3, 4)), (1, torch.full((2, 4, 3), float('nan')))):
            with self.assertRaises(ValueError):
                assembler.add(start, image)
            self.assertTrue(torch.equal(snapshot, assembler.total))

    def test_affine_valid_oblique_and_invalid(self):
        affine = torch.eye(4, dtype=torch.float64)
        affine[0, 1] = 0.2
        validate_geometry((5, 7, 9), affine)
        for bad in (torch.zeros(4, 4), torch.eye(3), torch.full((4, 4), float('nan'))):
            with self.assertRaises(ValueError):
                validate_geometry((5, 7, 9), bad)

    def test_resizing_white_noise_changes_distribution(self):
        # A deterministic fixture proving nonidentity; no empirical threshold tuning.
        generator = torch.Generator().manual_seed(7)
        noise = torch.randn(64, 4, 32, 32, generator=generator)
        resized = F.interpolate(noise, size=(64, 64), mode="bilinear", align_corners=False)
        self.assertLess(float(resized.var()), float(noise.var()))
        self.assertGreater(float((resized[:, :, :, :-1] * resized[:, :, :, 1:]).mean()), 0)

    def test_safetensors_header_read_only(self):
        header = {"conv_in.weight": {"dtype": "F32", "shape": [2, 8, 1, 1],
                                    "data_offsets": [0, 64]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            raw_header = json.dumps(header).encode()
            raw = struct.pack('<Q', len(raw_header)) + raw_header + bytes(64)
            path.write_bytes(raw)  # Test-only disposable fixture, not model/data output.
            self.assertEqual(inspect_safetensors(path)["conv_in_shape"], [2, 8, 1, 1])
            self.assertEqual(path.read_bytes(), raw)
            with self.assertRaises(ValueError):
                inspect_safetensors(path, expected_input_channels=4)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                inspect_safetensors(path)


if __name__ == '__main__':
    unittest.main()

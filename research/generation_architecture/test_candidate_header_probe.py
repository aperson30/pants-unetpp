import struct
import unittest
from candidate_header_probe import parse_header


class HeaderTests(unittest.TestCase):
    def fixture(self, endian):
        data = bytearray(348)
        struct.pack_into(endian + "i", data, 0, 348)
        struct.pack_into(endian + "8h", data, 40, 3, 512, 512, 50, 1, 1, 1, 1)
        struct.pack_into(endian + "8f", data, 76, 1, .7, .7, 2.5, 1, 1, 1, 1)
        data[344:348] = b"n+1\0"
        return data

    def test_endianness(self):
        for endian in ("<", ">"):
            self.assertEqual(parse_header(self.fixture(endian))["shape"], [512, 512, 50])

    def test_invalid_shape_and_spacing(self):
        for offset, kind, value in ((42, "h", 0), (80, "f", float("nan"))):
            data = self.fixture("<")
            struct.pack_into("<" + kind, data, offset, value)
            with self.assertRaises(ValueError):
                parse_header(data)

    def test_no_invented_affine(self):
        self.assertIsNone(parse_header(self.fixture("<"))["sform_rows"])
        with self.assertRaises(ValueError):
            parse_header(b"short")

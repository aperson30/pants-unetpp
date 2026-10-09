import gzip
import struct
import unittest
from candidate_label_extension import count_mask


class LabelExtensionTests(unittest.TestCase):
    def fixture(self, shape=(2, 2, 2), datatype=2, payload=b"\0\1\0\1\0\0\0\0"):
        header = bytearray(352)
        struct.pack_into("<i", header, 0, 348)
        struct.pack_into("<8h", header, 40, 3, *shape, 1, 1, 1, 1)
        struct.pack_into("<2h", header, 70, datatype, 8)
        struct.pack_into("<3f", header, 108, 352, 1, 0)
        header[344:348] = b"n+1\0"
        return gzip.compress(header + payload)

    def test_count_without_correspondence_claim(self):
        result = count_mask(self.fixture())
        self.assertEqual(result["foreground_voxels"], 2)
        self.assertFalse(result["tumor_correspondence_verified"])

    def test_reject_unsupported_or_short_payload(self):
        for fixture in (self.fixture(datatype=4), self.fixture(payload=b"\0"), self.fixture(payload=b"\0" * 9)):
            with self.assertRaises(ValueError):
                count_mask(fixture)

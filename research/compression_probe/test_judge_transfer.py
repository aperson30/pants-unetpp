import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from fetch_judge_fold4 import stream_member


class TransferTests(unittest.TestCase):
    def archive(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('weights.pth', b'not executable weights' * 100)
        payload.seek(0)
        return zipfile.ZipFile(payload)

    def test_bounded_stream_and_digest(self):
        with self.archive() as archive, tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'weights.partial'
            member = archive.getinfo('weights.pth')
            result = stream_member(archive, member, path)
            self.assertEqual(result['bytes'], member.file_size)
            self.assertEqual(result['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_existing_file_preserved(self):
        with self.archive() as archive, tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'existing'
            path.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                stream_member(archive, archive.getinfo('weights.pth'), path)
            self.assertEqual(path.read_bytes(), b'preserve')

    def test_oversize_member_rejected(self):
        with self.archive() as archive, tempfile.TemporaryDirectory() as root:
            member = archive.getinfo('weights.pth')
            member.file_size = 300_000_001
            with self.assertRaises(ValueError):
                stream_member(archive, member, Path(root) / 'out')


if __name__ == '__main__':
    unittest.main()

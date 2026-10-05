import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.recover_geometry_sources import download, recover


class RecoveryTests(unittest.TestCase):
    def test_download_hash_gate_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); body = b'archive fixture'
            response = io.BytesIO(body); response.headers = {'Content-Length': str(len(body))}
            digest = hashlib.sha256(body).hexdigest()
            with patch('urllib.request.urlopen', return_value=response):
                target = download(root, 'a.tar.gz', 'https://example.invalid/', digest, len(body))
            self.assertEqual(target.read_bytes(), body)
            self.assertEqual(download(root, 'a.tar.gz', '', digest, len(body)), target)
            with self.assertRaises(RuntimeError):
                download(root, 'a.tar.gz', '', 'wrong', len(body))
            self.assertEqual(target.read_bytes(), body)

    def test_wrong_download_digest_never_publishes_archive(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); response = io.BytesIO(b'bad')
            response.headers = {'Content-Length': '3'}
            with patch('urllib.request.urlopen', return_value=response), self.assertRaises(RuntimeError):
                download(root, 'a', '', 'wrong', 3)
            self.assertFalse((root / 'a').exists())

    def test_extracts_only_selected_regular_member_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); archive = root / 'a.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                for member_name in ('PanTS_00009232/ct.nii.gz', 'PanTS_00009001/ct.nii.gz'):
                    member = tarfile.TarInfo(member_name); member.size = 3
                    tar.addfile(member, io.BytesIO(b'abc'))
            destination = root / 'selected'
            self.assertEqual(len(recover(archive, destination)), 1)
            self.assertFalse((destination / 'PanTS_00009001').exists())
            with self.assertRaises(FileExistsError):
                recover(archive, destination)

    def test_unsafe_member_is_rejected_without_extracting(self):
        for bad in ('../escape.nii.gz', '/absolute.nii.gz'):
            with self.subTest(path=bad), tempfile.TemporaryDirectory() as name:
                root = Path(name); archive = root / 'a.tar.gz'
                with tarfile.open(archive, 'w:gz') as tar:
                    member = tarfile.TarInfo(bad); member.size = 3
                    tar.addfile(member, io.BytesIO(b'abc'))
                with self.assertRaises(RuntimeError):
                    recover(archive, root / 'selected')


if __name__ == '__main__':
    unittest.main()

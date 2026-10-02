import io
import unittest
import zipfile
from list_remote_zip import RangeReader, inspect


class Response(io.BytesIO):
    status = 206

    def __init__(self, body, left, right, total):
        super().__init__(body)
        self.headers = {'Content-Range': f'bytes {left}-{right}/{total}'}


class ZipRangeTests(unittest.TestCase):
    def test_directory_only_matches_standard_parser(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('case.nii.gz', b'not a real CT'*100)
            archive.writestr('metadata.txt', b'metadata')
        payload = stream.getvalue()
        def opener(request, timeout):
            left, right = map(int, request.headers['Range'].split('=')[1].split('-'))
            return Response(payload[left:right+1], left, right, len(payload))
        result = inspect('https://zenodo.org/test', opener=opener)
        self.assertEqual([m['name'] for m in result['entries']], ['case.nii.gz', 'metadata.txt'])
        self.assertEqual(result['member_count'], 2)

    def test_ignored_range_rejected_before_body_read(self):
        class BadResponse(Response):
            status = 200
            def read(self, *args):
                raise AssertionError('Must not read full archive response')
        with self.assertRaisesRegex(ValueError, 'did not honor Range'):
            RangeReader('https://zenodo.org/test', opener=lambda *a, **k: BadResponse(b'', 0, 0, 100))

    def test_budget_and_unbounded_read_rejected(self):
        reader = RangeReader('https://zenodo.org/test', byte_limit=2,
                             opener=lambda *a, **k: Response(b'x', 0, 0, 100))
        with self.assertRaisesRegex(ValueError, 'Unbounded'):
            reader.read()
        with self.assertRaisesRegex(ValueError, 'budget'):
            reader.read(3)


if __name__ == '__main__':
    unittest.main()

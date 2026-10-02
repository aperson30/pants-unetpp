"""Read public ZIP directory metadata with strict HTTP-range byte bounds.

Never extracts CTs/checkpoints. Reject a server ignoring Range before reading
its response body. Standard-library ZIP parser, no package/runtime changes.
"""
import argparse
import io
import json
import re
import time
import urllib.request
import zipfile


class RangeReader(io.RawIOBase):
    def __init__(self, url, opener=urllib.request.urlopen, byte_limit=8*1024**2,
                 max_calls=32):
        if not isinstance(max_calls, int) or not 1 <= max_calls <= 512:
            raise ValueError('Invalid bounded request limit')
        self.max_calls = max_calls
        self.url, self.opener, self.byte_limit = url, opener, byte_limit
        self.used, self.calls, self.position, self.total = 0, 0, 0, None
        self.etag = None
        self.deadline = time.monotonic() + 180
        self._fetch(0, 1)

    def _fetch(self, start, count):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Range probe total deadline exceeded')
        # Reserve the extra byte used to detect an oversized server response.
        if count > 4*1024**2 or self.used+count+1 > self.byte_limit or self.calls >= self.max_calls:
            raise ValueError('Range metadata budget exceeded')
        request = urllib.request.Request(self.url, headers={
            'Range': f'bytes={start}-{start+count-1}', 'Accept-Encoding': 'identity'})
        if self.etag and not self.etag.startswith('W/'):
            request.add_header('If-Match', self.etag)
        self.calls += 1
        with self.opener(request, timeout=min(15, remaining)) as response:
            if response.status != 206:
                raise ValueError('Server did not honor Range; body not read')
            etag = response.headers.get('ETag')
            if self.calls == 1:
                self.etag = etag
            elif self.etag != etag:
                raise ValueError('Archive identity changed between requests')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match:
                raise ValueError('Missing/invalid Content-Range')
            left, right, total = map(int, match.groups())
            if left != start or right != start+count-1 or total <= right:
                raise ValueError('Unexpected range bounds')
            if self.total is not None and self.total != total:
                raise ValueError('Archive changed between requests')
            self.total = total
            data = response.read(count+1)
            self.used += len(data)
            if len(data) != count:
                raise ValueError('Truncated/oversized range body')
            return data

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.total}[whence]
        self.position = base+offset
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def read(self, size=-1):
        if size < 0:
            # Python3.11 zipfile asks read() after seeking to its bounded
            # end-directory tail. Never permit that from the file start.
            remaining = max(0, self.total-self.position)
            if self.position <= 0 or remaining > 65558:
                raise ValueError('Unbounded read prohibited')
            size = remaining
        size = min(size, max(0, self.total-self.position))
        if size == 0:
            return b''
        data = self._fetch(self.position, size)
        self.position += len(data)
        return data


def inspect(url, opener=urllib.request.urlopen):
    reader = RangeReader(url, opener=opener)
    with zipfile.ZipFile(reader) as archive:
        members = archive.infolist()
        # Names/compression metadata only; no ZipFile.open/extract/read calls.
        summary = dict(directory_read=True, remote_bytes=reader.total,
                       fetched_bytes=reader.used, requests=reader.calls,
                       member_count=len(members),
                       entries=[dict(name=m.filename, uncompressed=m.file_size,
                                     compressed=m.compress_size, compression=m.compress_type,
                                     header_offset=m.header_offset, encrypted=bool(m.flag_bits & 1))
                                for m in members])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    args = parser.parse_args()
    if not args.url.startswith('https://zenodo.org/'):
        raise ValueError('This probe is restricted to the public Zenodo source')
    print(json.dumps(inspect(args.url)), flush=True)


if __name__ == '__main__':
    main()

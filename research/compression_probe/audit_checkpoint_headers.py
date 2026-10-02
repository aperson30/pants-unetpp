"""Read nnU-Net v2 checkpoint metadata from a remote ZIP without downloading weights.

Research audit tool (CPU, network-only). For each requested checkpoint inside a
remote ZIP (e.g. the PANORAMA baseline Zenodo archives), it:

1. reads the outer ZIP central directory with an HTTP Range request,
2. streams and inflates only the start of the checkpoint member until the
   inner PyTorch archive's first record (``<name>/data.pkl``) is complete,
3. parses that pickle with a restricted unpickler that never imports a class
   or executes code (every global becomes an inert stub) and never loads tensor
   storage (persistent IDs resolve to None),
4. prints init_args.fold / configuration, dataset_json.numTraining,
   current_epoch, trainer_name and the inner archive name.

It saves nothing except stdout. Typical transfer is a few MB per checkpoint.
Evidence scope: ``init_args.fold`` proves the fold index the trainer was
constructed with, not the exact split file used during training.

Example:
  python audit_checkpoint_headers.py \
    https://zenodo.org/records/11160381/files/Dataset104_PANORAMA_baseline_PDAC_Detection.zip \
    fold_0/checkpoint_best_panorama.pth
"""
from __future__ import annotations

import collections
import io
import pickle
import struct
import sys
import urllib.request
import zlib

CHUNK = 4 << 20
MAX_STREAM = 64 << 20
TAIL = 262144


def fetch(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(request) as response:
        if response.status != 206:
            raise RuntimeError(f'server ignored Range request (HTTP {response.status})')
        return response.read()


def content_length(url: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD')) as response:
        return int(response.headers['Content-Length'])


def find_member(url: str, suffix: str):
    size = content_length(url)
    tail = fetch(url, max(0, size - TAIL), size - 1)
    position = 0
    while True:
        position = tail.find(b'PK\x01\x02', position)
        if position < 0:
            raise LookupError(f'{suffix!r} not found in the last {TAIL} bytes of the directory')
        method = struct.unpack('<H', tail[position + 10:position + 12])[0]
        compressed, uncompressed = struct.unpack('<II', tail[position + 20:position + 28])
        name_len, extra_len, comment_len = struct.unpack('<HHH', tail[position + 28:position + 34])
        offset = struct.unpack('<I', tail[position + 42:position + 46])[0]
        name = tail[position + 46:position + 46 + name_len].decode()
        extra = tail[position + 46 + name_len:position + 46 + name_len + extra_len]
        if name.endswith(suffix):
            if 0xFFFFFFFF in (compressed, uncompressed, offset):
                cursor = 0
                while cursor < len(extra):
                    header_id, header_len = struct.unpack('<HH', extra[cursor:cursor + 4])
                    if header_id == 1:
                        values = iter(struct.unpack('<' + 'Q' * (header_len // 8),
                                                    extra[cursor + 4:cursor + 4 + header_len]))
                        if uncompressed == 0xFFFFFFFF:
                            uncompressed = next(values)
                        if compressed == 0xFFFFFFFF:
                            compressed = next(values)
                        if offset == 0xFFFFFFFF:
                            offset = next(values)
                    cursor += 4 + header_len
            return name, method, offset
        position += 46 + name_len + extra_len + comment_len


class _Inert(dict):
    """Stand-in for every pickled global; accepts construction and state silently."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return _Inert()


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        return _Inert

    def persistent_load(self, pid):
        return None


def first_inner_pickle(url: str, outer_method: int, outer_offset: int):
    local = fetch(url, outer_offset, outer_offset + 29)
    if local[:4] != b'PK\x03\x04':
        raise RuntimeError('bad outer local header')
    name_len, extra_len = struct.unpack('<HH', local[26:30])
    data_start = outer_offset + 30 + name_len + extra_len
    inflater = zlib.decompressobj(-15) if outer_method == 8 else None
    if outer_method not in (0, 8):
        raise RuntimeError(f'unsupported outer compression method {outer_method}')
    buffer, cursor = b'', data_start
    while cursor - data_start < MAX_STREAM:
        chunk = fetch(url, cursor, cursor + CHUNK - 1)
        cursor += CHUNK
        buffer += inflater.decompress(chunk) if inflater else chunk
        if len(buffer) >= 30 and buffer[:4] == b'PK\x03\x04':
            inner_len, inner_extra = struct.unpack('<HH', buffer[26:30])
            inner_name = buffer[30:30 + inner_len].decode()
            if not inner_name.endswith('data.pkl'):
                raise RuntimeError(f'first inner record is {inner_name!r}, not data.pkl')
            payload = io.BytesIO(buffer[30 + inner_len + inner_extra:])
            try:
                # The pickle ends at its STOP opcode; trailing bytes are ignored.
                return inner_name, _RestrictedUnpickler(payload).load()
            except (EOFError, pickle.UnpicklingError):
                continue
    raise RuntimeError('data.pkl not complete within the streaming budget')


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    url, suffix = argv[1], argv[2]
    member, method, offset = find_member(url, suffix)
    inner_name, state = first_inner_pickle(url, method, offset)
    init_args = state.get('init_args', {}) if isinstance(state, dict) else {}
    dataset_json = init_args.get('dataset_json') if isinstance(init_args, dict) else None
    print(f'member: {member}')
    print(f'inner_record: {inner_name}')
    print(f"init_args.fold: {init_args.get('fold')}")
    print(f"init_args.configuration: {init_args.get('configuration')}")
    print(f"dataset_json.numTraining: {dataset_json.get('numTraining') if isinstance(dataset_json, dict) else None}")
    print(f"current_epoch: {state.get('current_epoch')}")
    print(f"trainer_name: {state.get('trainer_name')}")


if __name__ == '__main__':
    main(sys.argv)

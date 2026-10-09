"""Bounded label-archive pass for new cross-accession candidates only.

No extraction, CT download, raw clinical identifiers, GPU use or eligibility pass.
Use previous complete mask evidence for the original candidates.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import struct
import tarfile
import time
import urllib.request

import numpy as np

ARCHIVE = "https://www.cs.jhu.edu/~zongwei/dataset/CancerVerse_Label.tar.gz"
REV = "c15802c28cb31b71dee2e680319d522b3bc4cbbf"
META_HASH = "3e58b8a1b8cb3b9664d446ef4d7ca393c51495af5073119c25596b2d4bb903d8"
ARCHIVE_SIZE = 551163721
ARCHIVE_SHA = "d78e18a46a896df71cd49db12817738439f35df4d243979a2aad0f03b39a473a"


def cache_archive(path):
    """Pin to the completed prior audit; cache once, never overwrite/retry."""
    path = Path(path)
    if path.exists():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        if path.stat().st_size != ARCHIVE_SIZE or digest.hexdigest() != ARCHIVE_SHA:
            raise ValueError("Existing cache differs; preserve and stop")
        return path
    if shutil.disk_usage(path.parent).free < 2 * 1024**3:
        raise ValueError("Need 2GiB free for isolated archive cache")
    partial = path.with_suffix(path.suffix + ".partial")
    digest, count, deadline = hashlib.sha256(), 0, time.monotonic() + 180
    with urllib.request.urlopen(ARCHIVE, timeout=25) as source, partial.open("xb") as output:
        while True:
            if time.monotonic() > deadline:
                raise Limit("Archive cache download deadline")
            block = source.read(min(1024**2, ARCHIVE_SIZE - count + 1))
            if not block:
                break
            count += len(block)
            if count > ARCHIVE_SIZE:
                raise ValueError("Archive cache exceeded pinned size")
            output.write(block)
            digest.update(block)
    if count != ARCHIVE_SIZE or digest.hexdigest() != ARCHIVE_SHA:
        raise ValueError("Archive version differs from previous complete audit")
    partial.rename(path)
    return path


class Limit(Exception):
    pass


class Reader:
    def __init__(self, source):
        self.source, self.count = source, 0
        self.deadline = time.monotonic() + 150
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        if time.monotonic() > self.deadline or self.count >= 560_000_000:
            raise Limit("Compressed byte/time limit reached")
        data = self.source.read(min(65536 if size < 0 else size, 560_000_000 - self.count))
        self.count += len(data)
        self.digest.update(data)
        return data


def count_mask(raw):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        header = stream.read(352)
        if len(header) != 352:
            raise ValueError("Short mask header")
        e = "<" if struct.unpack_from("<i", header)[0] == 348 else ">"
        if struct.unpack_from(e + "i", header)[0] != 348 or header[344:348] != b"n+1\0":
            raise ValueError("Unsupported mask")
        dims = struct.unpack_from(e + "8h", header, 40)
        datatype, bitpix = struct.unpack_from(e + "2h", header, 70)
        offset, slope, intercept = struct.unpack_from(e + "3f", header, 108)
        if dims[0] != 3 or min(dims[1:4]) <= 0 or datatype != 2 or bitpix != 8:
            raise ValueError("Expected scalar uint8 3D mask")
        if not math.isfinite(offset) or offset != int(offset) or not 352 <= offset <= 65536:
            raise ValueError("Invalid mask offset")
        if slope != 0 and math.isfinite(slope) and (slope <= 0 or intercept != 0):
            raise ValueError("Unexpected mask scaling")
        if len(stream.read(int(offset) - 352)) != int(offset) - 352:
            raise ValueError("Short mask extension")
        remaining = math.prod(dims[1:4])
        if remaining > 512 * 1024**2:
            raise ValueError("Decoded mask cap")
        foreground = 0
        while remaining:
            payload = stream.read(min(remaining, 1024**2))
            if not payload:
                raise ValueError("Short mask payload")
            foreground += int(np.count_nonzero(np.frombuffer(payload, dtype=np.uint8)))
            remaining -= len(payload)
        if stream.read(1):
            raise ValueError("Extra mask payload")
    return {"foreground_voxels": foreground, "shape": list(dims[1:4]),
            "tumor_correspondence_verified": False, "registration_verified": False}


def inspect(strict, expanded, archive_cache=None):
    for manifest in (strict, expanded):
        if manifest["revision"] != REV or manifest["metadata_sha256"] != META_HASH:
            raise ValueError("Candidate metadata changed")
    cases = lambda m: {r["case_id"] for g in m["candidate_groups"] for phase in ("noncontrast", "contrast") for r in g[phase]}
    targets = cases(expanded) - cases(strict)
    if not 0 < len(targets) <= 40:
        raise ValueError("Unexpected extension size")
    records, complete, reason = {}, False, None
    with (cache_archive(archive_cache).open("rb") if archive_cache else urllib.request.urlopen(ARCHIVE, timeout=25)) as source:
        archive_headers = {"cached_archive_sha256": ARCHIVE_SHA, "Content-Length": str(ARCHIVE_SIZE)} if archive_cache else {key: source.headers.get(key) for key in ("ETag", "Last-Modified", "Content-Length")}
        reader = Reader(source)
        try:
            with gzip.GzipFile(fileobj=reader) as decoded, tarfile.open(fileobj=decoded, mode="r|", bufsize=65536) as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    ids = [part for part in path.parts if part in targets]
                    if path.name != "pancreatic_lesion.nii.gz" or not ids:
                        continue
                    if len(ids) != 1 or ids[0] in records or not member.isfile() or member.size > 4 * 1024**2:
                        raise ValueError("Duplicate/oversized/non-file target mask")
                    with archive.extractfile(member) as mask:
                        raw = mask.read(member.size + 1)
                    if len(raw) != member.size:
                        raise ValueError("Mask archive length mismatch")
                    records[ids[0]] = count_mask(raw)
                    print(json.dumps({"masks_seen": len(records), "target_count": len(targets), "bytes_read": reader.count}), flush=True)
                    if set(records) == targets:
                        reason = "All extension targets inspected; archive not exhausted"
                        break
                else:
                    complete = True
        except Limit as error:
            reason = str(error)
    return {"revision": REV, "metadata_sha256": META_HASH, "archive": ARCHIVE,
            "archive_headers": archive_headers, "compressed_bytes_read": reader.count,
            "compressed_prefix_sha256": reader.digest.hexdigest(), "archive_scan_complete": complete,
            "stop_reason": reason, "target_cases": sorted(targets), "masks": records,
            "unseen_targets": sorted(targets - set(records)), "gpu_hours": 0,
            "training_eligible": False, "raw_identifiers_saved": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", required=True)
    parser.add_argument("--expanded", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--archive-cache")
    args = parser.parse_args()
    with open(args.strict) as source:
        strict = json.load(source)
    with open(args.expanded) as source:
        expanded = json.load(source)
    result = inspect(strict, expanded, args.archive_cache)
    with open(args.out, "x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({"targets": len(result["target_cases"]), "seen": len(result["masks"]),
                      "unseen": len(result["unseen_targets"])}), flush=True)

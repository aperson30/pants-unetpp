"""Read at most 64 KiB per public CT; parse only 348 NIfTI header bytes.

Not voxel inspection, full-file integrity, phase verification or registration.
No patient identifiers, reports or image payloads are retained.
"""
import argparse
import hashlib
import json
import math
import re
import struct
import urllib.request
import zlib

REVISION = "c15802c28cb31b71dee2e680319d522b3bc4cbbf"
CASES = ("CV_00010488", "CV_00004804")
CAP = 65536


def parse_header(raw):
    if len(raw) != 348:
        raise ValueError("Require exactly one NIfTI-1 header")
    endian = next((e for e in ("<", ">") if struct.unpack_from(e + "i", raw)[0] == 348), None)
    if endian is None or raw[344:348] != b"n+1\0":
        raise ValueError("Unsupported NIfTI header")
    dims = struct.unpack_from(endian + "8h", raw, 40)
    pixdim = struct.unpack_from(endian + "8f", raw, 76)
    if dims[0] != 3 or any(d <= 0 for d in dims[1:4]):
        raise ValueError("Require a positive 3D image shape")
    if any(not math.isfinite(p) or p <= 0 for p in pixdim[1:4]):
        raise ValueError("Invalid voxel spacing")
    qform, sform = struct.unpack_from(endian + "2h", raw, 252)
    rows = [list(struct.unpack_from(endian + "4f", raw, offset)) for offset in (280, 296, 312)]
    if sform and not all(math.isfinite(v) for row in rows for v in row):
        raise ValueError("Nonfinite sform")
    return {"shape": list(dims[1:4]), "spacing_header": list(pixdim[1:4]),
            "spatial_units_code": raw[123] & 7, "qform_code": qform,
            "sform_code": sform, "sform_rows": rows if sform else None,
            "datatype_code": struct.unpack_from(endian + "h", raw, 70)[0],
            "header_sha256": hashlib.sha256(raw).hexdigest()}


def inspect(case):
    if not re.fullmatch(r"CV_[0-9]{8}", case):
        raise ValueError("Require a public CancerVerse case filename ID")
    url = f"https://huggingface.co/datasets/BodyMaps/CancerVerse/resolve/{REVISION}/CancerVerse/{case}/ct.nii.gz"
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{CAP-1}"})
    with urllib.request.urlopen(request, timeout=20) as response:
        status = response.status
        content_range = response.headers.get("Content-Range")
        if status == 206 and (content_range is None or not content_range.startswith("bytes 0-")):
            raise ValueError("Partial response does not begin at header")
        prefix = response.read(CAP)
    header = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(prefix, 348)
    return {"case_id": case, "url": url, "http_status": status,
            "bytes_read": len(prefix), "geometry": parse_header(header),
            "full_file_integrity_verified": False, "registration_verified": False,
            "tumor_annotations_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    cases = args.case or CASES
    if len(cases) > 8 or len(cases) != len(set(cases)):
        raise ValueError("At most eight distinct case headers per bounded run")
    results = [inspect(case) for case in cases]
    with open(args.out, "x", encoding="utf-8") as output:
        json.dump({"revision": REVISION, "scope": "header-only candidate screen",
                   "gpu_hours": 0, "results": results}, output, indent=2, allow_nan=False)
    print(json.dumps(results, indent=2, allow_nan=False))

"""Original CPU preflight fixtures; not a replacement SMILE pipeline.

Explicit HWD convention for volume assembly. No file edits, model loads,
downloads, protected data access, or implicit resampling/clipping.
"""

import json
import math
import struct
from pathlib import Path

import torch


def finite_float(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")


def validate_latents(noisy, source, residual=None):
    finite_float(noisy, "noisy latent")
    finite_float(source, "source latent")
    if noisy.ndim != 4 or noisy.shape[1] != 4 or min(noisy.shape) < 1:
        raise ValueError("Expected nonempty Bx4xHxW noisy latent, NOT Bx8xHxW")
    if (source.shape != noisy.shape or source.device != noisy.device
            or source.dtype != noisy.dtype):
        raise ValueError("Source/noisy latents must match shape/device/dtype")
    if residual is not None:
        finite_float(residual, "residual")
        if (residual.shape != noisy.shape or residual.device != noisy.device
                or residual.dtype != noisy.dtype):
            raise ValueError("Denoiser must return four-channel residual matching noisy latent")


def hu_to_unit(hu):
    """Baseline input clipping [-1000,1000] followed by [0,1] normalization."""
    finite_float(hu, "CT HU")
    return (hu.clamp(-1000, 1000) + 1000) / 2000


def unit_to_hu(unit):
    """Decoded [0,1] to float HU; reject overshoot rather than hide clipping."""
    finite_float(unit, "decoded CT")
    if (unit < 0).any() or (unit > 1).any():
        raise ValueError("Decoded unit range violated; inspect interpolation, do not silently clip")
    return unit * 2000 - 1000


def validate_geometry(shape_hwd, affine):
    if (len(shape_hwd) != 3 or any(isinstance(n, bool) or not isinstance(n, int)
                                 or n < 1 for n in shape_hwd)):
        raise ValueError("Expected three positive integer HWD dimensions")
    finite_float(affine, "affine")
    if affine.shape != (4, 4):
        raise ValueError("Expected 4x4 affine")
    expected = affine.new_tensor([0, 0, 0, 1])
    if not torch.allclose(affine[3], expected, rtol=0, atol=1e-8):
        raise ValueError("Invalid homogeneous affine row")
    if torch.linalg.det(affine[:3, :3].double()).abs() <= 1e-12:
        raise ValueError("Singular spatial geometry")
    # Valid oblique/sheared NIfTI geometry is not automatically orthonormal.


class TripletAssembler:
    """CPU overlap average; each valid start appears once, all slices required."""
    def __init__(self, shape_hwd):
        validate_geometry(shape_hwd, torch.eye(4))
        if shape_hwd[2] < 3:
            raise ValueError("Three-slice baseline needs depth >=3")
        self.shape = tuple(shape_hwd)
        self.total = torch.zeros(self.shape, dtype=torch.float32)
        self.counts = torch.zeros(self.shape[2], dtype=torch.int64)
        self.starts = set()

    def add(self, start, triplet_hwc):
        if (isinstance(start, bool) or not isinstance(start, int)
                or not 0 <= start <= self.shape[2] - 3 or start in self.starts):
            raise ValueError("Invalid or repeated slice start")
        finite_float(triplet_hwc, "triplet")
        if triplet_hwc.device.type != "cpu" or triplet_hwc.shape != (*self.shape[:2], 3):
            raise ValueError("Expected CPU HxWx3 triplet at original output resolution")
        # Validate before modifying accumulator state.
        self.total[:, :, start:start + 3] += triplet_hwc.float()
        self.counts[start:start + 3] += 1
        self.starts.add(start)

    def finish(self):
        if self.starts != set(range(self.shape[2] - 2)) or (self.counts == 0).any():
            raise ValueError("Incomplete triplet set: refusing missing-slice output")
        output = self.total / self.counts.to(torch.float32)[None, None]
        finite_float(output, "assembled CT")
        return output


def inspect_safetensors(path, expected_input_channels=8):
    """Read-only shape/length preflight, not a full integrity/authenticity check.

    Does not allocate tensor payload, execute pickle or claim checkpoint loading.
    The expected checkpoint is the finetuned denoiser, not VAE/optimizer weights.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, size - 8):
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header object")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
              "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    ranges = []
    for key, info in tensors.items():
        if not isinstance(info, dict):
            raise ValueError(f"Invalid tensor descriptor: {key}")
        shape, offsets, dtype = info.get("shape"), info.get("data_offsets"), info.get("dtype")
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets) or dtype not in widths):
            raise ValueError(f"Unsupported/malformed tensor: {key}")
        lo, hi = offsets
        if not 0 <= lo <= hi <= size - 8 - length:
            raise ValueError(f"Out-of-file tensor: {key}")
        if hi - lo != math.prod(shape) * widths[dtype]:
            raise ValueError(f"Tensor byte size mismatch: {key}")
        if hi > lo:
            ranges.append((lo, hi))
    ranges.sort()
    if (not ranges or ranges[0][0] != 0 or ranges[-1][1] != size - 8 - length
            or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:]))):
        raise ValueError("Noncontiguous/overlapping/incomplete tensor payload")
    conv = tensors.get("conv_in.weight", {})
    shape = conv.get("shape", [])
    if len(shape) != 4 or shape[1] != expected_input_channels:
        raise ValueError("Finetuned denoiser conv_in channel contract violated")
    return {"size_bytes": size, "tensor_count": len(tensors), "conv_in_shape": shape,
            "scope": "header only; hash, full loader and GPU inference still required"}

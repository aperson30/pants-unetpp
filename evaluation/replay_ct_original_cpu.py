"""Bounded original-CPU CT replay; never edits protected evaluation inputs.

No CUDA imports or model execution. A matching hash is diagnostic evidence,
not a full source-audit certificate or permission to score predictions.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import platform
import shutil
import socket
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.recover_geometry_sources import sha


def replay(raw, destination, correction, expected, raw_expected, correction_expected):
    if sha(raw) != raw_expected or sha(correction) != correction_expected:
        raise RuntimeError('raw CT/frozen correction identity differs')
    destination.mkdir(parents=True, exist_ok=False)
    copied = destination / raw.name
    shutil.copyfile(raw, copied)
    # Import only the checksum-verified, previously used correction; no pyc
    # writes into the frozen source tree.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        specification = importlib.util.spec_from_file_location('frozen_ct_correction', correction)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        corrected_count = module.fix_folder(destination)
    finally:
        sys.dont_write_bytecode = previous
    source = nib.load(raw); fixed = nib.load(copied)
    if not np.array_equal(np.asanyarray(source.dataobj), np.asanyarray(fixed.dataobj)):
        raise RuntimeError('frozen correction changed CT voxels')
    digest = sha(copied)
    if sha(raw) != raw_expected or sha(correction) != correction_expected:
        raise RuntimeError('original evidence changed during replay')
    runtime = io.StringIO()
    with contextlib.redirect_stdout(runtime):
        np.show_runtime()
    report = {'schema': 'pants-original-cpu-ct-replay-v1', 'diagnostic_only': True,
              'hostname': socket.gethostname(), 'machine': platform.machine(),
              'numpy': np.__version__, 'nibabel': nib.__version__, 'numpy_runtime': runtime.getvalue(),
              'raw_sha256': raw_expected, 'correction_sha256': correction_expected,
              'replayed_sha256': digest, 'expected_prediction_sha256': expected,
              'exact_prediction_input_match': digest == expected, 'source_voxels_unchanged': True,
              'corrected_count': corrected_count, 'shape': list(fixed.shape), 'affine': fixed.affine.tolist()}
    with (destination / 'original_cpu_replay.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--frozen-correction', type=Path, required=True)
    parser.add_argument('--raw-sha256', required=True)
    parser.add_argument('--correction-sha256', required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    args = parser.parse_args()
    cid = args.raw.name.removesuffix('_0000.nii.gz')
    first = json.loads((args.evaluation_dir / 'unetpp_ds/prediction_provenance.json').read_text())
    second = json.loads((args.evaluation_dir / 'unetpp_nods/prediction_provenance.json').read_text())
    if first['inputs'] != second['inputs'] or cid not in first['inputs']:
        raise RuntimeError('frozen prediction input identities differ')
    replay(args.raw, args.destination, args.frozen_correction, first['inputs'][cid],
           args.raw_sha256, args.correction_sha256)
    print('ORIGINAL_CPU_REPLAY_DONE_NOT_A_COHORT_CERTIFICATE', flush=True)

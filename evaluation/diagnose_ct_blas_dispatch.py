"""CPU-only replay with documented, CPU-compatible OpenBLAS dispatches.

Test baseline x86 kernels only: no unsupported AVX512 dispatch on RM nodes.
Exact original SHA256 remains the gate; no model predictions/scores are used.
See https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/ .
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.recover_geometry_sources import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--frozen-correction', type=Path, required=True)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    if args.worker:
        np.show_runtime()
        subprocess.run([sys.executable, str(args.frozen_correction), str(args.destination)],
                       check=True, timeout=240)
        return
    if not args.expected_sha256:
        parser.error('--expected-sha256 required for parent')
    args.destination.mkdir(parents=True, exist_ok=False)
    original_hash = sha(args.raw)
    original_data = np.asanyarray(nib.load(args.raw).dataobj)
    rows = {}
    # Nehalem/Sandybridge/Haswell require only SSE4/AVX/AVX2 supported by the
    # x86 RM nodes used here. Do not force newer Intel AVX512 kernels.
    for architecture in ('NEHALEM', 'SANDYBRIDGE', 'HASWELL'):
        folder = args.destination / architecture; folder.mkdir()
        copied = folder / args.raw.name
        shutil.copyfile(args.raw, copied)
        environment = dict(os.environ, OPENBLAS_CORETYPE=architecture,
                           OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with (folder / 'dispatch.log').open('x') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                                     '--raw', str(args.raw), '--destination', str(folder),
                                     '--frozen-correction', str(args.frozen_correction)],
                                    env=environment, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=300, check=False)
        row = {'returncode': result.returncode}
        if result.returncode == 0:
            reloaded = nib.load(copied)
            unchanged = np.array_equal(np.asanyarray(reloaded.dataobj), original_data)
            row.update(sha256=sha(copied), source_voxels_unchanged=bool(unchanged),
                       matches_prediction_input=sha(copied) == args.expected_sha256,
                       affine=reloaded.affine.tolist())
            if not unchanged:
                raise RuntimeError('dispatch replay changed source voxels')
        rows[architecture] = row
        print(json.dumps({'dispatch': architecture, **row}, allow_nan=False), flush=True)
    if sha(args.raw) != original_hash:
        raise RuntimeError('raw diagnostic source changed')
    with (args.destination / 'dispatch_replay_report.json').open('x') as output:
        json.dump({'raw_sha256': original_hash, 'expected_sha256': args.expected_sha256,
                   'correction_sha256': sha(args.frozen_correction), 'dispatches': rows},
                  output, indent=2, allow_nan=False)
    print('CPU_DISPATCH_DIAGNOSIS_DONE_NOT_A_COHORT_PASS', flush=True)


if __name__ == '__main__':
    main()

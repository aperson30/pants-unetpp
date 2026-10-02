"""Download only two official fold-4 checkpoint members, without loading them.

Fixed public source/member allowlist, bounded streaming, ZIP CRC and local SHA256.
Never executes checkpoint contents, extracts an archive tree, or overwrites files.
Failed partials are retained; no retries. Fold4 is for the existing transport case,
not a claim of independent clinical validation or final cohort selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import zipfile

from list_remote_zip import RangeReader


SOURCES = [
    ('Dataset103_PANORAMA_baseline_Pancreas_Segmentation',
     'nnUNetTrainer__nnUNetPlans__3d_fullres', 'checkpoint_final.pth', 132172006),
    ('Dataset104_PANORAMA_baseline_PDAC_Detection',
     'nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres',
     'checkpoint_best_panorama.pth', 246418428),
]


def stream_member(archive, member, destination):
    if member.flag_bits & 1 or member.file_size > 300_000_000:
        raise ValueError('Encrypted or oversized checkpoint')
    if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError('Unexpected compression')
    digest, written = hashlib.sha256(), 0
    with archive.open(member) as source, destination.open('xb') as target:
        while chunk := source.read(4*1024**2):
            written += len(chunk)
            if written > member.file_size:
                raise ValueError('Checkpoint exceeds declared size')
            target.write(chunk)
            digest.update(chunk)
        target.flush()
        os.fsync(target.fileno())
    if written != member.file_size:
        raise ValueError('Incomplete checkpoint')
    return dict(bytes=written, sha256=digest.hexdigest(), zip_crc32=member.CRC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    if shutil.disk_usage(args.output).free < 2_000_000_000:
        raise ValueError('Require 2GB free local space')
    watchdog = threading.Timer(660, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    results = []
    try:
        for dataset, trainer, checkpoint, expected_size in SOURCES:
            url = f'https://zenodo.org/records/11160381/files/{dataset}.zip?download=1'
            reader = RangeReader(url, byte_limit=320*1024**2, max_calls=512,
                                 total_timeout=300)
            folder = args.output / dataset / trainer
            (folder / 'fold_4').mkdir(parents=True)
            with zipfile.ZipFile(reader) as archive:
                member = archive.getinfo(f'{dataset}/{trainer}/fold_4/{checkpoint}')
                if member.file_size != expected_size:
                    raise ValueError('Published checkpoint size changed')
                final = folder / 'fold_4' / checkpoint
                partial = final.with_suffix(final.suffix + '.partial')
                provenance = stream_member(archive, member, partial)
                partial.rename(final)
                for name in ['plans.json', 'dataset.json']:
                    metadata = archive.getinfo(f'{dataset}/{trainer}/{name}')
                    if metadata.file_size > 100000:
                        raise ValueError('Oversized metadata')
                    data = archive.read(metadata)
                    json.loads(data)
                    with (folder / name).open('xb') as target:
                        target.write(data)
                results.append(dict(dataset=dataset, fold=4, checkpoint=checkpoint,
                                    fetched_bytes=reader.used, requests=reader.calls,
                                    archive_etag=reader.etag, **provenance))
                print(json.dumps(results[-1]), flush=True)
        with (args.output / 'completion.json').open('x', encoding='utf8') as target:
            json.dump(dict(completed=True, weights_loaded=False, results=results), target, indent=2)
        print('MODEL_TRANSFER_COMPLETE_NO_LOAD', flush=True)
    finally:
        watchdog.cancel()


if __name__ == '__main__':
    main()

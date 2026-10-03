"""CPU-only, bounded checkpoint metadata audit; never loads tensor storage.

Uses the existing restricted reader: globals become inert stubs, not imports.
This proves saved names/configuration, not optimizer updates of a given head.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).parent/'compression_probe'))
from audit_checkpoint_headers import _RestrictedUnpickler


def audit(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root/'checkpoint_manifest.json').read_text())
    rows = {}
    for cell, recorded in manifest['cells'].items():
        path = (root/cell/'checkpoint_final.pth').resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('Checkpoint escaped backup root')
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''):
                sha.update(block)
        if sha.hexdigest() != recorded['checkpoint']['sha256']:
            raise ValueError(f'{cell}: manifest SHA mismatch')
        with zipfile.ZipFile(path) as archive:
            entries = [n for n in archive.namelist() if n.endswith('/data.pkl')]
            if len(entries) != 1:
                raise ValueError('Ambiguous metadata member')
            info = archive.getinfo(entries[0])
            if info.file_size > 8 << 20:
                raise ValueError('Metadata exceeds fixed bound')
            state = _RestrictedUnpickler(io.BytesIO(archive.read(info))).load()
        debug = json.loads((root/cell/'debug.json').read_text())
        plans = json.loads((root/cell/'plans.json').read_text())
        network = state['network_weights']
        heads = [k for k in network if 'seg_layers' in k and k.endswith('weight')]
        rows[cell] = dict(checkpoint_sha256=sha.hexdigest(),
                          manifest_sha_matches=True,
                          trainer=state.get('trainer_name'),
                          current_epoch=state.get('current_epoch'),
                          fold=state.get('init_args', {}).get('fold'),
                          deep_supervision_debug=debug.get('enable_deep_supervision'),
                          skip_shallowest_debug=debug.get('skip_shallowest_deep_supervision_head'),
                          patch=plans['configurations']['3d_fullres']['patch_size'],
                          stage_count=plans['configurations']['3d_fullres']['architecture']['arch_kwargs']['n_stages'],
                          saved_segmentation_head_keys=heads,
                          header_bytes=info.file_size,
                          scope='Metadata/config only; keys do not prove training or head accuracy')
    return rows


if __name__ == '__main__':
    print(json.dumps(audit(sys.argv[1]), indent=2))

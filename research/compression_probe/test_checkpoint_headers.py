"""Offline tests for audit_checkpoint_headers.py (no network)."""
import collections
import io
import pickle

from audit_checkpoint_headers import _Inert, _RestrictedUnpickler


class _Exploit:
    def __reduce__(self):
        import os
        return (os.system, ('echo SHOULD_NOT_RUN',))


def test_globals_are_never_imported_or_called():
    payload = pickle.dumps({'trap': _Exploit(), 'fold': 3})
    state = _RestrictedUnpickler(io.BytesIO(payload)).load()
    assert state['fold'] == 3
    assert isinstance(state['trap'], _Inert)


def test_ordered_dict_and_plain_values_survive():
    original = {'init_args': {'fold': 1, 'configuration': '3d_fullres',
                              'dataset_json': {'numTraining': 2238}},
                'network_weights': collections.OrderedDict(a=1), 'current_epoch': 950}
    state = _RestrictedUnpickler(io.BytesIO(pickle.dumps(original))).load()
    assert state['init_args']['fold'] == 1
    assert state['init_args']['dataset_json']['numTraining'] == 2238
    assert isinstance(state['network_weights'], collections.OrderedDict)
    assert state['current_epoch'] == 950


if __name__ == '__main__':
    test_globals_are_never_imported_or_called()
    test_ordered_dict_and_plain_values_survive()
    print('2 tests passed')

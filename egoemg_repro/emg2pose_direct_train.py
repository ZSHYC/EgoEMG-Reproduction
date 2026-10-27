"""Exact native independent-role fit over verified original ordinary files.

Full training requires the actual complete source and this entry's frozen CUDA
complete-epoch qualification. Engineering fits never publish scientific results.
Each CLI and its data workers require an external process owner.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import FunctionType

from . import emg2pose_memory_train as _kernel
from . import emg2pose_direct_runtime as runtime
from . import emg2pose_direct_recovery as recovery
from .emg2pose_direct_source import VerifiedDirectSource, open_source

storage = _kernel.storage
FORMAT = 'emg2pose_native_direct_fit_v1'
_ROOT = Path(__file__).parent
_FILES = ('emg2pose_direct_train.py', 'emg2pose_memory_train.py',
    'emg2pose_direct_recovery.py', 'emg2pose_memory_recovery.py',
    'classic_emg_train.py', 'emg2pose_check.py')
_IMPORTED = {name: storage._sha256(_ROOT/name) for name in _FILES}


def _sources():
    actual = {name: storage._sha256(_ROOT/name) for name in _FILES}
    if actual != _IMPORTED or actual['emg2pose_memory_train.py'] != _kernel._IMPORTED_SHA:
        raise ValueError('imported direct fit implementation changed on disk')
    return actual


_namespace = dict(_kernel.__dict__, __name__=__name__, __file__=__file__,
    FORMAT=FORMAT, runtime=runtime, recovery=recovery,
    VerifiedSource=VerifiedDirectSource, open_source=open_source, _sources=_sources)
_namespace.pop('main', None)


def _bind(function):
    bound = FunctionType(function.__code__, _namespace, name=function.__name__,
        argdefs=function.__defaults__, closure=function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = dict(function.__annotations__)
    return bound


for _name, _value in vars(_kernel).items():
    if isinstance(_value, FunctionType) and _value.__module__ == _kernel.__name__ and _name not in ('main', '_sources'):
        _namespace[_name] = globals()[_name] = _bind(_value)

# Inheriting the old witness would keep its old globals. Every method uses
# exactly its existing code object with the same private bindings as fit_source.
FitWitness = type('FitWitness', (_kernel.Callback,),
    dict(__module__=__name__, __doc__=_kernel.FitWitness.__doc__,
        **{name: _bind(value) for name, value in vars(_kernel.FitWitness).items()
            if isinstance(value, FunctionType)}))
_namespace['FitWitness'] = FitWitness


def main(argv=None):
    if 'EGOEMG_STAGE_FD_ARCHIVE' in os.environ:
        raise ValueError('direct fit does not accept an inherited archive descriptor')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'source-root', 'metadata-csv', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-source-completion-sha256', required=True)
    parser.add_argument('--size', choices=tuple(runtime.native.SIZES), required=True)
    parser.add_argument('--scope', choices=('full', 'cpu_engineering', 'cuda_engineering'), default='full')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--expected-last-sha256')
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--expected-qualification-sha256')
    args = parser.parse_args(argv)
    with open_source(args.source_root, expected_completion_sha256=args.expected_source_completion_sha256,
            upstream=args.upstream, metadata_csv=args.metadata_csv) as source:
        return fit_source(args.upstream, source, args.metadata_csv, args.size, args.output,
            scope=args.scope, resume=args.resume, expected_last_sha256=args.expected_last_sha256,
            stop_after_epoch=args.stop_after_epoch, qualification=args.qualification,
            expected_qualification_sha256=args.expected_qualification_sha256)


if __name__ == '__main__':
    main()

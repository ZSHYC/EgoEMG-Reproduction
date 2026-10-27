"""Bounded native state recovery over verified original ordinary files.

The exact qualified recovery algorithms run in private globals bound to the
direct source/runtime. This is an engineering fixture, not a full experiment.
Every CLI process and its data workers must have an external owner.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import FunctionType

from . import emg2pose_memory_recovery as _kernel
from . import emg2pose_direct_runtime as runtime
from .emg2pose_direct_source import VerifiedDirectSource, open_source

checks = _kernel.checks
storage = _kernel.storage
FORMAT = 'emg2pose_native_direct_recovery_v1'
_ROOT = Path(__file__).parent
_FILES = ('emg2pose_direct_recovery.py', 'emg2pose_memory_recovery.py', 'classic_emg_train.py')
_IMPORTED = {name: storage._sha256(_ROOT/name) for name in _FILES}


def _producer():
    actual = {name: storage._sha256(_ROOT/name) for name in _FILES}
    if actual != _IMPORTED or actual['emg2pose_memory_recovery.py'] != _kernel._IMPORTED_SHA:
        raise ValueError('imported direct recovery implementation changed on disk')
    return actual


# Shared code objects retain native checkpoint/loop/RNG observation semantics.
# Only these private bindings differ; no old module global is mutated.
_namespace = dict(_kernel.__dict__, __name__=__name__, __file__=__file__,
    FORMAT=FORMAT, runtime=runtime, VerifiedSource=VerifiedDirectSource,
    open_source=open_source, _producer=_producer)
_namespace.pop('main', None)
for _name, _value in vars(_kernel).items():
    if isinstance(_value, FunctionType) and _value.__module__ == _kernel.__name__ and _name not in ('main', '_producer'):
        _bound = FunctionType(_value.__code__, _namespace, name=_value.__name__,
            argdefs=_value.__defaults__, closure=_value.__closure__)
        _bound.__kwdefaults__ = _value.__kwdefaults__
        _bound.__annotations__ = dict(_value.__annotations__)
        _namespace[_name] = _bound
        globals()[_name] = _bound


def main(argv=None):
    """Open ordinary source files; inherited archive rights are never consumed."""
    if 'EGOEMG_STAGE_FD_ARCHIVE' in os.environ:
        raise ValueError('direct recovery does not accept an inherited archive descriptor')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'source-root', 'metadata-csv', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-source-completion-sha256', required=True)
    parser.add_argument('--size', choices=tuple(runtime.native.SIZES), required=True)
    parser.add_argument('--phase', choices=('prefix', 'restore'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), required=True)
    parser.add_argument('--expected-prefix-sha256')
    args = parser.parse_args(argv)
    with open_source(args.source_root, expected_completion_sha256=args.expected_source_completion_sha256,
            upstream=args.upstream, metadata_csv=args.metadata_csv) as source:
        return run_recovery_phase(args.upstream, source, args.metadata_csv, args.size, args.output,
            phase=args.phase, device=args.device, expected_prefix_sha256=args.expected_prefix_sha256)


if __name__ == '__main__':
    main()

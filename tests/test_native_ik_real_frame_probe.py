"""Fail before data/model work for invalid real-frame engineering requests."""
from argparse import Namespace

import pytest

from scripts.paper import probe_native_mano_ik_real_frames as probe


def arguments(tmp_path, **updates):
    value = dict(action='produce', data=tmp_path/'data', geometry=tmp_path/'geometry',
                 upstream=tmp_path/'upstream', manotorch=tmp_path/'manotorch',
                 chumpy=tmp_path/'chumpy', mano_assets=tmp_path/'assets',
                 center_index=331075, original=None, output=tmp_path/'output')
    return Namespace(**dict(value, **updates))


def test_existing_output_rejected_before_source_admission(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.output.mkdir()
    monkeypatch.setattr(probe, 'admit_native_ik_inputs', lambda *a, **k: pytest.fail('admission must not run'))
    with pytest.raises(ValueError, match='existing|exists'):
        probe.run(args)


@pytest.mark.parametrize('index', [-1, 0, True, 1.5])
def test_invalid_engineering_index_rejected_before_admission(tmp_path, monkeypatch, index):
    monkeypatch.setattr(probe, 'admit_native_ik_inputs', lambda *a, **k: pytest.fail('admission must not run'))
    with pytest.raises(ValueError, match='index'):
        probe.run(arguments(tmp_path, center_index=index))


def test_verify_requires_original_before_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'admit_native_ik_inputs', lambda *a, **k: pytest.fail('admission must not run'))
    with pytest.raises(ValueError, match='original'):
        probe.run(arguments(tmp_path, action='verify'))


def test_unknown_action_cannot_start_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'admit_native_ik_inputs', lambda *a, **k: pytest.fail('admission must not run'))
    with pytest.raises(ValueError, match='action'):
        probe.run(arguments(tmp_path, action='full'))

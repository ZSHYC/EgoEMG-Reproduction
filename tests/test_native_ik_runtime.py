"""Source boundaries with genuine existing assets; no fake full IK producer."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pytest

from egoemg_repro.native_ik_runtime import IKSource, admit_ik_sources
from egoemg_repro import native_ik_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_sources(tmp_path):
    names = ('EGOEMG_IK_MANOTORCH', 'EGOEMG_IK_CHUMPY', 'EGOEMG_IK_ASSETS')
    if not all(os.environ.get(n) for n in names):
        pytest.skip('existing admitted source and private MANO assets are required')
    manotorch, chumpy, assets = [Path(os.environ[n]) for n in names]
    manifest = json.loads((ROOT / 'assets/native_ik_dependency_sources.json').read_text())
    destinations = [tmp_path / n for n in ('manotorch', 'chumpy', 'assets')]
    for source, destination, records in (
        (manotorch, destinations[0], manifest['manotorch_files_sha256']),
        (chumpy, destinations[1], manifest['chumpy_files_sha256']),
        (assets, destinations[2], {'models/MANO_RIGHT.pkl': manifest['mano_right_sha256']}),
    ):
        for name, expected in records.items():
            original = source / name
            assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    return [ROOT / 'upstream', *destinations]


def test_genuine_source_admission_is_torch_free_and_rng_preserving(native_sources):
    py, npr = random.getstate(), np.random.get_state()
    assert 'torch' not in sys.modules
    source = admit_ik_sources(*native_sources)
    assert 'torch' not in sys.modules
    identity = source.identity
    assert identity['manotorch_revision'] == 'a2a70c591f91551078b7bb2af9b5d9f275b626e0'
    assert identity['actual_dependency_source_files'] == {'manotorch': 25, 'chumpy': 20}
    assert identity['model_constructed'] is False
    assert identity['scientific_eligible'] is False
    assert identity['author_external_checkout_identity_proven'] is False
    assert identity['native_policy']['max_outer_iterations'] == 200
    assert identity['native_policy']['lbfgs_max_iter'] == 50
    assert identity['native_policy']['beta'] == 'internal zero'
    assert identity['native_policy']['right_mano_for_both_hands'] is True
    assert random.getstate() == py
    after = np.random.get_state()
    assert npr[0] == after[0] and np.array_equal(npr[1], after[1]) and npr[2:] == after[2:]
    copy = source.identity
    copy['native_policy']['beta'] = 'participant'
    assert source.identity == identity


@pytest.mark.parametrize('part,relative', [(1, 'manotorch/manolayer.py'),
                                         (2, 'chumpy/ch.py'), (3, 'models/MANO_RIGHT.pkl')])
def test_wrong_dependency_or_model_bytes_rejected_before_torch(native_sources, part, relative):
    p = native_sources[part] / relative
    p.write_bytes(p.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='source|asset|MANO'):
        admit_ik_sources(*native_sources)
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('part', [1, 2])
def test_extra_python_provider_rejected(native_sources, part):
    p = native_sources[part] / ('manotorch' if part == 1 else 'chumpy') / 'new_provider.py'
    p.write_text('raise RuntimeError("must not import")\n')
    with pytest.raises(ValueError, match='source|provider'):
        admit_ik_sources(*native_sources)


def test_symlink_private_asset_rejected_before_import(native_sources):
    p = native_sources[3] / 'models/MANO_RIGHT.pkl'
    target = p.with_suffix('.original'); p.rename(target); p.symlink_to(target)
    with pytest.raises(ValueError, match='symlink'):
        admit_ik_sources(*native_sources)


def test_wrong_upstream_tree_rejected_before_import(native_sources, tmp_path):
    fake = tmp_path / 'not-upstream'; fake.mkdir()
    with pytest.raises(ValueError, match='original|source|upstream'):
        admit_ik_sources(fake, *native_sources[1:])


@pytest.mark.parametrize('part,relative', [(1, 'manotorch/manolayer.py'),
                                         (2, 'chumpy/ch.py'), (3, 'models/MANO_RIGHT.pkl')])
def test_admitted_generations_remain_bound(native_sources, part, relative):
    source = admit_ik_sources(*native_sources)
    p = native_sources[part] / relative
    p.write_bytes(p.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed'):
        source.check()


def test_source_mutation_during_hash_admission_rejected(native_sources, monkeypatch):
    original = runtime.file_sha256
    target = native_sources[1] / 'manotorch/manolayer.py'
    changed = False

    def digest(path):
        nonlocal changed
        result = original(path)
        if Path(path) == target and not changed:
            target.write_bytes(target.read_bytes() + b'\n')
            changed = True
        return result

    monkeypatch.setattr(runtime, 'file_sha256', digest)
    with pytest.raises(ValueError, match='changed'):
        admit_ik_sources(*native_sources)


def test_new_hardlink_invalidates_private_geometry_asset(native_sources):
    source = admit_ik_sources(*native_sources)
    p = native_sources[3] / 'models/MANO_RIGHT.pkl'
    os.link(p, p.with_suffix('.linked'))
    with pytest.raises(ValueError, match='changed'):
        source.check()


def test_unregistered_source_cannot_build_geometry():
    with pytest.raises(ValueError, match='registered'):
        IKSource().build()
    assert 'torch' not in sys.modules


def test_unknown_device_rejected_before_model(native_sources):
    source = admit_ik_sources(*native_sources)
    with pytest.raises(ValueError, match='device'):
        source.build(device='cuda:3')
    assert 'torch' not in sys.modules

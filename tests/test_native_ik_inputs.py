"""Real-file input boundaries; no native model or full-source producer mock."""
import ast
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from egoemg_repro.native_ik_inputs import NativeIKInputs, admit_native_ik_inputs
from egoemg_repro import native_ik_inputs as native_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sources(tmp_path):
    data, geometry, upstream = [tmp_path / n for n in ('data', 'geometry', 'upstream')]
    for p in (data, geometry, upstream):
        p.mkdir()
    for rel in ('scripts/ik/batch_ik_mesh.py', 'egoemg/UmeTrack/dataset/generic_hand_model.json'):
        target = upstream / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'upstream' / rel, target)
    n = 53
    fields = {}
    arrays = {}
    for i, hand in enumerate(('left', 'right')):
        a = (np.arange(n * 48, dtype=np.float32).reshape(n, 48) + i) / 100
        a[[0, 11, n - 1]] = 0
        a.tofile(geometry / f'generated_mano_{hand}_pose.dat')
        arrays[hand] = a
        fields[f'generated_mano_{hand}_pose'] = dict(
            filename=f'generated_mano_{hand}_pose.dat', dtype='float32', shape=[n, 48])
    (data / 'manifest.json').write_text(json.dumps(dict(total_rows=n, fields=fields)))
    np.savez(data / 'metadata.npz', episode_start=np.array([0]), episode_end=np.array([n]))
    return SimpleNamespace(data=data, geometry=geometry, upstream=upstream, arrays=arrays, n=n)


def admit(s, **kwargs):
    return admit_native_ik_inputs(s.data, s.geometry, s.upstream, engineering=True, **kwargs)


def original_indices(s, start, end):
    """Compile the actual pinned original initializer, without model imports."""
    tree = ast.parse((s.upstream / 'scripts/ik/batch_ik_mesh.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AnchorPoseDataset')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    module = ast.Module(body=[method], type_ignores=[])
    scope = {}
    exec(compile(module, '<actual-original-AnchorPoseDataset-initializer>', 'exec'), scope)
    obj = SimpleNamespace()
    scope['__init__'](obj, s.arrays['left'], start, end, 1, None)
    return obj.indices


def snapshots(s):
    paths = [*s.data.iterdir(), *s.geometry.iterdir(),
             *s.upstream.rglob('*.py'), *s.upstream.rglob('*.json')]
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths}


def test_real_io_complete_membership_matches_original_initializer(sources):
    s = sources
    inputs = admit(s, batch_size=3, chunk_rows=7)
    assert inputs.shards == ((0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 53))
    for hand in ('right', 'left'):
        complete = []
        for shard, (start, end) in enumerate(inputs.shards):
            batches = list(inputs.batches(hand, shard))
            actual = np.concatenate([idx for idx, _, _ in batches]).tolist()
            assert actual == original_indices(s, start, end)
            assert [len(row[0]) for row in batches] == [3] * ((end - start) // 3) + ([((end - start) % 3)] if (end - start) % 3 else [])
            for idx, raw, canonical in batches:
                assert idx.dtype == np.int64
                assert raw.dtype == canonical.dtype == np.float32
                np.testing.assert_array_equal(raw, s.arrays[hand][idx])
                np.testing.assert_array_equal(canonical[:, 3:], raw[:, 3:])
                assert not canonical[:, :3].any()
                assert not np.shares_memory(raw, canonical)
            complete += actual
        assert complete == list(range(s.n))
    identity = inputs.identity
    assert identity['hands_order'] == ['right', 'left']
    assert identity['samples_per_hand'] == s.n
    assert identity['full_available_release'] is False
    assert identity['scientific_eligible'] is False
    assert identity['full_frame_labels_regenerated'] is False
    assert identity['scan']['left']['zero_rows'] == identity['scan']['right']['zero_rows'] == 3
    assert identity['scan']['left']['elements'] == s.n * 48
    assert identity['files_sha256']['geometry/generated_mano_left_pose.dat'] == hashlib.sha256(s.arrays['left'].tobytes()).hexdigest()


def test_default_batch_and_partial_shard_preserved(sources):
    inputs = admit(sources)
    assert inputs.identity['batch_size'] == 10096
    assert inputs.identity['logical_worker_partitions'] == 6
    assert [len(list(inputs.batches('right', k))) for k in range(6)] == [1] * 6
    assert inputs.identity['batches_per_hand'] == [1] * 6


def test_source_bytes_rng_and_raw_canonical_copies_remain_independent(sources):
    before = snapshots(sources)
    py_rng, np_rng = random.getstate(), np.random.get_state()
    inputs = admit(sources, batch_size=4)
    batches = inputs.batches('left', 0)
    _, raw, canonical = next(batches)
    raw[:] = 99
    canonical[:] = -99
    batches.close()
    _, raw_again, canonical_again = next(inputs.batches('left', 0))
    np.testing.assert_array_equal(raw_again, sources.arrays['left'][:4])
    assert not canonical_again[:, :3].any()
    assert snapshots(sources) == before
    assert random.getstate() == py_rng
    after = np.random.get_state()
    assert np_rng[0] == after[0] and np.array_equal(np_rng[1], after[1]) and np_rng[2:] == after[2:]


def test_identity_copy_and_registered_object_required(sources):
    inputs = admit(sources)
    original = inputs.identity
    copy = inputs.identity
    copy['full_available_release'] = True
    copy['shards'][0][1] = 999
    assert inputs.identity == original
    with pytest.raises((AttributeError, FrozenInstanceError)):
        inputs.shards = ((0, 999),)
    with pytest.raises(ValueError, match='registered'):
        NativeIKInputs().check()


@pytest.mark.parametrize('key,value', [('total_rows', True), ('total_rows', 5), ('total_rows', 53.0)])
def test_wrong_total_rows_rejected(sources, key, value):
    p = sources.data / 'manifest.json'
    d = json.loads(p.read_text()); d[key] = value; p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        admit(sources)


@pytest.mark.parametrize('field,value', [('dtype', 'float64'), ('shape', [53, 47]),
                                       ('filename', '../escape.dat'), ('filename', 'other.dat')])
def test_wrong_pose_schema_rejected(sources, field, value):
    p = sources.data / 'manifest.json'
    d = json.loads(p.read_text()); d['fields']['generated_mano_left_pose'][field] = value
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        admit(sources)


@pytest.mark.parametrize('location', ['data', 'geometry'])
def test_optional_global_valid_never_silently_substitutes_selection(sources, location):
    np.save(getattr(sources, location) / 'valid.npy', np.ones(sources.n, dtype=bool))
    with pytest.raises(ValueError, match='valid.npy'):
        admit(sources)


@pytest.mark.parametrize('location', ['data', 'geometry'])
def test_optional_global_valid_appearance_invalidates_admitted_use(sources, location):
    inputs = admit(sources)
    np.save(getattr(sources, location) / 'valid.npy', np.ones(sources.n, dtype=bool))
    with pytest.raises(ValueError, match='valid.npy'):
        inputs.check()


@pytest.mark.parametrize('column,value', [(0, np.nan), (47, np.nan), (3, np.inf), (3, -np.inf)])
def test_nonfinite_raw_pose_rejected_including_last_row(sources, column, value):
    p = sources.geometry / 'generated_mano_left_pose.dat'
    a = np.memmap(p, np.float32, mode='r+', shape=(sources.n, 48))
    a[-1, column] = value; a.flush(); del a
    with pytest.raises(ValueError, match='finite'):
        admit(sources, chunk_rows=7)


@pytest.mark.parametrize('kind', ['truncated', 'extended', 'symlink'])
def test_wrong_physical_pose_file_rejected(sources, kind):
    p = sources.geometry / 'generated_mano_left_pose.dat'
    b = p.read_bytes()
    if kind == 'symlink':
        q = p.with_suffix('.original'); p.rename(q); p.symlink_to(q)
    else:
        p.write_bytes(b[:-4] if kind == 'truncated' else b + b'0000')
    with pytest.raises(ValueError):
        admit(sources)


@pytest.mark.parametrize('kind', ['same_bytes_replaced', 'hardlink', 'source_edit', 'data_edit'])
def test_source_generation_changes_rejected(sources, kind):
    inputs = admit(sources)
    p = sources.geometry / 'generated_mano_left_pose.dat'
    if kind == 'same_bytes_replaced':
        q = p.with_suffix('.copy'); q.write_bytes(p.read_bytes()); os.replace(q, p)
    elif kind == 'hardlink':
        os.link(p, p.with_suffix('.linked'))
    elif kind == 'source_edit':
        p = sources.upstream / 'scripts/ik/batch_ik_mesh.py'
        p.write_text(p.read_text() + '\n')
    else:
        with p.open('r+b') as f:
            f.write(np.array([1], np.float32).tobytes())
    with pytest.raises(ValueError, match='changed'):
        inputs.check()


def test_source_change_during_generator_consumption_rejected(sources):
    inputs = admit(sources, batch_size=2)
    iterator = inputs.batches('left', 0)
    next(iterator)
    p = sources.data / 'manifest.json'; p.write_text(p.read_text() + ' ')
    with pytest.raises(ValueError, match='changed'):
        next(iterator)


def test_original_source_change_between_initial_hash_and_guard_rejected(sources, monkeypatch):
    original = native_inputs.file_sha256
    target = sources.upstream / 'scripts/ik/batch_ik_mesh.py'
    changed = False

    def digest(path):
        nonlocal changed
        value = original(path)
        if Path(path) == target and not changed:
            target.write_text(target.read_text() + '\n')
            changed = True
        return value

    monkeypatch.setattr(native_inputs, 'file_sha256', digest)
    with pytest.raises(ValueError, match='source|changed'):
        admit(sources)


def test_manifest_change_between_two_guard_snapshots_rejected(sources, monkeypatch):
    original = native_inputs._Guard
    calls = 0

    def guard(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            p = sources.data / 'manifest.json'
            value = json.loads(p.read_text())
            value['total_rows'] += 1
            p.write_text(json.dumps(value))
        return original(*args, **kwargs)

    monkeypatch.setattr(native_inputs, '_Guard', guard)
    with pytest.raises(ValueError, match='changed'):
        admit(sources)


@pytest.mark.parametrize('batch_size', [True, 0, -1, 1.5])
def test_invalid_batch_size(sources, batch_size):
    with pytest.raises(ValueError):
        admit(sources, batch_size=batch_size)


@pytest.mark.parametrize('hand,shard', [('both', 0), ('left', -1), ('right', 6), ('left', True)])
def test_invalid_batch_selection(sources, hand, shard):
    inputs = admit(sources)
    with pytest.raises(ValueError):
        list(inputs.batches(hand, shard))


def test_small_data_cannot_claim_full_release(sources):
    with pytest.raises(ValueError, match='full|registered'):
        admit_native_ik_inputs(sources.data, sources.geometry, ROOT / 'upstream')


def test_forged_full_reference_rejected_before_data_admission(sources):
    ref = sources.data / 'fake-reference.json'
    ref.write_text(json.dumps(dict(total_rows=sources.n, full_numeric_scan_complete=True)))
    with pytest.raises(ValueError, match='reference'):
        admit_native_ik_inputs(sources.data, sources.geometry, ROOT / 'upstream', reference=ref)


def test_input_import_is_torch_free():
    code = 'import sys; import egoemg_repro.native_ik_inputs; assert "torch" not in sys.modules'
    r = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

"""Direct EMG2Pose conversion preserves source bits and official identities."""
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1] / 'upstream'))


def module():
    return importlib.import_module('egoemg_repro.emg2pose_data')


@pytest.fixture(autouse=True)
def sufficient_fixture_storage(monkeypatch):
    # Tiny data fixtures should not depend on the runner's physical free space.
    # Capacity cases override this patch inside their test, after fixture setup.
    monkeypatch.setattr(module().shutil, 'disk_usage', lambda path: SimpleNamespace(
        total=128 * 1024**3, used=64 * 1024**3, free=64 * 1024**3))


def fixture(root, *, explicit_ik=False):
    source = root / 'private-source'
    source.mkdir()
    paths, rows, arrays = [], [], {}
    for filename, split, generalization, side in [
        ('a-val', 'val', 'user', 'right'),
        ('b-train', 'train', 'none', 'left'),
        ('c-stage', 'test', 'stage', 'left'),
        ('d-both', 'test', 'user_stage', 'right'),
    ]:
        values = np.zeros(8, dtype=[('time', '<f8'), ('joint_angles', '<f4', (20,)),
                                   ('emg', '<f4', (16,))])
        values['time'] = np.arange(8, dtype=np.float64) / 2000 + 12345678
        values['emg'] = np.arange(128, dtype=np.float32).reshape(8, 16) - 64
        values['emg'][0, 0] = -0.0
        values['joint_angles'] = 1
        values['joint_angles'][0] = 0
        values['joint_angles'][1] = 1e-9
        values['joint_angles'][3, 0] = np.nan
        values['joint_angles'][4, 1] = np.inf
        values['joint_angles'][5] = -0.0
        path = source / (filename + '.hdf5')
        row = dict(session='repeated-session', filename=filename, user='private-user-中文',
                   stage='private-stage', side=side, moving_hand='both', split=split,
                   generalization=generalization, held_out_user=generalization in ('user', 'user_stage'),
                   held_out_stage=generalization in ('stage', 'user_stage'), start=12345678,
                   end=12345678 + 8 / 2000)
        with h5py.File(path, 'w') as f:
            g = f.create_group('emg2pose')
            g.create_dataset('timeseries', data=values)
            if explicit_ik:
                g.create_dataset('ik_failure_mask', data=np.array([False, True, False, False, False, True, False, False]))
            for key, value in row.items():
                g.attrs[key] = value
            g.attrs['sample_rate'] = 2000.0
            g.attrs['num_channels'] = 16
            g.attrs['typed_numeric'] = np.array([1, 2], dtype='<i2')
            extra = g.create_dataset('preserved_extra', data=np.array([7, 9], dtype=np.int16))
            extra.attrs['units'] = 'original-extra'
        rows.append(row)
        paths.append(path)
        arrays[filename] = values
    metadata = root / 'metadata.csv'
    with metadata.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return paths, metadata, arrays


def inspect(paths, metadata, **kwargs):
    return module().inspect_corpus(paths, metadata, cohort='engineering', upstream=UPSTREAM, **kwargs)


def receipt(root, inventory):
    path = root / 'source-receipt.json'
    path.write_text(json.dumps(inventory, allow_nan=False), encoding='utf-8')
    return path


def test_repeated_session_is_not_deduplicated_and_false_strings_are_false(tmp_path):
    paths, metadata, _ = fixture(tmp_path)
    inventory = inspect(paths, metadata)
    assert len(inventory['sessions']) == 4
    assert [s['filename'] for s in inventory['sessions']] == ['b-train', 'a-val', 'c-stage', 'd-both']
    assert inventory['sessions'][0]['held_out_user'] is False
    assert inventory['sessions'][0]['held_out_stage'] is False
    assert inventory['sessions'][0]['session_id'] == 'repeated-session'
    assert inventory['full_corpus_complete'] is False
    assert inventory['inspected_file_set_complete'] is False
    assert inventory['total_rows'] == 32
    assert inventory['counts']['split'] == {'train': 1, 'val': 1, 'test': 2}
    assert '/emg2pose/preserved_extra' in inventory['sessions'][0]['hdf5_objects']
    assert inventory['sessions'][0]['hdf5_objects']['/emg2pose']['attrs']['typed_numeric']['dtype'] == 'int16'


@pytest.mark.parametrize('change', ['attrs', 'missing_field', 'wrong_emg', 'wrong_angles', 'wrong_time', 'truncated', 'external_link', 'soft_link', 'external_storage'])
def test_rejects_conflicting_or_unsafe_hdf5(tmp_path, change):
    paths, metadata, arrays = fixture(tmp_path)
    path = paths[0]
    if change == 'truncated':
        path.write_bytes(path.read_bytes()[:-128])
    else:
        with h5py.File(path, 'r+') as f:
            g = f['emg2pose']
            if change == 'attrs':
                g.attrs['split'] = 'train'
            elif change in ('missing_field', 'wrong_emg', 'wrong_angles', 'wrong_time'):
                dtype = [('time', '<f4' if change == 'wrong_time' else '<f8'),
                         ('joint_angles', '<f4', (22 if change == 'wrong_angles' else 20,))]
                if change != 'missing_field':
                    dtype += [('emg', '<f4', (8 if change == 'wrong_emg' else 16,))]
                del g['timeseries']
                g.create_dataset('timeseries', data=np.zeros(8, dtype=dtype))
            elif change == 'external_link':
                g['outside'] = h5py.ExternalLink(str(paths[1]), '/emg2pose')
            elif change == 'soft_link':
                g['alias'] = h5py.SoftLink('/emg2pose/timeseries')
            else:
                g.create_dataset('external', (8,), dtype='i4', external=[('outside.dat', 0, 32)])
    with pytest.raises(ValueError):
        inspect(paths, metadata)


@pytest.mark.parametrize('kind', ['file', 'parent'])
def test_rejects_symlink_sources_before_following_them(tmp_path, kind):
    paths, metadata, _ = fixture(tmp_path)
    link = tmp_path / 'linked'
    link.symlink_to(paths[0] if kind == 'file' else paths[0].parent)
    paths[0] = link if kind == 'file' else link / paths[0].name
    with pytest.raises(ValueError, match='symlink|unsafe'):
        inspect(paths, metadata)


def test_rejects_duplicate_filename_and_parent_traversal(tmp_path):
    paths, metadata, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='duplicate'):
        inspect(paths + [paths[0]], metadata)
    bad = paths[0].parent / '..' / paths[0].parent.name / paths[0].name
    with pytest.raises(ValueError, match='unsafe'):
        inspect([bad] + paths[1:], metadata)


@pytest.mark.parametrize('cohort', ['full', 'mini'])
def test_arbitrary_fixture_csv_cannot_claim_official_complete_cohort(tmp_path, cohort):
    paths, metadata, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='official|metadata|acquisition'):
        module().inspect_corpus(paths, metadata, cohort=cohort, upstream=UPSTREAM)


def test_unknown_or_dirty_upstream_rejected(tmp_path):
    paths, metadata, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='upstream'):
        module().inspect_corpus(paths, metadata, cohort='engineering', upstream=tmp_path)


def test_fractional_channel_metadata_cannot_be_truncated_to_sixteen(tmp_path):
    paths, metadata, _ = fixture(tmp_path)
    with h5py.File(paths[0], 'r+') as f:
        f['emg2pose'].attrs['num_channels'] = 16.5
    with pytest.raises(ValueError, match='channel'):
        inspect(paths, metadata)


def test_optional_empty_utf8_attributes_are_preserved(tmp_path):
    paths, metadata, _ = fixture(tmp_path)
    with h5py.File(paths[0], 'r+') as f:
        f['emg2pose'].attrs['optional_empty'] = ''
    inventory = inspect(paths, metadata)
    source = next(s for s in inventory['sessions'] if s['filename'] == 'a-val')
    assert source['hdf5_objects']['/emg2pose']['attrs']['optional_empty']['text_values'] == ['']


def test_csv_changed_while_inspecting_cannot_get_a_stale_identity(tmp_path, monkeypatch):
    paths, metadata, _ = fixture(tmp_path)
    original = module()._hdf5_objects
    def racing(file):
        result = original(file)
        metadata.write_text(metadata.read_text() + '\n')
        return result
    monkeypatch.setattr(module(), '_hdf5_objects', racing)
    with pytest.raises(ValueError, match='metadata.*changed|changed.*metadata'):
        inspect(paths, metadata)


@pytest.mark.parametrize('value', ['False-ish', '', '0'])
def test_ambiguous_boolean_attribute_is_not_truthy(tmp_path, value):
    paths, metadata, _ = fixture(tmp_path)
    with h5py.File(paths[0], 'r+') as f:
        f['emg2pose'].attrs['held_out_stage'] = value
    with pytest.raises(ValueError, match='bool'):
        inspect(paths, metadata)


@pytest.mark.parametrize('explicit_ik', [False, True])
def test_all_field_bits_masks_blocks_and_unicode_sidecar_match_source(tmp_path, explicit_ik):
    paths, metadata, arrays = fixture(tmp_path, explicit_ik=explicit_ik)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    inventory = inspect(paths, metadata)
    result = module().convert_corpus(receipt(tmp_path, inventory), tmp_path / 'converted', chunk_rows=3, upstream=UPSTREAM)
    out = tmp_path / 'converted'
    assert result['complete'] is True
    assert result['full_corpus_complete'] is False
    assert result['engineering_only'] is True
    assert not (tmp_path / 'converted.incomplete').exists()
    assert (out / 'completion.json').exists()
    manifest = json.loads((out / 'manifest.json').read_text())
    order = ['b-train', 'a-val', 'c-stage', 'd-both']
    for field in ('emg', 'joint_angles', 'time'):
        spec = manifest['fields'][field]
        actual = np.memmap(out / spec['filename'], dtype=spec['dtype'], shape=tuple(spec['shape']), mode='r')
        expected = np.concatenate([arrays[name][field] for name in order])
        assert actual.tobytes() == expected.tobytes()
        assert spec['sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
    valid = np.fromfile(out / 'valid_mask.dat', dtype=bool)
    expected_mask = [True, False, True, False, False, False, True, True] if explicit_ik else [False, False, True, False, False, False, True, True]
    assert valid.tolist() == expected_mask * 4
    with np.load(out / 'metadata.npz', allow_pickle=False) as md:
        assert all(md[k].dtype.kind != 'O' for k in md.files)
        assert md['session_start_idx'].tolist() == [0, 8, 16, 24]
        assert md['session_end_idx'].tolist() == [8, 16, 24, 32]
        assert md['session_session_id'].tolist() == [b'repeated-session'] * 4
        assert md['session_generalization'].tolist() == [b'none', b'user', b'stage', b'user_stage']
        assert md['users_user'][0].decode() == 'private-user-中文'
        expected_blocks = [(0, 1), (2, 3), (6, 8)] if explicit_ik else [(2, 3), (6, 8)]
        assert list(zip(md['blocks_start'], md['blocks_end'])) == [(a + i * 8, b + i * 8) for i in range(4) for a, b in expected_blocks]
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == before


@pytest.mark.parametrize('field', ['emg', 'time'])
@pytest.mark.parametrize('kind', ['nan_payload', 'positive_infinity', 'negative_infinity'])
def test_nonfinite_raw_emg_and_time_preserve_bits_without_changing_label_validity(tmp_path, field, kind):
    paths, metadata, arrays = fixture(tmp_path)
    if kind == 'nan_payload':
        value = (np.array([0x7fc12345], dtype='<u4').view('<f4')[0] if field == 'emg'
                 else np.array([0x7ff8000000004321], dtype='<u8').view('<f8')[0])
    else:
        value = np.inf if kind == 'positive_infinity' else -np.inf
    # Row2 has finite nonzero joint angles: raw EMG/time cannot change its
    # official angle/IK validity. The explicit NaN payload must survive too.
    with h5py.File(paths[0], 'r+') as file:
        ts = file['emg2pose/timeseries']
        row = ts[2]
        if field == 'emg':
            row['emg'][3] = value
            arrays['a-val']['emg'][2, 3] = value
        else:
            row['time'] = value
            arrays['a-val']['time'][2] = value
        ts[2] = row
    source_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = module().convert_corpus(receipt(tmp_path, inspect(paths, metadata)),
                                     tmp_path / 'converted', chunk_rows=3, upstream=UPSTREAM)
    order = ['b-train', 'a-val', 'c-stage', 'd-both']
    for name in ('emg', 'joint_angles', 'time'):
        expected = np.concatenate([arrays[key][name] for key in order]).tobytes()
        actual = (tmp_path / 'converted' / (name + '.dat')).read_bytes()
        assert actual == expected
        assert result['fields'][name]['sha256'] == hashlib.sha256(expected).hexdigest()
    valid = np.fromfile(tmp_path / 'converted' / 'valid_mask.dat', dtype=bool)
    assert valid.tolist() == [False, False, True, False, False, False, True, True] * 4
    assert result['fields']['valid_mask']['sha256'] == hashlib.sha256(valid.tobytes()).hexdigest()
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == source_hashes


def test_conversion_refuses_changed_source_and_retains_incomplete_evidence(tmp_path):
    paths, metadata, _ = fixture(tmp_path)
    source_receipt = receipt(tmp_path, inspect(paths, metadata))
    with h5py.File(paths[0], 'r+') as f:
        f['emg2pose'].attrs['changed_after_inspection'] = True
    with pytest.raises(ValueError, match='changed|identity|receipt'):
        module().convert_corpus(source_receipt, tmp_path / 'converted', upstream=UPSTREAM)
    assert not (tmp_path / 'converted').exists()
    stage = tmp_path / 'converted.incomplete'
    assert stage.is_dir()
    assert not (stage / 'completion.json').exists()
    assert json.loads((stage / 'status.json').read_text())['complete'] is False


def test_conversion_never_overwrites_existing_or_incomplete_output(tmp_path):
    paths, metadata, _ = fixture(tmp_path)
    source_receipt = receipt(tmp_path, inspect(paths, metadata))
    out = tmp_path / 'converted'
    out.mkdir()
    (out / 'keep.txt').write_text('original')
    with pytest.raises(FileExistsError):
        module().convert_corpus(source_receipt, out, upstream=UPSTREAM)
    assert (out / 'keep.txt').read_text() == 'original'
    stage = tmp_path / 'another.incomplete'
    stage.mkdir()
    with pytest.raises(FileExistsError):
        module().convert_corpus(source_receipt, tmp_path / 'another', upstream=UPSTREAM)


def test_source_changed_after_field_copy_is_rejected_before_publication(tmp_path, monkeypatch):
    paths, metadata, _ = fixture(tmp_path)
    source_receipt = receipt(tmp_path, inspect(paths, metadata))
    original = module()._sidecar
    def racing(sessions, blocks):
        result = original(sessions, blocks)
        with h5py.File(paths[0], 'r+') as f:
            f['emg2pose'].attrs['changed_late'] = True
        return result
    monkeypatch.setattr(module(), '_sidecar', racing)
    with pytest.raises(ValueError, match='changed'):
        module().convert_corpus(source_receipt, tmp_path / 'converted', upstream=UPSTREAM)
    assert not (tmp_path / 'converted').exists()
    assert not (tmp_path / 'converted.incomplete' / 'completion.json').exists()


def test_racing_destination_directory_is_not_replaced(tmp_path, monkeypatch):
    paths, metadata, _ = fixture(tmp_path)
    source_receipt = receipt(tmp_path, inspect(paths, metadata))
    original = module()._publish_directory
    def racing(stage, output):
        output.mkdir()
        original(stage, output)
    monkeypatch.setattr(module(), '_publish_directory', racing)
    with pytest.raises(FileExistsError):
        module().convert_corpus(source_receipt, tmp_path / 'converted', upstream=UPSTREAM)
    assert (tmp_path / 'converted').is_dir()
    assert list((tmp_path / 'converted').iterdir()) == []
    assert not (tmp_path / 'converted.incomplete' / 'completion.json').exists()


@pytest.mark.parametrize('during_write', [False, True])
def test_storage_admission_and_dynamic_reserve_abort_without_complete_marker(tmp_path, monkeypatch, during_write):
    paths, metadata, _ = fixture(tmp_path)
    source_receipt = receipt(tmp_path, inspect(paths, metadata))
    calls = 0
    def available(path):
        nonlocal calls
        calls += 1
        free = 100 * 1024**3 if during_write and calls == 1 else 1
        return SimpleNamespace(total=100 * 1024**3, used=0, free=free)
    monkeypatch.setattr(module().shutil, 'disk_usage', available)
    with pytest.raises(RuntimeError, match='storage|reserve'):
        module().convert_corpus(source_receipt, tmp_path / 'converted', chunk_rows=3, upstream=UPSTREAM)
    assert not (tmp_path / 'converted').exists()
    assert not (tmp_path / 'converted.incomplete' / 'completion.json').exists()


def test_official_reader_preserves_fallback_and_normalizes_the_written_data(tmp_path):
    torch = pytest.importorskip('torch')
    paths, metadata, arrays = fixture(tmp_path)
    module().convert_corpus(receipt(tmp_path, inspect(paths, metadata)), tmp_path / 'converted', chunk_rows=3, upstream=UPSTREAM)
    import sys
    sys.path.insert(0, str(UPSTREAM.resolve()))
    from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
    from egoemg.transforms import Compose, ExtractToTensor
    ds = Emg2PoseDataset(memmap_dir=tmp_path / 'converted', window_length=2, stride=2,
                        skip_ik_failures=False, allowed_splits=['val'], norm_mode='per-dataset',
                        norm_stats_path=str(UPSTREAM / 'assets/per_dataset_norm_stats.json'),
                        transform=Compose([ExtractToTensor(field='emg')]))
    assert len(ds) == 4
    sample = ds[0]
    expected = (torch.from_numpy(arrays['a-val']['emg'][:2].copy()) / (18.49 + 1e-6)).numpy().T
    assert sample['emg'].tobytes() == expected.tobytes()
    # The two original all-zero/near-zero label frames are raw-invalid;
    # the unchanged official reader promotes an entirely invalid window.
    assert sample['label_valid_mask'].tolist() == [True, True]
    assert np.fromfile(tmp_path / 'converted' / 'valid_mask.dat', dtype=bool)[8:10].tolist() == [False, False]

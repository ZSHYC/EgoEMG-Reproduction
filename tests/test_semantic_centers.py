"""Hand-counted frozen centers; no regeneration, payload scan or model errors."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from test_semantic_tables import metadata_fixture

UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1] / 'upstream'))


def module():
    return importlib.import_module('egoemg_repro.semantic_centers')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(partition):
    partition['provenance'].pop('partition_sha256', None)
    raw = json.dumps(partition, sort_keys=True, separators=(',', ':')).encode()
    partition['provenance']['partition_sha256'] = hashlib.sha256(raw).hexdigest()
    return partition


def rewrite(fixture, change):
    value = json.loads(fixture.partition.read_text())
    change(value)
    fixture.partition.write_text(json.dumps(seal(value)))


@pytest.fixture
def fixture(tmp_path):
    from types import SimpleNamespace
    root = metadata_fixture(tmp_path / 'PRIVATE-SOURCE-PATH')
    arrays = {'label_gesture_class': np.array([0, 0, 1, 50, -1, 33, 33, 59], dtype='<i4'),
              'label_gesture_active': np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=bool),
              'frame_split_id': np.array([0, 0, 1, 2, 3, 2, 2, 3], dtype='i1'),
              'generated_label_valid': np.ones((8, 2), dtype=bool)}
    arrays['generated_label_valid'][2] = [True, False]
    for hand in ('left', 'right'):
        arrays[f'generated_joint_angles_{hand}'] = np.full((8, 20), np.pi / 2, dtype='<f4')
        arrays[f'mocap_{hand}_wrist_pitch'] = np.full(8, 90, dtype='<f4')
        arrays[f'mocap_{hand}_wrist_yaw'] = np.full(8, -45, dtype='<f4')
        arrays[f'mocap_{hand}_wrist_angles_valid'] = np.ones(8, dtype=bool)
    arrays['generated_joint_angles_left'][2, 3] = np.nan
    arrays['mocap_left_wrist_angles_valid'][2] = False
    fields = json.loads((root / 'manifest.json').read_text())['fields']
    ledger = {'ledger_version': 1, 'files': {}, 'PRIVATE-LEDGER': 'PRIVATE-UNKNOWN-USER'}
    for name, array in arrays.items():
        path = root / fields[name]['filename']
        array.tofile(path)
        ledger['files'][path.name] = dict(bytes=path.stat().st_size, sha256=sha(path),
                                        checksum_basis='local_transfer', original_user='PRIVATE-LEDGER-USER')
    receipt = root / 'acquisition_ledger.json'
    receipt.write_text(json.dumps(ledger))
    partition = dict(schema_version=1, settings=dict(window_length=4, train_stride=1, eval_stride=4,
                                                    val_fraction=.12, jitter=False),
                     episodes=[dict(index=i, start=i * 4, end=(i + 1) * 4,
                                    id='PRIVATE-EPISODE', subject='PRIVATE-PARTICIPANT') for i in range(2)],
                     windows=[dict(episode=0, start=0, end=4, center=2, role='test', split='user'),
                              dict(episode=1, start=4, end=8, center=6, role='test', split='gesture')],
                     coverage={role: dict(windows=2 if role == 'test' else 0,
                                          hand_samples=4 if role == 'test' else 0) for role in ('train', 'val', 'test')},
                     provenance=dict(manifest_sha256=sha(root / 'manifest.json'),
                                     metadata_sha256=sha(root / 'metadata.npz'),
                                     frame_split_sha256=sha(root / 'frame_split_id.dat'),
                                     protocol='legacy-prefiltered-temporal-validation',
                                     interval_convention='half-open [start,end)'),
                     PRIVATE_EXTRA='PRIVATE-PARTITION-PATH')
    path = tmp_path / 'PRIVATE-FROZEN-PARTITIONS.json'
    path.write_text(json.dumps(seal(partition)))
    return SimpleNamespace(root=root, partition=path, arrays=arrays, receipt=receipt)


def collect(fixture):
    return module().collect_center_coverage(fixture.root, fixture.partition, upstream=UPSTREAM,
                                           partition_sha256=sha(fixture.partition),
                                           transfer_receipt_sha256=sha(fixture.receipt))


def test_two_windows_four_hands_and_angle_denominators(fixture):
    audit = collect(fixture)
    public = audit['public_projection']
    assert public['denominators']['window_centers'] == 2
    assert public['denominators']['hand_centers'] == 4
    assert public['denominators']['possible_angle_elements'] == 88
    assert public['denominators']['valid_angle_elements'] == 65
    assert public['overall']['active_window_centers'] == 1
    assert public['overall']['active_hand_centers'] == 2
    assert public['overall']['generated_valid_hand_centers'] == 3
    assert public['overall']['wrist_valid_hand_centers'] == 3
    assert public['overall']['valid_by_angle'] == [3, 3, 3, 2] + [3] * 18
    assert public['by_role']['test'] == public['overall']
    assert public['by_split']['user']['window_centers'] == 1
    assert public['by_hand']['left']['valid_angle_elements'] == 41
    assert public['by_hand']['right']['valid_angle_elements'] == 24
    assert public['full60_view']['definition_count'] == 60
    assert public['full60_view']['observed_class_count'] == 2
    assert public['full60_view']['active_observed_class_count'] == 1
    assert public['timeline_frames'] == {'status': 'not_computed', 'count': None}
    assert public['raw_mask_hand_frames'] == {'status': 'not_computed', 'count': None}
    assert public['model_error_statistics'] == {'status': 'not_computed', 'count': None}
    assert audit['private_receipt']['point_read_payload_bytes'] == 372
    assert public['source_identity']['full_payload_rehash_performed'] is False


def test_units_and_mask_match_independent_legacy_dataset(fixture):
    pytest.importorskip('torch')
    from egoemg_repro.data import LegacyDataset
    # Old Dataset needs EMG and identity arrays only for this independent oracle.
    manifest = json.loads((fixture.root / 'manifest.json').read_text())
    for hand in ('left', 'right'):
        name = f'emg_{hand}_filtered'
        np.zeros((8, 8), dtype='<f4').tofile(fixture.root / (name + '.dat'))
        manifest['fields'][name] = dict(filename=name + '.dat', dtype='float32', shape=[8, 8])
    (fixture.root / 'manifest.json').write_text(json.dumps(manifest))
    with np.load(fixture.root / 'metadata.npz', allow_pickle=False) as data:
        md = {key: data[key] for key in data.files}
    md['episode_id'] = np.array(['PRIVATE-ONE', 'PRIVATE-TWO'], dtype='S20')
    np.savez(fixture.root / 'metadata.npz', **md)
    rewrite(fixture, lambda p: p['provenance'].update(manifest_sha256=sha(fixture.root / 'manifest.json'),
                                                   metadata_sha256=sha(fixture.root / 'metadata.npz')))
    windows = json.loads(fixture.partition.read_text())['windows']
    dataset = LegacyDataset(fixture.root, windows)
    angles, masks = hashlib.sha256(), hashlib.sha256()
    for index in range(4):
        sample = dataset[index]
        value = sample['joint_angles'].numpy()[:, 2].copy()
        mask = sample['angle_valid_mask'].numpy()[:, 2].copy()
        angles.update(value.tobytes())
        masks.update(mask.tobytes())
        assert value[0] == np.float32(np.pi / 2)
        assert value[20] == np.float32(np.pi / 2)
        assert value[21] == np.float32(-np.pi / 4)
        if index == 0:
            assert value[3] == 0 and not mask[3] and not mask[20:].any()
        if index == 1:
            assert not mask[:20].any() and mask[20:].all()
    private = collect(fixture)['private_receipt']
    assert private['model_angle_radians_sha256'] == angles.hexdigest()
    assert private['angle_valid_mask_sha256'] == masks.hexdigest()


def test_raw50_retained_and_explicit_59_conservation(fixture):
    values = fixture.arrays['label_gesture_class'].copy()
    values[6] = 50
    path = fixture.root / 'label_gesture_class.dat'
    values.tofile(path)
    receipt = json.loads(fixture.receipt.read_text())
    receipt['files'][path.name]['sha256'] = sha(path)
    fixture.receipt.write_text(json.dumps(receipt))
    public = collect(fixture)['public_projection']
    full, reduced = public['full60_view'], public['exclude_raw59_view']
    assert full['definition_count'] == 60 and reduced['definition_count'] == 59
    raw = next(row for row in full['classes'] if row['class_id'] == 50)
    assert raw['window_centers'] == 1 and raw['hand_centers'] == 2
    assert full['totals']['window_centers'] - reduced['totals']['window_centers'] == 1
    assert full['totals']['hand_centers'] - reduced['totals']['hand_centers'] == 2
    assert reduced['excluded_class_ids'] == [50]
    assert sum(row['window_centers'] for row in full['families']) == 2


@pytest.mark.parametrize('damage', ['duplicate', 'wrong_center', 'cross_episode', 'unknown_split', 'unknown_role',
                                    'wrong_role_split', 'wrong_episode_bounds', 'wrong_coverage',
                                    'wrong_manifest_hash', 'wrong_metadata_hash', 'wrong_frame_hash', 'jitter',
                                    'bool_coordinate', 'wrong_window_length'])
def test_malformed_frozen_partition_rejected(fixture, damage):
    def change(p):
        w = p['windows'][0]
        if damage == 'duplicate': p['windows'].append(dict(w))
        elif damage == 'wrong_center': w['center'] = 1
        elif damage == 'cross_episode': w.update(start=2, end=6, center=4)
        elif damage == 'unknown_split': w['split'] = 'PRIVATE-SPLIT'
        elif damage == 'unknown_role': w['role'] = 'validation'
        elif damage == 'wrong_role_split': w['role'] = 'train'
        elif damage == 'wrong_episode_bounds': p['episodes'][0]['end'] = 5
        elif damage == 'wrong_coverage': p['coverage']['test']['hand_samples'] = 42
        elif damage.startswith('wrong_') and damage.endswith('_hash'):
            p['provenance'][{'wrong_manifest_hash':'manifest_sha256', 'wrong_metadata_hash':'metadata_sha256',
                             'wrong_frame_hash':'frame_split_sha256'}[damage]] = '0' * 64
        elif damage == 'jitter': p['settings']['jitter'] = True
        elif damage == 'bool_coordinate': w['episode'] = False
        elif damage == 'wrong_window_length': p['settings']['window_length'] = 3
    rewrite(fixture, change)
    with pytest.raises(ValueError): collect(fixture)


def test_external_and_internal_partition_hashes_are_enforced(fixture):
    with pytest.raises(ValueError, match='partition'):
        module().collect_center_coverage(fixture.root, fixture.partition, upstream=UPSTREAM,
                                        partition_sha256='0' * 64, transfer_receipt_sha256=sha(fixture.receipt))
    value = json.loads(fixture.partition.read_text())
    value['provenance']['partition_sha256'] = '0' * 64
    fixture.partition.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='partition'): collect(fixture)


@pytest.mark.parametrize('damage', ['missing_record', 'wrong_bytes', 'wrong_basis', 'bad_sha', 'truncated',
                                    'symlink', 'wrong_center_split', 'bad_class', 'noncanonical_bool'])
def test_transfer_field_and_center_domains_fail(fixture, damage):
    ledger = json.loads(fixture.receipt.read_text())
    name = 'label_gesture_active.dat'
    if damage == 'missing_record': del ledger['files'][name]
    elif damage == 'wrong_bytes': ledger['files'][name]['bytes'] += 1
    elif damage == 'wrong_basis': ledger['files'][name]['checksum_basis'] = 'unverified'
    elif damage == 'bad_sha': ledger['files'][name]['sha256'] = 'bad'
    elif damage == 'truncated': (fixture.root / name).write_bytes(b'bad')
    elif damage == 'symlink':
        path = fixture.root / name
        path.rename(fixture.root / 'original.dat')
        path.symlink_to('original.dat')
    elif damage == 'wrong_center_split':
        path = fixture.root / 'frame_split_id.dat'
        with path.open('r+b') as stream: stream.seek(2); stream.write(b'\x02')
    elif damage == 'bad_class':
        with (fixture.root / 'label_gesture_class.dat').open('r+b') as stream:
            stream.seek(8); stream.write(struct.pack('<i', 60))
    else:
        with (fixture.root / name).open('r+b') as stream: stream.seek(2); stream.write(b'\x02')
    fixture.receipt.write_text(json.dumps(ledger))
    with pytest.raises(ValueError): collect(fixture)


def test_source_change_during_point_reads_is_rejected(fixture, monkeypatch):
    real = module()._point_row
    changed = []
    def changing(*args, **kwargs):
        value = real(*args, **kwargs)
        if not changed:
            path = fixture.root / 'generated_joint_angles_left.dat'
            with path.open('r+b') as stream: stream.write(np.float32(.75).tobytes())
            changed.append(True)
        return value
    monkeypatch.setattr(module(), '_point_row', changing)
    with pytest.raises(ValueError, match='changed'): collect(fixture)


def test_public_projection_does_not_copy_private_or_unknown_fields(fixture):
    audit = collect(fixture)
    public = json.dumps(audit['public_projection'])
    for sentinel in ('PRIVATE-', 'episode', 'subject', 'source_root', str(fixture.root), str(fixture.partition)):
        assert sentinel not in public
    assert audit['private_receipt']['partition_path'] == str(fixture.partition.absolute())


def test_cli_private_audit_and_existing_output_not_overwritten(fixture, tmp_path):
    output = tmp_path / 'audit.json'
    args = [sys.executable, '-m', 'egoemg_repro.semantic_centers', '--data-root', str(fixture.root),
            '--partition', str(fixture.partition), '--partition-sha256', sha(fixture.partition),
            '--transfer-receipt-sha256', sha(fixture.receipt),
            '--upstream', str(UPSTREAM), '--output', str(output)]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original = output.read_bytes()
    assert json.loads(original)['public_projection']['denominators']['hand_centers'] == 4
    assert output.stat().st_mode & 0o777 == 0o600
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0 and output.read_bytes() == original


def test_transfer_receipt_requires_explicit_frozen_hash(fixture):
    with pytest.raises(ValueError, match='transfer'):
        module().collect_center_coverage(fixture.root, fixture.partition, upstream=UPSTREAM,
                                        partition_sha256=sha(fixture.partition),
                                        transfer_receipt_sha256='0' * 64)


def test_nan_wrist_only_invalidates_that_angle_before_cleanup(fixture):
    path = fixture.root / 'mocap_right_wrist_pitch.dat'
    values = fixture.arrays['mocap_right_wrist_pitch'].copy()
    values[2] = np.nan
    values.tofile(path)
    ledger = json.loads(fixture.receipt.read_text())
    ledger['files'][path.name]['sha256'] = sha(path)
    fixture.receipt.write_text(json.dumps(ledger))
    public = collect(fixture)['public_projection']
    assert public['overall']['valid_angle_elements'] == 64
    assert public['overall']['valid_by_angle'][20:] == [2, 3]
    assert public['overall']['finite_by_angle'][20:] == [3, 4]
    assert public['by_hand']['right']['valid_angle_elements'] == 23


def test_unassigned_center_is_not_a_defined_class(fixture):
    path = fixture.root / 'label_gesture_class.dat'
    values = fixture.arrays['label_gesture_class'].copy()
    values[6] = -1
    values.tofile(path)
    ledger = json.loads(fixture.receipt.read_text())
    ledger['files'][path.name]['sha256'] = sha(path)
    fixture.receipt.write_text(json.dumps(ledger))
    public = collect(fixture)['public_projection']
    assert public['overall']['window_centers'] == 2
    assert public['full60_view']['observed_class_count'] == 1
    assert public['full60_view']['totals']['window_centers'] == 1
    assert public['unassigned_centers']['window_centers'] == 1
    assert public['unassigned_centers']['hand_centers'] == 2


def test_role_footprints_cannot_overlap_even_with_distinct_centers(fixture):
    def change(p):
        p['settings']['window_length'] = 2
        p['windows'] = [dict(episode=0, start=0, end=2, center=1, role='train', split='train'),
                        dict(episode=0, start=1, end=3, center=2, role='test', split='user')]
        p['coverage']['train'] = dict(windows=1, hand_samples=2)
        p['coverage']['test'] = dict(windows=1, hand_samples=2)
    rewrite(fixture, change)
    with pytest.raises(ValueError, match='overlap'): collect(fixture)


def test_window_order_does_not_change_point_hashes_or_aggregates(fixture):
    first = collect(fixture)
    rewrite(fixture, lambda p: p['windows'].reverse())
    second = collect(fixture)
    assert first['private_receipt']['model_angle_radians_sha256'] == second['private_receipt']['model_angle_radians_sha256']
    first['public_projection'].pop('source_identity')
    second['public_projection'].pop('source_identity')
    assert first['public_projection'] == second['public_projection']


def test_small_frozen_source_change_is_rejected(fixture, monkeypatch):
    real = module()._point_row
    changed = []
    def changing(*args, **kwargs):
        value = real(*args, **kwargs)
        if not changed:
            fixture.partition.write_bytes(fixture.partition.read_bytes())
            changed.append(True)
        return value
    monkeypatch.setattr(module(), '_point_row', changing)
    with pytest.raises(ValueError, match='changed'): collect(fixture)


def test_late_source_change_before_return_is_rejected(fixture, monkeypatch):
    real = module().definitions._check_upstream
    calls = []
    def change_on_final_check(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(True)
        if len(calls) == 3:
            with (fixture.root / 'mocap_right_wrist_yaw.dat').open('r+b') as stream:
                stream.write(np.float32(.5).tobytes())
        return result
    monkeypatch.setattr(module().definitions, '_check_upstream', change_on_final_check)
    with pytest.raises(ValueError, match='changed'): collect(fixture)


def test_class_activity_contingency_keeps_role_and_split_denominators(fixture):
    def change(p):
        p['settings']['window_length'] = 2
        p['windows'] = [dict(episode=0, start=0, end=2, center=1, role='val', split='train'),
                        dict(episode=1, start=5, end=7, center=6, role='test', split='gesture')]
        p['coverage']['val'] = dict(windows=1, hand_samples=2)
        p['coverage']['test'] = dict(windows=1, hand_samples=2)
    rewrite(fixture, change)
    public = collect(fixture)['public_projection']
    cells = public['center_contingency']
    assert [(r['role'], r['split'], r['class_id'], r['active'], r['window_centers'], r['hand_centers'])
            for r in cells] == [('val', 'train', 0, False, 1, 2), ('test', 'gesture', 33, False, 1, 2)]
    assert sum(row['valid_angle_elements'] for row in cells) == public['overall']['valid_angle_elements']
    assert {row['class_id'] for row in cells if row['role'] == 'test'} == {33}
    assert public['full60_view']['observed_class_count'] == 2

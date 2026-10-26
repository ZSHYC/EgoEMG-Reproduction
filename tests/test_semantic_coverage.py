"""Frame audit projection must conserve real denominators and remain private-safe."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_semantic_tables import metadata_fixture, mutate_json


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1] / 'upstream'))


def module():
    return importlib.import_module('egoemg_repro.semantic_coverage')


def audit_fixture(root):
    path = root / 'PRIVATE-audit.json'
    # Eight hand-counted shared time rows; these are not hand samples.
    rows = [(-1, 'both', False), (0, 'train', True), (0, 'train', False),
            (1, 'user', True), (50, 'gesture', True), (33, 'gesture', True),
            (33, 'gesture', False), (59, 'both', True)]
    fields = {}
    for name, dtype, size, digest in [('label_gesture_class', '<i4', 32, 'a' * 64),
                                      ('label_gesture_active', '|b1', 8, 'b' * 64),
                                      ('frame_split_id', '|i1', 8, 'c' * 64)]:
        fields[name] = dict(bytes=size, dtype=dtype, shape=[8], sha256=digest,
                            matches_acquisition_receipt=True, checksum_basis='local_transfer',
                            private_path='PRIVATE-FIELD-PATH')
    doc = dict(format='private_legacy_gesture_frame_audit_v1', observed_utc='2026-10-26T03:57:25+00:00',
               total_synchronized_rows=8, no_hand_duplication=True, class_range=[-1, 59],
               actual_class_ids=[-1, 0, 1, 33, 50, 59], actual_active_class_ids=[0, 1, 33, 50, 59],
               split_labels=['train', 'user', 'gesture', 'both'], fields=fields,
               counts=[dict(class_id=c, split=s, active=a, synchronized_rows=1) for c, s, a in rows],
               scope='engineering fixture only', private_subject='PRIVATE-SUBJECT-ALICE')
    path.write_text(json.dumps(doc))
    return path


def test_hand_counted_frames_activity_families_and_raw_exclusion(tmp_path):
    root = metadata_fixture(tmp_path / 'metadata')
    audit = audit_fixture(tmp_path)
    result = module().build_frame_coverage(audit, upstream=UPSTREAM, metadata_root=root)
    assert result['timeline_frames'] == dict(total=8, active=5, inactive=3, defined=7,
                                             unassigned=1, active_defined=5, raw50=1)
    assert result['definition_count'] == 60
    assert result['observed_class_count'] == 5 and result['active_observed_class_count'] == 5
    assert result['data_scope'] == 'engineering_fixture'
    by_id = {r['id']: r for r in result['by_gesture']}
    assert len(by_id) == 60
    assert {i: by_id[i]['total'] for i in (0, 1, 33, 50, 59)} == {0: 2, 1: 1, 33: 2, 50: 1, 59: 1}
    assert by_id[2]['total'] == 0 and by_id[2]['observed'] is False
    assert result['full60_view']['family_rows'] == {'Single-hand': 3, 'Symmetric bimanual': 1,
                                                   'Asymmetric bimanual': 3}
    excluded = result['exclude_raw59_frame_view']
    assert excluded['definition_count'] == 59 and excluded['observed_class_count'] == 4
    assert excluded['before_count'] == 7 and excluded['raw_excluded_count'] == 1 and excluded['after_count'] == 6
    assert excluded['before_active'] == 5 and excluded['raw_excluded_active'] == 1 and excluded['after_active'] == 4
    assert excluded['family_definition_counts'] == {'Single-hand': 30, 'Symmetric bimanual': 17,
                                                    'Asymmetric bimanual': 12}
    assert excluded['family_rows'] == {'Single-hand': 3, 'Symmetric bimanual': 0, 'Asymmetric bimanual': 3}
    assert {s: r['total'] for s, r in result['by_split'].items()} == {'train': 2, 'user': 1, 'gesture': 3, 'both': 2}


def test_record_order_does_not_change_coverage_or_invent_windows(tmp_path):
    root = metadata_fixture(tmp_path / 'metadata')
    audit = audit_fixture(tmp_path)
    before = module().build_frame_coverage(audit, upstream=UPSTREAM, metadata_root=root)
    mutate_json(audit, lambda d: d['counts'].reverse())
    after = module().build_frame_coverage(audit, upstream=UPSTREAM, metadata_root=root)
    for key in ('timeline_frames', 'by_gesture', 'by_split', 'full60_view', 'exclude_raw59_frame_view', 'records'):
        assert after[key] == before[key]
    for key in ('window_centers', 'hand_centers', 'raw_mask_hand_frames', 'center_valid_angles', 'model_error_statistics'):
        assert after[key]['status'] == 'not_computed' and after[key]['count'] is None
    assert after['provenance']['computation_input'] == 'verified_frame_audit'
    assert after['provenance']['source_array_rehash_performed_by'] == 'recorded_audit'


@pytest.mark.parametrize('fault', ['missing_field', 'dtype', 'shape', 'bytes', 'receipt', 'sha256',
                                  'basis', 'duplicate', 'class', 'split', 'active_string', 'negative_count',
                                  'float_count', 'bool_count', 'sum', 'ids', 'active_ids', 'range',
                                  'hand_duplication', 'timestamp', 'total', 'format'])
def test_invalid_or_inconsistent_audit_fails_without_outputs(tmp_path, fault):
    root = metadata_fixture(tmp_path / 'metadata')
    audit = audit_fixture(tmp_path)
    def change(d):
        field = d['fields']['label_gesture_class']
        if fault == 'missing_field': del d['fields']['label_gesture_active']
        elif fault == 'dtype': field['dtype'] = 'float32'
        elif fault == 'shape': field['shape'] = [8, 1]
        elif fault == 'bytes': field['bytes'] += 1
        elif fault == 'receipt': field['matches_acquisition_receipt'] = False
        elif fault == 'sha256': field['sha256'] = 'PRIVATE-HASH'
        elif fault == 'basis': field['checksum_basis'] = 'unverified'
        elif fault == 'duplicate': d['counts'].append(dict(d['counts'][0]))
        elif fault == 'class': d['counts'][0]['class_id'] = 60
        elif fault == 'split': d['counts'][0]['split'] = 'PRIVATE-SPLIT'
        elif fault == 'active_string': d['counts'][0]['active'] = 'False'
        elif fault == 'negative_count': d['counts'][0]['synchronized_rows'] = -1
        elif fault == 'float_count': d['counts'][0]['synchronized_rows'] = 1.0
        elif fault == 'bool_count': d['counts'][0]['synchronized_rows'] = True
        elif fault == 'sum': d['counts'][0]['synchronized_rows'] = 2
        elif fault == 'ids': d['actual_class_ids'].append(2)
        elif fault == 'active_ids': d['actual_active_class_ids'].remove(50)
        elif fault == 'range': d['class_range'] = [0, 59]
        elif fault == 'hand_duplication': d['no_hand_duplication'] = 1
        elif fault == 'timestamp': d['observed_utc'] = 'PRIVATE-TIME'
        elif fault == 'total': d['total_synchronized_rows'] = 9
        elif fault == 'format': d['format'] = 'unknown'
    mutate_json(audit, change)
    output = tmp_path / 'out'
    with pytest.raises(ValueError):
        module().export_frame_coverage(audit, upstream=UPSTREAM, metadata_root=root, output=output)
    assert not output.exists()


def test_missing_audit_is_not_a_zero_coverage_result(tmp_path):
    root = metadata_fixture(tmp_path / 'metadata')
    with pytest.raises(ValueError, match='source|audit|missing'):
        module().build_frame_coverage(tmp_path / 'missing.json', upstream=UPSTREAM, metadata_root=root)


def test_cli_whitelists_public_fields_and_preserves_audit_identity(tmp_path):
    root = metadata_fixture(tmp_path / 'PRIVATE-METADATA')
    audit = audit_fixture(tmp_path)
    output = tmp_path / 'out'
    run = subprocess.run([sys.executable, '-m', 'egoemg_repro.semantic_coverage', '--frame-audit', str(audit),
                          '--upstream', str(UPSTREAM), '--metadata-root', str(root), '--output', str(output)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert {p.name for p in output.iterdir()} == {'coverage.json', 'coverage.md'}
    for path in output.iterdir():
        blob = path.read_bytes()
        for private in (b'PRIVATE', str(tmp_path).encode(), b'private_subject', b'private_path', b'episode_subject'):
            assert private not in blob, (path.name, private)
    doc = json.loads((output / 'coverage.json').read_text())
    assert doc['sources']['frame_audit']['sha256'] == hashlib.sha256(audit.read_bytes()).hexdigest()
    assert doc['sources']['frame_audit']['filename'] == 'legacy-gesture-frame-audit.json'
    assert doc['sources']['fields']['label_gesture_class']['sha256'] == 'a' * 64
    assert doc['provenance']['observed_utc'] == '2026-10-26T03:57:25+00:00'
    text = (output / 'coverage.md').read_text()
    assert 'synchronized timeline rows' in text and 'not_computed' in text
    assert 'not a new raw-array scan' in text and 'raw' in text

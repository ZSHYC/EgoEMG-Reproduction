"""Definition exports must preserve semantics without claiming data coverage."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1] / 'upstream'))
NAMES = ('THUMB_CMC_FE THUMB_CMC_AA THUMB_MCP_FE THUMB_IP_FE '
         'INDEX_MCP_AA INDEX_MCP_FE INDEX_PIP_FE INDEX_DIP_FE '
         'MIDDLE_MCP_AA MIDDLE_MCP_FE MIDDLE_PIP_FE MIDDLE_DIP_FE '
         'RING_MCP_AA RING_MCP_FE RING_PIP_FE RING_DIP_FE '
         'PINKY_MCP_AA PINKY_MCP_FE PINKY_PIP_FE PINKY_DIP_FE').split()
LEGACY_NOTE = ("Derived from generated_mano_* by decomposing MANO relative joint rotations "
               "into scalar projections onto UmeTrack's predefined anatomical rotation axes "
               "(axis-angle projection, inverse of UmeTrack FK).")


def module():
    return importlib.import_module('egoemg_repro.semantic_tables')


def metadata_fixture(root, rows=8):
    root.mkdir()
    fields = {}
    for name, dtype, shape in [('label_gesture_class', 'int32', [rows]),
                               ('label_gesture_active', 'bool', [rows]),
                               ('frame_split_id', 'int8', [rows]),
                               ('generated_label_valid', 'bool', [rows, 2])]:
        fields[name] = dict(filename=name + '.dat', dtype=dtype, shape=shape)
    for hand in ('left', 'right'):
        for name, dtype, shape in [(f'generated_joint_angles_{hand}', 'float32', [rows, 20]),
                                   (f'mocap_{hand}_wrist_pitch', 'float32', [rows]),
                                   (f'mocap_{hand}_wrist_yaw', 'float32', [rows]),
                                   (f'mocap_{hand}_wrist_angles_valid', 'bool', [rows])]:
            fields[name] = dict(filename=name + '.dat', dtype=dtype, shape=shape)
    semantics = [n.lower() for n in NAMES]
    manifest = dict(format_version='egoemg_v2_memmap', total_rows=rows, num_episodes=2,
                    fields=fields, generated_joint_angles_semantics=semantics,
                    left_hand_strategy='flip_local_z',
                    frame_split_labels=['train', 'user', 'gesture', 'both'],
                    source_root='PRIVATE-SOURCE-ALICE', unknown_private='PRIVATE-UNKNOWN-BOB')
    summary = dict(format_version='egoemg_v2_memmap', total_rows=rows,
                   generated_joint_angles_semantics=semantics, left_hand_strategy='flip_local_z',
                   generated_joint_angles_note=LEGACY_NOTE, source_root='PRIVATE-SUMMARY-PATH')
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'metadata_summary.json').write_text(json.dumps(summary))
    np.savez(root / 'metadata.npz', episode_start_idx=np.array([0, rows // 2], dtype='i8'),
             episode_end_idx=np.array([rows // 2, rows], dtype='i8'),
             episode_length=np.array([rows // 2, rows - rows // 2], dtype='i8'),
             splits_split=np.array(['train', 'user', 'gesture', 'both'], dtype='S7'),
             splits_split_id=np.arange(4, dtype='i4'),
             episode_subject=np.array(['PRIVATE-SUBJECT-ALICE'] * 2))
    return root


def mutate_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_complete_60_definitions_keep_raw_and_noncontiguous_families(tmp_path):
    result = module().build_semantic_tables(UPSTREAM, metadata_fixture(tmp_path / 'metadata'))
    doc = result['gesture_vocabulary']
    assert doc['definition_count'] == 60
    assert doc['family_definition_counts'] == {'Single-hand': 30, 'Symmetric bimanual': 18,
                                              'Asymmetric bimanual': 12}
    assert [r['id'] for r in doc['rows']] == list(range(60))
    assert doc['observed_class_count'] is None and doc['coverage_status'] == 'not_computed'
    rows = {r['id']: r for r in doc['rows']}
    assert rows[33]['family'] == 'Asymmetric bimanual'
    assert [rows[i]['family'] for i in (50, 51, 52)] == ['Symmetric bimanual'] * 3
    assert rows[50]['name'] == 'raw' and rows[50]['analysis59_excluded'] is True
    assert sum(r['analysis59_excluded'] for r in rows.values()) == 1


def test_22_semantics_preserve_finger_order_units_and_mask_sources(tmp_path):
    doc = module().build_semantic_tables(UPSTREAM, metadata_fixture(tmp_path / 'metadata'))['joint_angle_semantics']
    assert [r['name'] for r in doc['rows']] == NAMES + ['WRIST_FE', 'WRIST_RU']
    assert [r['index'] for r in doc['rows']] == list(range(22))
    for row in doc['rows'][:20]:
        assert row['disk_unit'] == 'rad' and row['model_unit'] == 'rad' and row['report_unit'] == 'deg'
        assert row['disk_component'] == row['index']
        assert row['disk_field'] == 'generated_joint_angles_{hand}'
        assert row['hand_index'] == {'left': 0, 'right': 1}
        assert row['mask_field'] == 'generated_label_valid'
    assert [r['disk_field'] for r in doc['rows'][20:]] == [
        'mocap_{hand}_wrist_pitch', 'mocap_{hand}_wrist_yaw']
    assert [r['disk_unit'] for r in doc['rows'][20:]] == ['deg', 'deg']
    assert [r['mask_field'] for r in doc['rows'][20:]] == ['mocap_{hand}_wrist_angles_valid'] * 2
    assert set(doc['mask_protocols']) == {'official_literal_loader', 'legacy_reproduction_angle_mask'}
    assert doc['mask_protocols']['legacy_reproduction_angle_mask']['finite_before_cleanup'] is True
    assert doc['left_hand_strategy'] == 'flip_local_z'
    assert doc['generation']['legacy_method'] == 'axis_angle_projection'
    assert doc['generation']['paper_method'] == 'landmark_level_ik'


def test_nominal_finger_limits_do_not_become_mocap_wrist_limits(tmp_path):
    rows = module().build_semantic_tables(UPSTREAM, metadata_fixture(tmp_path / 'metadata'))['joint_angle_semantics']['rows']
    assert rows[0]['nominal_limit_rad'] == pytest.approx([-0.7872664332389832, 1.287266492843628])
    assert rows[19]['nominal_limit_rad'] == pytest.approx([0.0, 1.4336633682250977])
    assert [r['nominal_limit_rad'] for r in rows[20:]] == [None, None]
    assert all('not a measured' in r['limit_scope'] for r in rows)


def test_pdf_descriptions_are_fixed_reference_not_unverified_equal(tmp_path):
    doc = module().build_semantic_tables(UPSTREAM, metadata_fixture(tmp_path / 'metadata'))['gesture_vocabulary']
    rows = doc['rows']
    assert rows[0]['description_official'] == 'ASL digit 1'
    assert rows[0]['paper_reference']['row_name'] == 'ASL1-ASL9'
    assert rows[0]['description_matches_paper'] is False
    assert rows[54]['description_official'] == "Two hands thumb make 'Checky' sign"
    assert 'with thumbs forming' in rows[54]['paper_reference']['description']
    assert rows[54]['description_matches_paper'] is False
    assert doc['paper_reference']['sha256'] == '124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725'
    assert 'fixed' in doc['paper_reference']['kind']


@pytest.mark.parametrize('fault', ['missing_id', 'duplicate_id', 'duplicate_member', 'missing_member', 'unknown_family'])
def test_gesture_constant_parser_rejects_missing_duplicate_or_unknown_members(fault):
    source = (UPSTREAM / 'scripts/paper/generate_per_gesture_appendix.py').read_text()
    if fault == 'missing_id':
        source = '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('0:'))
    elif fault == 'duplicate_id':
        source = source.replace('    59: ("ThumbOppo",', '    0: ("DUP", "duplicate"),\n    59: ("ThumbOppo",')
    elif fault == 'duplicate_member':
        source = source.replace('list(range(0, 30))', repr([0, 0] + list(range(1, 30))))
    elif fault == 'missing_member':
        source = source.replace('list(range(0, 30))', repr(list(range(1, 30))))
    else:
        source = source.replace('"Asymmetric bimanual":', '"Unknown family":')
    with pytest.raises(ValueError):
        module()._gesture_constants(source)


def test_ast_rejects_executable_expression_without_running_it(tmp_path):
    source = (UPSTREAM / 'scripts/paper/generate_per_gesture_appendix.py').read_text()
    sentinel = tmp_path / 'executed'
    source = source.replace('list(range(0, 30))', f"__import__('pathlib').Path({str(sentinel)!r}).write_text('bad')")
    with pytest.raises(ValueError, match='AST|constant|expression'):
        module()._gesture_constants(source)
    assert not sentinel.exists()


@pytest.mark.parametrize('fault', ['revision', 'tracked', 'untracked'])
def test_unknown_or_dirty_upstream_is_rejected(tmp_path, fault):
    metadata = metadata_fixture(tmp_path / 'metadata')
    root = tmp_path / 'upstream'
    subprocess.run(['git', 'clone', '--quiet', '--shared', str(UPSTREAM), str(root)], check=True)
    if fault == 'revision':
        subprocess.run(['git', '-C', str(root), 'checkout', '--quiet', 'HEAD~1'], check=True)
    elif fault == 'tracked':
        with (root / 'egoemg/constants.py').open('a') as f:
            f.write('\n# changed\n')
    else:
        (root / 'untracked-private.py').write_text('raise RuntimeError()')
    with pytest.raises(ValueError, match='upstream|official'):
        module().build_semantic_tables(root, metadata)


@pytest.mark.parametrize('fault', ['order', 'wrist_dtype', 'split', 'summary', 'filename', 'length'])
def test_metadata_schema_disagreement_is_rejected(tmp_path, fault):
    root = metadata_fixture(tmp_path / 'metadata')
    if fault == 'order':
        mutate_json(root / 'manifest.json', lambda d: d['generated_joint_angles_semantics'].reverse())
    elif fault == 'wrist_dtype':
        mutate_json(root / 'manifest.json', lambda d: d['fields']['mocap_left_wrist_pitch'].__setitem__('dtype', 'float64'))
    elif fault == 'split':
        mutate_json(root / 'manifest.json', lambda d: d['frame_split_labels'].reverse())
    elif fault == 'summary':
        mutate_json(root / 'metadata_summary.json', lambda d: d.__setitem__('generated_joint_angles_note', 'PRIVATE-UNKNOWN'))
    elif fault == 'filename':
        mutate_json(root / 'manifest.json', lambda d: d['fields']['generated_label_valid'].__setitem__('filename', '../private.dat'))
    else:
        mutate_json(root / 'manifest.json', lambda d: d.__setitem__('total_rows', 9))
    with pytest.raises(ValueError, match='metadata|manifest|semantics|summary|schema'):
        module().build_semantic_tables(UPSTREAM, root)


def test_cli_exports_only_definitions_and_safe_content_identities(tmp_path):
    root = metadata_fixture(tmp_path / 'PRIVATE-METADATA')
    output = tmp_path / 'out'
    run = subprocess.run([sys.executable, '-m', 'egoemg_repro.semantic_tables', '--upstream', str(UPSTREAM),
                          '--metadata-root', str(root), '--output', str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert {p.name for p in output.iterdir()} == {'gesture-vocabulary.json', 'gesture-vocabulary.md',
                                               'joint-angle-semantics.json', 'joint-angle-semantics.md'}
    for path in output.iterdir():
        value = path.read_bytes()
        for private in (b'PRIVATE', str(tmp_path).encode(), b'episode_subject', b'source_root'):
            assert private not in value, (path.name, private)
    doc = json.loads((output / 'gesture-vocabulary.json').read_text())
    assert doc['sources']['manifest']['sha256'] == hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    assert doc['sources']['official']['revision'] == 'f7cbd048b7092139852270491d3859cd472e0311'
    markdown = (output / 'joint-angle-semantics.md').read_text()
    assert 'official_literal_loader' in markdown and 'legacy_reproduction_angle_mask' in markdown
    assert 'not measured' in markdown and 'degrees on disk' in markdown

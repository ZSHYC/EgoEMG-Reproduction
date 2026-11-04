"""New same-center population/IO contract; existing model math is untouched."""
import copy
import importlib
import json

import numpy as np
import importlib.util
import pytest

# Pure metadata checks remain available without the optional model runtime.
requires_official_runtime = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason='uses the complete optional official Torch reader/model/evaluation runtime')

from test_emg_initializer_data import source
from test_emgformer_literal import UPSTREAM


def api():
    try:
        return importlib.import_module('egoemg_repro.center_reference_data')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.center_reference_data':
            pytest.fail('complete official same-center EMG input adapter missing')
        raise


def fixture(root, *, reordered=False):
    n = 4 * 7790 + 500
    arrays = source(root, n=n, optional_labels=True)
    arrays['episode_index'] = np.zeros(n, np.int64)
    # The official helper filters the center's split, not the full footprint.
    arrays['frame_split_id'][:] = 1 if reordered else 2
    arrays['frame_split_id'][19475] = 3 if reordered else 1
    arrays['generated_label_valid'][11685, 0] = False
    manifest = json.loads((root / 'manifest.json').read_text())
    for name in ('episode_index', 'frame_split_id', 'generated_label_valid'):
        a = arrays[name]
        a.tofile(root / (name + '.dat'))
        manifest['fields'][name] = dict(filename=name + '.dat', dtype=str(a.dtype), shape=list(a.shape))
    if reordered:
        labels = ['both', 'gesture', 'train', 'user']
        manifest['frame_split_labels'] = labels
        with np.load(root / 'metadata.npz', allow_pickle=False) as z:
            metadata = {k: z[k] for k in z.files}
        metadata['splits_split'] = np.asarray(labels, dtype='S8')
        np.savez(root / 'metadata.npz', **metadata)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return arrays


def config(size='small'):
    from egoemg_repro.emgformer_literal import compose_emgformer
    return compose_emgformer(UPSTREAM, size)


@requires_official_runtime
@pytest.mark.parametrize('reordered', [False, True])
def test_complete_official_grid_keeps_center_split_and_max_window_boundary(tmp_path, reordered):
    # A window-length grid or numeric split-id hardcode breaks these literals.
    fixture(tmp_path, reordered=reordered)
    grid = api().reference_windows(tmp_path, UPSTREAM)
    assert [(w['start'], w['center'], w['end'], w['split']) for w in grid['windows']] == [
        (5685, 11685, 17685, 'gesture'), (13475, 19475, 25475, 'user')]
    assert grid['counts'] == dict(windows=2, hand_requests=4,
        by_split=dict(gesture=2, user=2, both=0))
    assert grid['required_window_length'] == 12000
    assert grid['reference_window_length'] == 7790
    assert grid['full_paper_complete'] is False


@pytest.mark.parametrize('length', [True, 0, 7790, float('nan')])
def test_another_reference_budget_is_not_silently_accepted(tmp_path, length):
    with pytest.raises(ValueError):
        api().reference_windows(tmp_path, UPSTREAM, required_window_length=length)


@requires_official_runtime
@pytest.mark.parametrize('size', ['small', 'middle', 'large'])
def test_actual_pure_emg_inputs_bind_hand_center_target_and_keep_invalid_request(tmp_path, size):
    arrays = fixture(tmp_path)
    module = api()
    ds = module.CenterReferenceDataset(tmp_path, UPSTREAM, config(size),
        module.reference_windows(tmp_path, UPSTREAM))
    assert len(ds) == 4
    samples = [ds[i] for i in range(4)]
    assert [(x['hand'], x['center'], x['reference_sample_index']) for x in samples] == [
        ('left', 11685, 0), ('left', 19475, 1), ('right', 11685, 2), ('right', 19475, 3)]
    assert [bool(x['label_valid_mask'][0]) for x in samples] == [False, True, True, True]
    for sample in samples:
        assert sample['emg'].shape == (8, 12000) and sample['emg'].dtype == np.float32
        assert np.isfinite(sample['emg']).all()
        assert sample['joint_angles'].shape == (22, 1)
        assert sample['joint_angles'][:20, 0].tobytes() == arrays[
            'generated_joint_angles_' + sample['hand']][sample['center']].tobytes()
        assert sample['label_valid_mask'].shape == (1,) and sample['label_valid_mask'].dtype == bool
        assert sample['reference_unique_sample_index'] == sample['reference_sample_index']
        assert sample['reference_repeat_index'] == 0 and sample['jitter_offset'] == 0
        assert sample['end'] - sample['start'] == 12000
        assert 'vision_img' not in sample and 'vision_valid_mask' not in sample


@requires_official_runtime
@pytest.mark.parametrize('damage', ['subset', 'center', 'counts'])
def test_caller_cannot_choose_a_smaller_or_shifted_scientific_grid(tmp_path, damage):
    fixture(tmp_path)
    module = api()
    grid = copy.deepcopy(module.reference_windows(tmp_path, UPSTREAM))
    if damage == 'subset':
        grid['windows'].pop()
    elif damage == 'center':
        grid['windows'][0]['center'] += 1
    else:
        grid['counts']['hand_requests'] = 2
    with pytest.raises(ValueError):
        module.CenterReferenceDataset(tmp_path, UPSTREAM, config(), grid)


@requires_official_runtime
@pytest.mark.parametrize('field', ['emg_left_filtered', 'generated_joint_angles_left'])
def test_nonfinite_request_fails_instead_of_skipping_or_filling(tmp_path, field):
    arrays = fixture(tmp_path)
    row = 5685 if field.startswith('emg_') else 11685
    arrays[field][row, 0] = np.nan
    arrays[field].tofile(tmp_path / (field + '.dat'))
    module = api()
    ds = module.CenterReferenceDataset(tmp_path, UPSTREAM, config(),
        module.reference_windows(tmp_path, UPSTREAM))
    with pytest.raises(ValueError, match='finite'):
        ds[0]


@requires_official_runtime
@pytest.mark.parametrize('filename', ['manifest.json', 'episode_index.dat', 'frame_split_id.dat'])
def test_changed_source_generation_is_rejected_after_construction(tmp_path, filename):
    fixture(tmp_path)
    module = api()
    ds = module.CenterReferenceDataset(tmp_path, UPSTREAM, config(),
        module.reference_windows(tmp_path, UPSTREAM))
    changed = tmp_path / filename
    changed.write_bytes(changed.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='chang|source|generation'):
        ds[0]

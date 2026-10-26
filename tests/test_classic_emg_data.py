"""Real small memmaps exercise the new official-literal classic data family."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from egoemg_repro.data import LegacyDataset, LegacyStore


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1] / 'upstream')).resolve()
W = 7790
POLICY = 'official_literal_loader'
NORM_SHA = 'b787a1ef98d48f434cb808e0e61e49c1ee0c09fae5b7ca9fc6a05ba4b14ee122'
VARIANTS = ('vemg2pose', 'neuropose', 'vemg2pose_with_aug', 'neuropose_with_aug')


def runtime():
    return importlib.import_module('egoemg_repro.classic_emg_data')


def recipe(variant='vemg2pose'):
    pytest.importorskip('hydra')
    from egoemg_repro.classic_emg import compose_classic
    return compose_classic(UPSTREAM, variant)


def torch_runtime():
    torch = pytest.importorskip('torch')
    pytest.importorskip('hydra')
    torch.set_num_threads(2)
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    return torch


def make_store(root, n=W, *, splits=None, signals=True):
    root.mkdir(exist_ok=True)
    splits = np.zeros(n, dtype=np.int8) if splits is None else np.asarray(splits, dtype=np.int8)
    arrays = {'frame_split_id': splits}
    if signals:
        arrays['generated_label_valid'] = np.ones((n, 2), dtype=bool)
        arrays['generated_label_valid'][1, 0] = False
        for h, hand in enumerate(('left', 'right')):
            arrays[f'emg_{hand}_filtered'] = ((np.arange(n, dtype=np.float32)[:, None] % 17)
                                                  + np.arange(8, dtype=np.float32)[None, :] * .25 + h * 30)
            arrays[f'generated_joint_angles_{hand}'] = np.broadcast_to(
                np.arange(20, dtype=np.float32)[None, :] * .01 + h, (n, 20)).copy()
            arrays[f'mocap_{hand}_wrist_pitch'] = np.full(n, 90 + h * 10, dtype=np.float32)
            arrays[f'mocap_{hand}_wrist_yaw'] = np.full(n, -180 + h * 10, dtype=np.float32)
            arrays[f'mocap_{hand}_wrist_angles_valid'] = np.ones(n, dtype=bool)
        arrays['mocap_left_wrist_angles_valid'][2] = False
    fields = {}
    for name, arr in arrays.items():
        filename = name + '.dat'
        arr.tofile(root / filename)
        fields[name] = {'filename': filename, 'dtype': str(arr.dtype), 'shape': list(arr.shape)}
    (root / 'manifest.json').write_text(json.dumps({'format_version': 'egoemg_v2_memmap', 'total_rows': n,
        'fields': fields, 'episode_fields': {}, 'frame_split_labels': ['train', 'user', 'gesture', 'both']}))
    np.savez(root / 'metadata.npz', episode_id=np.array(['ep'], dtype='S8'),
        episode_subject=np.array(['synthetic'], dtype='S12'), episode_subject_id=np.array([0]),
        episode_source_parquet=np.array(['synthetic.parquet'], dtype='S24'),
        episode_zed_video_path=np.array([''], dtype='S1'), episode_head_video_path=np.array([''], dtype='S1'),
        episode_start_idx=np.array([0]), episode_end_idx=np.array([n]), episode_length=np.array([n]),
        episode_beta_idx=np.array([0]), subjects_subject=np.array(['synthetic'], dtype='S12'),
        splits_split=np.array(['train', 'user', 'gesture', 'both'], dtype='S8'), splits_split_id=np.arange(4))
    return arrays


def window(*, role='train', split='train', start=0):
    return dict(episode=0, start=start, end=start + W, center=start + W // 2, role=role, split=split)


def dataset(root, variant='vemg2pose', role='val', windows=None, normalization=None, **kwargs):
    rt = runtime()
    return rt.ClassicDataset(root, windows or [window(role=role)], upstream=UPSTREAM, config=recipe(variant),
        normalization=normalization or rt.classic_normalization(UPSTREAM), role=role, mask_policy=POLICY, **kwargs)


def test_fixed_normalization_is_field_aware_pooled_and_policy_bound():
    norm = runtime().classic_normalization(UPSTREAM)
    assert norm['asset_sha256'] == NORM_SHA
    assert norm['key'] == 'egoemg__filtered'
    assert (norm['mean'], norm['std'], norm['epsilon']) == (.000832, 8.6551, 1e-6)
    assert norm['dataset_name'] == 'egoemg' and norm['field'] == 'filtered'
    assert norm['mask_policy'] == POLICY
    assert norm['estimated_from_project_train_rows'] is False
    assert 'path' not in norm and str(UPSTREAM) not in json.dumps(norm)


@pytest.mark.parametrize('variant,batch,channels', [('vemg2pose', 600, 8), ('neuropose', 256, 16)])
def test_published_base_recipe_is_admitted_against_resolved_official_recipe(variant, batch, channels):
    pytest.importorskip('hydra')
    cfg = runtime().load_classic_config(Path(__file__).parents[1] / 'configs' / (variant + '.json'), UPSTREAM)
    assert cfg['variant'] == variant and cfg['batch_size'] == batch
    assert (cfg['window_length'], cfg['train_stride'], cfg['eval_stride']) == (7790, 780, 7790)
    assert cfg['mask_policy'] == POLICY and cfg['normalization'] == runtime().classic_normalization(UPSTREAM)
    assert cfg['layout']['input_channels'] == channels
    assert cfg['reader_field_selection']['declared_explicit_fields'] is None
    assert cfg['reader_field_selection']['declared_modalities'] == ['emg', 'joint_angles', 'labels']
    assert cfg['reader_field_selection']['effective_filtered_fields'] == ['emg_left_filtered', 'emg_right_filtered']
    assert cfg['raw_loss_weights'] == {'mae': 1, 'landmark/fingertip': .01, 'fingertip_distance': .01}
    assert cfg['effective_loss_weights'] == {'mae': 1.0, 'landmark/fingertip': .02}
    assert (cfg['monitor'], cfg['early_stopping_patience'], cfg['max_epochs']) == ('val_loss', 50, 100)


@pytest.mark.parametrize('damage', ['extra', 'mask', 'norm', 'layout', 'loss', 'window', 'initializer'])
def test_recipe_changes_cannot_keep_the_official_data_identity(tmp_path, damage):
    pytest.importorskip('hydra')
    src = Path(__file__).parents[1] / 'configs/vemg2pose.json'
    cfg = json.loads(src.read_text())
    if damage == 'extra': cfg['extra'] = 'unknown'
    elif damage == 'mask': cfg['mask_policy'] = 'legacy_reproduction_angle_mask'
    elif damage == 'norm': cfg['normalization']['std'] = 12.275496274662686
    elif damage == 'layout': cfg['layout']['effective'] = 'emg2pose_interpolate16'
    elif damage == 'loss': del cfg['raw_loss_weights']['fingertip_distance']
    elif damage == 'window': cfg['window_length'] = 12000
    else: cfg['checkpoint'] = '/private/initializer.pt'
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='recipe|config'):
        runtime().load_classic_config(path, UPSTREAM)


@pytest.mark.parametrize('damage', ['mean', 'std', 'epsilon', 'asset_sha256', 'mask_policy'])
def test_bad_fixed_norm_is_rejected_before_data_access(tmp_path, damage):
    cfg = recipe()
    norm = runtime().classic_normalization(UPSTREAM)
    norm[damage] = 'bad' if damage in ('asset_sha256', 'mask_policy') else 1.0
    with pytest.raises(ValueError, match='normalization'):
        runtime().ClassicDataset(tmp_path / 'absent', [], upstream=UPSTREAM, config=cfg,
            normalization=norm, role='train', mask_policy=POLICY)


def test_unknown_mask_protocol_rejected_before_source_or_data_access(tmp_path):
    with pytest.raises(ValueError, match='mask'):
        runtime().ClassicDataset(tmp_path, [], upstream=tmp_path / 'absent', config=None,
            normalization={}, role='train', mask_policy='legacy_reproduction_angle_mask')


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('role', ['train', 'val', 'test'])
def test_emg_labels_and_generated_mask_match_actual_official_reader_bitwise(tmp_path, variant, role):
    torch = torch_runtime()
    from hydra.utils import instantiate
    from egoemg.transforms import Compose
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    cfg = recipe(variant)
    split = 'user' if role == 'test' else 'train'
    make_store(tmp_path, splits=np.full(W, 1 if role == 'test' else 0))
    ds = dataset(tmp_path, variant, role, [window(role=role, split=split)])
    assert len(ds) == 2
    for h, hand in enumerate(('left', 'right')):
        official = EgoEmgMemmapDataset(tmp_path, W, stride=780, target_hand=hand,
            modalities=['emg', 'joint_angles', 'labels'], emg_field_preference='filtered',
            fields=[f'emg_{hand}_filtered'],  # Explicitly repair the pinned modality omission, not its maths.
            emg_layout='target_hand' if variant.startswith('vemg') else 'emg2pose_interpolate16',
            emg2pose_channel_indices=[10, 12, 0, 1, 2, 4, 5, 6], channel_interpolate=False,
            norm_mode='per-dataset', norm_stats_path=str(UPSTREAM / 'assets/per_dataset_norm_stats.json'),
            transform=Compose([instantiate(t) for t in cfg.transforms[role]]))
        source = tmp_path / f'emg_{hand}_filtered.dat'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for seed in (0, 42, 123):
            np.random.seed(seed); torch.manual_seed(seed)
            expected = official[0]
            numpy_after, torch_after = np.random.get_state(), torch.get_rng_state()
            np.random.seed(seed); torch.manual_seed(seed)
            actual = ds[h]
            for key in ('emg', 'joint_angles', 'label_valid_mask'):
                value = actual[key].numpy()
                assert value.dtype == expected[key].dtype
                assert value.shape == expected[key].shape
                assert value.tobytes() == np.ascontiguousarray(expected[key]).tobytes(), key
            actual_after = np.random.get_state()
            assert numpy_after[0] == actual_after[0] and np.array_equal(numpy_after[1], actual_after[1])
            assert numpy_after[2:] == actual_after[2:] and torch.equal(torch_after, torch.get_rng_state())
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        assert actual['hand'] == hand and actual['subject'] == 'synthetic'
        assert actual['dataset_name'] == 'egoemg'
        assert actual['mask_policy'] == POLICY and 'angle_valid_mask' not in actual
        assert actual['role'] == role and actual['split'] == split
        runtime().admit_classic_sample(actual, ds.contract)
    # Wrist validity is deliberately false at row2 while generated validity is true.
    assert ds[0]['label_valid_mask'][2]


@pytest.mark.parametrize('variant', ['vemg2pose', 'neuropose'])
def test_original_modality_selection_really_omits_filtered_before_explicit_adapter(tmp_path, variant):
    torch_runtime()
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = recipe(variant)
    make_store(tmp_path)
    original = OmegaConf.create(OmegaConf.to_container(cfg.dataset.train[0], resolve=True))
    original.memmap_dir = str(tmp_path)
    official = instantiate(original, transform=None)
    with pytest.raises(KeyError, match='emg_left_filtered'):
        official[0]
    # The new loader makes this field selection decision explicit, while its
    # subsequent layout/norm/augmentation maths are checked against the reader.
    assert dataset(tmp_path, variant)[0]['emg'].shape == (8 if variant == 'vemg2pose' else 16, W)


def test_neuropose_zeros_are_filled_before_scalar_norm_and_not_left_zero(tmp_path):
    torch_runtime()
    make_store(tmp_path)
    sample = dataset(tmp_path, 'neuropose')[0]
    value = sample['emg'].numpy()
    zeros = sorted(set(range(16)) - {10, 12, 0, 1, 2, 4, 5, 6})
    expected = np.float32((np.zeros(1, dtype=np.float32) - .000832) / (8.6551 + 1e-6))[0]
    assert value[zeros, :].tobytes() == np.full((8, W), expected, np.float32).tobytes()
    assert float(expected) == pytest.approx(-9.612828609999269e-5)


@pytest.mark.parametrize('field', ['generated_joint_angles_left', 'emg_left_filtered', 'mocap_left_wrist_pitch'])
def test_literal_nonfinite_values_survive_until_explicit_admission_rejects(tmp_path, field):
    torch_runtime()
    arrays = make_store(tmp_path)
    arrays[field].flat[0] = np.nan
    arrays[field].tofile(tmp_path / (field + '.dat'))
    ds = dataset(tmp_path)
    sample = ds[0]
    key = 'emg' if field.startswith('emg') else 'joint_angles'
    assert np.isnan(sample[key].numpy()).any()
    assert sample['label_valid_mask'][0]
    with pytest.raises(ValueError, match='finite'):
        runtime().admit_classic_sample(sample, ds.contract)
    if field == 'generated_joint_angles_left':
        old = LegacyDataset(tmp_path, [window()])[0]
        assert old['joint_angles'][0, 0] == 0  # The new reader must avoid this cleanup branch.
        assert not old['angle_valid_mask'][0, 0]


def test_new_partition_grid_is_disjoint_and_fixed_norm_provenance_is_honest(tmp_path):
    split = np.repeat([0, 1, 2, 3], [8 * W, W, W, W])
    make_store(tmp_path, len(split), splits=split, signals=False)
    manifest = runtime().build_classic_partitions(LegacyStore(tmp_path), upstream=UPSTREAM)
    assert manifest['settings'] == dict(window_length=7790, train_stride=780, eval_stride=7790, val_fraction=.12, jitter=False)
    assert manifest['coverage']['train']['windows'] == 60
    assert manifest['coverage']['val']['windows'] == 1
    assert manifest['coverage']['test']['windows'] == 3
    assert manifest['coverage']['test']['by_split'] == {'user': 1, 'gesture': 1, 'both': 1}
    roles = {role: set() for role in ('train', 'val', 'test')}
    for w in manifest['windows']:
        roles[w['role']].update(range(w['start'], w['end']))
        assert w['end'] - w['start'] == W
        assert manifest['coverage'][w['role']]['hand_samples'] == 2 * manifest['coverage'][w['role']]['windows']
        if w['role'] == 'train': assert w['start'] % 780 == 0
        if w['role'] == 'test': assert (w['center'] - 3895) % W == 0
        else: assert w['split'] == 'train'
    assert not roles['train'] & roles['val'] and not roles['train'] & roles['test'] and not roles['val'] & roles['test']
    assert manifest['provenance']['mask_policy'] == POLICY
    assert manifest['provenance']['normalization'] == runtime().classic_normalization(UPSTREAM)
    original = copy.deepcopy(manifest)
    digest = original['provenance'].pop('partition_sha256')
    assert digest == hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def partitioned_store(root):
    split = np.repeat([0, 1, 2], [2 * W, W, W])
    make_store(root, len(split), splits=split)
    return runtime().build_classic_partitions(LegacyStore(root), upstream=UPSTREAM)


def test_rehashed_overlapping_partition_is_rejected_before_sample_reads(tmp_path):
    torch_runtime()
    partitions = partitioned_store(tmp_path)
    overlapping = next(copy.deepcopy(w) for w in partitions['windows'] if w['role'] == 'val')
    overlapping['role'] = 'train'
    partitions['windows'].append(overlapping)
    partitions['provenance'].pop('partition_sha256')
    partitions['provenance']['partition_sha256'] = hashlib.sha256(json.dumps(
        partitions, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with pytest.raises(ValueError, match='partition'):
        runtime().build_classic_datasets(tmp_path, partitions, upstream=UPSTREAM, config=recipe(),
            normalization=runtime().classic_normalization(UPSTREAM), mask_policy=POLICY)


def test_datasets_keep_three_roles_and_pair_hands_with_generated_mask(tmp_path):
    torch_runtime()
    partitions = partitioned_store(tmp_path)
    datasets = runtime().build_classic_datasets(tmp_path, partitions, upstream=UPSTREAM, config=recipe(),
        normalization=runtime().classic_normalization(UPSTREAM), mask_policy=POLICY)
    assert set(datasets) == {'train', 'val', 'test'}
    for role, ds in datasets.items():
        assert len(ds) == partitions['coverage'][role]['hand_samples']
        assert ds[0]['start'] == ds[1]['start'] and ds[0]['hand'] == 'left' and ds[1]['hand'] == 'right'


def test_identity_binds_actual_consumed_bytes_but_survives_data_root_move(tmp_path):
    pytest.importorskip('hydra')
    src = tmp_path / 'one'
    partitions = partitioned_store(src)
    kwargs = dict(upstream=UPSTREAM, config=recipe(), normalization=runtime().classic_normalization(UPSTREAM), mask_policy=POLICY)
    identity = runtime().classic_data_identity(src, partitions, **kwargs)
    moved = tmp_path / 'two'
    shutil.copytree(src, moved)
    assert runtime().classic_data_identity(moved, partitions, **kwargs) == identity
    assert str(tmp_path) not in json.dumps(identity)
    assert identity['mask_policy'] == POLICY and identity['normalization']['asset_sha256'] == NORM_SHA
    assert identity['reader_field_selection']['effective_filtered_fields'] == ['emg_left_filtered', 'emg_right_filtered']
    assert identity['fields']['generated_joint_angles_left']['shape'] == [4 * W, 20]
    with (moved / 'emg_left_filtered.dat').open('r+b') as stream:
        stream.write(np.float32(123).tobytes())
    changed = runtime().classic_data_identity(moved, partitions, **kwargs)
    assert changed['fields']['emg_left_filtered']['sha256'] != identity['fields']['emg_left_filtered']['sha256']
    assert changed['sha256'] != identity['sha256']


@pytest.mark.parametrize('damage', ['dtype', 'shape', 'symlink'])
def test_data_identity_rejects_wrong_schema_and_unsafe_field_path(tmp_path, damage):
    pytest.importorskip('hydra')
    partitions = partitioned_store(tmp_path)
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    field = manifest['fields']['emg_left_filtered']
    if damage == 'dtype': field['dtype'] = 'float16'
    elif damage == 'shape': field['shape'][1] = 16
    else:
        old = tmp_path / field['filename']
        relocated = tmp_path / 'relocated.dat'
        old.rename(relocated)
        old.symlink_to(relocated)
    path.write_text(json.dumps(manifest))
    # Rebuild to avoid making the manifest receipt mismatch the reason for failure.
    partitions = runtime().build_classic_partitions(LegacyStore(tmp_path), upstream=UPSTREAM)
    with pytest.raises(ValueError, match='schema|dtype|shape|safe|symlink'):
        runtime().classic_data_identity(tmp_path, partitions, upstream=UPSTREAM, config=recipe(),
            normalization=runtime().classic_normalization(UPSTREAM), mask_policy=POLICY)


@pytest.mark.parametrize('phase', ['after_partition', 'after_field_hash', 'before_final_admission'])
def test_identity_rejects_field_mutation_during_any_admission_phase(tmp_path, monkeypatch, phase):
    pytest.importorskip('hydra')
    rt = runtime()
    partitions = partitioned_store(tmp_path)
    def mutate():
        with (tmp_path / 'emg_left_filtered.dat').open('r+b') as stream:
            stream.write(np.float32(456).tobytes())
    if phase == 'after_partition':
        original = rt._partitions
        def intervening(*args, **kwargs):
            original(*args, **kwargs)
            mutate()
        monkeypatch.setattr(rt, '_partitions', intervening)
    elif phase == 'after_field_hash':
        original = rt.file_sha256
        def intervening(path):
            result = original(path)
            if Path(path).name == 'emg_left_filtered.dat': mutate()
            return result
        monkeypatch.setattr(rt, 'file_sha256', intervening)
    else:
        original = rt.classic_emg.resolved_classic_identity
        def intervening(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(rt.classic_emg, 'resolved_classic_identity', intervening)
    with pytest.raises(ValueError, match='changed'):
        rt.classic_data_identity(tmp_path, partitions, upstream=UPSTREAM, config=recipe(),
            normalization=rt.classic_normalization(UPSTREAM), mask_policy=POLICY)


@pytest.mark.parametrize('damage', ['dtype', 'shape', 'mask', 'angle_mask'])
def test_explicit_admission_rejects_schema_or_mask_protocol_drift(tmp_path, damage):
    torch = torch_runtime()
    make_store(tmp_path)
    ds = dataset(tmp_path)
    sample = ds[0]
    if damage == 'dtype': sample['joint_angles'] = sample['joint_angles'].double()
    elif damage == 'shape': sample['emg'] = sample['emg'][:, :-1]
    elif damage == 'mask': sample['label_valid_mask'] = sample['label_valid_mask'].to(torch.int8)
    else: sample['angle_valid_mask'] = torch.ones(22, W, dtype=torch.bool)
    with pytest.raises(ValueError, match='schema|protocol'):
        runtime().admit_classic_sample(sample, ds.contract)


def test_mutated_resolved_transforms_are_rejected_before_opening_dataset(tmp_path):
    cfg = recipe()
    cfg.transforms.train[1].channel_dim = 0
    with pytest.raises(ValueError, match='recipe'):
        runtime().ClassicDataset(tmp_path / 'absent', [], upstream=UPSTREAM, config=cfg,
            normalization=runtime().classic_normalization(UPSTREAM), role='train', mask_policy=POLICY)

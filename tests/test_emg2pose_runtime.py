"""Independent runtime admission and actual fixed EMG2Pose factory checks."""
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from egoemg_repro import emg2pose_data


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1] / 'upstream'))


def runtime():
    return importlib.import_module('egoemg_repro.emg2pose_runtime')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp('private-runtime-source')
    files, rows, arrays = [], [], {}
    count = 8570
    for name, split, generalization, side in [('train-a', 'train', 'none', 'left'),
                                              ('val-a', 'val', 'user', 'right'),
                                              ('test-a', 'test', 'user_stage', 'left')]:
        values = np.zeros(count, dtype=[('time', '<f8'), ('joint_angles', '<f4', (20,)),
                                        ('emg', '<f4', (16,))])
        values['time'] = 100 + np.arange(count, dtype=np.float64) / 2000
        values['joint_angles'] = np.linspace(.1, .9, 20, dtype=np.float32)
        values['emg'] = np.sin(np.arange(count * 16, dtype=np.float32) / 97).reshape(count, 16)
        row = dict(filename=name, session='same-original-session', user='private-runtime-user',
                   stage='private-stage', side=side, moving_hand='both', split=split,
                   generalization=generalization, held_out_user=generalization != 'none',
                   held_out_stage=generalization == 'user_stage', start=100, end=100 + count / 2000)
        path = root / (name + '.hdf5')
        with h5py.File(path, 'w') as file:
            group = file.create_group('emg2pose')
            group.create_dataset('timeseries', data=values)
            for key, value in row.items():
                group.attrs[key] = value
            group.attrs['sample_rate'] = 2000.0
            group.attrs['num_channels'] = 16
        files.append(path)
        rows.append(row)
        arrays[split] = values
    csv_path = root / 'metadata.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inventory = emg2pose_data.inspect_corpus(files, csv_path, cohort='engineering', upstream=UPSTREAM)
    source_receipt = root / 'original-inventory.json'
    dump(source_receipt, inventory)
    output = root / 'complete-data'
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(emg2pose_data.shutil, 'disk_usage', lambda path: SimpleNamespace(
            total=128 * 1024**3, used=64 * 1024**3, free=64 * 1024**3))
        emg2pose_data.convert_corpus(source_receipt, output, upstream=UPSTREAM)
    return SimpleNamespace(root=root, data=output, source_receipt=source_receipt,
                           metadata_csv=csv_path, arrays=arrays)


def config(corpus, size='small'):
    pytest.importorskip('hydra')
    return runtime().compose_config(UPSTREAM, size, corpus.data, source_receipt=corpus.source_receipt)


def framework():
    torch = pytest.importorskip('torch')
    pytest.importorskip('pytorch_lightning')
    pytest.importorskip('hydra')
    return torch


def test_completed_data_admission_binds_exact_receipts_without_private_identifiers(corpus):
    identity = runtime().validate_completed_data(UPSTREAM, corpus.data, corpus.source_receipt)
    assert identity['rows'] == 25710
    assert identity['input_channels'] == 16 and identity['angle_dimensions'] == 20
    assert identity['full_corpus_complete'] is False and identity['engineering_only'] is True
    assert identity['cohort'] == 'engineering'
    assert identity['present_splits'] == ['test', 'train', 'val']
    assert identity['source_receipt_sha256'] == digest(corpus.source_receipt)
    # The converter's embedded copy intentionally adds a newline. Verify the
    # original exact bytes and copied semantic content, never guess a new SHA.
    assert digest(corpus.data / 'source-receipt.json') != digest(corpus.source_receipt)
    text = json.dumps(identity)
    assert 'private-runtime-user' not in text and str(corpus.root) not in text


@pytest.mark.parametrize('damage', ['missing_completion', 'incomplete', 'wrong_format', 'fake_full',
                                   'legacy_dimensions', 'changed_field', 'changed_metadata',
                                   'forged_metadata_hash', 'unsafe_field', 'symlink_field',
                                   'object_sidecar', 'missing_snapshot', 'missing_time_dtype', 'float_index_sidecar'])
def test_incomplete_forged_legacy_or_changed_data_is_rejected(corpus, tmp_path, damage):
    data = tmp_path / 'damaged'
    shutil.copytree(corpus.data, data)
    receipt = json.loads((data / 'completion.json').read_text())
    manifest = json.loads((data / 'manifest.json').read_text())
    if damage == 'missing_completion':
        (data / 'completion.json').unlink()
    elif damage == 'incomplete':
        receipt['complete'] = False
    elif damage == 'wrong_format':
        receipt['format'] = 'legacy_egoemg_completed'
    elif damage == 'fake_full':
        receipt.update(cohort='full', full_corpus_complete=True, engineering_only=False)
    elif damage == 'legacy_dimensions':
        manifest['fields']['joint_angles']['shape'][1] = 22
        receipt['fields']['joint_angles']['shape'][1] = 22
        dump(data / 'manifest.json', manifest)
        receipt['manifest_sha256'] = digest(data / 'manifest.json')
    elif damage == 'missing_time_dtype':
        del manifest['fields']['time']['dtype']
        del receipt['fields']['time']['dtype']
        dump(data / 'manifest.json', manifest)
        receipt['manifest_sha256'] = digest(data / 'manifest.json')
    elif damage == 'changed_field':
        path = data / 'emg.dat'
        with path.open('r+b') as stream:
            stream.write(b'wrong raw bits')
    elif damage in {'changed_metadata', 'forged_metadata_hash', 'object_sidecar', 'float_index_sidecar'}:
        with np.load(data / 'metadata.npz', allow_pickle=False) as source:
            md = {key: source[key].copy() for key in source.files}
        if damage == 'object_sidecar':
            md['session_generalization'] = np.asarray(['none'] * 3, dtype=object)
        elif damage == 'float_index_sidecar':
            md['session_start_idx'] = md['session_start_idx'].astype(np.float64)
        else:
            md['session_split_id'][1] = md['session_split_id'][0]
        np.savez(data / 'metadata.npz', **md)
        if damage != 'changed_metadata':
            receipt['metadata_sha256'] = digest(data / 'metadata.npz')
    elif damage == 'unsafe_field':
        manifest['fields']['emg']['filename'] = '../emg.dat'
        receipt['fields']['emg']['filename'] = '../emg.dat'
        dump(data / 'manifest.json', manifest)
        receipt['manifest_sha256'] = digest(data / 'manifest.json')
    elif damage == 'symlink_field':
        (data / 'emg.dat').unlink()
        (data / 'emg.dat').symlink_to(corpus.data / 'emg.dat')
    else:
        (data / 'source-receipt.json').unlink()
    if damage != 'missing_completion':
        dump(data / 'completion.json', receipt)
    with pytest.raises(ValueError):
        runtime().validate_completed_data(UPSTREAM, data, corpus.source_receipt)


def test_original_receipt_bytes_must_match_conversion_hash(corpus, tmp_path):
    changed = tmp_path / 'different-formatting.json'
    changed.write_text(corpus.source_receipt.read_text() + '\n')
    with pytest.raises(ValueError, match='receipt|identity'):
        runtime().validate_completed_data(UPSTREAM, corpus.data, changed)


@pytest.mark.parametrize('filename', ['completion.json', 'source-receipt.json'])
def test_transaction_identity_cannot_change_during_runtime_admission(corpus, tmp_path, monkeypatch, filename):
    data = tmp_path / 'racing-data'
    shutil.copytree(corpus.data, data)
    original = emg2pose_data._sha256
    def racing(path):
        result = original(path)
        if Path(path) == data / 'emg.dat':
            content = json.loads((data / filename).read_text())
            content['complete'] = False
            dump(data / filename, content)
        return result
    monkeypatch.setattr(emg2pose_data, '_sha256', racing)
    with pytest.raises(ValueError, match='changed|identity|snapshot'):
        runtime().validate_completed_data(UPSTREAM, data, corpus.source_receipt)


@pytest.mark.parametrize('filename', ['emg.dat', 'joint_angles.dat', 'time.dat', 'valid_mask.dat'])
@pytest.mark.parametrize('when', ['after_field_hash', 'at_admission_end'])
def test_hashed_fields_must_remain_unchanged_until_admission_returns(corpus, tmp_path, monkeypatch, filename, when):
    data = tmp_path / 'racing-field-data'
    shutil.copytree(corpus.data, data)
    original = emg2pose_data._sha256
    metadata_hashes = 0
    def racing(path):
        nonlocal metadata_hashes
        result = original(path)
        if Path(path) == data / 'metadata.npz':
            metadata_hashes += 1
        if ((when == 'after_field_hash' and Path(path) == data / filename) or
                (when == 'at_admission_end' and Path(path) == data / 'metadata.npz' and metadata_hashes == 2)):
            with (data / filename).open('r+b') as stream:
                first = stream.read(1)
                stream.seek(0)
                stream.write(bytes([first[0] ^ 1]))
        return result
    monkeypatch.setattr(emg2pose_data, '_sha256', racing)
    with pytest.raises(ValueError, match='changed|stable|identity'):
        runtime().validate_completed_data(UPSTREAM, data, corpus.source_receipt)


@pytest.mark.parametrize('change', ['restore_mtime', 'replace_same_bytes', 'symlink_same_bytes'])
def test_field_identity_and_safe_path_are_rechecked_at_admission_end(corpus, tmp_path, monkeypatch, change):
    data = tmp_path / 'racing-field-identity'
    shutil.copytree(corpus.data, data)
    field = data / 'emg.dat'
    original = emg2pose_data._sha256
    metadata_hashes = 0
    def racing(path):
        nonlocal metadata_hashes
        result = original(path)
        if Path(path) == data / 'metadata.npz':
            metadata_hashes += 1
            if metadata_hashes == 2:
                before = field.stat()
                if change == 'restore_mtime':
                    with field.open('r+b') as stream:
                        first = stream.read(1)
                        stream.seek(0)
                        stream.write(bytes([first[0] ^ 1]))
                    os.utime(field, ns=(before.st_atime_ns, before.st_mtime_ns))
                else:
                    replacement = tmp_path / 'replacement.dat'
                    shutil.copy2(field, replacement)
                    if change == 'replace_same_bytes':
                        os.replace(replacement, field)
                    else:
                        field.unlink()
                        field.symlink_to(replacement)
        return result
    monkeypatch.setattr(emg2pose_data, '_sha256', racing)
    with pytest.raises(ValueError, match='changed|stable|identity|symlink'):
        runtime().validate_completed_data(UPSTREAM, data, corpus.source_receipt)


def test_relocated_data_and_receipt_keep_equal_content_identity(corpus, tmp_path):
    data = tmp_path / 'relocated'
    shutil.copytree(corpus.data, data)
    source = tmp_path / 'moved-original-receipt.json'
    shutil.copyfile(corpus.source_receipt, source)
    metadata = tmp_path / 'moved-metadata.csv'
    shutil.copyfile(corpus.metadata_csv, metadata)
    first = runtime().validate_completed_data(UPSTREAM, corpus.data, corpus.source_receipt)
    second = runtime().validate_completed_data(UPSTREAM, data, source, metadata_csv=metadata)
    assert first == second
    pytest.importorskip('hydra')
    a = config(corpus)
    b = runtime().compose_config(UPSTREAM, 'small', data, source_receipt=source, metadata_csv=metadata)
    assert runtime().runtime_identity(UPSTREAM, a, corpus.source_receipt) == runtime().runtime_identity(UPSTREAM, b, source)


@pytest.mark.parametrize(('size', 'stride'), [('small', 3900), ('middle', 780), ('large', 780)])
def test_official_recipe_and_partition_paths_are_exact(corpus, size, stride):
    cfg = config(corpus, size)
    assert cfg.module.out_channels == cfg.module.head.out_channels == 20
    assert cfg.module.featurizer.conv_blocks[0].in_channels == 16
    assert cfg.datamodule.window_length == cfg.datamodule.val_test_window_length == 7790
    assert cfg.datamodule.stride == stride and cfg.datamodule.val_test_stride == 7790
    assert cfg.trainer.max_epochs == 250 and cfg.batch_size == 320
    assert cfg.optimizer.lr == .001 and cfg.optimizer.weight_decay == .01
    assert cfg.monitor_metric == 'val_mae' and cfg.monitor_mode == 'min'
    for role in ('train', 'val', 'test'):
        entry = cfg.dataset[role][0]
        assert entry.allowed_splits == [role]
        assert entry.memmap_dir == str(corpus.data)
        assert entry.jitter == (role == 'train')
        assert entry.skip_ik_failures is False


@pytest.mark.parametrize('damage', ['out_channels', 'input_channels', 'lr', 'initializer', 'wrong_path'])
def test_config_changes_fail_before_constructing_an_unapproved_runtime(corpus, damage):
    cfg = config(corpus)
    if damage == 'out_channels':
        cfg.module.out_channels = 22
    elif damage == 'input_channels':
        cfg.module.featurizer.conv_blocks[0].in_channels = 8
    elif damage == 'lr':
        cfg.optimizer.lr = .2
    elif damage == 'initializer':
        cfg.pretrained_checkpoint = '/unapproved/private/checkpoint.pt'
    else:
        cfg.dataset.val[0].memmap_dir = '/unapproved/data'
    with pytest.raises(ValueError):
        runtime().build_runtime(UPSTREAM, cfg)


@pytest.mark.parametrize(('size', 'parameters'), [('small', 3481636), ('middle', 6638884), ('large', 16342308)])
def test_three_real_official_modules_forward_align_and_partition_datasets(corpus, size, parameters):
    torch = framework()
    cfg = config(corpus, size)
    model, dm = runtime().build_runtime(UPSTREAM, cfg)
    assert type(model).__module__ == 'egoemg.lightning'
    assert sum(p.numel() for p in model.model.parameters()) == parameters
    assert (model.model.left_context, model.model.right_context, model.model.out_channels) == (510, 0, 20)
    dm.setup()
    assert (len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset)) == ((1 if size == 'small' else 2), 1, 1)
    assert dm.train_dataset.jitter is True and dm.val_dataset.jitter is False
    assert {dm.train_dataset[0]['session_idx'], dm.val_dataset[0]['session_idx'], dm.test_dataset[0]['session_idx']} == {0, 1, 2}
    batch = dm._collate_fn([dm.val_dataset[0]])
    assert tuple(batch['emg'].shape) == (1, 16, 7790)
    model.eval()
    with torch.no_grad():
        prediction, target, mask = model(batch)
    assert tuple(prediction.shape) == tuple(target.shape) == (1, 20, 7280)
    assert tuple(mask.shape) == (1, 7280)
    assert torch.isfinite(prediction).all()
    assert torch.equal(target, batch['joint_angles'][..., 510:])


def test_actual_eval_normalization_and_raw_augmentation_order(corpus):
    torch = framework()
    cfg = config(corpus)
    _, dm = runtime().build_runtime(UPSTREAM, cfg)
    dm.setup()
    validation = dm.val_dataset[0]
    expected = (torch.from_numpy(corpus.arrays['val']['emg'][:7790].copy()) / (18.49 + 1e-6)).numpy().T
    assert validation['emg'].tobytes() == expected.tobytes()
    names = [type(t).__name__ for t in dm.train_transforms.transforms]
    assert names == ['ExtractField', 'RotationAugmentation', 'ToFloatTensor', 'RandomChannelMask',
                     'RandomTimeMask', 'RandomFrequencyMask', 'RandomGaussianNoise']
    assert dm.train_transforms.transforms[3].mask_prob == .25
    assert dm.train_transforms.transforms[5].num_masks == 3
    assert dm.train_transforms.transforms[6].apply_prob == .5
    dm.train_dataset.jitter = False  # Isolate transform order from random window offset.
    np.random.seed(71)
    torch.manual_seed(71)
    sample = dm.train_dataset[0]
    np.random.seed(71)
    torch.manual_seed(71)
    raw = corpus.arrays['train']['emg'][:7790].copy()
    augmented = dm.train_transforms({'emg': raw})
    expected_train = (torch.as_tensor(augmented) / (18.49 + 1e-6)).numpy().T
    assert sample['emg'].tobytes() == expected_train.tobytes()


def test_seed_is_applied_before_actual_model_initialization(corpus):
    torch = framework()
    cfg = config(corpus)
    torch.manual_seed(12)
    first, _ = runtime().build_runtime(UPSTREAM, cfg)
    expected = {k: v.clone() for k, v in first.state_dict().items()}
    del first
    torch.manual_seed(987)
    second, _ = runtime().build_runtime(UPSTREAM, cfg)
    assert all(torch.equal(expected[k], v) for k, v in second.state_dict().items())


def test_actual_strict_offline_reload_preserves_predictions(corpus, tmp_path, monkeypatch):
    torch = framework()
    cfg = config(corpus)
    module, dm = runtime().build_runtime(UPSTREAM, cfg)
    dm.setup('validate')
    batch = dm._collate_fn([dm.val_dataset[0]])
    module.eval()
    with torch.no_grad():
        expected = module(batch)[0].clone()
    identity = runtime().runtime_identity(UPSTREAM, cfg, corpus.source_receipt)
    checkpoint = tmp_path / 'engineering-random-weights.pt'
    torch.save({'identity': identity, 'state_dict': module.state_dict()}, checkpoint)
    del module
    def offline(*args, **kwargs):
        raise AssertionError('offline runtime attempted a network connection')
    monkeypatch.setattr('socket.socket.connect', offline)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    restored, _ = runtime().reload_runtime(UPSTREAM, cfg, saved['state_dict'], identity=saved['identity'])
    restored.eval()
    with torch.no_grad():
        assert torch.equal(expected, restored(batch)[0])


@pytest.mark.parametrize('damage', ['missing_key', 'wrong_head_shape', 'wrong_identity', 'wrong_tensor_dtype'])
def test_reload_rejects_partial_wrong_shape_or_wrong_provenance(corpus, damage):
    torch = framework()
    cfg = config(corpus)
    model, _ = runtime().build_runtime(UPSTREAM, cfg)
    state = model.state_dict()
    identity = runtime().runtime_identity(UPSTREAM, cfg, corpus.source_receipt)
    if damage == 'missing_key':
        del state[next(iter(state))]
    elif damage == 'wrong_head_shape':
        key = next(k for k, v in state.items() if v.ndim == 2 and v.shape == (20, 512))
        state[key] = torch.zeros(22, 512)
    elif damage == 'wrong_tensor_dtype':
        key = next(k for k, v in state.items() if v.dtype == torch.float32)
        state[key] = state[key].double()
    else:
        identity['config_sha256'] = '0' * 64
    with pytest.raises((ValueError, RuntimeError)):
        runtime().reload_runtime(UPSTREAM, cfg, state, identity=identity)

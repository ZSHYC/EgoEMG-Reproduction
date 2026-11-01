"""Full held-out target IO, never a synthetic full-release experiment."""
import copy
import hashlib
import importlib
from pathlib import Path
import pickle
import random

import numpy as np
import pytest

from test_emgformer_data import UPSTREAM, W, cfg
from test_emg_initializer_data import source
from test_emgformer_cohort import write_value


def api():
    try:
        return importlib.import_module('egoemg_repro.emgformer_trajectory_data')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.emgformer_trajectory_data':
            pytest.fail('native full-trajectory input admission missing')
        raise


@pytest.fixture
def files(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('pytorch_lightning')
    from egoemg_repro.emgformer_data import emgformer_data_contract
    source(tmp_path, optional_labels=True)
    config = cfg()
    return tmp_path, config, emgformer_data_contract(UPSTREAM, config)['normalization']


def admit(files, **kwargs):
    root, config, norm = files
    return api().admit_trajectory_inputs(root, upstream=UPSTREAM, config=config,
        normalization=norm, **kwargs)


@pytest.mark.parametrize('role', ['val', 'test'])
@pytest.mark.parametrize('hand', ['left', 'right'])
def test_original_full_reader_targets_masks_emg_and_point_parent_preserved(files, role, hand):
    m = api(); inputs = admit(files); ds = inputs.datasets[role]
    index = 0 if hand == 'left' else len(ds.reader.windows)
    got = ds[index]; point = inputs.point_cohort.datasets[role][index]
    window = ds.reader.windows[0]; lo, hi = window['start'], window['end']; store = ds.reader.store
    expected = np.concatenate([np.asarray(store.field('generated_joint_angles_'+hand)[lo:hi]).T,
        np.stack([np.deg2rad(np.asarray(store.field('mocap_'+hand+'_wrist_'+axis)[lo:hi], np.float32))
            for axis in ('pitch', 'yaw')])])
    assert got['joint_angles'].shape == (22, W) and got['joint_angles'].dtype == np.float32
    assert got['joint_angles'].tobytes() == expected.tobytes()
    assert got['label_valid_mask'].shape == (W,) and got['label_valid_mask'].dtype == np.bool_
    assert got['label_valid_mask'].tobytes() == store.field('generated_label_valid')[lo:hi, int(hand == 'right')].tobytes()
    assert got['emg'].tobytes() == point['emg'].tobytes()
    for name in ('start', 'end', 'center', 'episode', 'hand', 'role', 'split', 'emgformer_sample_index'):
        assert got[name] == point[name]
    assert point['joint_angles'].shape == (22, 1)
    assert inputs.point_cohort.datasets[role].views[0]._center_target_only is True
    assert ds.reader.views[0]._center_target_only is False
    inputs.check()


def test_complete_bytes_all_new_target_rows_rng_and_no_model_permit(files, monkeypatch):
    import torch
    def deny(*a, **kw): raise AssertionError('data admission must not construct a model or optimizer')
    monkeypatch.setattr(torch.nn.Module, '__init__', deny)
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', deny)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files[0].iterdir() if p.is_file()}
    rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    inputs = admit(files); identity = inputs.identity
    assert identity['format'] == 'original_emgformer_full_heldout_trajectory_inputs_v1'
    assert identity['counts'] == dict(val=4, test=8)
    assert identity['payload_file_count'] == 16 and len(identity['payload_files_sha256']) == 16
    assert identity['point_cohort_sha256'] == inputs.point_cohort.identity['sha256']
    assert identity['payload_files_sha256'] == inputs.point_cohort.identity['payload_files_sha256']
    for role, rows in [('val', 24000), ('test', 48000)]:
        scan = identity['finite_heldout_targets'][role]
        assert scan['unique_target_rows'] == rows and scan['consumed_target_elements'] == rows*44
        assert scan['nonfinite_target_entries'] == scan['nonfinite_converted_wrist_entries'] == 0
        assert scan['target_mode'] == 'full_trajectory'
    for flag in ('metadata_matches_registered_legacy_release', 'scientific_eligible', 'full_fit_or_evaluation_permit', 'full_paper_complete'):
        assert identity[flag] is False
    assert torch.equal(rng[0], torch.get_rng_state()) and rng[2] == random.getstate()
    after = np.random.get_state()
    assert np.array_equal(rng[1][1], after[1]) and rng[1][2:] == after[2:]
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files[0].iterdir() if p.is_file()}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('role', ['val', 'test'])
@pytest.mark.parametrize('edge', ['first_noncenter', 'last'])
@pytest.mark.parametrize('field', ['generated_joint_angles_left', 'mocap_right_wrist_pitch'])
def test_masked_offcenter_and_tail_nonfinite_targets_reject(files, role, edge, field):
    m = api()
    from egoemg_repro.emgformer_data import build_emgformer_partitions
    from egoemg_repro.data import LegacyStore
    partitions = build_emgformer_partitions(LegacyStore(files[0]))
    window = next(w for w in partitions['windows'] if w['role'] == role)
    row = window['start']+1 if edge == 'first_noncenter' else window['end']-1
    write_value(files[0], field, row, np.nan)
    mask = np.memmap(files[0]/'generated_label_valid.dat', dtype=bool, mode='r+', shape=(24*W, 2))
    mask[row] = False; mask.flush(); del mask
    with pytest.raises(ValueError, match='nonfinite'):
        admit(files)


@pytest.mark.parametrize('damage', ['point_receipt', 'trajectory_receipt'])
def test_resealed_wrong_receipts_reject_after_actual_complete_read(files, damage):
    from egoemg_repro.classic_emg import _hash
    inputs = admit(files)
    value = copy.deepcopy(inputs.point_cohort.identity if damage == 'point_receipt' else inputs.identity)
    value['payload_files_sha256']['field:emg_right_filtered'] = '0'*64
    value['sha256'] = _hash({k: v for k, v in value.items() if k != 'sha256'})
    kwargs = {'expected_point_identity': value} if damage == 'point_receipt' else {'expected': value}
    with pytest.raises(ValueError, match='identity|receipt|content'):
        admit(files, **kwargs)


@pytest.mark.parametrize('damage', ['center', 'norm', 'stride', 'windows', 'config', 'identity', 'roles', 'unregistered_copy', 'parent_config', 'parent_reader'])
def test_full_reader_and_admission_computations_remain_bound(files, damage):
    m = api(); inputs = admit(files); ds = inputs.datasets['test']; view = ds.reader.views[1]
    if damage == 'center': view._center_target_only = True
    elif damage == 'norm': view._emg_std = view._emg_std.astype(np.float64)
    elif damage == 'stride': view._frame_memmaps['generated_joint_angles_right'].strides = (0, 0)
    elif damage == 'windows': ds.reader.windows[0]['start'] += 1
    elif damage == 'config': ds.reader.config.optimizer.lr = 1.
    elif damage == 'identity': inputs.identity['counts']['test'] += 2
    elif damage == 'roles': object.__setattr__(inputs, 'datasets', {'val': ds, 'test': ds})
    elif damage == 'parent_config': inputs.point_cohort.config.optimizer.lr = 1.
    elif damage == 'parent_reader': inputs.point_cohort.datasets['val'].views[0]._emg_std *= 2
    else: inputs = copy.copy(inputs)
    with pytest.raises(ValueError, match='bound|binding|changed|registered|mapping'):
        inputs.check()


def test_pickle_rebuilds_full_views_and_rejects_changed_generation(files):
    inputs = admit(files); ds = inputs.datasets['test']; serialized = pickle.dumps(ds)
    restored = pickle.loads(serialized)
    assert len(serialized) < 1024**2 and restored.reader is not ds.reader
    for index in (0, len(ds)//2, len(ds)-1):
        a, b = ds[index], restored[index]
        for name in ('emg', 'emg_channel_mask', 'joint_angles', 'label_valid_mask'):
            assert a[name].shape == b[name].shape and a[name].dtype == b[name].dtype
            assert a[name].tobytes() == b[name].tobytes()
    write_value(files[0], 'generated_joint_angles_left', 287999, .123)
    with pytest.raises(ValueError, match='generation|stamp|changed'):
        pickle.loads(serialized)


def test_literal_loader_and_explicit_engineering_batch_policy(files):
    m = api(); inputs = admit(files)
    loader = m.trajectory_loader(inputs, 'test')
    assert loader.batch_size == 500 and loader.num_workers == 6 and loader.prefetch_factor == 1
    assert loader.persistent_workers and not loader.pin_memory and not loader.drop_last
    assert loader.generator is None and type(loader.sampler).__name__ == 'SequentialSampler'
    engineering = m.trajectory_loader(inputs, 'test', cpu_engineering=True)
    batch = next(iter(engineering))
    assert engineering.batch_size == 2 and engineering.num_workers == 0
    assert batch['joint_angles'].shape == (2, 22, W)
    with pytest.raises(ValueError): m.trajectory_loader(inputs, 'train')
    with pytest.raises(ValueError): m.trajectory_loader(inputs, 'test', cpu_engineering=1)


def test_changed_imported_new_source_is_rejected(files):
    m = api(); inputs = admit(files); path = Path(m.__file__); original = path.read_bytes()
    try:
        path.write_bytes(original+b'\n# isolated test capsule generation change\n')
        with pytest.raises(ValueError, match='source|stamp|changed'):
            inputs.check()
    finally:
        path.write_bytes(original)

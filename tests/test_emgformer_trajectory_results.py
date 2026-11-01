"""Engineering native-forward alignment/raw packets, never full-source results."""
import copy
import importlib
import math

import numpy as np
import pytest
import torch

from test_emgformer_literal import UPSTREAM

W = 12000


def api():
    try: return importlib.import_module('egoemg_repro.emgformer_trajectory_results')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.emgformer_trajectory_results':
            pytest.fail('native full-trajectory capture, raw packets and statistics missing')
        raise


def toy(dtype=torch.float32, *, right=2):
    """Original Lightning forward and alignment around an explicit toy head."""
    from egoemg_repro.classic_emg import _activate
    _activate(UPSTREAM)
    from egoemg.lightning import EmgPredictionModule
    from egoemg.models.modules.base import BaseModule
    class Model(torch.nn.Module):
        align_predictions = BaseModule.align_predictions
        align_mask = BaseModule.align_mask
        def __init__(self):
            super().__init__(); self.head = torch.nn.Identity()
            self.left_context = W-13; self.right_context = right
            self.center_supervised = self.supervise_at_prediction_rate = False
            self.register_buffer('values', torch.arange(66).reshape(1, 22, 3).to(dtype)/10)
        def forward(self, batch):
            return self.head(self.values.expand(len(batch['emg']), -1, -1))
    class Wrapper(torch.nn.Module):
        forward = EmgPredictionModule.forward
        def __init__(self):
            super().__init__(); self.model = Model(); self.task_type = 'regression'; self.ignore_head_tail_dims = 0
    return Wrapper().eval()


def batch(n=2):
    return dict(emg=torch.zeros(n, 8, W),
        joint_angles=torch.arange(n*22*W, dtype=torch.float32).reshape(n, 22, W)/100000,
        label_valid_mask=(torch.arange(W)[None].expand(n, -1)%3 != 0).clone(),
        emg_channel_mask=torch.ones(n, 8, dtype=torch.bool))


@pytest.mark.parametrize('right', [0, 2])
def test_same_original_forward_context_alignment_and_head_witness(right):
    m = api(); module = toy(right=right); source = batch(); state = {n: x.clone() for n, x in module.state_dict().items()}
    expected = module(source)
    got = m.capture_native_trajectory(module, source, upstream=UPSTREAM)
    assert got['actual_model_forwards'] == 1 and got['same_forward_original_alignment_exact']
    for name, value in zip(('prediction', 'target', 'mask'), expected):
        assert torch.equal(got['arrays'][name], value) and got['arrays'][name].dtype == value.dtype
    assert torch.equal(got['arrays']['raw_prediction'], module.model.values.expand(2, -1, -1))
    assert torch.equal(got['arrays']['source_target'], source['joint_angles'])
    assert torch.equal(got['arrays']['source_mask'], source['label_valid_mask'])
    a = got['alignment']; assert a['left_context'] == W-13 and a['right_context'] == right
    assert a['aligned_prediction_time_steps'] == 13-right and a['raw_prediction_time_steps'] == 3
    assert a['aligned_center_index'] == (13-right)//2
    assert a['source_aligned_center_index'] == W-13+(13-right)//2
    assert a['source_aligned_center_index'] != W//2
    assert not module.model.head._forward_hooks
    assert all(torch.equal(x, state[n]) for n, x in module.state_dict().items())


@pytest.mark.parametrize('damage', ['training', 'point_target', 'prediction_rate', 'wrong_tuple', 'exception'])
def test_bad_full_forward_and_late_failure_do_not_leave_hooks(damage):
    m = api(); module = toy(); source = batch()
    if damage == 'training': module.train()
    elif damage == 'point_target': source['joint_angles'] = source['joint_angles'][..., :1]
    elif damage == 'prediction_rate': module.model.supervise_at_prediction_rate = True
    elif damage == 'wrong_tuple':
        original = module.forward
        def wrong(b):
            p, t, mask = original(b); return p, t+1, mask
        module.forward = wrong
    else:
        def broken(x): raise RuntimeError('engineering head exception')
        module.model.head.forward = broken
    with pytest.raises((ValueError, RuntimeError)):
        m.capture_native_trajectory(module, source, upstream=UPSTREAM)
    assert not module.model.head._forward_hooks


def metadata(start, n, samples=6, role='test'):
    indexes = list(range(start, start+n)); windows = [i%(samples//2) for i in indexes]
    return dict(index=indexes, unique=indexes.copy(), repeat=[0]*n,
        start=[i*W for i in windows], end=[(i+1)*W for i in windows],
        center=[i*W+W//2 for i in windows], episode=[0]*n,
        hand=['left' if i < samples//2 else 'right' for i in indexes],
        split=[('user', 'gesture', 'both')[i%3] if role == 'test' else 'train' for i in windows],
        user=['engineering_user_'+str(i) for i in windows])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_lossless_typed_raw_packet_roundtrip_including_bfloat16(tmp_path, dtype):
    m = api(); captured = m.capture_native_trajectory(toy(dtype), batch(), upstream=UPSTREAM)
    record = m.write_trajectory_packet(tmp_path, 0, arrays=captured['arrays'], alignment=captured['alignment'],
        metadata=metadata(0, 2), input_sha256='a'*64, sample_input_sha256=['b'*64]*2, upstream=UPSTREAM)
    got = m.read_trajectory_packet(tmp_path, record, upstream=UPSTREAM)
    for name, value in captured['arrays'].items():
        assert torch.equal(value, got['arrays'][name]) and value.dtype == got['arrays'][name].dtype
    assert got['alignment'] == captured['alignment'] and got['metadata'] == metadata(0, 2)
    assert got['arrays']['raw_prediction'].dtype == dtype
    assert got['arrays']['prediction'].dtype == dtype


@pytest.mark.parametrize('damage', ['file', 'rehashed_file', 'unsafe_name', 'source_target', 'mask_dtype'])
def test_raw_corruption_or_invalid_original_target_schema_rejects(tmp_path, damage):
    m = api(); captured = m.capture_native_trajectory(toy(), batch(), upstream=UPSTREAM)
    arrays = {k: v.clone() for k, v in captured['arrays'].items()}
    if damage == 'source_target': arrays['source_target'][0, 0, -1] = float('nan')
    elif damage == 'mask_dtype': arrays['mask'] = arrays['mask'].float()
    if damage in ('source_target', 'mask_dtype'):
        with pytest.raises(ValueError):
            m.write_trajectory_packet(tmp_path, 0, arrays=arrays, alignment=captured['alignment'],
                metadata=metadata(0, 2), input_sha256='a'*64, sample_input_sha256=['b'*64]*2, upstream=UPSTREAM)
        return
    record = m.write_trajectory_packet(tmp_path, 0, arrays=arrays, alignment=captured['alignment'],
        metadata=metadata(0, 2), input_sha256='a'*64, sample_input_sha256=['b'*64]*2, upstream=UPSTREAM)
    if damage == 'unsafe_name': record['file'] = '../escaped.pt'
    else:
        path = tmp_path/record['file']; packet = torch.load(path, weights_only=True)
        packet['prediction'][0, 0, 0] += 1; torch.save(packet, path)
        if damage == 'rehashed_file': record['file_sha256'] = m._sha(path)
    record = m._seal({k: v for k, v in record.items() if k != 'sha256'})
    with pytest.raises(ValueError): m.read_trajectory_packet(tmp_path, record, upstream=UPSTREAM)


def engineering_packets(role='test'):
    """Arbitrary engineering errors; these are not trained full-source passes."""
    m = api(); captured = m.capture_native_trajectory(toy(), batch(6), upstream=UPSTREAM)
    arrays = {k: v.clone() for k, v in captured['arrays'].items()}
    arrays['source_target'].zero_(); arrays['target'].zero_()
    time = arrays['prediction'].shape[-1]
    arrays['prediction'] = torch.arange(6*22*time, dtype=torch.float32).reshape(6, 22, time)/100
    arrays['source_mask'].zero_(); lo = captured['alignment']['left_context']
    for i, count in enumerate([1, 0, 7, 11, 2, 3]): arrays['source_mask'][i, lo:lo+count] = True
    arrays['mask'] = arrays['source_mask'][:, lo:W-2].clone()
    return [dict(arrays={k: v[start:start+n].clone() for k, v in arrays.items()},
        alignment=captured['alignment'], metadata=metadata(start, n, role=role),
        input_sha256='c'*64, sample_input_sha256=['d'*64]*n) for start, n in ((0, 4), (4, 2))]


@pytest.mark.parametrize('role', ['val', 'test'])
def test_full_trajectory_and_aligned_center_statistics_conserve_with_tail(role):
    m = api(); parts = engineering_packets(role)
    report = m.trajectory_statistics(iter(parts), samples=6, batch_size=4, role=role, upstream=UPSTREAM)
    assert report['consumed_samples'] == report['unique_samples'] == 6 and report['duplicate_consumptions'] == 0
    assert report['batch_sizes'] == [4, 2]
    weighted = torch.tensor(0., dtype=torch.float32)
    for part in parts:
        a = part['arrays']; mask = a['mask'][:, None].expand_as(a['target'])
        mean = (a['prediction'].float()-a['target']).abs()[mask].mean() if mask.any() else a['prediction'].sum()*0
        weighted += mean*len(a['prediction'])
    assert report['native_batch_weighted_mae_rad'] == float(weighted/6)
    for mode in ('trajectory', 'aligned_center'):
        error = 0.; count = 0
        for part in parts:
            a = part['arrays']; p, t, valid = a['prediction'].double().numpy(), a['target'].double().numpy(), a['mask'].numpy()
            if mode == 'aligned_center':
                center = report['alignment']['aligned_center_index']; p, t, valid = p[..., center:center+1], t[..., center:center+1], valid[..., center:center+1]
            mask = np.broadcast_to(valid[:, None], p.shape); error += float(np.abs(p-t)[mask].sum()); count += int(mask.sum())
        stats = report[mode]; assert stats['overall']['valid_angle_count'] == count
        assert math.isclose(stats['overall']['absolute_error_sum_rad'], error, rel_tol=1e-12)
        assert sum(x['valid_angle_count'] for x in stats['by_joint']) == count
        assert math.isclose(sum(x['absolute_error_sum_rad'] for x in stats['by_joint']), error, rel_tol=1e-12)
        assert set(stats['by_finger']) == {'thumb', 'index', 'middle', 'ring', 'pinky'}
        assert set(stats['by_pd_group']) == {'proximal', 'mid', 'distal'} and len(stats['by_hand']) == 2
        assert len(stats['by_joint_group']) == 10
        assert stats['by_joint_group']['wrist_FE']['valid_angle_count'] == stats['by_joint'][20]['valid_angle_count']
    assert not math.isclose(report['native_batch_weighted_mae_rad'], report['trajectory']['overall']['mae_deg']*math.pi/180)


def test_all_invalid_batch_keeps_native_zero_and_the_complete_population():
    m = api(); parts = engineering_packets()
    parts[0]['arrays']['mask'].zero_(); parts[0]['arrays']['source_mask'].zero_()
    report = m.trajectory_statistics(iter(parts), samples=6, batch_size=4, role='test', upstream=UPSTREAM)
    a = parts[1]['arrays']; valid = a['mask'][:, None].expand_as(a['target'])
    expected = (a['prediction']-a['target']).abs()[valid].mean()*2/6
    assert report['native_batch_weighted_mae_rad'] == float(expected)
    assert report['consumed_samples'] == 6 and report['trajectory']['overall']['valid_angle_count'] == 22*5


@pytest.mark.parametrize('damage', ['missing_tail', 'duplicate_index', 'wrong_hand', 'unpaired_window'])
def test_partial_or_falsified_trajectory_population_rejects(damage):
    m = api(); parts = engineering_packets()
    if damage == 'missing_tail': parts.pop()
    elif damage == 'duplicate_index': parts[1]['metadata']['index'][0] = 0
    elif damage == 'wrong_hand': parts[0]['metadata']['hand'][0] = 'right'
    else:
        parts[1]['metadata']['start'][0] += W; parts[1]['metadata']['end'][0] += W; parts[1]['metadata']['center'][0] += W
    with pytest.raises(ValueError): m.trajectory_statistics(iter(parts), samples=6, batch_size=4, role='test', upstream=UPSTREAM)

"""Control source and raw transport; fixtures do not grant a full-run permit."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="native tensor/runtime checks require the optional Torch tier")

ROOT = Path(__file__).resolve().parents[1]
ORDER = [(e,v) for e in ('aligned','normalized_zero','half_window_roll') for v in ('native','missing')]


def api():
    try: return importlib.import_module('egoemg_repro.modality_control_run')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.modality_control_run':
            pytest.fail('complete source-bound control workflow missing')
        raise


def key(i):
    return dict(sequence=i, split='user', hand='left', ep_idx=0, center_idx=3895+i,
        window_start=i, window_end=7790+i, group_index=0, native_index=i)


def packets():
    target = np.arange(66, dtype=np.float32).reshape(3,22)/17
    fusion = np.stack([target+np.float32(.25+c*.1)for c in range(6)])
    vision = np.stack([target+np.float32(-.7 if c%2==0 else .9)for c in range(6)])
    crop = np.array([[False],[True],[False]])
    arrays = dict(fusion_prediction=fusion, vision_prediction=vision, targets=target,
        label_valid_mask=np.array([[False],[True],[False]]), natural_crop_valid_mask=crop,
        condition_vision_valid_mask=np.stack([crop if c%2==0 else np.zeros_like(crop)for c in range(6)]))
    for role in ('fusion','vision'):
        arrays['mae_'+role+'_rad'] = np.array([[float((torch.from_numpy(arrays[role+'_prediction'][c,i])
            -torch.from_numpy(target[i])).abs().mean())for i in range(3)]for c in range(6)],np.float64)
    baseline = dict(fusion_prediction=fusion[0].copy(), vision_prediction=vision[0].copy(),
        emg_prediction=np.zeros((3,22),np.float32), emg_full_prediction=np.zeros((3,22,146),np.float32),
        targets=target.copy(), label_valid_mask=arrays['label_valid_mask'].copy(), crop_valid_mask=crop.copy())
    for role in ('fusion','vision','emg'):
        baseline['mae_'+role+'_rad'] = np.array([float((torch.from_numpy(baseline[role+'_prediction'][i])
            -torch.from_numpy(target[i])).abs().mean())for i in range(3)],np.float64)
    return arrays,baseline


def test_raw_six_condition_packet_roundtrip_keeps_all_flags_targets_and_native_baseline(tmp_path):
    m=api();a,b=packets();keys=[key(i)for i in range(3)]
    rec=m.write_control_batch(tmp_path,0,keys=keys,arrays=a,baseline=b,
        baseline_ordinal=0,inputs_sha256={'row:0':'a'*64})
    out=m.read_control_batch(tmp_path,rec,expected_keys=keys,baseline=b,baseline_ordinal=0)
    assert set(out)==set(a)
    for n in a:assert out[n].dtype==a[n].dtype and np.array_equal(out[n],a[n])
    assert rec['condition_order']==[list(x)for x in ORDER]
    assert rec['full_inference_permit']is False and not(tmp_path/'result.json').exists()
    assert not out['condition_vision_valid_mask'][1].any()
    assert out['label_valid_mask'][1,0]  # Artificial image loss retains this valid label.


@pytest.mark.parametrize('damage',['nonfinite','precision','condition_missing','wrong_error',
    'targets','label_mask','natural_crop_mask','artificial_vision_mask','vision_depends_on_emg',
    'aligned_fusion','aligned_vision','duplicate','wrong_window','mixed_hand'])
def test_semantically_wrong_control_packet_fails_before_write(tmp_path,damage):
    m=api();a,b=packets();keys=[key(i)for i in range(3)]
    if damage=='nonfinite':a['fusion_prediction'][1,0,0]=np.nan
    elif damage=='precision':a['fusion_prediction']=a['fusion_prediction'].astype(np.float64)
    elif damage=='condition_missing':a['fusion_prediction']=a['fusion_prediction'][:-1]
    elif damage=='wrong_error':a['mae_fusion_rad'][3,1]+=.01
    elif damage=='targets':a['targets'][0,0]+=1
    elif damage=='label_mask':a['label_valid_mask'][:]=True
    elif damage=='natural_crop_mask':a['natural_crop_valid_mask'][:]=True
    elif damage=='artificial_vision_mask':a['condition_vision_valid_mask'][1,1,0]=True
    elif damage=='vision_depends_on_emg':a['vision_prediction'][2,0,0]+=1
    elif damage=='aligned_fusion':a['fusion_prediction'][0,0,0]+=1
    elif damage=='aligned_vision':a['vision_prediction'][0,0,0]+=1
    elif damage=='duplicate':keys[1]=dict(keys[0],sequence=1)
    elif damage=='wrong_window':keys[1]['window_end']-=1
    elif damage=='mixed_hand':keys[1]['hand']='right'
    with pytest.raises(ValueError):
        m.write_control_batch(tmp_path,0,keys=keys,arrays=a,baseline=b,
            baseline_ordinal=0,inputs_sha256={'row:0':'a'*64})
    assert not list(tmp_path.glob('*.npz'))


@pytest.mark.parametrize('damage',['reorder_conditions','baseline_ordinal','permit','path','bytes','symlink'])
def test_resealed_or_changed_raw_packet_cannot_replay_as_another_condition(tmp_path,damage):
    m=api();a,b=packets();keys=[key(i)for i in range(3)]
    rec=m.write_control_batch(tmp_path,0,keys=keys,arrays=a,baseline=b,
        baseline_ordinal=0,inputs_sha256={'row:0':'a'*64})
    if damage=='reorder_conditions':rec['condition_order'].reverse()
    elif damage=='baseline_ordinal':rec['baseline_ordinal']=1
    elif damage=='permit':rec['full_inference_permit']=True
    elif damage=='path':rec['file']='../outside.npz'
    elif damage=='bytes':
        p=tmp_path/rec['file'];p.write_bytes(p.read_bytes()[:-9])
    elif damage=='symlink':
        p=tmp_path/rec['file'];original=tmp_path/'original.npz';p.rename(original);p.symlink_to(original)
    from egoemg_repro.classic_emg import _hash
    rec['sha256']=_hash({k:v for k,v in rec.items()if k!='sha256'})
    with pytest.raises(ValueError):m.read_control_batch(tmp_path,rec,expected_keys=keys,baseline=b,baseline_ordinal=0)


def arguments():
    return dict(recipe='vits_s',evidence=str(ROOT/'results/native-dense-occlusion'),
        upstream='/home/shihao/projects/EgoEMG-Reproduction/upstream',
        index_root='/data/shihao/egoemg-native-dense-occlusion-complete-indices-20261101',
        paired_root='/data/shihao/egoemg-native-dense-paired-inputs-v2-20261101',
        geometry_root='/data/shihao/egoemg-native-dense-geometry-vits_s-full-20261101',
        baseline='/data/shihao/egoemg-vits-dense-frozen-full-cpu-2d55b25d9303-20261101/private/flow',
        data='/home/shihao/projects/EgoEMG-Reproduction/data/legacy',
        crops='/home/shihao/projects/EgoEMG-Reproduction/data/crops',
        fit='/data/shihao/egoemg-native-fusion-full-59b5ed5f525f-20261028/private/full-fit')


def test_rn18_completed_baseline_is_required_before_control_source_admission():
    m=api()
    # RN18's original full pass is still running; its milestone cannot be used
    # as a complete baseline or filled in by a matching-sized ViTS artifact.
    with pytest.raises(ValueError,match='completed.*baseline|baseline.*pending'):
        m.load_control_evidence(dict(arguments(),recipe='rn18_s'))


@pytest.fixture(scope='module')
def real_control_evidence():
    m=api()
    if not Path(arguments()['paired_root']).is_dir():
        if os.environ.get('EGOEMG_REQUIRE_CONTROL_SOURCE')=='1':pytest.fail('real complete control source roots required')
        pytest.skip('real source roots unavailable; no synthetic full positive')
    return m.load_control_evidence(arguments())


def test_real_evidence_binds_complete_source_and_all_native_baseline_packet_hashes(real_control_evidence):
    e=real_control_evidence;e.check()
    assert len(e.keys())==17168 and len(e.baseline_records)==272
    assert e.identity['recipe']=='vits_s'
    assert e.identity['condition_order']==[list(x)for x in ORDER]
    assert e.identity['all_baseline_payload_sha256_verified']
    assert e.identity['baseline_model_identity_sha256']=='7543495ccd161d11bc2db25f277a7dfb1fddd4fd0051b9f1d0531e5faeea41fd'
    assert not e.identity['full_inference_permit']
    # Real first batch, original target masks and existing native predictions.
    b=e.baseline_packet(0)
    assert b['targets'].shape==(64,22)and b['emg_full_prediction'].shape==(64,22,146)
    assert b['label_valid_mask'].dtype==np.bool_


@pytest.mark.parametrize('damage',['copy','identity','guard','native','records','request'])
def test_replaced_context_or_resealed_identity_does_not_grant_control_source(real_control_evidence,damage,monkeypatch):
    from types import SimpleNamespace
    from egoemg_repro.classic_emg import _hash
    e=real_control_evidence
    if damage=='copy':
        with pytest.raises(ValueError,match='registered'):copy.copy(e).check()
        return
    old=copy.deepcopy(e.identity)
    with monkeypatch.context()as patch:
        if damage=='identity':e.identity['recipe']='rn18_s';e.identity['sha256']=_hash({k:v for k,v in e.identity.items()if k!='sha256'})
        elif damage=='guard':patch.setattr(e,'_guard',SimpleNamespace(check=lambda:None))
        elif damage=='native':patch.setattr(e,'native',copy.copy(e.native))
        elif damage=='records':patch.setattr(e,'baseline_records',e.baseline_records[::-1])
        elif damage=='request':patch.setattr(e,'request',dict(e.request,max_samples=1))
        try:
            with pytest.raises(ValueError):e.check()
        finally:e.identity.clear();e.identity.update(old)
    e.check()


def test_source_loader_stays_model_free_in_fresh_process(real_control_evidence):
    code='''import json,sys
from egoemg_repro.modality_control_run import load_control_evidence
assert 'torch'not in sys.modules
e=load_control_evidence(json.loads(sys.argv[1]));e.check()
assert len(e.keys())==17168 and 'torch'not in sys.modules
print('source-only-complete-17168')
'''
    p=subprocess.run([sys.executable,'-B','-c',code,json.dumps(arguments())],capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stderr[-1500:]
    assert p.stdout.strip()=='source-only-complete-17168'

"""RN18 exact cached-key remapping and new source-bound visual consumer."""
import copy
import importlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from egoemg_repro.classic_emg import _hash
from egoemg_repro.occlusion_dense_data import _typed_sha

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(os.environ.get('EGOEMG_RN18_QUAL_CACHE','/data/shihao/egoemg-rn18-ordinary-cache-source-read-guard-contract-v4-86a2f9f0a7b3-20261105'))


def api():
    try:return importlib.import_module('egoemg_repro.rn18_qualitative_render')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.rn18_qualitative_render':pytest.fail('RN18 cached visual consumer is missing')
        raise


def fixture():
    ordinary=dict(sequence=0,split='gesture',hand='right',ep_idx=2,center_idx=18000,
        window_start=12000,window_end=24000,native_index=1,group_index=1)
    dense=dict(ordinary,sequence=4,native_index=4)
    target=np.linspace(.1,.6,22,dtype=np.float32)[:,None]
    fusion=dict(emg=np.arange(96000,dtype=np.float32).reshape(8,12000)/9000,
        emg_channel_mask=np.ones(8,bool),joint_angles=target.copy(),label_valid_mask=np.array([False],bool),
        vision_img=np.zeros((3,256,256),np.float32),vision_valid_mask=np.array([True],bool),
        vision_frame_indices=np.array([87],np.int64))
    standalone={n:fusion[n].copy()for n in('emg','emg_channel_mask','joint_angles','label_valid_mask')}
    typed={'fusion:'+n:_typed_sha(fusion[n])for n in('emg','emg_channel_mask','joint_angles','label_valid_mask','vision_img','vision_valid_mask','vision_frame_indices')}
    typed.update({'standalone:'+n:_typed_sha(standalone[n])for n in('emg','emg_channel_mask','joint_angles','label_valid_mask')})
    record=dict(key=ordinary,typed_input_sha256=typed,verified_input_sha256=_hash(typed))
    join=dict(ordinary_sequence=0,dense_sequence=4,ordinary_key=copy.deepcopy(ordinary),dense_key=dense,
        original_batch_ordinal=0,original_row_in_batch=4,verified_input_sha256=_hash(typed))
    sample=dict(key=copy.deepcopy(dense),fusion=fusion,standalone_emg=standalone,verified_input_sha256=_hash(typed))
    arrays=dict(targets=target[:,0].copy(),vision_prediction=target[:,0]+np.float32(.2),
        emg_prediction=target[:,0]+np.float32(.3),fusion_prediction=target[:,0]+np.float32(.1),
        label_valid_mask=np.array([False],bool),crop_valid_mask=np.array([True],bool),
        mae_vision_rad=np.float64(.2),mae_emg_rad=np.float64(.3),mae_fusion_rad=np.float64(.1))
    arrays['emg_full_prediction']=np.repeat(arrays['emg_prediction'][:,None],230,axis=1)
    return record,join,arrays,sample


# Catches treating dense sequence numbers as ordinary indices or changing the
# physical key while retaining a matching center and self-reported hash.
def test_exact_dense_source_is_remapped_only_to_the_original_ordinary_key_without_mutation():
    r,j,a,s=fixture();before=copy.deepcopy((r,j,a,s))
    out=api().materialize_rn18_cache_row(r,j,a,s)
    assert out['sample']['key']==out['prediction']['key']==r['key']
    assert out['sample']['key']['sequence']==0 and out['cache_lineage']['dense_key']['sequence']==4
    assert out['cache_lineage']['ordinary_key']==r['key']and not out['trained_source_admitted']
    assert out['sample']['verified_input_sha256']==r['verified_input_sha256']
    assert out['sample']['typed_input_sha256']==r['typed_input_sha256']
    np.testing.assert_array_equal(out['prediction']['arrays']['targets'],s['fusion']['joint_angles'][:,0])
    assert out['prediction']['arrays']['label_valid_mask'].tolist()==[False]
    assert out['prediction']['arrays']['emg_full_prediction'].shape==(22,230)
    assert s['key']==before[3]['key']and r==before[0]and j==before[1]
    out['prediction']['arrays']['vision_prediction'][0]+=1
    np.testing.assert_array_equal(a['vision_prediction'],before[2]['vision_prediction'])


@pytest.mark.parametrize('damage',['center','window','hand','split','ordinary_sequence','dense_sequence',
    'fresh_key','fresh_claim','fusion_emg','standalone_emg','fresh_target','fresh_mask','saved_target',
    'saved_mask','saved_crop','record_hash','join_hash','wrong_window_length'])
def test_cache_remapping_rejects_physical_or_actual_tensor_mismatch(damage):
    r,j,a,s=fixture()
    if damage in('center','window','hand','split'):
        field=dict(center='center_idx',window='window_start',hand='hand',split='split')[damage]
        j['dense_key'][field]='left'if damage=='hand'else'user'if damage=='split'else j['dense_key'][field]+1
    elif damage=='ordinary_sequence':j['ordinary_sequence']=1
    elif damage=='dense_sequence':j['dense_sequence']=5
    elif damage=='fresh_key':s['key']['ep_idx']=3
    elif damage=='fresh_claim':s['verified_input_sha256']='a'*64
    elif damage=='fusion_emg':s['fusion']['emg'][0,0]+=1
    elif damage=='standalone_emg':s['standalone_emg']['emg'][0,0]+=1
    elif damage=='fresh_target':s['fusion']['joint_angles'][0,0]+=1
    elif damage=='fresh_mask':s['fusion']['label_valid_mask'][0]=True
    elif damage=='saved_target':a['targets'][0]+=1
    elif damage=='saved_mask':a['label_valid_mask'][0]=True
    elif damage=='saved_crop':a['crop_valid_mask'][0]=False
    elif damage=='record_hash':r['verified_input_sha256']='b'*64
    elif damage=='join_hash':j['verified_input_sha256']='c'*64
    else:
        for k in(r['key'],j['ordinary_key'],j['dense_key'],s['key']):k['window_end']-=1
    with pytest.raises(ValueError):api().materialize_rn18_cache_row(r,j,a,s)


def test_display_selection_keeps_original_ranks_and_explicit_failure_requests_separate():
    rows=[dict(key={'sequence':i},crop_valid=i!=1,source_filter_pass=True,delta_mae=float(5-i))for i in range(6)]
    out=api().select_rn18_render_rows(rows,limit=2,extra_ranks=(6,))
    assert [(v['rank'],v['row']['key']['sequence'],v['diagnostic_extra'])for v in out]==[(1,0,False),(3,2,False),(6,5,True)]
    assert out[-1]['row']['delta_mae']==0 and rows[1]['crop_valid']is False


@pytest.mark.parametrize('limit,extra',[(0,()),(True,()),(1,(1,)),(1,(0,)),(1,(7,)),(1,(2,)),(1,(6,6)),(1,(True,))])
def test_invalid_display_caps_or_silent_rank_repair_are_rejected(limit,extra):
    rows=[dict(key={'sequence':i},crop_valid=i!=1,source_filter_pass=True,delta_mae=float(5-i))for i in range(6)]
    with pytest.raises(ValueError):api().select_rn18_render_rows(rows,limit=limit,extra_ranks=extra)


def test_unregistered_cache_cannot_create_output_or_admit_actual_inputs(tmp_path):
    m=api();out=tmp_path/'not-created'
    with pytest.raises(ValueError,match='registered'):
        m.export_rn18_candidates(object(),upstream='/missing',data='/missing',crops='/missing',output=out,limit=1)
    assert not out.exists()


@pytest.fixture(scope='module')
def actual_cache():
    if not CACHE.exists():pytest.skip('actual independently accepted RN18 cache required; no fabricated full positive')
    return api().open_rn18_cache(CACHE)


def test_actual_closed_cache_retains_all2208_original_keys_and_exact_full_EMG_center(actual_cache):
    c=actual_cache;c.check();v=c.identity
    assert v['selected']==2208 and v['source_filter_pass']==2119
    assert v['source_model_identity_sha256']=='1bb167a3f47a2b8912e24502b252ebfa81fea2e90c02f11b204259608b88c6ee'
    assert not v['trained_model_or_inference_permit']and not v['figure7_reproduced']
    first=c.ranked_rows()[0];assert first['sequence']==1891 and first['delta_mae']==0.22168400883674622
    row=c.row(1891);np.testing.assert_array_equal(row['emg_full_prediction'][:,115],row['emg_prediction'])
    assert row['targets'].shape==(22,)and row['targets'].dtype==np.float32
    with pytest.raises(ValueError,match='registered'):copy.copy(c).check()
    for index in(-1,2208,True):
        with pytest.raises((ValueError,IndexError)):c.row(index)


def test_resealed_or_changed_actual_completion_proof_rejects_before_cache_io(tmp_path,monkeypatch):
    m=api();v=json.loads(m.REFERENCE.read_text());v['source_audit']['actual_selected']=1
    p=tmp_path/'wrong-proof.json';p.write_text(json.dumps(v));monkeypatch.setattr(m,'REFERENCE',p)
    with pytest.raises(ValueError):m.open_rn18_cache(tmp_path/'must-not-read')


@pytest.mark.parametrize('damage',['manifest_map','raw_arrays','ranking'])
def test_changed_real_source_capsule_or_raw_output_cannot_reuse_the_same_proof(actual_cache,tmp_path,damage):
    source=tmp_path/'real-cache-copy';shutil.copytree(CACHE,source)
    if damage=='manifest_map':
        p=source/'private/stage-receipt.json';v=json.loads(p.read_text());v['files_sha256'].pop(next(iter(v['files_sha256'])));p.write_text(json.dumps(v))
    elif damage=='raw_arrays':
        p=source/'private/selected/arrays.npz';raw=bytearray(p.read_bytes());raw[len(raw)//2]^=1;p.write_bytes(raw)
    else:
        p=source/'private/selected/ranking.json';v=json.loads(p.read_text());v['ranked_rows'].reverse();p.write_text(json.dumps(v))
    with pytest.raises(ValueError):api().open_rn18_cache(source)


def test_real_top_and_explicit_failure_panel_use_original_RGB_raw_arrays_and_full_source(actual_cache,tmp_path):
    pytest.importorskip('torch')
    from PIL import Image
    from test_occlusion_dense_data import UPSTREAM
    out=tmp_path/'actual-panels'
    result=api().export_rn18_candidates(actual_cache,upstream=UPSTREAM,
        data='/home/shihao/projects/EgoEMG-Reproduction/data/legacy',crops='/home/shihao/projects/EgoEMG-Reproduction/data/crops',
        output=out,limit=1,extra_ranks=(2119,))
    assert result['rendered_candidates']==2 and result['default_rank_candidates']==1 and result['diagnostic_extra_candidates']==1
    assert [s['rank']for s in result['samples']]==[1,2119]
    assert [s['key']['sequence']for s in result['samples']]==[1891,1460]
    assert result['full_source_payload_files']==53 and result['full_source_payload_bytes']==61377230438
    assert not result['condition_examples_selected']and not result['figure7_reproduced']
    for item in result['samples']:
        assert Image.open(out/item['png_file']).size==(2160,512)
        with np.load(out/item['arrays_file'],allow_pickle=False)as z:
            np.testing.assert_array_equal(z['pose:gt'],actual_cache.row(item['key']['sequence'])['targets'])
            assert z['rgb'].dtype==np.uint8 and z['rgb'].shape==(256,256,3)
        meta=json.loads((out/item['metadata_file']).read_text())
        assert meta['source_model_identity_sha256']==actual_cache.identity['source_model_identity_sha256']
        assert meta['panel_metadata']['key']==item['key']
    with pytest.raises(FileExistsError):
        api().export_rn18_candidates(actual_cache,upstream=UPSTREAM,data='/missing',crops='/missing',output=out,limit=1)

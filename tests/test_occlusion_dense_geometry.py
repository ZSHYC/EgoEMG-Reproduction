"""Authenticated dense derived-packet loading; no native operator executes."""
import builtins
from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from egoemg_repro.classic_emg import _hash
from egoemg_repro.occlusion_dense_data import _typed_sha
from test_occlusion_dense_geometry_data import fixture,sha,write


def api():return importlib.import_module('egoemg_repro.occlusion_dense_geometry')


def files(tmp_path,monkeypatch):
    m=api();kw=fixture(tmp_path,monkeypatch)
    from egoemg_repro.occlusion_dense_geometry_data import export_dense_geometry_inputs
    out=tmp_path/'export';identity=export_dense_geometry_inputs(output=out,**kw)
    proof=dict(format='actual_complete_dense_geometry_inputs_export_verified_v1',actual_exit_code=0,
        actual_complete_raw_input_admissions=2,all_six_original_owners_closed=True,
        exact_complete_identity_and_typed_exports_rebuilt=True,all_capsule_and_export_bytes_after_exit_exact=True,
        identity=identity,files_sha256={p.name:sha(p)for p in out.iterdir()},native_geometry_executed=False,
        learned_pose_models_constructed=0,full_paper_complete=False)
    p=tmp_path/'input-proof.json';write(p,proof);monkeypatch.setattr(m,'INPUT_PROOF_SHA',sha(p))
    return dict(export=out,evidence=p,metadata=kw['data']/'metadata.npz',calibration=kw['geometry']/'meta/GX010023_standard_calibration.json')


def load(kw,recipe='vits_s'):return api().load_dense_geometry_inputs(recipe=recipe,**kw)


@pytest.mark.parametrize('recipe',['vits_s','rn18_s'])
def test_exact_all_native_requests_retain_zero_poses_and_pool_source_order(tmp_path,monkeypatch,recipe):
    m=api();kw=files(tmp_path,monkeypatch);v=load(kw,recipe);v.check()
    rows=m.select_dense_requests(v,'complete_dense')
    assert [r['native_sequence']for r in rows]==[0,1,2,3]
    assert [r['hand']for r in rows]==['left','right','left','right']
    assert [r['pool_index']for r in rows]==[0,0,1,1]and[r['center_idx']for r in rows]==[10,10,60,60]
    assert v.identity['expected_complete_hand_requests']==4 and v.identity['native_geometry_executed']is False
    assert not v.data['arrays']['left_pose'][0].any()
    assert v.data['arrays']['image_stale'][0]and not v.data['arrays']['camera_tracked'][0]


def test_bilateral_probe_selects_actual_nonzero_keys_without_full_role_claim(tmp_path,monkeypatch):
    m=api();kw=files(tmp_path,monkeypatch);v=load(kw);rows=m.select_dense_requests(v,'bilateral_probe')
    assert [r['native_sequence']for r in rows]==[2,3]and{r['hand']for r in rows}=={'left','right'}
    with pytest.raises(ValueError):m.select_dense_requests(v,'complete_role')
    with pytest.raises(ValueError):load(kw,'rn50_m')


@pytest.mark.parametrize('target',['evidence','metadata','calibration','points','selection','identity','completion'])
def test_any_current_consumed_byte_drift_rejects(tmp_path,monkeypatch,target):
    kw=files(tmp_path,monkeypatch);load(kw).check()
    if target in kw:p=kw[target]
    else:p=kw['export']/({'points':'vits_s-points.npz','selection':'vits_s-selection.npz','identity':'identity.json','completion':'completion.json'}[target])
    with p.open('r+b')as f:f.write(b'X')
    with pytest.raises(ValueError):load(kw)


def test_caller_resealed_proof_rejects_before_private_reads(tmp_path):
    p=tmp_path/'fake.json';write(p,dict(actual_exit_code=0,all_six_original_owners_closed=True))
    with pytest.raises(ValueError):api().load_dense_geometry_inputs(tmp_path/'missing',p,recipe='vits_s',metadata=tmp_path/'missing',calibration=tmp_path/'missing')


def test_source_order_pool_mapping_must_agree_even_when_test_fixture_reseals_every_file(tmp_path,monkeypatch):
    m=api();kw=files(tmp_path,monkeypatch);p=kw['export']/'vits_s-selection.npz'
    with np.load(p,allow_pickle=False)as f:a={k:f[k]for k in f.files}
    a['pool_index'][0]=1;np.savez(p,**a)
    i=json.loads((kw['export']/'identity.json').read_text());i['grids']['vits_s']['selection_sha256']={k:_typed_sha(v)for k,v in a.items()}
    i['sha256']=_hash({k:v for k,v in i.items()if k!='sha256'});write(kw['export']/'identity.json',i)
    c=json.loads((kw['export']/'completion.json').read_text());c['identity_sha256']=i['sha256']
    c['files_sha256']={n:sha(kw['export']/n)for n in c['files_sha256']};c['sha256']=_hash({k:v for k,v in c.items()if k!='sha256'});write(kw['export']/'completion.json',c)
    pr=json.loads(kw['evidence'].read_text());pr['identity']=i;pr['files_sha256']={p.name:sha(p)for p in kw['export'].iterdir()};write(kw['evidence'],pr)
    monkeypatch.setattr(m,'INPUT_PROOF_SHA',sha(kw['evidence']))
    with pytest.raises(ValueError):load(kw)


@pytest.mark.parametrize('damage',['clone','data_token','selection','identity'])
def test_unregistered_or_mutated_live_dense_packet_cannot_select_native_work(tmp_path,monkeypatch,damage):
    m=api();v=load(files(tmp_path,monkeypatch));v.check()
    if damage=='clone':v=replace(v)
    elif damage=='data_token':v.data['_token']='0'*64
    elif damage=='selection':
        a=v.selection['pool_index'];a.setflags(write=True);a[0]=1;a.setflags(write=False)
    else:v.identity['expected_complete_hand_requests']=2;v.identity['sha256']=_hash({k:x for k,x in v.identity.items()if k!='sha256'})
    with pytest.raises(ValueError):m.select_dense_requests(v,'complete_dense')


def test_loading_and_selecting_cannot_import_torch(tmp_path,monkeypatch):
    m=api();kw=files(tmp_path,monkeypatch);old=builtins.__import__
    def reject(name,*a,**k):
        if name=='torch'or name.startswith('torch.'):raise AssertionError('loading/selection cannot import Torch')
        return old(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',reject)
    rows=m.select_dense_requests(load(kw),'complete_dense');assert len(rows)==4


def zero_source(tmp_path,monkeypatch):
    import test_occlusion_dense_geometry_data as fixtures
    original=fixtures.values
    def zeros():
        p,f,b,md=original()
        for h in('left','right'):f['generated_mano_'+h+'_pose'][:]=0
        return p,f,b,md
    monkeypatch.setattr(fixtures,'values',zeros)
    v=load(files(tmp_path,monkeypatch));calls=[]
    def mesh(pose,beta,R,t,flip,device):
        assert not np.any(pose)and device=='cpu';calls.append((flip,float(beta[0])));return None,None
    native=SimpleNamespace(get_mano_verts_world=mesh)
    monkeypatch.setattr(v._core,'_native_runtime',lambda a:(native,SimpleNamespace(check=lambda:None),dict(actual_right_mano_geometry_layers_constructed=1)))
    kernel=(Path(__file__).parents[1]/'upstream/egoemg/occlusion.py').resolve()
    monkeypatch.setattr(v._core,'_verify_runtime',lambda runtime,guard:kernel)
    # Keep this zero-only unit's fake runtime on the reloaded verifier core.
    # Genuine provider/asset verification is separately exercised on server.
    monkeypatch.setattr(api(),'_core',lambda path:v._core)
    return v,calls


def test_native_execution_passes_every_source_key_and_zero_pose_to_frozen_frame_function(tmp_path,monkeypatch):
    m=api();v,calls=zero_source(tmp_path,monkeypatch);out=tmp_path/'run'
    r=m.run_dense_geometry(v,out,mode='complete_dense',upstream='test',mano_assets='test',chumpy_dir='test')
    assert calls==[(True,1.),(False,2.),(True,3.),(False,4.)]
    assert r['rows']==r['actual_mesh_calls']==r['invalid_zero_pose_rows']==4 and r['actual_occlusion_calls']==0
    assert r['complete_dense_geometry']is True and not r['complete_standard_role']and not r['figure13_reproduced']
    checked=m.verify_dense_geometry(out);assert checked==r


def test_probe_and_existing_output_cannot_silently_grant_full_geometry(tmp_path,monkeypatch):
    m=api();v,calls=zero_source(tmp_path,monkeypatch);out=tmp_path/'present';out.mkdir()
    with pytest.raises(ValueError):m.run_dense_geometry(v,out,mode='complete_dense',upstream='test',mano_assets='test',chumpy_dir='test')
    with pytest.raises(ValueError):m.run_dense_geometry(v,tmp_path/'probe',mode='bilateral_probe',upstream='test',mano_assets='test',chumpy_dir='test')
    assert not calls


def test_frozen_original_verifier_row_body_rejects_resealed_flag_change(tmp_path,monkeypatch):
    m=api();v,calls=zero_source(tmp_path,monkeypatch);out=tmp_path/'run'
    m.run_dense_geometry(v,out,mode='complete_dense',upstream='test',mano_assets='test',chumpy_dir='test');m.verify_dense_geometry(out)
    p=out/'points.jsonl';rows=[json.loads(x)for x in p.read_text().splitlines()];rows[0]['mocap_tracked']=not rows[0]['mocap_tracked']
    p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n'for x in rows))
    result=json.loads((out/'result.json').read_text());result['journal_file_sha256']=sha(p)
    result['sha256']=_hash({k:x for k,x in result.items()if k!='sha256'});write(out/'result.json',result)
    with pytest.raises(ValueError):m.verify_dense_geometry(out)

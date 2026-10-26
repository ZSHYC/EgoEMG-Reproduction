"""Task A: source admission and hand-computed reducers, without model runtime."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def api(name):
    return importlib.import_module('tools.legacy_derived.' + name)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@pytest.fixture
def admitted_fixture(tmp_path, monkeypatch):
    """Synthetic policy anchors test metadata logic; never represent real runs."""
    p = api('preflight')
    root = tmp_path/'private-source'; root.mkdir()
    files = {}
    for name in p.LEGACY_FILES:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('SYNTHETIC_SOURCE = 1\n')
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(p, 'LEGACY_FILES', files)
    # Unit fixture only: separate tests exercise actual preimport rejection.
    monkeypatch.setattr(p, '_reject_preimports', lambda:None)
    monkeypatch.setattr(p, '_upstream', lambda root:dict(revision=p.UPSTREAM_COMMIT, sha256=p.UPSTREAM_SHA))
    partition = {'windows':[], 'private_sentinel':'/private/participant-SECRET'}
    norm = {'left':{'mean':[0]*8,'std':[1]*8}, 'right':{'mean':[0]*8,'std':[1]*8}}
    monkeypatch.setattr(p, 'REFERENCE', dict(partition_sha256=canonical(partition),
        normalization_sha256=canonical(norm), data_sha256='d'*64))
    runs, baselines = [], {}
    for model, original in p.BASELINES.items():
        cfg = dict(seed=42, model_kind=original['kind'], upstream_experiment=original['experiment'],
                   epochs=2, precision='bfloat16', matmul_precision='high', batch_size=2, num_workers=0)
        recipe = (json.dumps(cfg, indent=2)+'\n').encode()
        digest = hashlib.sha256()
        runtime_names = list(p.CORE_FILES) + (list(p.VISUAL_FILES) if original['kind']!='emg' else [])
        for name in runtime_names:
            digest.update(name.encode()); digest.update((root/name).read_bytes())
        digest.update(b'selected_recipe'); digest.update(recipe)
        init = {'policy':original['policy'], 'seed_before_model':42}
        prov = dict(model_kind=original['kind'], source=dict(revision=p.UPSTREAM_COMMIT, sha256=p.UPSTREAM_SHA),
            reproduction=dict(sha256=digest.hexdigest(), selected_recipe_sha256=hashlib.sha256(recipe).hexdigest(),
                runtime_files_sha256={name:files[name] for name in runtime_names}),
            data={'sha256':'d'*64}, config_sha256=canonical(cfg), partition_sha256=canonical(partition),
            normalization_sha256=canonical(norm), initialization=init,
            recipe=dict(source_utf8=recipe.decode(), effective_config=cfg))
        summary = dict(status='training_complete', test_evaluated=True, test={'bounded':False},
                       global_step=4, selected_checkpoint_step=original['step'], provenance=prov)
        env = dict(initialization=init, precision='bfloat16', matmul_precision='high')
        run = tmp_path/('private-run-'+model); run.mkdir()
        values = {'config.json':cfg,'partitions.json':partition,'normalization.json':norm,
                  'provenance.json':prov,'summary.json':summary,'environment.json':env,'run.json':{'run_id':'SECRET'}}
        for name,value in values.items():
            (run/name).write_text(json.dumps(value, indent=2)+'\n')
        (run/'recipe_source.json').write_bytes(recipe)
        hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in run.iterdir()}
        baselines[model]=dict(original, recipe_sha256=hashlib.sha256(recipe).hexdigest(),
            config_sha256=canonical(cfg), summary_sha256=hashes['summary.json'], initialization_sha256=canonical(init))
        runs.append(dict(model=model, run=str(run), files_sha256=hashes))
    monkeypatch.setattr(p, 'BASELINES', baselines)
    monkeypatch.setattr(p, 'RAW_FILES', {x['model']:dict(x['files_sha256']) for x in runs},raising=False)
    return dict(format=p.FORMAT, legacy_root=str(root), upstream_root=str(tmp_path/'upstream'), runs=runs)


def test_preflight_reads_only_small_metadata_and_never_imports_runtime(admitted_fixture, monkeypatch):
    p=api('preflight'); original=p._read_small
    def read(path):
        assert Path(path).suffix not in ('.pt','.ckpt','.dat')
        return original(path)
    monkeypatch.setattr(p,'_read_small',read)
    result=p.preflight_manifest(admitted_fixture)
    assert result['status']=='preflight_only' and result['scope']=='no_inference'
    assert result['benchmark_eligible'] is False and result['checkpoint_tensors_read'] is False
    rendered=json.dumps(result)
    assert all(x not in rendered for x in ('SECRET','private-source','private-run','windows','run_id'))
    assert len(result['models'])==4
    assert len(result['identities']['tool_source_sha256'])==64


@pytest.mark.parametrize('damage',['unknown_model','missing_model','new_source','raw_hash','canonical_hash','norm','recipe','source_pin'])
def test_preflight_fail_closed_before_any_tensor_or_array_access(admitted_fixture,monkeypatch,damage):
    p=api('preflight'); value=copy.deepcopy(admitted_fixture)
    if damage=='unknown_model':value['runs'][0]['model']='resnet50'
    elif damage=='missing_model':value['runs'].pop()
    elif damage=='new_source':
        (Path(value['legacy_root'])/'egoemg_repro/train.py').write_text('NEW = True\n')
    else:
        spec=value['runs'][0]; root=Path(spec['run'])
        if damage=='raw_hash':spec['files_sha256']['normalization.json']='0'*64
        elif damage=='canonical_hash':
            cfg=json.loads((root/'config.json').read_text()); cfg['seed']=43
            (root/'config.json').write_text(json.dumps(cfg));spec['files_sha256']['config.json']=hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
        elif damage=='norm':
            (root/'normalization.json').write_text('{}');spec['files_sha256']['normalization.json']=hashlib.sha256(b'{}').hexdigest()
        elif damage=='recipe':(root/'recipe_source.json').unlink()
        else:monkeypatch.setattr(p,'_upstream',lambda root:dict(revision='0'*40,sha256=p.UPSTREAM_SHA))
    with pytest.raises(ValueError):p.preflight_manifest(value)


def test_same_named_preimport_and_source_symlink_are_rejected(tmp_path,monkeypatch):
    p=api('preflight')
    monkeypatch.setitem(sys.modules,'egoemg_repro.already_loaded',SimpleNamespace(__file__='/new/source.py'))
    with pytest.raises(ValueError,match='preimport'):p.validate_source_manifest(tmp_path,tmp_path)


def test_source_identity_rechecks_late_mutation(admitted_fixture,monkeypatch):
    p=api('preflight'); path=Path(admitted_fixture['legacy_root'])/'egoemg_repro/train.py'
    original=p._upstream
    def mutate(root):
        result=original(root);path.write_text('late source mutation\n');return result
    monkeypatch.setattr(p,'_upstream',mutate)
    with pytest.raises(ValueError):p.preflight_manifest(admitted_fixture)


@pytest.fixture
def center_fixture():
    r=api('reducers'); n=4
    cohort=dict(ordinal=np.arange(n,dtype='<u8'),hand=np.array([0,1,0,1],dtype='u1'),
        split=np.array([0,0,1,2],dtype='u1'),gesture=np.array([30,33,50,-1],dtype='<i4'),
        active=np.array([True,False,True,False]),crop_present=np.array([True,True,False,True]),
        user_ordinal=np.array([0,1,0,1],dtype='<u4'))
    target=np.zeros((n,22),dtype='<f4'); valid=np.zeros((n,22),dtype=bool)
    valid[0,:2]=True;valid[1,0]=True;valid[2,:]=True
    pred=np.zeros_like(target);pred[0,:]=1;pred[1,:]=3;pred[2,:]=2
    identity=r.cohort_identity(target,valid,cohort)
    frames={name:dict(prediction=pred.copy(),ordinal=cohort['ordinal'].copy(),cohort_sha256=identity) for name in r.MODELS}
    return frames,target,valid,cohort


def test_two_weightings_users_and_zero_valid_are_hand_computed(center_fixture):
    r=api('reducers'); frames,target,valid,cohort=center_fixture
    out=r.reduce_centers(frames,target,valid,cohort); m=out['models']['s']
    a=m['by_split']['user'];deg=180/np.pi
    assert a['absolute_error_sum_rad']==5 and a['valid_angle_count']==3
    assert a['angle_pooled_mae_deg']==pytest.approx(5/3*deg)
    assert a['sample_error_sum_rad']==4 and a['metric_sample_count']==2
    assert a['sample_weighted_mae_deg']==pytest.approx(2*deg)
    assert a['user_summary']['mean_mae_deg']==pytest.approx(2*deg)
    assert a['user_summary']['std_mae_deg']==pytest.approx(deg)
    assert a['user_summary']['ddof']==0
    empty=m['by_gesture']['-1']
    assert empty['requested_samples']==1 and empty['metric_sample_count']==0
    assert empty['angle_pooled_mae_deg'] is None and empty['sample_weighted_mae_deg'] is None
    assert m['overall']['valid_angle_count']==25 and m['overall']['requested_samples']==4


def test_noncontiguous_families_raw_unknown_crop_and_group_overlap(center_fixture):
    r=api('reducers');args=center_fixture;m=r.reduce_centers(*args)['models']['s']
    assert m['by_family']['symmetric']['requested_samples']==2  # 30 and raw50
    assert m['by_family']['asymmetric']['requested_samples']==1 # 33
    assert m['views']['defined59']['requested_samples']==2
    assert m['views']['raw50']['requested_samples']==1 and m['views']['unknown_minus1']['requested_samples']==1
    assert m['views']['crop_present_defined59']['requested_samples']==2
    assert m['by_joint']['0']['valid_angle_count']==3
    assert m['by_group']['thumb']['valid_angle_count']==7
    assert m['by_group']['proximal']['valid_angle_count']==13
    assert len(m['by_joint'])==22 and len(m['by_group'])==10
    assert sum(x['valid_angle_count'] for x in m['by_joint'].values())==m['overall']['valid_angle_count']
    assert sum(x['valid_angle_count'] for x in m['by_group'].values())>m['overall']['valid_angle_count']
    assert len(m['by_gesture'])==61 and m['by_gesture']['0']['angle_pooled_mae_deg'] is None


def test_physical_center_is_5490_never_aligned_or_token_midpoint():
    r=api('reducers'); a=np.zeros((22,11490),dtype='<f4');a[:,5490]=1;a[:,5745]=9
    p,t,v=r.physical_center(a,a.copy(),np.ones_like(a,dtype=bool),window_length=12000,left_context=510,right_context=0)
    assert np.all(p==1) and p.shape==(22,1)
    with pytest.raises(ValueError):r.physical_center(a[:,:100],a[:,:100],np.ones((22,100),bool))


@pytest.mark.parametrize('damage',['missing_model','unknown_model','order','identity','target','mask','dtype','shape','nan','private_extra'])
def test_cache_cross_model_identity_and_typed_schema_are_fail_closed(center_fixture,damage):
    r=api('reducers'); frames,target,valid,cohort=copy.deepcopy(center_fixture)
    if damage=='missing_model':frames.pop('m')
    elif damage=='unknown_model':frames['unknown']=frames.pop('m')
    elif damage=='order':frames['m']['ordinal']=frames['m']['ordinal'][::-1].copy()
    elif damage=='identity':frames['m']['cohort_sha256']='0'*64
    elif damage=='target':target[0,0]=1
    elif damage=='mask':valid[0,0]=False
    elif damage=='dtype':frames['m']['prediction']=frames['m']['prediction'].astype('f8')
    elif damage=='shape':frames['m']['prediction']=frames['m']['prediction'][:,:20]
    elif damage=='nan':frames['m']['prediction'][0,0]=np.nan
    else:frames['m']['private_path']='/private/SECRET'
    with pytest.raises(ValueError):r.reduce_centers(frames,target,valid,cohort)


def test_private_cache_is_0600_no_pickle_and_public_is_aggregate_only(tmp_path,center_fixture):
    r=api('reducers');path=tmp_path/'private.npz';receipt=r.write_private_cache(path,*center_fixture)
    assert path.stat().st_mode&0o777==0o600
    loaded=r.read_private_cache(path,expected_sha256=receipt['sha256'])
    assert r.reduce_centers(*loaded)==r.reduce_centers(*center_fixture)
    with pytest.raises(FileExistsError):r.write_private_cache(path,*center_fixture)
    out=r.reduce_centers(*loaded);out['private_path']='/private/SECRET';out['models']['s']['private_id']='SECRET'
    rendered=json.dumps(r.public_projection(out),allow_nan=False)
    for value in ['SECRET','prediction','target','ordinal','cohort_sha256','/private/']:
        assert value not in rendered
    path.chmod(0o644)
    with pytest.raises(ValueError):r.read_private_cache(path,expected_sha256=receipt['sha256'])


def test_cli_only_preflight_no_full_entry_and_no_success_overwrite(tmp_path,admitted_fixture,capsys):
    c=api('cli'); manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(admitted_fixture))
    output=tmp_path/'new';assert c.main(['preflight','--manifest',str(manifest),'--output',str(output)])==0
    result=json.loads((output/'preflight.json').read_text())
    assert result['status']=='preflight_only' and result['benchmark_eligible'] is False
    assert 'SECRET' not in capsys.readouterr().out
    assert c.main(['preflight','--manifest',str(manifest),'--output',str(output)])==1
    with pytest.raises(SystemExit):c.main(['run','--manifest',str(manifest)])


def test_frozen_anchors_match_actual_public_completed_receipts():
    p=api('preflight'); published=json.loads((Path(__file__).parents[1]/'results/completed/publication-manifest.json').read_text())
    assert p.LEGACY_FILES=={k:v for k,v in published['code_validation']['tested_files_sha256'].items() if k in p.LEGACY_FILES}
    for model,baseline in p.BASELINES.items():
        audit=published['models'][model]['identities']['audit']
        for key,source in [('recipe_sha256','selected_recipe_sha256'),('config_sha256','config_sha256'),
                           ('summary_sha256','summary_sha256'),('initialization_sha256','initialization_sha256')]:
            assert baseline[key]==audit[source]


def test_end_of_whole_admission_rejects_same_bytes_late_metadata_replacement(admitted_fixture,monkeypatch):
    p=api('preflight');original=p._run;first=Path(admitted_fixture['runs'][0]['run'])/'normalization.json'
    def replace(spec,root):
        result=original(spec,root)
        if spec['model']=='fusion':
            before=first.stat();raw=first.read_bytes();first.unlink();first.write_bytes(raw)
            os.utime(first,ns=(before.st_atime_ns,before.st_mtime_ns))
        return result
    monkeypatch.setattr(p,'_run',replace)
    with pytest.raises(ValueError):p.preflight_manifest(admitted_fixture)


def test_actual_source_symlink_is_rejected(admitted_fixture):
    p=api('preflight');path=Path(admitted_fixture['legacy_root'])/'egoemg_repro/train.py'
    raw=path.read_bytes();path.unlink();other=path.with_name('other.py');other.write_bytes(raw);path.symlink_to(other)
    with pytest.raises(ValueError,match='namespace'):p.preflight_manifest(admitted_fixture)


def test_real_cache_reader_rejects_object_extra_shape_and_model_missing(tmp_path,center_fixture):
    r=api('reducers');original=tmp_path/'original.npz';r.write_private_cache(original,*center_fixture)
    with np.load(original,allow_pickle=False) as file:arrays={k:file[k] for k in file.files}
    for damage in ['object','extra','shape','missing']:
        changed=copy.deepcopy(arrays)
        if damage=='object':changed['target']=np.array([{'private':'SECRET'}],dtype=object)
        elif damage=='extra':changed['private_path']=np.array(['SECRET'])
        elif damage=='shape':changed['target']=changed['target'][:,:20]
        else:changed.pop('prediction__m')
        path=tmp_path/(damage+'.npz');np.savez(path,**changed);path.chmod(0o600)
        with pytest.raises(ValueError):r.read_private_cache(path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_trajectory_stream_reduces_hand_time_weights_groups_and_hist_without_retaining_arrays():
    r=api('reducers');state={};pred=np.zeros((22,2),dtype='<f4');target=np.zeros_like(pred);valid=np.zeros_like(pred,dtype=bool)
    pred[:,0]=1;pred[:,1]=3;valid[:2,0]=True;valid[0,1]=True
    r.reduce_trajectory_chunk(pred,target,valid,{'hand':0,'split':0,'user_ordinal':0},state)
    r.reduce_trajectory_chunk(pred[:,:1],target[:,:1],valid[:,:1],{'hand':1,'split':0,'user_ordinal':1},state)
    out=r.trajectory_report(state);a=out['overall']
    assert a['requested_samples']==3 and a['valid_angle_count']==5 and a['absolute_error_sum_rad']==7
    assert a['sample_error_sum_rad']==5 and a['sample_weighted_mae_deg']==pytest.approx(5/3*180/np.pi)
    assert a['user_summary']['mean_mae_deg']==pytest.approx((5/3+1)/2*180/np.pi)
    assert out['by_group']['thumb']['valid_angle_count']==5
    assert sum(out['overall']['histogram_valid_angles'])==5
    assert len(out['by_joint'])==22
    assert not any(isinstance(x,np.ndarray) for x in state.values())


def test_cache_source_is_still_bound_when_deserialization_finishes(tmp_path,center_fixture,monkeypatch):
    r=api('reducers');path=tmp_path/'cache.npz';receipt=r.write_private_cache(path,*center_fixture);original=r.np.load
    def mutate(*args,**kwargs):
        result=original(*args,**kwargs)
        with path.open('r+b') as stream:stream.write(b'XX')
        return result
    monkeypatch.setattr(r.np,'load',mutate)
    with pytest.raises(ValueError):r.read_private_cache(path,expected_sha256=receipt['sha256'])


def test_cli_does_not_unlink_someone_elses_marker_on_exclusive_create_collision(tmp_path,admitted_fixture,monkeypatch):
    c=api('cli');manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(admitted_fixture));output=tmp_path/'new'
    original=c.os.open
    def collide(path,*args,**kwargs):
        if Path(path)==output/'preflight.json':
            Path(path).write_bytes(b'not owned by this publication');raise FileExistsError('collision')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(c.os,'open',collide)
    assert c.main(['preflight','--manifest',str(manifest),'--output',str(output)])==1
    assert (output/'preflight.json').read_bytes()==b'not owned by this publication'


def test_rebinding_raw_hash_cannot_hide_semantically_equal_config_reformat(admitted_fixture):
    p=api('preflight');spec=admitted_fixture['runs'][0];path=Path(spec['run'])/'config.json'
    path.write_bytes(path.read_bytes()+b'\n')
    spec['files_sha256']['config.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):p.preflight_manifest(admitted_fixture)


def test_private_prediction_bytes_require_explicit_external_cache_receipt(tmp_path,center_fixture):
    r=api('reducers');path=tmp_path/'cache.npz';receipt=r.write_private_cache(path,*center_fixture)
    assert receipt['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):r.read_private_cache(path)
    with np.load(path,allow_pickle=False) as file:values={k:file[k] for k in file.files}
    values['prediction__s'][0,0]+=.5;np.savez(path,**values)
    with pytest.raises(ValueError):r.read_private_cache(path,expected_sha256=receipt['sha256'])


def test_preflight_fsync_failure_has_no_success_marker_or_unrelated_deletion(tmp_path,admitted_fixture,monkeypatch,capsys):
    c=api('cli');manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(admitted_fixture))
    sentinel=tmp_path/'unrelated';sentinel.write_bytes(b'keep');output=tmp_path/'new'
    monkeypatch.setattr(c.os,'fsync',lambda *a:(_ for _ in ()).throw(OSError('SECRET path error')))
    assert c.main(['preflight','--manifest',str(manifest),'--output',str(output)])==1
    assert not (output/'preflight.json').exists() and sentinel.read_bytes()==b'keep'
    assert 'SECRET' not in capsys.readouterr().err


def test_relocated_source_and_protocol_bytes_have_identical_safe_identity(tmp_path,admitted_fixture):
    p=api('preflight');before=p.preflight_manifest(admitted_fixture);moved=copy.deepcopy(admitted_fixture)
    new_root=tmp_path/'relocated-source';shutil.copytree(moved['legacy_root'],new_root);moved['legacy_root']=str(new_root)
    for spec in moved['runs']:
        destination=tmp_path/('relocated-'+spec['model']);shutil.copytree(spec['run'],destination);spec['run']=str(destination)
    assert p.preflight_manifest(moved)==before

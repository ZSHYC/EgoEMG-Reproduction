"""Exact original ordinary-data source, factory, recipe and strict weight reload."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from test_emg2pose_runtime import corpus as native_corpus,UPSTREAM


def source_api():return importlib.import_module('egoemg_repro.emg2pose_direct_source')
def runtime():return importlib.import_module('egoemg_repro.emg2pose_direct_runtime')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@pytest.fixture(scope='module')
def pipeline_corpus(native_corpus,tmp_path_factory):
    from test_emg2pose_disk_prepare import case as scratch
    from test_emg2pose_memory_data import make_tar
    from egoemg_repro import emg2pose_disk_pipeline as pipeline,emg2pose_tar_preservation as kept
    root=tmp_path_factory.mktemp('direct-original-pipeline')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(kept,'available_memory_bytes',lambda:2*1024**4)
        c=scratch.__wrapped__(root,patch)
        raw=make_tar([(p.name,b'0',p.read_bytes()) for p in sorted(native_corpus.root.glob('*.hdf5'))])
        c['archive'].chmod(0o600);c['archive'].write_bytes(raw);c['archive'].chmod(0o400)
        r=c['receipt'];r['source']['expected_bytes']=len(raw);r['source_bytes_received']=len(raw);r['full_sha256']=hashlib.sha256(raw).hexdigest()
        (c['download']/'receipt.json').write_text(json.dumps(r))
        out=root/'source';pipeline.prepare_original_data(c['download'],out,upstream=UPSTREAM,metadata_csv=native_corpus.metadata_csv,
            cohort='engineering',expected_download_receipt_sha256=sha(c['download']/'receipt.json'))
    return SimpleNamespace(root=out,csv=native_corpus.metadata_csv)

@pytest.fixture
def case(pipeline_corpus,tmp_path):
    root=tmp_path/'source';shutil.copytree(pipeline_corpus.root,root)
    csv=tmp_path/'metadata.csv';shutil.copy2(pipeline_corpus.csv,csv)
    return SimpleNamespace(root=root,csv=csv)

def open_source(c,**kwargs):
    return source_api().open_source(c.root,upstream=UPSTREAM,metadata_csv=c.csv,
        expected_completion_sha256=kwargs.pop('expected',sha(c.root/'completion.json')),**kwargs)

def config(c,s,size='small'):
    pytest.importorskip('hydra')
    return runtime().compose_config(UPSTREAM,size,s.index,source=s,metadata_csv=c.csv)

def framework():
    t=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');pytest.importorskip('hydra');return t


def test_actual_direct_source_admits_original_fields_and_returns_private_safe_copies(case):
    with open_source(case) as s:
        assert type(s) is source_api().VerifiedDirectSource and not hasattr(s,'fileno')
        assert s.index==case.root/'data' and s.data_identity['file_count']==3 and s.data_identity['rows']==25710
        assert s.data_identity['input_channels']==16 and s.data_identity['angle_dimensions']==20
        assert s.data_identity['archive_bytes']>0 and s.data_identity['full_corpus_complete'] is False
        identity=s.data_identity;identity['full_corpus_complete']=True;identity['counts'].clear()
        assert s.data_identity['full_corpus_complete'] is False and s.data_identity['counts']
        assert str(case.root) not in json.dumps(s.data_identity) and 'private-runtime-user' not in json.dumps(s.data_identity)
    with pytest.raises(ValueError):s.assert_current()

def test_constructing_unadmitted_direct_source_is_not_public(case):
    with pytest.raises(TypeError):source_api().VerifiedDirectSource(case.root)

def test_separately_frozen_pipeline_marker_is_required(case):
    with pytest.raises(ValueError):open_source(case,expected='0'*64)

@pytest.mark.parametrize('fault',['format','incomplete','full','path','producer','data_identity','preparation','retired','science'])
def test_resealed_invalid_pipeline_cannot_admit_direct_data(case,fault):
    p=case.root/'completion.json';r=json.loads(p.read_text())
    if fault=='format':r['format']='emg2pose_memory_source_v1'
    elif fault=='incomplete':r['pipeline_complete']=False
    elif fault=='full':r['full_corpus_complete']=True
    elif fault=='path':r['data_directory']='../other'
    elif fault=='producer':r['producer_files_sha256']['emg2pose_disk_pipeline.py']='0'*64
    elif fault=='data_identity':r['data_identity']['fields']['emg']['sha256']='0'*64
    elif fault=='preparation':r['preparation_sha256']='0'*64
    elif fault=='retired':r['redundant_tar_retired']=False
    else:r['scientific_complete']=True
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):open_source(case)

@pytest.mark.parametrize('name',['data/emg.dat','data/joint_angles.dat','data/time.dat','data/valid_mask.dat',
    'data/metadata.npz','data/manifest.json','data/completion.json','prepared/source-inventory.json','completion.json'])
def test_every_field_and_transaction_binding_is_live_after_admission(case,name):
    with open_source(case) as s:
        p=case.root/name;b=p.read_bytes();p.write_bytes(b[:-1]+bytes([b[-1]^1]))
        with pytest.raises(ValueError):s.assert_current()

def test_csv_and_restored_field_identity_changes_are_detected(case):
    with open_source(case) as s:
        p=case.root/'data/emg.dat';b=p.read_bytes();st=p.stat();p.write_bytes(b);os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
        with pytest.raises(ValueError):s.assert_current()
    with open_source(case) as s:
        case.csv.write_text(case.csv.read_text()+'\n')
        with pytest.raises(ValueError):s.assert_current()

@pytest.mark.parametrize('fault',['missing','replace','corrupt'])
@pytest.mark.parametrize('boundary',['live','fresh'])
def test_retained_original_payload_changes_refuse_live_and_fresh_admission(case,fault,boundary):
    preserved=case.root/'prepared/preserved'
    path=next((preserved/'contents').rglob('*.hdf5'))
    with open_source(case) as source:
        if fault=='missing':path.unlink()
        elif fault=='replace':
            replacement=path.with_suffix('.replacement');replacement.write_bytes(b'X'*path.stat().st_size)
            os.replace(replacement,path)
        else:
            with path.open('r+b') as stream:
                stream.seek(-1,2);last=stream.read(1);stream.seek(-1,2);stream.write(bytes([last[0]^1]))
        if boundary=='live':
            with pytest.raises(ValueError):source.assert_current()
        else:
            with pytest.raises(ValueError):open_source(case)


def test_field_replacement_after_original_deep_admission_is_rejected(case,monkeypatch):
    from egoemg_repro import emg2pose_runtime as native
    original=native.validate_completed_data
    def changed(*a,**k):
        result=original(*a,**k);p=case.root/'data/emg.dat';copy=p.with_name('replacement.dat');shutil.copy2(p,copy);os.replace(copy,p);return result
    monkeypatch.setattr(native,'validate_completed_data',changed)
    with pytest.raises(ValueError):open_source(case)

@pytest.mark.parametrize('size,parameters',[('small',3481636),('middle',6638884),('large',16342308)])
def test_original_seeded_factory_forward_optimizer_and_strict_reload(case,size,parameters):
    torch=framework();from egoemg_repro import emg2pose_runtime as native
    with open_source(case) as s:
        cfg=config(case,s,size);identity=runtime().runtime_identity(UPSTREAM,cfg,source=s)
        module,dm=runtime().build_runtime(UPSTREAM,cfg,source=s)
        reference_cfg=native.compose_config(UPSTREAM,size,s.index,source_receipt=case.root/'prepared/source-inventory.json',metadata_csv=case.csv)
        reference,rdm=native.build_runtime(UPSTREAM,reference_cfg)
        assert sum(p.numel() for p in module.model.parameters())==parameters
        assert all(torch.equal(v,reference.state_dict()[k]) for k,v in module.state_dict().items())
        dm.setup();rdm.setup();a=dm._collate_fn([dm.val_dataset[0]]);b=rdm._collate_fn([rdm.val_dataset[0]])
        assert all(torch.equal(a[k],b[k]) for k in ('emg','joint_angles','label_valid_mask'))
        import random,numpy as np
        rng=(random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone())
        train_a=dm.train_dataset[0]
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.random.set_rng_state(rng[2])
        train_b=rdm.train_dataset[0]
        for k in ('emg','joint_angles','label_valid_mask'):
            assert np.asarray(train_a[k]).dtype==np.asarray(train_b[k]).dtype and np.asarray(train_a[k]).tobytes()==np.asarray(train_b[k]).tobytes()
        assert train_a['window_start_idx']==train_b['window_start_idx']
        module.eval();reference.eval()
        with torch.no_grad():actual=module(a);expected=reference(b)
        assert all(torch.equal(x,y) for x,y in zip(actual,expected))
        assert tuple(actual[0].shape)==tuple(actual[1].shape)==(1,20,7280) and tuple(actual[2].shape)==(1,7280)
        opt=module.configure_optimizers();assert opt['optimizer'].defaults['lr']==.001
        assert opt['lr_scheduler']['scheduler'].T_max==250
        restored,other=runtime().reload_runtime(UPSTREAM,cfg,module.state_dict(),identity=identity,source=s)
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in module.state_dict().items())
        for data in (dm,rdm,other):runtime().close_data_module(data)
        assert not dm.val_dataset._memmaps

@pytest.mark.parametrize('fault',['lr','window','batch','dataset_target','weights','binding'])
def test_config_changes_rejected_before_rng_or_model_construction(case,fault):
    torch=framework()
    with open_source(case) as s:
        cfg=config(case,s)
        if fault=='lr':cfg.optimizer.lr=.2
        elif fault=='window':cfg.dataset.val[0].window_length=9000
        elif fault=='batch':cfg.batch_size=1
        elif fault=='dataset_target':cfg.dataset.val[0]._target_='different.Dataset'
        elif fault=='weights':cfg.pretrained_checkpoint='/unapproved/weights.pt'
        else:cfg[runtime().CONTEXT_KEY].source_completion_sha256='0'*64
        before=torch.random.get_rng_state().clone()
        with pytest.raises(ValueError):runtime().build_runtime(UPSTREAM,cfg,source=s)
        assert torch.equal(before,torch.random.get_rng_state())

def test_cached_runtime_uses_one_deep_admission_and_keeps_original_globals(case,monkeypatch):
    framework();from egoemg_repro import emg2pose_runtime as native
    original=native.build_runtime.__globals__['_validated_config']
    with open_source(case) as s:
        def forbid(*a,**k):raise AssertionError('repeated full field hashing after deep source admission')
        monkeypatch.setattr(native,'validate_completed_data',forbid)
        cfg=config(case,s);module,dm=runtime().build_runtime(UPSTREAM,cfg,source=s)
        assert runtime().runtime_identity(UPSTREAM,cfg,source=s)['data']==s.data_identity
        assert native.build_runtime.__globals__['_validated_config'] is original
        runtime().close_data_module(dm)

def test_closed_and_dictionary_sources_cannot_construct_models(case):
    torch=framework();fake=SimpleNamespace(index=case.root/'data',assert_current=lambda:None,data_identity={'full_corpus_complete':True})
    before=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):runtime().compose_config(UPSTREAM,'small',fake.index,source=fake,metadata_csv=case.csv)
    s=open_source(case);cfg=config(case,s);s.close()
    with pytest.raises(ValueError):runtime().build_runtime(UPSTREAM,cfg,source=s)
    assert torch.equal(before,torch.random.get_rng_state())

def test_exact_relocation_keeps_content_identity_but_wrong_concrete_source_fails(case,tmp_path):
    framework();moved=tmp_path/'relocated';shutil.copytree(case.root,moved)
    other=SimpleNamespace(root=moved,csv=case.csv)
    with open_source(case) as a,open_source(other) as b:
        first=config(case,a);second=config(other,b)
        assert runtime().runtime_identity(UPSTREAM,first,source=a)==runtime().runtime_identity(UPSTREAM,second,source=b)
        with pytest.raises(ValueError):runtime().build_runtime(UPSTREAM,first,source=b)

def test_strict_reload_rejects_half_precision_tensor_state(case):
    torch=framework()
    with open_source(case) as s:
        cfg=config(case,s);module,dm=runtime().build_runtime(UPSTREAM,cfg,source=s);state=dict(module.state_dict());name=next(k for k,v in state.items() if v.dtype==torch.float32);state[name]=state[name].half()
        with pytest.raises(ValueError):runtime().reload_runtime(UPSTREAM,cfg,state,identity=runtime().runtime_identity(UPSTREAM,cfg,source=s),source=s)
        runtime().close_data_module(dm)

def test_foreign_old_runtime_binding_is_not_silently_dropped(case):
    framework();from egoemg_repro import emg2pose_runtime as native
    from omegaconf import OmegaConf
    with open_source(case) as s:
        cfg=config(case,s);OmegaConf.update(cfg,native.CONTEXT_KEY,{'format':'foreign-original-binding','data':'/other'},force_add=True)
        with pytest.raises(ValueError):runtime().build_runtime(UPSTREAM,cfg,source=s)

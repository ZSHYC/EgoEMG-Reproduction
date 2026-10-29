"""Recovery admission must reject unsafe parents before loading model tensors."""
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest


def api():
    try:
        return importlib.import_module('egoemg_repro.wilor_recovery')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.wilor_recovery':
            pytest.fail('explicit native WiLoR recovery admission is missing')
        raise


def seal(value):
    value = dict(value)
    value['sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return value


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def journal(path,identity,count=3,*,bad_epoch=False):
    previous='0'*64;lines=[]
    for n in range(count):
        record=dict(epoch=n+1 if bad_epoch else n,optimizer_executions=(n+1)*11224,
            train_samples=202020,train_batches=11224,global_step=(n+1)*11224,train_loss=1.,
            validation=dict(samples=3152,val_mae=.1))
        entry=seal(dict(previous_sha256=previous,identity_sha256=identity['sha256'],record=record))
        previous=entry['sha256'];lines.append(json.dumps(entry,sort_keys=True,separators=(',',':'))+'\n')
    path.write_text(''.join(lines))


@pytest.fixture
def parent(tmp_path):
    fit=tmp_path/'private/fit';(fit/'checkpoints').mkdir(parents=True)
    # These files exercise metadata rejection only and are never native permits.
    identity=seal(dict(format='native_wilor_uninterrupted_fit_v1',policy=dict(scope='full',max_epochs=30),
        counts=dict(train=202020,val=3152,test=8612),scientific_eligible=False,full_paper_complete=False))
    put(fit/'identity.json',identity);journal(fit/'epochs.jsonl',identity)
    for n in range(3):(fit/'checkpoints'/f'vit-small-epoch={n:03d}-val_mae=0.1000.ckpt').write_bytes(b'metadata-only')
    (fit/'checkpoints/last.ckpt').write_bytes(b'metadata-only')
    dead=lambda pid:dict(pid=pid,start_ticks='1',uid=os.getuid(),sid=pid,pgrp=pid,state='R')
    reg=dict(owner=dead(2147483644),worker=dead(2147483645),guardian=dead(2147483646),descriptor_bindings={})
    put(fit.parent/'owned-fit/started.json',reg)
    put(fit.parent/'owned-fit/stage-result.json',dict(reg,status='failed',worker_exit_code=1,all_owned_consumers_exited=True))
    return fit


def test_metadata_inspection_never_claims_native_admission(parent):
    result=api()._parent_metadata(parent)
    assert result['raw_checkpoint_admitted'] is False
    assert len(result['records'])==3
    assert result['identity']['counts']==dict(train=202020,val=3152,test=8612)


@pytest.mark.parametrize('change', ['identity-seal','identity-format','scope','bool-count','full-count',
    'partial-journal','empty-journal','broken-chain','nonconsecutive-epoch','completed-budget',
    'missing-terminal','unclean-terminal','bool-exit','mismatched-registration','live-worker','live-guardian',
    'symlink-checkpoint','directory-checkpoint'])
def test_unsafe_parent_rejects_before_native_checkpoint_loading(parent,change):
    identity=json.loads((parent/'identity.json').read_text())
    if change in ('identity-seal','identity-format','scope','bool-count','full-count'):
        if change=='identity-seal':identity['sha256']='f'*64
        else:
            identity.pop('sha256')
            if change=='identity-format':identity['format']='unrelated_fit'
            elif change=='scope':identity['policy']['scope']='pretend_full'
            elif change=='bool-count':identity['counts']['train']=True
            else:identity['counts']['train']=2
            identity=seal(identity)
        put(parent/'identity.json',identity)
    elif change=='partial-journal':
        p=parent/'epochs.jsonl';p.write_bytes(p.read_bytes()[:-1])
    elif change=='empty-journal':(parent/'epochs.jsonl').write_bytes(b'')
    elif change=='broken-chain':
        p=parent/'epochs.jsonl';p.write_text(p.read_text().replace('"previous_sha256":"000','"previous_sha256":"111',1))
    elif change=='nonconsecutive-epoch':journal(parent/'epochs.jsonl',identity,bad_epoch=True)
    elif change=='completed-budget':journal(parent/'epochs.jsonl',identity,30)
    elif change=='missing-terminal':(parent.parent/'owned-fit/stage-result.json').unlink()
    elif change in ('symlink-checkpoint','directory-checkpoint'):
        p=parent/'checkpoints/last.ckpt';p.unlink()
        if change=='symlink-checkpoint':p.symlink_to(parent/'identity.json')
        else:p.mkdir()
    else:
        p=parent.parent/'owned-fit/stage-result.json';result=json.loads(p.read_text())
        if change=='unclean-terminal':result['all_owned_consumers_exited']=False
        elif change=='bool-exit':result['worker_exit_code']=False
        elif change=='mismatched-registration':result['worker']['start_ticks']='2'
        else:
            from egoemg_repro.emg2pose_memory_process import process_identity
            key='worker'if change=='live-worker'else'guardian';result[key]=process_identity()
            reg=json.loads((parent.parent/'owned-fit/started.json').read_text());reg[key]=result[key]
            put(parent.parent/'owned-fit/started.json',reg)
        put(p,result)
    with pytest.raises((ValueError,FileNotFoundError),match='(parent|journal|identity|scope|count|owner|checkpoint|terminal|budget|registration)'):
        api().admit_wilor_recovery(parent)


def test_full_parent_cannot_select_older_packet_by_argument(parent):
    with pytest.raises(ValueError,match='(full|checkpoint)'):
        api().admit_wilor_recovery(parent,checkpoint=parent/'checkpoints/vit-small-epoch=000-val_mae=0.1000.ckpt')


def test_no_checkpoints_is_not_a_recoverable_parent(parent):
    for path in (parent/'checkpoints').iterdir():path.unlink()
    with pytest.raises(ValueError,match='checkpoint'):
        api().admit_wilor_recovery(parent)


def test_existing_native_fit_api_keeps_explicit_no_resume_contract(parent):
    api()
    from egoemg_repro.wilor_native_train import fit_wilor
    with pytest.raises(FileExistsError,match='no overwrite/resume'):
        fit_wilor(None,None,upstream=None,config=None,assets=None,output=parent,scope='cpu_engineering')


@pytest.fixture
def callback_packet(parent):
    p=str(parent/'checkpoints/vit-small-epoch=000-val_mae=0.1000.ckpt')
    key="ModelCheckpoint{'monitor': 'val_mae', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"
    early="EarlyStopping{'monitor': 'val_mae', 'mode': 'min'}"
    # Full native state_dict shapes; mode belongs to state_key, not state_dict.
    checkpoint=dict(callbacks={key:dict(monitor='val_mae',best_model_score=.1,best_model_path=p,
        current_score=.1,dirpath=str(parent/'checkpoints'),best_k_models={p:.1},
        kth_best_model_path='',kth_value=float('inf'),last_model_path=''),
        early:dict(wait_count=0,stopped_epoch=0,best_score=.1,patience=30)})
    return checkpoint,[dict(validation=dict(val_mae=.1))],p


def test_native_callback_mode_is_admitted_from_state_key(parent,callback_packet):
    packet,records,path=callback_packet
    assert api()._checkpoint_callbacks(packet,parent,records)=={Path(path):.1}


@pytest.mark.parametrize('change', ['mode','monitor','bool-interval','duplicate-key','dirpath',
    'best-score','topk-score','missing-best','early-best','early-wait','early-patience','bool-wait','bool-stopped'])
def test_callback_state_cannot_forge_original_minimum_and_history(parent,callback_packet,change):
    packet,records,path=callback_packet;callbacks=packet['callbacks']
    key=next(k for k in callbacks if k.startswith('ModelCheckpoint'))
    cb=callbacks[key];early=next(v for k,v in callbacks.items()if k.startswith('EarlyStopping'))
    if change in ('mode','monitor','bool-interval'):
        new=key.replace("'mode': 'min'","'mode': 'max'")if change=='mode'else (
            key.replace("'monitor': 'val_mae'","'monitor': 'test_mae'")if change=='monitor'else
            key.replace("'every_n_train_steps': 0","'every_n_train_steps': False"))
        callbacks[new]=callbacks.pop(key)
    elif change=='duplicate-key':callbacks[key+' ']=dict(cb)
    elif change=='dirpath':cb['dirpath']=str(parent.parent/'other')
    elif change=='best-score':cb['best_model_score']=.2
    elif change=='topk-score':cb['best_k_models'][path]=.2
    elif change=='missing-best':cb['best_model_path']=''
    elif change=='early-best':early['best_score']=.2
    elif change=='early-wait':early['wait_count']=1
    elif change=='early-patience':early['patience']=29
    elif change=='bool-wait':early['wait_count']=False
    else:early['stopped_epoch']=False
    with pytest.raises(ValueError,match='callback'):
        api()._checkpoint_callbacks(packet,parent,records)


def test_metadata_changed_during_parent_inspection_is_not_pinned_as_original(parent,monkeypatch):
    module=api();original=module._producer
    def producer(path):
        result=original(path)
        identity=json.loads((parent/'identity.json').read_text());identity.pop('sha256')
        identity['counts']['train']=2;put(parent/'identity.json',seal(identity))
        return result
    monkeypatch.setattr(module,'_producer',producer)
    with pytest.raises(ValueError,match='(metadata|parent).*changed'):
        module._parent_metadata(parent)

"""Real historical LAST reader and future-holdout admission, never mocked fits."""
import copy
import importlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',ROOT/'upstream')).resolve()
RUN=Path(os.environ.get('EGOEMG_LEGACY_RN18_RUN',str(ROOT/'runs/rn18-seed42')))
SOURCE=Path(os.environ.get('EGOEMG_LEGACY_RN18_SOURCE',str(UPSTREAM.parent)))


def api():
    try:return importlib.import_module('egoemg_repro.rn18_fusion_handoff')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.rn18_fusion_handoff':pytest.fail('RN18 trained producer handoff missing')
        raise


@pytest.fixture(scope='module')
def legacy():
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');torch.set_num_threads(2)
    if not(RUN/'last.pt').is_file():
        if os.environ.get('EGOEMG_REQUIRE_RN18_LEGACY_PRODUCER_TEST')=='1':pytest.fail('real completed legacy RN18 run required')
        pytest.skip('real completed legacy RN18 run absent; no synthetic full-trained positive')
    packet=torch.load(str(RUN/'last.pt'),map_location='cpu',weights_only=False,mmap=True)
    events=[json.loads(line)for line in(RUN/'progress.jsonl').read_text().splitlines()]
    return packet,events,json.loads((RUN/'config.json').read_text())


def test_real_legacy_last_reopens_all_training_domains_without_models_or_rng_changes(legacy,monkeypatch):
    import torch
    def denied(*a,**kw):raise AssertionError('legacy producer reader cannot construct a neural module/optimizer')
    monkeypatch.setattr(torch.nn.Module,'__init__',denied);monkeypatch.setattr(torch.optim.AdamW,'__init__',denied)
    before=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
    p=api().read_legacy_rn18_producer(RUN,source_root=SOURCE,upstream=UPSTREAM);v=p.identity
    assert v['producer_family']=='legacy-resnet18-150-0.01-nojitter'
    assert v['checkpoint_selection']=='LAST'and v['completed_epochs']==150 and v['optimizer_executions']==10650
    assert v['native_state_tensors']==124 and v['native_optimizer_parameters']==64
    assert v['source_checkpoint_sha256']=='e9bf7c7fa2abee7997c2c7c50f750137ded4ae8dad1cba998f52392516db5fc0'
    assert v['source_validation_selected_step']==6319 and v['source_validation_selected_step']!=v['optimizer_executions']
    assert v['actual_history_counts']==dict(start=1,train_step=10650,validation=150,final_test=1)
    assert len(v['tensor_inventory'])==124 and v['source_recipe']['fingertip_weight']==.01
    assert v['producer_training_verified']and not v['current_raw_input_bytes_revalidated']and not v['fusion_initializers_complete']
    assert torch.equal(before[0],torch.get_rng_state())and before[2]==random.getstate()
    after=np.random.get_state();assert np.array_equal(before[1][1],after[1])and before[1][2:]==after[2:]
    assert not torch.cuda.is_initialized();p.check()


@pytest.mark.parametrize('damage',['optimizer_step','optimizer_shape','scheduler_budget','scheduler_epoch','rng','loop_epoch','model_dtype','batchnorm_counter'])
def test_actual_native_saved_domain_damage_cannot_pass(legacy,damage):
    import torch
    original,_,cfg=legacy;packet=copy.copy(original)
    if damage.startswith('optimizer'):
        packet['optimizer']=copy.deepcopy(original['optimizer']);key=next(iter(packet['optimizer']['state']))
        if damage=='optimizer_step':packet['optimizer']['state'][key]['step'].sub_(1)
        else:packet['optimizer']['state'][key]['exp_avg']=torch.ones(1)
    elif damage.startswith('scheduler'):
        packet['scheduler']=dict(original['scheduler']);packet['scheduler']['T_max'if damage=='scheduler_budget'else'last_epoch']=149
    elif damage=='rng':packet['rng']=dict(original['rng'],torch=torch.zeros(1,dtype=torch.uint8))
    elif damage=='loop_epoch':packet['epoch']=149
    else:
        packet['model']=dict(original['model'])
        if damage=='model_dtype':key=next(iter(packet['model']));packet['model'][key]=packet['model'][key].half()
        else:key=next(k for k in packet['model']if k.endswith('num_batches_tracked'));packet['model'][key]=packet['model'][key]-1
    with pytest.raises(ValueError):api()._validate_legacy_domains(packet,cfg)


@pytest.mark.parametrize('damage',['duplicate_update','missing_validation','bounded_validation','best_score','start_scope','final_step'])
def test_actual_complete_history_damage_cannot_pass(legacy,damage):
    packet,original,cfg=legacy;events=list(original)
    if damage=='duplicate_update':i=next(i for i,x in enumerate(events)if x['event']=='train_step');events.insert(i,events[i])
    elif damage=='missing_validation':i=next(i for i,x in enumerate(events)if x['event']=='validation');events.pop(i)
    elif damage=='bounded_validation':
        i=next(i for i,x in enumerate(events)if x['event']=='validation');events[i]=copy.deepcopy(events[i]);events[i]['metrics']['bounded']=True
    elif damage=='best_score':
        i=next(i for i,x in enumerate(events)if x['event']=='validation');events[i]=dict(events[i],best_val_mae_deg=0.)
    elif damage=='start_scope':events[0]=dict(events[0],bounded_integration=True)
    else:events[-1]=dict(events[-1],global_step=10649)
    with pytest.raises(ValueError):api()._legacy_history(events,packet,cfg)


@pytest.mark.parametrize('damage',['config','source','missing_history','best_for_last'])
def test_changed_run_or_unknown_reader_rejects_before_constructors(legacy,tmp_path,monkeypatch,damage):
    import torch
    m=api();run=tmp_path/'run';run.mkdir()
    for p in RUN.iterdir():
        if p.is_file()and p.name in m.LEGACY_FILES:
            shutil.copyfile(p,run/p.name)
    source=SOURCE
    if damage=='config':
        p=run/'config.json';p.unlink();cfg=dict(legacy[2],epochs=149);p.write_text(json.dumps(cfg))
    elif damage=='source':
        source=tmp_path/'source'
        for n in m.READER_SOURCES:
            p=source/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/n,p)
        with(source/'egoemg_repro/train.py').open('a')as f:f.write('\n# changed\n')
    elif damage=='missing_history':(run/'progress.jsonl').unlink()
    else:
        (run/'last.pt').unlink();shutil.copyfile(RUN/'best.pt',run/'last.pt')
    def denied(*a,**kw):raise AssertionError('invalid producer reached model construction')
    monkeypatch.setattr(torch.nn.Module,'__init__',denied)
    with pytest.raises(ValueError):m.read_legacy_rn18_producer(run,source_root=source,upstream=UPSTREAM)


def test_copied_or_resealed_producer_identity_is_not_registered(legacy):
    from egoemg_repro.classic_emg import _hash
    p=api().read_legacy_rn18_producer(RUN,source_root=SOURCE,upstream=UPSTREAM)
    fake=copy.copy(p)
    with pytest.raises(ValueError):fake.check()
    p.identity['checkpoint_selection']='BEST';p.identity['sha256']=_hash({k:v for k,v in p.identity.items()if k!='sha256'})
    with pytest.raises(ValueError):p.check()


def test_unregistered_cohort_cannot_export_even_a_real_completed_vision_branch(tmp_path):
    with pytest.raises(ValueError):api().admit_rn18_fusion_vision_last(object(),RUN,tmp_path/'output',vision_source_root=SOURCE)


def test_missing_legacy_run_has_no_metadata_or_json_admission(tmp_path):
    with pytest.raises(ValueError):api().read_legacy_rn18_producer(tmp_path,source_root=SOURCE,upstream=UPSTREAM)


@pytest.fixture(scope='module')
def full_cohort():
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    request=os.environ.get('EGOEMG_RN18_FUSION_FULL_INPUTS_JSON')
    if not request:
        pytest.skip('registered complete actual fusion input request unavailable; no fake full-positive cohort')
    q=json.loads(Path(request).read_text())
    from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
    from egoemg_repro.rn18_fusion_cohort import admit_rn18_fusion_cohort
    c=admit_rn18_fusion_cohort(q['data'],q['crops'],upstream=UPSTREAM,config=compose_rn18_fusion(UPSTREAM))
    assert c.identity['counts']==dict(train=62470,val=1052,test=2792)
    return c


def test_actual_complete_vision_last_export_and_independent_source_bound_reopen(full_cohort,legacy,tmp_path,monkeypatch):
    import torch
    def denied(*a,**kw):raise AssertionError('trained branch admission/reopening must not construct a neural module or optimizer')
    monkeypatch.setattr(torch.nn.Module,'__init__',denied);monkeypatch.setattr(torch.optim.AdamW,'__init__',denied)
    m=api();out=tmp_path/'vision-last'
    p=m.admit_rn18_fusion_vision_last(full_cohort,RUN,out,vision_source_root=SOURCE)
    p.check();v=p.identity;binding=v['current_input_binding']
    assert v['producer_training_verified']and v['current_raw_input_bytes_revalidated']
    assert binding['footprints']['overlap_rows']==dict(val=0,test=0)
    assert binding['source_counts']==dict(train=63172,val=1052,test=2792)
    assert binding['current_fusion_counts']==dict(train=62470,val=1052,test=2792)
    assert binding['shared_actual_payload_files']==53 and len(binding['legacy_only_wrist_valid_payloads'])==2
    assert len(v['tensor_inventory'])==124 and v['checkpoint_selection']=='LAST'
    assert not v['fusion_initializers_complete']and not v['emg_initializer_admitted']and not v['scientific_eligible']
    saved=torch.load(str(p.checkpoint),map_location='cpu',weights_only=False,mmap=True)
    assert set(saved['state_dict'])==set(legacy[0]['model'])
    assert all(torch.equal(saved['state_dict'][n],t)for n,t in legacy[0]['model'].items())
    reopened=m.read_rn18_fusion_vision_last(full_cohort,RUN,out,vision_source_root=SOURCE)
    assert reopened.identity==v and not torch.cuda.is_initialized()
    fake=copy.copy(reopened)
    with pytest.raises(ValueError):fake.check()
    receipt=out/'vision-last-admission.json';changed=copy.deepcopy(v)
    changed['producer']['completed_epochs']=149
    from egoemg_repro.classic_emg import _hash
    changed['sha256']=_hash({k:x for k,x in changed.items()if k!='sha256'})
    receipt.write_text(json.dumps(changed))
    with pytest.raises(ValueError):reopened.check()
    with pytest.raises(ValueError):m.read_rn18_fusion_vision_last(full_cohort,RUN,out,vision_source_root=SOURCE)

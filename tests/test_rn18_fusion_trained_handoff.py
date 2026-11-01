"""Real fixed-holdout producer reads; no forged full-trained positive."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest

from test_fusion_literal_data import joint_store,UPSTREAM
from test_vision_literal_data import crop_store,release_fixture_readers

ROOT=Path(__file__).resolve().parents[1]


def api():
    try:return importlib.import_module('egoemg_repro.rn18_fusion_trained_handoff')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.rn18_fusion_trained_handoff':pytest.fail('RN18 combined trained-producer handoff missing')
        raise


@pytest.fixture(scope='module')
def producer(tmp_path_factory):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');api()
    retained=os.environ.get('EGOEMG_RN18_HANDOFF_ENGINEERING')
    if retained:
        root=Path(retained);assert(root/'fit/fit-result.json').is_file()
        try:yield root,root/'data',root/'crops',root/'fit',root/'emg-best.pt'
        finally:release_fixture_readers(root)
        return
    root=tmp_path_factory.mktemp('rn18-matched-producer');data=root/'data';joint_store(data,n=9*12000)
    np.repeat([0,1,2,3],[6*12000,12000,12000,12000]).astype(np.int8).tofile(data/'frame_split_id.dat')
    crops=root/'crops';crop_store(crops,flat=True);fit=root/'fit'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-u','-B','-m','egoemg_repro.rn18_initializer_train','fit','--data',str(data),'--upstream',str(UPSTREAM),'--output',str(fit),'--scope','cpu_engineering']
    for label,args in [('prefix',['--stop-after-epoch','1']),('resume',['--resume'])]:
        p=subprocess.run(command+args,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        (root/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-3500:]
    from egoemg_repro.rn18_initializer_train import export_rn18_initializer_best
    exported=root/'emg-best.pt';export_rn18_initializer_best(fit,exported,require_full=False)
    try:yield root,data,crops,fit,exported
    finally:release_fixture_readers(root)


def cohort(producer):
    from egoemg_repro.rn18_fusion_literal import compose_rn18_fusion
    from egoemg_repro.rn18_fusion_cohort import admit_rn18_fusion_cohort
    _,data,crops,_,_=producer
    return admit_rn18_fusion_cohort(data,crops,upstream=UPSTREAM,config=compose_rn18_fusion(UPSTREAM),require_full=False)


def test_real_best_reader_checks_every_tensor_and_fixed_future_io_without_models_or_rng_changes(producer,monkeypatch):
    import torch
    from egoemg_repro.rn18_initializer_train import read_rn18_initializer_export
    c=cohort(producer);_,_,_,fit,export=producer
    lineage=read_rn18_initializer_export(export,source_fit=fit,require_full=False)
    def denied(*a,**k):raise AssertionError('handoff reader cannot construct a model or optimizer')
    monkeypatch.setattr(torch.nn.Module,'__init__',denied);monkeypatch.setattr(torch.optim.Optimizer,'__init__',denied)
    before=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
    p=api().read_rn18_fusion_emg_best(c,fit,export,source_root=ROOT,require_full=False);v=p.identity
    assert v['producer_family']=='rn18-fixed-holdout-cotrain-small-150-0.01'
    assert v['scope']=='engineering'and v['completed_epochs']==2 and v['optimizer_executions']==4
    assert v['checkpoint_selection']=='native validation BEST'and v['source_selected_checkpoint_sha256']==lineage['source_checkpoint_sha256']
    assert v['tensor_inventory']==lineage['tensor_inventory']and len(v['tensor_inventory'])==110
    assert v['model_state_sha256']==lineage['state_sha256']
    assert v['future_fusion_partition_content_sha256']==c.identity['partition_content_sha256']
    assert v['input_binding']['footprints']['overlap_rows']==dict(val=0,test=0)
    assert v['input_binding']['shared_payload_files']>=12 and not v['full_initializer']
    assert v['scientific_eligible']is v['full_paper_complete']is False
    p.check(require_full=False)
    with pytest.raises(ValueError,match='full|engineering'):p.check(require_full=True)
    assert torch.equal(before[0],torch.get_rng_state())and before[2]==random.getstate()
    after=np.random.get_state();assert np.array_equal(before[1][1],after[1])and before[1][2:]==after[2:]
    assert not torch.cuda.is_initialized()and '/home/'not in json.dumps(v)and '/tmp/'not in json.dumps(v)


@pytest.mark.parametrize('damage',['clone','resealed_scope','checkpoint_path','producer_path','config'])
def test_cached_registered_best_cannot_be_cloned_or_rebound(producer,damage):
    from egoemg_repro.classic_emg import _hash
    c=cohort(producer);_,_,_,fit,export=producer
    p=api().read_rn18_fusion_emg_best(c,fit,export,source_root=ROOT,require_full=False)
    if damage=='clone':p=copy.copy(p)
    elif damage=='resealed_scope':
        p.identity=copy.deepcopy(p.identity);p.identity.update(scope='full',full_initializer=True);p.identity['sha256']=_hash({k:v for k,v in p.identity.items()if k!='sha256'})
    elif damage=='checkpoint_path':p.checkpoint=fit/'checkpoints/last.ckpt'
    elif damage=='producer_path':p.source_fit=fit.parent/'other-fit'
    else:p._cohort.config.batch_size=100
    with pytest.raises(ValueError):p.check(require_full=False)


def test_actual_engineering_producer_is_rejected_by_full_gate_before_any_output(producer,tmp_path):
    _,_,_,fit,export=producer;m=api()
    with pytest.raises(ValueError,match='engineering|full'):
        m._read_emg_producer(fit,export,upstream=UPSTREAM,source_root=ROOT,require_full=True)
    destination=tmp_path/'full-handoff'
    with pytest.raises(ValueError):m.admit_rn18_fusion_initializers(cohort(producer),tmp_path/'vision',fit,destination,vision_source_root=ROOT,emg_source_root=ROOT)
    assert not destination.exists()


def test_current_live_partial_initializer_cannot_supply_a_full_handoff(tmp_path,monkeypatch):
    import torch
    fit=Path(os.environ.get('EGOEMG_RN18_LIVE_INITIALIZER','/data/shihao/egoemg-rn18-cotrain-original-full-cli-20261101/private/full-fit'))
    if not fit.exists():pytest.skip('actual live RN18 initializer unavailable in this environment')
    if(fit/'fit-result.json').exists():pytest.skip('real initializer is no longer partial; use complete-producer acceptance')
    def denied(*a,**k):raise AssertionError('partial producer reached a learned constructor')
    monkeypatch.setattr(torch.nn.Module,'__init__',denied)
    with pytest.raises(ValueError):api()._read_emg_producer(fit,tmp_path/'absent.pt',upstream=UPSTREAM,source_root=ROOT,require_full=True)


@pytest.mark.parametrize('damage',['tensor','family','selected_lineage','file_generation'])
def test_changed_real_export_or_wrong_family_cannot_be_admitted(producer,tmp_path,damage):
    import torch
    _,_,_,fit,export=producer;c=cohort(producer);dest=tmp_path/'best.pt';shutil.copyfile(export,dest)
    if damage=='file_generation':
        p=api().read_rn18_fusion_emg_best(c,fit,dest,source_root=ROOT,require_full=False)
        with dest.open('ab')as f:f.write(b'changed')
        with pytest.raises(ValueError):p.check(require_full=False)
        return
    packet=torch.load(dest,map_location='cpu',weights_only=False)
    if damage=='tensor':key=next(iter(packet['state_dict']));packet['state_dict'][key]=packet['state_dict'][key]+.25
    elif damage=='family':packet['emg_initializer_lineage']=packet.pop('rn18_initializer_lineage')
    else:packet['rn18_initializer_lineage']['selected_epoch']+=1
    torch.save(packet,dest)
    with pytest.raises(ValueError):api().read_rn18_fusion_emg_best(c,fit,dest,source_root=ROOT,require_full=False)


def test_full_reader_requires_real_source_root_and_actual_producer_sources(producer,tmp_path):
    _,_,_,fit,export=producer
    with pytest.raises(ValueError,match='source'):
        api().read_rn18_fusion_emg_best(cohort(producer),fit,export,source_root=tmp_path,require_full=False)


@pytest.mark.parametrize('damage',['future_fold','paired_emg_bytes'])
def test_actual_reconstructed_inputs_reject_another_fold_or_different_emg(producer,damage):
    from egoemg_repro.classic_emg_cohort import _StatGuard
    _,_,_,fit,_=producer;c=cohort(producer);identity=json.loads((fit/'identity.json').read_text())
    if damage=='future_fold':identity['cohort']['future_fusion_partition_content_sha256']='0'*64
    else:identity['cohort']['payload_files_sha256']['field:emg_left_filtered']='0'*64
    with pytest.raises(ValueError,match='fold|payload|bytes'):
        api()._input_binding(c,identity,_StatGuard([]))

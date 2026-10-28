"""Fresh native NeuroPose, never an old resumed-trajectory acceptance."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT=Path(__file__).parents[1]
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',PROJECT/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.neuropose_native_train')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.neuropose_native_train':pytest.fail('native uninterrupted NeuroPose entry is missing')
        raise


def test_full_policy_keeps_original_model_population_batch_precision_and_stopping():
    p=api().neuropose_policy()
    assert p['scope']=='full' and p['variant']=='neuropose'
    assert p['batch_size']==256 and p['counts']==dict(train=101714,val=1576,test=4306)
    assert p['trainer']['precision']=='16-mixed' and p['trainer']['max_epochs']==100
    assert p['trainer']['accumulate_grad_batches']==1 and p['trainer']['gradient_clip_val']==0
    assert p['trainer']['limit_train_batches']==p['trainer']['limit_val_batches']==1.0
    assert p['monitor']=='val_loss' and p['patience']==50 and p['num_workers']==0
    assert p['checkpoint_continuation_supported'] is False
    assert p['training_trajectory_agreement_claimed'] is False


@pytest.mark.parametrize('scope,batch,precision,train',[('cuda_engineering',256,'16-mixed',512),('cpu_engineering',2,'32-true',4)])
def test_engineering_scope_is_prospectively_bounded_and_never_full(scope,batch,precision,train):
    p=api().neuropose_policy(scope=scope)
    assert p['scope']==scope and p['batch_size']==batch and p['trainer']['precision']==precision
    assert p['trainer']['max_epochs']==2 and p['engineering_train_limit']==train
    assert p['scientific_eligible'] is False


@pytest.mark.parametrize('scope',[None,True,'cuda','formal_training'])
def test_unknown_scope_is_rejected(scope):
    with pytest.raises((ValueError,TypeError)):api().neuropose_policy(scope=scope)


@pytest.mark.parametrize('override',[dict(batch_size=128),dict(precision='bf16-mixed'),dict(max_epochs=2),dict(resume='last.ckpt'),dict(gradient_clip_val=1)])
def test_caller_cannot_replace_original_recipe_or_enable_continuation(override):
    with pytest.raises(TypeError):api().neuropose_policy(**override)


def test_full_entry_requires_raw_new_qualification_before_data_or_output(tmp_path):
    with pytest.raises(ValueError,match='qualification|precondition|artifact'):
        api().fit_neuropose(tmp_path/'missing',upstream=tmp_path/'missing',config=tmp_path/'missing',
            partitions={},output=tmp_path/'out',qualification={'passed':True},preconditions={'passed':True})
    assert not (tmp_path/'out').exists()


def test_old_failed_gate_source_is_not_changed_by_the_new_entry():
    assert hashlib.sha256((PROJECT/'egoemg_repro/classic_emg_formal.py').read_bytes()).hexdigest()=='9f2ecd61c0731571137a85bd61f75a624fa910c8a0fec3eb454ed4f3ef40ab31'


def test_other_original_variant_is_rejected_before_opening_data(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('hydra')
    with pytest.raises(ValueError,match='NeuroPose'):
        api().fit_neuropose(tmp_path/'missing',upstream=UPSTREAM,config=PROJECT/'configs/vemg2pose.json',
            partitions=tmp_path/'missing-partitions',output=tmp_path/'out',scope='cpu_engineering')
    assert not (tmp_path/'out').exists()


def test_cpu_rng_snapshot_never_queries_or_initializes_visible_cuda(monkeypatch):
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:pytest.fail('CPU scope must not read CUDA RNG'))
    value=api()._rng_packet({'train_loader':torch.Generator().manual_seed(42)},cpu=True)
    assert value['cuda']==[] and set(value['generators'])=={'train_loader'}
    assert value['torch'].dtype==torch.uint8


@pytest.fixture(scope='module')
def native_fit(tmp_path_factory):
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch_lightning');pytest.importorskip('hydra')
    torch.set_num_threads(2)
    retained=os.environ.get('EGOEMG_NEURO_NATIVE_RETAINED_FIT')
    if retained:
        root=Path(retained)
        return root,api().read_neuropose_fit(root/'fit',require_full=False)
    from test_classic_emg_training import corpus
    root=tmp_path_factory.mktemp('native-neuro')
    parts=corpus(root/'data');(root/'partitions.json').write_text(json.dumps(parts))
    result=api().fit_neuropose(root/'data',upstream=UPSTREAM,config=PROJECT/'configs/neuropose.json',
        partitions=root/'partitions.json',output=root/'fit',scope='cpu_engineering')
    return root,result


def test_actual_original_cpu_fit_commits_complete_epochs_and_native_best(native_fit):
    root,result=native_fit
    assert result['completed_epochs']==2 and result['optimizer_executions']==2
    assert result['train_samples_per_epoch']==[2,2] and result['val_samples_per_epoch']==[2,2]
    assert result['initial_model_sha256']!=result['last_model_sha256']
    assert result['full_training_terminated'] is False and result['scope']=='cpu_engineering'
    assert result['selected_best']['file']!='last.ckpt'
    assert api().read_neuropose_fit(root/'fit',require_full=False)==result
    with pytest.raises(ValueError,match='engineering|full'):api().read_neuropose_fit(root/'fit')


def test_repeated_fit_cannot_overwrite_owned_native_checkpoint(native_fit):
    root,_=native_fit;last=root/'fit/checkpoints/last.ckpt';before=last.read_bytes()
    with pytest.raises(FileExistsError):
        api().fit_neuropose(root/'data',upstream=UPSTREAM,config=PROJECT/'configs/neuropose.json',
            partitions=root/'partitions.json',output=root/'fit',scope='cpu_engineering')
    assert last.read_bytes()==before


def test_new_process_strict_best_reload_reproduces_entire_native_validation(native_fit):
    root,result=native_fit
    code="from egoemg_repro.neuropose_native_train import reopen_neuropose_best,_validation;import json,sys,torch;p=reopen_neuropose_best(sys.argv[1],require_full=False);print(json.dumps(_validation(p)));assert not torch.cuda.is_initialized()"
    p=subprocess.run([sys.executable,'-B','-c',code,str(root/'fit')],cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stderr
    value=json.loads(p.stdout.strip().splitlines()[-1]);assert value['samples']==2
    assert value['prediction_sha256']==result['selected_best']['validation_prediction_sha256']
    assert value['model_sha256']==result['selected_best']['model_sha256']


@pytest.mark.parametrize('damage',['journal_tail','wrong_epoch','promote_scope','wrong_best','changed_model',
    'modified_initial','optimizer_corruption','optimizer_counter','missing_rng'])
def test_resealed_summaries_and_raw_checkpoint_corruption_are_rejected(native_fit,damage):
    import torch
    root,_=native_fit;fit=root/'fit'
    names=['identity.json','fit-result.json','epochs.jsonl','checkpoints/last.ckpt']
    result=json.loads((fit/'fit-result.json').read_text());names.append('checkpoints/'+result['selected_best']['file'])
    before={name:(fit/name).read_bytes() for name in names}
    try:
        if damage=='journal_tail':
            with (fit/'epochs.jsonl').open('ab') as f:f.write(b'{}\n')
        elif damage=='changed_model':
            path=fit/'checkpoints'/result['selected_best']['file'];c=torch.load(path,map_location='cpu',weights_only=False)
            key=next(k for k,v in c['state_dict'].items() if torch.is_floating_point(v));c['state_dict'][key].add_(1);torch.save(c,path)
        elif damage in ('optimizer_corruption','optimizer_counter','missing_rng'):
            path=fit/'checkpoints/last.ckpt';c=torch.load(path,map_location='cpu',weights_only=False)
            if damage=='missing_rng':del c['native_neuropose']['rng']
            else:
                state=next(iter(c['optimizer_states'][0]['state'].values()))
                if damage=='optimizer_counter':state['step'].add_(1)
                else:state['exp_avg'].fill_(float('nan'))
            torch.save(c,path);result['last_checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            result.pop('sha256');result['sha256']=api()._hash(result)
            (fit/'fit-result.json').write_text(json.dumps(result))
        else:
            if damage=='wrong_epoch':result['completed_epochs']=100
            elif damage=='promote_scope':result['scope']='full';result['full_training_terminated']=True
            elif damage=='modified_initial':result['initial_model_sha256']='0'*64
            else:result['selected_best']['file']='last.ckpt'
            result.pop('sha256');result['sha256']=api()._hash(result)
            (fit/'fit-result.json').write_text(json.dumps(result))
        with pytest.raises((ValueError,FileNotFoundError)):api().read_neuropose_fit(fit,require_full=False)
    finally:
        for name,b in before.items():(fit/name).write_bytes(b)
    assert api().read_neuropose_fit(fit,require_full=False)['completed_epochs']==2


@pytest.mark.parametrize('value',[None,0,''])
def test_selected_reopen_requires_an_explicit_boolean_full_requirement(native_fit,value):
    root,_=native_fit
    with pytest.raises(ValueError,match='boolean'):
        api().reopen_neuropose_best(root/'fit',require_full=value)


def test_changed_later_validation_payload_cannot_reopen_a_fit(native_fit):
    import numpy as np
    root,_=native_fit;path=root/'data/emg_right_filtered.dat';before=path.read_bytes()
    try:
        values=np.memmap(path,mode='r+',dtype=np.float32,shape=(path.stat().st_size//32,8))
        values[7790+7000,3]+=1;values.flush();values._mmap.close()
        with pytest.raises(ValueError):api().read_neuropose_fit(root/'fit',require_full=False)
    finally:path.write_bytes(before)
    assert api().read_neuropose_fit(root/'fit',require_full=False)['completed_epochs']==2


def test_checkpoint_without_finished_result_is_incomplete(native_fit):
    root,_=native_fit;path=root/'fit/fit-result.json';before=path.read_bytes()
    try:
        path.unlink()
        with pytest.raises(FileNotFoundError):api().read_neuropose_fit(root/'fit',require_full=False)
    finally:path.write_bytes(before)

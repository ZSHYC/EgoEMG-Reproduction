"""Native saved-state and cross-fit qualification rejection regressions."""
import copy
import importlib
import struct

import pytest


def train_api():
    return importlib.import_module('egoemg_repro.wilor_native_train')


def loop_packet(*,epochs=2,batches=2,val_batches=176):
    # Pinned Lightning2.5.0.post0 save timing, independently observed in the
    # genuine CPU and GPU checkpoints: epoch completed trails processed by one.
    steps=epochs*batches
    def progress(total,current,fields=('ready','completed')):
        return dict(total={k:total for k in fields},current={k:current for k in fields})
    def batch(total,current,last):
        return dict(**progress(total,current,('ready','completed','started','processed')),is_last_batch=last)
    return dict(fit_loop={
        'state_dict':{},'epoch_loop.state_dict':{'_batches_that_stepped':steps},
        'epoch_loop.batch_progress':batch(steps,batches,True),
        'epoch_loop.scheduler_progress':progress(epochs,1),
        'epoch_loop.automatic_optimization.state_dict':{},
        'epoch_loop.automatic_optimization.optim_progress':{'optimizer':{
            'step':progress(steps,batches),'zero_grad':progress(steps,batches,('ready','completed','started'))}},
        'epoch_loop.manual_optimization.state_dict':{},
        'epoch_loop.manual_optimization.optim_step_progress':progress(0,0),
        'epoch_loop.val_loop.state_dict':{},
        'epoch_loop.val_loop.batch_progress':batch(val_batches,val_batches,True),
        'epoch_progress':{'total':dict(ready=epochs,started=epochs,processed=epochs,completed=epochs-1),
                          'current':dict(ready=epochs,started=epochs,processed=epochs,completed=epochs-1)}},
        validate_loop={'state_dict':{},'batch_progress':batch(0,0,False)},
        test_loop={'state_dict':{},'batch_progress':batch(0,0,False)},
        predict_loop={'state_dict':{},'batch_progress':progress(0,0,('ready','completed','started','processed'))})


@pytest.mark.parametrize('epochs,batches,val_batches',[(1,2,2),(2,2,2),(2,2,176),(30,11224,176)])
def test_native_loop_validation_preserves_checkpoint_save_timing(epochs,batches,val_batches):
    train_api()._validate_loops(loop_packet(epochs=epochs,batches=batches,val_batches=val_batches),
                              epochs=epochs,steps=epochs*batches,batches=batches,val_batches=val_batches)


@pytest.mark.parametrize('damage',['none','missing_fit','empty_fit','stepped','batch_total','batch_current',
    'scheduler_total','scheduler_current','optimizer_total','optimizer_current','zero_grad','manual',
    'val_total','val_current','epoch_ready','epoch_started','epoch_processed','epoch_completed',
    'standalone_val','standalone_test','predict','bool_counter','last_batch','extra'])
def test_incomplete_or_contradictory_native_loop_state_is_rejected(damage):
    v=loop_packet();fit=v['fit_loop']
    if damage=='none':v=None
    elif damage=='missing_fit':del v['fit_loop']
    elif damage=='empty_fit':v['fit_loop']={}
    elif damage=='stepped':fit['epoch_loop.state_dict']['_batches_that_stepped']=3
    elif damage.startswith('batch_'):fit['epoch_loop.batch_progress'][damage.split('_')[1]]['completed']-=1
    elif damage.startswith('scheduler_'):fit['epoch_loop.scheduler_progress'][damage.split('_')[1]]['completed']-=1
    elif damage.startswith('optimizer_'):fit['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step'][damage.split('_')[1]]['completed']-=1
    elif damage=='zero_grad':fit['epoch_loop.automatic_optimization.optim_progress']['optimizer']['zero_grad']['total']['started']=3
    elif damage=='manual':fit['epoch_loop.manual_optimization.optim_step_progress']['total']['completed']=1
    elif damage.startswith('val_'):fit['epoch_loop.val_loop.batch_progress'][damage.split('_')[1]]['completed']=175
    elif damage.startswith('epoch_'):fit['epoch_progress']['total'][damage.removeprefix('epoch_')]=0
    elif damage=='standalone_val':v['validate_loop']['batch_progress']['total']['completed']=1
    elif damage=='standalone_test':v['test_loop']['batch_progress']['total']['completed']=1
    elif damage=='predict':v['predict_loop']['batch_progress']['current']['ready']=1
    elif damage=='bool_counter':fit['epoch_loop.scheduler_progress']['current']['ready']=True
    elif damage=='last_batch':fit['epoch_loop.batch_progress']['is_last_batch']=False
    else:fit['unknown_state']={}
    with pytest.raises(ValueError,match='loop|counter|checkpoint'):
        train_api()._validate_loops(v,epochs=2,steps=4,batches=2,val_batches=176)


@pytest.mark.parametrize('damage',['one_byte','legacy_seed_only','oversize','bad_offset','noncontiguous'])
def test_cuda_rng_saved_packet_schema_is_checked_without_cuda_initialization(damage):
    torch=pytest.importorskip('torch');m=train_api();v=m._rng_packet(cpu=True)
    # PyTorch2.2.2 get_state emits uint64 seed + int64 Philox offset, 16 bytes.
    raw=struct.pack('=Qq',42,1008)
    state=torch.tensor(list(raw),dtype=torch.uint8)
    if damage=='one_byte':state=state[:1]
    elif damage=='legacy_seed_only':state=state[:8]
    elif damage=='oversize':state=torch.cat((state,state))
    elif damage=='bad_offset':state=torch.tensor(list(struct.pack('=Qq',42,1009)),dtype=torch.uint8)
    else:state=torch.tensor(list(raw)*2,dtype=torch.uint8)[::2]
    v['cuda']=[state];before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='RNG|rng'):m._validate_rng(v,cpu=False)
    assert torch.equal(torch.get_rng_state(),before) and not torch.cuda.is_initialized()


def test_pinned_cuda_rng_schema_validation_does_not_restore_or_initialize_cuda():
    torch=pytest.importorskip('torch');m=train_api();v=m._rng_packet(cpu=True)
    v['cuda']=[torch.tensor(list(struct.pack('=Qq',42,1008)),dtype=torch.uint8)]
    before=torch.get_rng_state().clone();m._validate_rng(v,cpu=False)
    assert torch.equal(torch.get_rng_state(),before) and not torch.cuda.is_initialized()


@pytest.mark.parametrize('mismatch',['native','replay','both'])
def test_qualification_pair_from_another_fit_rejects_before_owned_receipt_io(mismatch,monkeypatch):
    m=importlib.import_module('egoemg_repro.wilor_native_eval')
    fit=dict(sha256='a'*64,scope='cuda_engineering',completed_epochs=2,optimizer_executions=4,
             train_samples_per_epoch=[36,36],val_samples_per_epoch=[3152,3152])
    identity=dict(policy=dict(precision='bf16-mixed',batch_size_by_role=dict(train=18,val=18,test=18)),
                  cohort=dict(registered_full_legacy=True,counts=dict(train=202020,val=3152,test=8612)),
                  runtime_device=dict(visible_device='GPU-fixture-not-a-qualification'))
    native=dict(fit_result_sha256='a'*64,scope='cuda_engineering',role='val',kind='native',sha256='c'*64,
                layout=dict(samples=3152,batch_size=18),execution_device=identity['runtime_device'])
    replay=copy.deepcopy(native);replay.update(kind='replay',reference_sha256=native['sha256'])
    if mismatch in ('native','both'):native['fit_result_sha256']='b'*64
    if mismatch in ('replay','both'):replay['fit_result_sha256']='b'*64
    def forbidden(*a,**k):raise AssertionError('mismatched fits must reject before producer or input reads')
    monkeypatch.setattr(m,'_owned',forbidden)
    with pytest.raises(ValueError,match='fit|matching'):
        m._qualification_dependencies(dict(input_admission='unread',fit='unread',native='unread',replay='unread'),
                                      fit,identity,native,replay,None)

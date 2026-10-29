"""Restore-boundary unit checks; small tensors are not WiLoR qualification."""
import copy
import importlib
from types import SimpleNamespace

import pytest
import torch

from test_wilor_recovery import parent,callback_packet


@pytest.fixture
def native_state(callback_packet):
    from egoemg_repro import classic_emg_train as storage
    callbacks,_,_=callback_packet
    torch.manual_seed(123)
    module=torch.nn.Linear(2,1)
    optimizer=torch.optim.AdamW(module.parameters(),lr=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30,eta_min=1e-5)
    module(torch.ones(1,2)).sum().backward();optimizer.step();scheduler.step();optimizer.zero_grad()
    checkpoint=dict(state_dict=copy.deepcopy(module.state_dict()),optimizer_states=[copy.deepcopy(optimizer.state_dict())],
        lr_schedulers=[copy.deepcopy(scheduler.state_dict())],callbacks=copy.deepcopy(callbacks['callbacks']),
        loops={'fit_loop':{'epoch_progress':{'completed':0,'processed':1}}},native_wilor={'rng':storage._rng({})})
    state={'callbacks':{},'loops':{}};corrupt={'domain':None};calls=[]
    loop=SimpleNamespace(_combined_loader=None,_data_fetcher=None,state_dict=lambda:state['loops'])
    holders=[]
    for key in checkpoint['callbacks']:
        holders.append(SimpleNamespace(state_key=key,state_dict=lambda key=key:state['callbacks'].get(key,{})))
    def native_restore():
        calls.append('native')
        module.load_state_dict(checkpoint['state_dict']);optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_states'][0]))
        scheduler.load_state_dict(checkpoint['lr_schedulers'][0]);state['callbacks']=copy.deepcopy(checkpoint['callbacks'])
        # Native PL2.5 deliberately does not restore this transient observation.
        next(v for k,v in state['callbacks'].items()if k.startswith('ModelCheckpoint'))['current_score']=None
        state['loops']=copy.deepcopy(checkpoint['loops']['fit_loop'])
        if corrupt['domain']=='model':
            with torch.no_grad():next(module.parameters()).add_(1)
        elif corrupt['domain']=='optimizer':optimizer.param_groups[0]['lr']=.1
        elif corrupt['domain']=='scheduler':scheduler.T_max=31
        elif corrupt['domain']=='callbacks':next(v for k,v in state['callbacks'].items()if k.startswith('EarlyStopping'))['wait_count']=1
        elif corrupt['domain']=='loops':state['loops']['epoch_progress']['processed']=2
        # Model construction or native restore may consume global RNG; the gate
        # must restore the saved stream after this call and before the iterator.
        torch.manual_seed(999);torch.rand(3)
        return 'native-return'
    trainer=SimpleNamespace(optimizers=[optimizer],lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
        callbacks=holders,fit_loop=loop,_checkpoint_connector=SimpleNamespace(restore_training_state=native_restore))
    with torch.no_grad():next(module.parameters()).add_(2)
    return module,trainer,checkpoint,corrupt,calls


def install(native_state):
    module,trainer,checkpoint,_,_=native_state
    api=importlib.import_module('egoemg_repro.wilor_recovery')
    return api._install_restore_gate(trainer,module,checkpoint,guard=lambda:None)


def test_all_six_domains_restore_after_native_call_before_iterator(native_state):
    _,trainer,checkpoint,_,calls=native_state
    expected=torch.Generator();expected.set_state(checkpoint['native_wilor']['rng']['torch'])
    want=torch.rand(4,generator=expected)
    receipt=install(native_state)
    assert trainer._checkpoint_connector.restore_training_state()=='native-return'
    assert calls==['native']
    assert set(receipt['equal'])=={'model','optimizer','scheduler','callbacks','loops','rng'}
    assert all(value is True for value in receipt['equal'].values())
    assert receipt['calls']==1 and receipt['restored_before_iterator'] is True
    assert torch.equal(torch.rand(4),want)


@pytest.mark.parametrize('domain',['model','optimizer','scheduler','callbacks','loops'])
def test_restore_mismatch_aborts_before_training_iterator(native_state,domain):
    _,trainer,_,corrupt,_=native_state;corrupt['domain']=domain
    receipt=install(native_state)
    with pytest.raises(ValueError,match='restor'):
        trainer._checkpoint_connector.restore_training_state()
    assert not receipt.get('restored_before_iterator',False)


def test_missing_rng_restoration_is_a_real_mismatch(native_state,monkeypatch):
    from egoemg_repro import classic_emg_train as storage
    _,trainer,_,_,_=native_state
    monkeypatch.setattr(storage,'_restore_rng',lambda *a:None)
    install(native_state)
    with pytest.raises(ValueError,match='restor'):
        trainer._checkpoint_connector.restore_training_state()


@pytest.mark.parametrize('field',['_combined_loader','_data_fetcher'])
def test_late_restore_after_iterator_construction_rejects(native_state,field):
    _,trainer,_,_,calls=native_state
    install(native_state);setattr(trainer.fit_loop,field,object())
    with pytest.raises(ValueError,match='iterator'):
        trainer._checkpoint_connector.restore_training_state()
    assert calls==[]


def test_restore_gate_is_single_use(native_state):
    _,trainer,_,_,calls=native_state
    install(native_state);trainer._checkpoint_connector.restore_training_state()
    with pytest.raises(ValueError,match='once'):
        trainer._checkpoint_connector.restore_training_state()
    assert calls==['native']


@pytest.fixture
def receipt_pair():
    fields=('model','optimizer','scheduler','callbacks','loops','rng')
    expected={k:format(i+1,'x')*64 for i,k in enumerate(fields)}
    return dict(calls=1,expected=dict(expected),actual=dict(expected),equal={k:True for k in fields},restored_before_iterator=True),expected


def test_restore_receipt_is_bound_to_independently_read_parent_state(receipt_pair):
    receipt,expected=receipt_pair
    api=importlib.import_module('egoemg_repro.wilor_recovery')
    api._validate_restore_receipt(receipt,expected)


@pytest.mark.parametrize('change',['both-hashes','bool-call','int-equal','int-iterator','missing-domain','extra-domain','bad-digest'])
def test_restore_receipt_cannot_self_attest_a_different_parent(receipt_pair,change):
    receipt,expected=receipt_pair
    if change=='both-hashes':receipt['expected']['model']=receipt['actual']['model']='0'*64
    elif change=='bool-call':receipt['calls']=True
    elif change=='int-equal':receipt['equal']['model']=1
    elif change=='int-iterator':receipt['restored_before_iterator']=1
    elif change=='missing-domain':receipt['actual'].pop('rng')
    elif change=='extra-domain':receipt['equal']['extra']=True
    else:receipt['expected']['rng']=receipt['actual']['rng']='x'*64
    api=importlib.import_module('egoemg_repro.wilor_recovery')
    with pytest.raises(ValueError,match='restor'):
        api._validate_restore_receipt(receipt,expected)

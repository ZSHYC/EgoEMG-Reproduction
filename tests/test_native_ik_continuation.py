"""Continuation admission is real file I/O; carriers never certify native IK."""
import copy
import importlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pytest

from egoemg_repro.data import file_sha256
from egoemg_repro.emg2pose_memory_process import process_identity
from egoemg_repro.native_ik_inputs import _hash, _state
from egoemg_repro import native_ik_results as results
from test_native_ik_results import inputs, carrier


def api():
    try:return importlib.import_module('egoemg_repro.native_ik_continuation')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.native_ik_continuation':pytest.fail('native IK prefix continuation admission is missing')
        raise


def _carrier_process(inputs, root):
    """Actual exited carrier writer, explicitly NOT geometry or optimization."""
    flow=root/'private/full-label-workflow';base=flow/'flow/right-shard-0/produce';base.mkdir(parents=True)
    actor=process_identity();runtime=dict(format='structural_prefix_IO_carrier_only',source_identity_sha256='2'*64)
    rows=[]
    for index in (0,1):
        batch=results.select_ik_batch(inputs,'right',0,index);arrays,record=carrier(batch)
        p=base/f'batch-{index:05d}';p.mkdir();np.savez(p/'state.npz',**arrays)
        value=results.seal(dict(format='original_native_ik_raw_batch_v1',batch=batch.identity,
            runtime_identity=runtime,actor=actor,record=record,statistics=results.validate_ik_state(batch,arrays,record),
            state_file='state.npz',state_file_sha256=file_sha256(p/'state.npz'),results_source_sha256=results.IMPORTED_SHA,
            pending_independent_geometry_verification=True,full_frame_labels_regenerated=False,
            original_default_batch10096_qualified=False,scientific_eligible=False,full_paper_complete=False))
        results.write_json(p/'result.json',value)
        rows.append(dict(batch_index=index,rows=3,start=index*3,end=(index+1)*3,
            actual_original_producer_actor=actor,result_file_sha256=file_sha256(p/'result.json'),
            state_file_sha256=file_sha256(p/'state.npz'),original_native_optimizer_steps=1,
            original_native_closure_evaluations=1,raw_zero_pose_rows=int(index==0)))
    # A real interrupted boundary, never adopted as a complete packet.
    (base/'batch-00002').mkdir();(base/'batch-00002/state.npz').write_bytes(b'incomplete-carrier')
    config=results.seal(dict(action='workflow',request=dict(engineering=True,batch_size=3,device='cpu',qualification=None),
        input_identity_sha256=_hash(inputs.identity),source_identity_sha256='2'*64,
        qualified_runtime_identity=runtime,qualification_file_sha256=None))
    results.write_json(flow/'configuration.json',config)
    state=dict(status='failed',actual_exit_code=1,producer=actor)
    results.write_json(root/'private/execution-state.json',results.seal(state))
    proof=results.seal(dict(format='original_native_ik_prefix_engineering_IO_audit_v1',engineering=True,
        original_terminal_state=results.seal(state),all_recorded_original_owning_actors_physically_closed=[actor],
        original_configuration_file_sha256=file_sha256(flow/'configuration.json'),original_source_bundle_sha256=None,
        original_qualified_runtime_identity=runtime,original_input_identity_sha256=_hash(inputs.identity),
        original_native_source_identity_sha256='2'*64,original_qualification_file_sha256=None,
        actual_hand='right',actual_shard=0,actual_completed_original_batches=2,actual_completed_original_rows=6,
        first_uncompleted_batch_index=2,original_batch_size=3,original_batches=rows,
        full_frame_labels_regenerated=False,scientific_eligible=False,full_paper_complete=False))
    results.write_json(root/'prefix-proof.json',proof)
    raise SystemExit(1)


@pytest.fixture
def prefix_source(inputs,tmp_path):
    root=tmp_path/'old-carrier';root.mkdir()
    p=multiprocessing.get_context('fork').Process(target=_carrier_process,args=(inputs,root));p.start();p.join(20)
    assert p.exitcode==1
    return inputs,root,root/'prefix-proof.json'


def open_prefix(fixture):
    i,r,p=fixture;return api().admit_ik_prefix(i,r,proof=p,engineering=True)


def reseal(path,change):
    x=json.loads(path.read_text());x.pop('sha256');change(x);path.write_text(json.dumps(results.seal(x)))


def test_missing_continuation_api_is_an_explicit_failure():
    assert callable(api().admit_ik_prefix)


# Catches restarting/renumbering old coupled batches or treating carriers as IK.
def test_real_file_prefix_keeps_exact_original_boundary_and_false_full_flags(prefix_source):
    p=open_prefix(prefix_source);i,r,_=prefix_source
    assert p.batch_prefix('right',0)==tuple(r/f'private/full-label-workflow/flow/right-shard-0/produce/batch-{n:05d}'for n in (0,1))
    assert p.batch_prefix('left',0)==() and p.batch_prefix('right',5)==()
    assert p.identity['reused_batches']==2 and p.identity['reused_rows']==6
    assert p.identity['engineering'] and not p.identity['full_available_release']
    assert not p.identity['full_frame_labels_regenerated'] and not p.identity['native_optimizer_execution_verified']
    assert not p.identity['scientific_eligible'] and not p.identity['full_paper_complete']
    with np.load(p.batch_prefix('right',0)[1]/'state.npz',allow_pickle=False)as z:
        assert z['indices'].tolist()==[3,4,5]
    assert (r/'private/full-label-workflow/flow/right-shard-0/produce/batch-00002/state.npz').read_bytes()==b'incomplete-carrier'


def test_returned_identity_cannot_rebind_original_packets(prefix_source):
    p=open_prefix(prefix_source);q=p.identity;q['reused_rows']=999;q['runtime_identity']['source_identity_sha256']='0'*64
    assert p.identity['reused_rows']==6 and p.identity['runtime_identity']['source_identity_sha256']=='2'*64
    with pytest.raises(ValueError,match='registered'):copy.copy(p).check()
    with pytest.raises(ValueError,match='registered'):api().IKPrefix().check()


@pytest.mark.parametrize('fault',['missing_result','missing_state','hole','extra_complete','extra_other_hand','reordered_proof','duplicate_proof','wrong_rows','wrong_runtime','wrong_input','swapped_actor','live_actor','full_claim'])
def test_old_prefix_fault_cannot_be_adopted(prefix_source,fault):
    i,r,proof=prefix_source;base=r/'private/full-label-workflow/flow/right-shard-0/produce'
    if fault=='missing_result':(base/'batch-00000/result.json').unlink()
    elif fault=='missing_state':(base/'batch-00000/state.npz').unlink()
    elif fault=='hole':(base/'batch-00001').rename(base/'batch-00003')
    elif fault in('extra_complete','extra_other_hand'):
        import shutil
        q=base/'batch-00003'if fault=='extra_complete'else r/'private/full-label-workflow/flow/left-shard-0/produce/batch-00000'
        shutil.copytree(base/'batch-00000',q)
    elif fault=='reordered_proof':reseal(proof,lambda x:x['original_batches'].reverse())
    elif fault=='duplicate_proof':reseal(proof,lambda x:x['original_batches'].__setitem__(1,x['original_batches'][0]))
    elif fault=='wrong_rows':reseal(proof,lambda x:x.update(actual_completed_original_rows=7))
    elif fault=='wrong_runtime':reseal(proof,lambda x:x['original_qualified_runtime_identity'].update(source_identity_sha256='0'*64))
    elif fault=='wrong_input':reseal(proof,lambda x:x.update(original_input_identity_sha256='0'*64))
    elif fault=='swapped_actor':reseal(proof,lambda x:x['original_batches'][0]['actual_original_producer_actor'].update(start_ticks='1'))
    elif fault=='live_actor':reseal(proof,lambda x:x['all_recorded_original_owning_actors_physically_closed'].append(process_identity()))
    else:reseal(proof,lambda x:x.update(scientific_eligible=True))
    with pytest.raises(ValueError):open_prefix(prefix_source)


@pytest.mark.parametrize('fault',['same_bytes_replacement','hardlink','optional_mask','identity','provider','late_extra','partial_mutation'])
def test_late_generation_and_provider_drift_invalidates_consumption(prefix_source,fault,tmp_path):
    i,r,_=prefix_source;p=open_prefix(prefix_source);path=p.batch_prefix('right',0)[0]/'state.npz'
    if fault=='same_bytes_replacement':
        other=path.with_name('replacement.npz');other.write_bytes(path.read_bytes());os.replace(other,path)
    elif fault=='hardlink':os.link(path,tmp_path/'linked.npz')
    elif fault=='optional_mask':(_state(i)['paths']['right'].parent/'valid.npy').write_bytes(b'new-mask')
    elif fault=='identity':api()._PREFIXES[p]['identity']['reused_rows']=999
    elif fault=='provider':
        a=_state(i)['arrays']['right'];_state(i)['arrays']['right']=np.asarray(a).copy()
    elif fault=='late_extra':(path.parent.parent/'batch-00099').mkdir()
    else:(path.parent.parent/'batch-00002/state.npz').write_bytes(b'changed-incomplete')
    with pytest.raises(ValueError):p.batch_prefix('right',0)


def test_hardlinked_packet_is_rejected_at_first_admission(prefix_source,tmp_path):
    _,r,_=prefix_source;path=r/'private/full-label-workflow/flow/right-shard-0/produce/batch-00000/state.npz'
    os.link(path,tmp_path/'hardlinked.npz')
    with pytest.raises(ValueError):open_prefix(prefix_source)


def test_copied_input_registration_cannot_supply_a_prefix(prefix_source):
    i,r,p=prefix_source
    with pytest.raises(ValueError,match='registered'):api().admit_ik_prefix(copy.copy(i),r,proof=p,engineering=True)


def test_forged_full_proof_rejects_before_touching_any_source(tmp_path):
    q=tmp_path/'resealed-full.json';q.write_text(json.dumps(results.seal(dict(format='actual_interrupted_original_native_IK68batch_prefix_postexit_raw_read_v1',actual_completed_original_batches=68))))
    with pytest.raises(ValueError):api().admit_ik_prefix(object(),tmp_path/'absent',proof=q)


@pytest.mark.parametrize('hand,shard',[('both',0),('left',True),('right',6),('right',-1)])
def test_no_arbitrary_hand_or_logical_shard(prefix_source,hand,shard):
    p=open_prefix(prefix_source)
    with pytest.raises(ValueError):p.batch_prefix(hand,shard)

"""Mixed IK receipt roles and exact persistence; carriers certify no geometry."""
import copy
import importlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from egoemg_repro.data import file_sha256
from egoemg_repro.emg2pose_memory_process import process_identity
from egoemg_repro import native_ik_results as results
from test_native_ik_results import inputs,carrier
from test_native_ik_continuation import prefix_source,open_prefix,reseal
from test_native_ik_workflow import request as native_request


def api():return importlib.import_module('egoemg_repro.native_ik_continuation')


def _new_carrier(inputs,root):
    p=root/'batch-00002';p.mkdir(parents=True);batch=results.select_ik_batch(inputs,'right',0,2);arrays,record=carrier(batch)
    np.savez(p/'state.npz',**arrays)
    results.write_json(p/'result.json',results.seal(dict(format='original_native_ik_raw_batch_v1',
        batch=batch.identity,runtime_identity=dict(format='structural_prefix_IO_carrier_only',source_identity_sha256='2'*64),
        actor=process_identity(),record=record,statistics=results.validate_ik_state(batch,arrays,record),
        state_file='state.npz',state_file_sha256=file_sha256(p/'state.npz'),results_source_sha256=results.IMPORTED_SHA,
        pending_independent_geometry_verification=True,full_frame_labels_regenerated=False,
        original_default_batch10096_qualified=False,scientific_eligible=False,full_paper_complete=False)))


def _map_process(inputs,prefix,new_root,actor,output):
    api().write_continued_shard(inputs,prefix,hand='right',shard=0,new_root=new_root,
        new_producer_actor=actor,configuration_sha256='a'*64,output=output)


@pytest.fixture
def continued(prefix_source,tmp_path):
    inputs,old_root,_=prefix_source;prefix=open_prefix(prefix_source);new_root=tmp_path/'new-producer'
    producer=multiprocessing.get_context('fork').Process(target=_new_carrier,args=(inputs,new_root));producer.start();producer.join(20);assert producer.exitcode==0
    actor=json.loads((new_root/'batch-00002/result.json').read_text())['actor'];output=tmp_path/'mixed-map'
    mapper=multiprocessing.get_context('fork').Process(target=_map_process,args=(inputs,prefix,new_root,actor,output));mapper.start();mapper.join(20);assert mapper.exitcode==0
    doc=json.loads((output/'mixed.json').read_text())
    return dict(inputs=inputs,prefix=prefix,new_root=new_root,directory=output,receipt_actor=doc['receipt_actor'],doc=doc)


def read(x,**changes):
    args=dict(configuration_sha256='a'*64,receipt_actor=x['receipt_actor'],prefix=x['prefix']);args.update(changes)
    return api().read_continued_shard(x['inputs'],x['directory'],**args)


def test_mixed_receipt_api_is_missing_until_implemented():
    assert callable(getattr(api(),'write_continued_shard',None))
    assert callable(getattr(api(),'read_continued_shard',None))


# Catches replacing two real old producers with the new aggregate actor, or
# silently running the partial old batch2 under the old producer's identity.
def test_mixed_receipt_preserves_all_original_and_new_packet_producers(continued):
    m=read(continued);entries=m.entries
    assert [e['batch_index']for e in entries]==[0,1,2]
    assert [e['reused']for e in entries]==[True,True,False]
    assert [e['rows']for e in entries]==[3,3,2]
    assert entries[0]['producer_actor']==entries[1]['producer_actor']
    assert entries[0]['producer_actor']!=entries[2]['producer_actor']!=continued['receipt_actor']
    assert all(e['producer_actor']!=continued['receipt_actor']for e in entries)
    assert Path(entries[2]['directory'])==continued['new_root']/'batch-00002'
    assert m.identity['rows']==8 and m.identity['reused_rows']==6 and m.identity['new_rows']==2
    assert not m.identity['geometric_reconstruction_verified']and not m.identity['full_paper_complete']
    entries[0]['producer_actor']['pid']=1
    assert m.entries[0]['producer_actor']['pid']!=1
    with pytest.raises(ValueError):copy.copy(m).check()


@pytest.mark.parametrize('fault',['actor_is_aggregate','aggregate_is_packet','changed_old_actor','changed_new_actor',
    'false_reused','false_new','wrong_configuration','wrong_input','wrong_runtime','missing_entry','reverse_entries',
    'replaced_directory','wrong_result','wrong_state','wrong_batch','claims_geometry'])
def test_resealed_mixed_map_cannot_change_actual_lineage(continued,fault):
    p=continued['directory']/'mixed.json'
    def damage(v):
        entries=v['entries']
        if fault=='actor_is_aggregate':entries[0]['producer_actor']=v['receipt_actor']
        elif fault=='aggregate_is_packet':v['receipt_actor']=entries[2]['producer_actor']
        elif fault in('changed_old_actor','changed_new_actor'):entries[0 if fault=='changed_old_actor'else 2]['producer_actor']['start_ticks']='1'
        elif fault=='false_reused':entries[0]['reused']=False
        elif fault=='false_new':entries[2]['reused']=True
        elif fault=='wrong_configuration':v['configuration_sha256']='b'*64
        elif fault=='wrong_input':v['input_identity_sha256']='b'*64
        elif fault=='wrong_runtime':v['runtime_identity']['source_identity_sha256']='b'*64
        elif fault=='missing_entry':entries.pop()
        elif fault=='reverse_entries':entries.reverse()
        elif fault=='replaced_directory':entries[2]['directory']=entries[1]['directory']
        elif fault=='wrong_result':entries[0]['result_sha256']='b'*64
        elif fault=='wrong_state':entries[2]['state_sha256']='b'*64
        elif fault=='wrong_batch':entries[2]['batch_index']=1
        else:v['geometric_reconstruction_verified']=True
    reseal(p,damage)
    with pytest.raises(ValueError):read(continued)


def test_actual_live_or_wrong_receipt_owner_never_supplies_verified_lineage(continued):
    with pytest.raises(ValueError):read(continued,receipt_actor=process_identity())
    actor=copy.deepcopy(continued['receipt_actor']);actor['start_ticks']='1'
    with pytest.raises(ValueError):read(continued,receipt_actor=actor)


@pytest.mark.parametrize('role',['old','new','map'])
def test_late_packet_or_receipt_changes_invalidate_consumption(continued,role):
    m=read(continued)
    p=Path(m.entries[0 if role=='old'else 2]['directory'])/'state.npz'if role!='map'else continued['directory']/'mixed.json'
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):m.check()


def test_new_map_does_not_overwrite_any_existing_output(continued):
    x=continued
    with pytest.raises(FileExistsError):api().write_continued_shard(x['inputs'],x['prefix'],hand='right',shard=0,
        new_root=x['new_root'],new_producer_actor=x['doc']['entries'][2]['producer_actor'],
        configuration_sha256='a'*64,output=x['directory'])


def runner():
    try:return importlib.import_module('scripts.paper.run_native_mano_ik_continuation')
    except ModuleNotFoundError as exc:
        if exc.name=='scripts.paper.run_native_mano_ik_continuation':pytest.fail('owned native IK continuation runner is missing')
        raise


def test_owned_continuation_runner_is_missing_until_implemented():
    assert callable(runner().run_continued_workflow)


@pytest.mark.parametrize('fault',['cpu_full','wrong_gpu','small_batch','small_reserve','unknown_key'])
def test_invalid_continuation_never_starts_input_or_GPU_io(tmp_path,fault):
    cfg=native_request(tmp_path)
    cfg.update(engineering=False,batch_size=10096,device='cuda:0',gpu_uuid='GPU-8b2cacdf-4470-21f2-01c5-e15475dbdfae',
        qualification=str(tmp_path/'qualification'),minimum_free_bytes=225*1024**3)
    if fault=='cpu_full':cfg.update(device='cpu',gpu_uuid=None)
    elif fault=='wrong_gpu':cfg['gpu_uuid']='GPU-not-the-qualified-card'
    elif fault=='small_batch':cfg['batch_size']=10095
    elif fault=='small_reserve':cfg['minimum_free_bytes']=224*1024**3
    else:cfg['new_math']=True
    with pytest.raises(ValueError):runner().validate_request(cfg)


def test_existing_continuation_never_adopts_or_overwrites_a_partial_root(tmp_path):
    out=tmp_path/'existing';out.mkdir();(out/'kept').write_text('old-failure')
    with pytest.raises(FileExistsError):runner().run_continued_workflow(native_request(tmp_path),tmp_path/'missing-old',tmp_path/'missing-proof',out)
    assert (out/'kept').read_text()=='old-failure'


def _exited_actor(path):
    path.write_text(json.dumps(process_identity()))


@pytest.fixture
def stage_carrier(tmp_path):
    actors=[]
    for index in (0,1):
        path=tmp_path/f'actor{index}.json';p=multiprocessing.get_context('fork').Process(target=_exited_actor,args=(path,));p.start();p.join(10);assert p.exitcode==0;actors.append(json.loads(path.read_text()))
    root=tmp_path/'stage';root.mkdir();owner=process_identity()
    started=dict(owner=owner,worker=actors[0],guardian=actors[1]);(root/'started.json').write_text(json.dumps(started))
    value=dict(started,status='success',worker_exit_code=0,all_owned_consumers_exited=True)
    (root/'stage-result.json').write_text(json.dumps(value))
    return root,owner,value


def test_closed_stage_binds_actual_packet_worker_and_guardian(stage_carrier):
    root,owner,wanted=stage_carrier;actual,bindings=runner().read_closed_stage(root,owner)
    assert actual==wanted
    (root/'stage-result.json').write_text(json.dumps(dict(wanted,worker_exit_code=1)))
    with pytest.raises(ValueError):bindings.check()


@pytest.mark.parametrize('fault',['stopped','failed_worker','still_live','swapped_start','wrong_owner','reused_guardian'])
def test_failed_live_or_swapped_owned_roles_cannot_complete_a_stage(stage_carrier,fault):
    root,owner,value=stage_carrier
    if fault=='stopped':value['all_owned_consumers_exited']=False
    elif fault=='failed_worker':value['worker_exit_code']=1
    elif fault=='still_live':value['worker']=process_identity()
    elif fault=='swapped_start':value['worker']['start_ticks']='1'
    elif fault=='wrong_owner':value['owner']['start_ticks']='1'
    else:value['guardian']=value['worker']
    (root/'stage-result.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):runner().read_closed_stage(root,owner)

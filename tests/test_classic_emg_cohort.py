"""CPU-only whole-footprint admission; no model, Torch or synthetic pass permit."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from egoemg_repro import classic_emg_data as data
from test_classic_emg_data import make_store, W, UPSTREAM


@pytest.fixture
def cohort():
    return importlib.import_module('egoemg_repro.classic_emg_cohort')


@pytest.fixture
def case(tmp_path, cohort, monkeypatch):
    pytest.importorskip('hydra')
    root = tmp_path/'data'
    arrays = make_store(root, 5*W, splits=np.r_[np.zeros(4*W), np.ones(W)])
    partition = data.build_classic_partitions(data.LegacyStore(root), upstream=UPSTREAM)
    assert {r:partition['coverage'][r]['unique_rows'] for r in ('train','val','test')} == {
        'train':22610,'val':7790,'test':7790}
    monkeypatch.setattr(cohort.shutil, 'disk_usage', lambda _: SimpleNamespace(free=64*1024**3))
    return SimpleNamespace(root=root, arrays=arrays, partitions=partition, out=tmp_path/'admission',
                           config=Path(__file__).parents[1]/'configs/vemg2pose.json')


def admit(cohort, case, **kwargs):
    return cohort.admit_classic_cohort(case.root, case.partitions, upstream=UPSTREAM,
                                      config=case.config, output=case.out, **kwargs)


def verify(cohort, case, receipt, **kwargs):
    return cohort.verify_cohort_receipt(case.root, case.partitions, upstream=UPSTREAM,
                                       config=case.config, receipt=receipt, **kwargs)


def rewrite(case, field, values):
    values.tofile(case.root/(field+'.dat'))


def test_all_roles_unique_rows_both_hands_and_no_formal_authorization(cohort, case):
    before = set(sys.modules)
    receipt = admit(cohort, case, chunk_bytes=4096)
    assert 'torch' not in set(sys.modules)-before
    assert receipt['complete'] is True and receipt['cpu_cohort_admitted'] is True
    assert receipt['production_ready'] is receipt['cuda_qualified'] is receipt['gpu_capacity_qualified'] is False
    assert receipt['footprints']['train']['unique_rows'] == 22610
    assert receipt['footprints']['train']['windows'] == 20
    assert receipt['footprints']['train']['hand_samples'] == 40
    assert set(receipt['raw_fields']) == set(case.arrays) and len(receipt['raw_fields']) == 12
    for role, rows in [('train',22610),('val',7790),('test',7790)]:
        for hand in ('left','right'):
            record = receipt['raw_fields'][f'generated_joint_angles_{hand}'][role]
            assert record['unique_rows'] == rows and record['elements'] == rows*20
            assert record['nonfinite_elements'] == 0
            assert receipt['fixed_values'][hand][role]['angles']['elements'] == rows*22
        assert receipt['supervision'][role]['generated_supervised_hand_rows'] > 0
    assert receipt['supervision']['train'] == {'left':22609,'right':22610,'generated_supervised_hand_rows':45219}
    assert receipt['supervision']['val'] == receipt['supervision']['test'] == {
        'left':7790,'right':7790,'generated_supervised_hand_rows':15580}
    assert verify(cohort, case, case.out/'receipt.json') == receipt
    assert str(case.root) not in json.dumps(receipt) and 'st_ino' not in json.dumps(receipt)


@pytest.mark.parametrize('field', ['emg_left_filtered','emg_right_filtered',
                                  'generated_joint_angles_left','generated_joint_angles_right',
                                  'mocap_left_wrist_pitch','mocap_right_wrist_yaw'])
def test_raw_nonfinite_away_from_center_is_not_repaired_even_if_invalid(cohort, case, field):
    values = case.arrays[field]; values.flat[100] = np.nan; rewrite(case, field, values)
    valid = case.arrays['generated_label_valid']; valid[:200] = False; rewrite(case,'generated_label_valid',valid)
    before = hashlib.sha256((case.root/(field+'.dat')).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='finite'):
        admit(cohort, case)
    assert hashlib.sha256((case.root/(field+'.dat')).read_bytes()).hexdigest() == before
    assert not (case.out/'receipt.json').exists()


def test_empty_role_supervision_rejects_without_filtering_windows(cohort, case):
    mask=case.arrays['generated_label_valid'];mask[4*W:]=False;rewrite(case,'generated_label_valid',mask)
    with pytest.raises(ValueError,match='supervision'):
        admit(cohort,case)
    assert case.partitions['coverage']['test']['hand_samples'] == 2
    assert not (case.out/'receipt.json').exists()


@pytest.mark.parametrize('damage',['subset','fraction','grid','role'])
def test_resealed_incomplete_or_changed_partition_cannot_admit(cohort,case,damage):
    p=case.partitions
    if damage=='subset':p['windows'].pop()
    if damage=='fraction':p['settings']['val_fraction']=.2
    if damage=='grid':p['settings']['window_length']=12000
    if damage=='role':p['windows'][0]['role']='engineering'
    p['provenance'].pop('partition_sha256');p['provenance']['partition_sha256']=data._hash(p)
    with pytest.raises(ValueError,match='partition|settings|role|cohort'):
        admit(cohort,case)


@pytest.mark.parametrize('damage',['passed','counts','identity','role','complete'])
def test_verify_recomputes_semantics_after_attacker_reseals(cohort,case,damage):
    receipt=admit(cohort,case)
    if damage=='passed':receipt={'passed':True,'complete':True}
    if damage=='counts':receipt['raw_fields']['emg_left_filtered']['train']['elements']=8
    if damage=='identity':receipt['data_identity']['fields']['emg_left_filtered']['sha256']='0'*64
    if damage=='role':receipt['footprints'].pop('test')
    if damage=='complete':receipt['production_ready']=True
    receipt.pop('sha256',None);receipt['sha256']=data._hash(receipt)
    with pytest.raises(ValueError,match='receipt|admission|identity'):
        verify(cohort,case,receipt)


def test_missing_consumed_field_is_not_a_partial_admission(cohort,case):
    path=case.root/'manifest.json';manifest=json.loads(path.read_text())
    del manifest['fields']['emg_right_filtered'];path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='field|schema'):
        admit(cohort,case)


def test_fixed_sparse_layout_and_degree_conversion_are_real_values(cohort,case):
    case.config=case.config.with_name('neuropose.json')
    result=admit(cohort,case)
    for role,span in [('val',(3*W,4*W)),('test',(4*W,5*W))]:
        lo,hi=span
        expected=np.zeros((hi-lo,16),np.float32)
        expected[:,[10,12,0,1,2,4,5,6]]=case.arrays['emg_left_filtered'][lo:hi]
        expected=(expected-.000832)/(8.6551+1e-6)
        assert result['fixed_values']['left'][role]['emg']['sha256']==hashlib.sha256(expected.tobytes()).hexdigest()
        angles=np.concatenate([case.arrays['generated_joint_angles_left'][lo:hi],
                               np.full((hi-lo,1),np.float32(np.pi/2)),
                               np.full((hi-lo,1),np.float32(-np.pi))],axis=1)
        assert result['fixed_values']['left'][role]['angles']['sha256']==hashlib.sha256(angles.tobytes()).hexdigest()


def test_identical_data_migration_preserves_content_receipt(cohort,case,tmp_path):
    result=admit(cohort,case);shutil.copytree(case.root,tmp_path/'moved')
    case.root=tmp_path/'moved'
    assert verify(cohort,case,result)==result


@pytest.mark.parametrize('where',['root','ancestor','output'])
def test_symlink_rejected_before_resolve(cohort,case,tmp_path,where):
    link=tmp_path/'link'
    if where=='root':link.symlink_to(case.root,target_is_directory=True);case.root=link
    elif where=='ancestor':
        link.symlink_to(tmp_path,target_is_directory=True);case.root=link/'data'
    else:link.symlink_to(tmp_path,target_is_directory=True);case.out=link/'output'
    with pytest.raises(ValueError,match='symlink'):
        admit(cohort,case)


def test_fast_guard_detects_late_data_and_source_stat_changes(cohort,case,monkeypatch):
    result=admit(cohort,case)
    guard=cohort.cohort_stat_guard(case.root,case.partitions,upstream=UPSTREAM,config=case.config,receipt=result)
    guard.check()
    with guard.read():pass
    values=case.arrays['emg_left_filtered'];values[0,0]+=1;rewrite(case,'emg_left_filtered',values)
    with pytest.raises(ValueError,match='changed|stamp'):
        guard.check()


def test_mutation_after_full_identity_is_rejected(cohort,case,monkeypatch):
    original=data.classic_data_identity
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        values=case.arrays['emg_left_filtered'];values[0,0]+=1;rewrite(case,'emg_left_filtered',values)
        return result
    monkeypatch.setattr(data,'classic_data_identity',changed)
    with pytest.raises(ValueError,match='changed|stamp'):
        admit(cohort,case)
    assert not (case.out/'receipt.json').exists()


def test_existing_output_and_low_capacity_never_overwrite(cohort,case,monkeypatch):
    case.out.mkdir();(case.out/'unrelated').write_text('keep')
    with pytest.raises((FileExistsError,ValueError)):admit(cohort,case)
    assert (case.out/'unrelated').read_text()=='keep'
    case.out=case.out.with_name('low')
    monkeypatch.setattr(cohort.shutil,'disk_usage',lambda _:SimpleNamespace(free=15*1024**3-1))
    with pytest.raises((ValueError,OSError),match='capacity|reserve'):
        admit(cohort,case)
    assert not (case.out/'receipt.json').exists()


def test_late_failure_removes_only_owned_success(cohort,case,monkeypatch):
    original=cohort._atomic_json
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        if args[0].name=='receipt.json':
            (case.out/'unrelated').write_text('keep')
            values=case.arrays['emg_left_filtered'];values[0,0]+=1;rewrite(case,'emg_left_filtered',values)
        return result
    monkeypatch.setattr(cohort,'_atomic_json',changed)
    with pytest.raises(ValueError,match='changed|stamp'):
        admit(cohort,case)
    assert not (case.out/'receipt.json').exists() and (case.out/'unrelated').read_text()=='keep'


def test_chunk_budget_rejects_unbounded_arrays_before_output(cohort,case):
    with pytest.raises(ValueError,match='chunk'):
        admit(cohort,case,chunk_bytes=64*1024**2+1)
    assert not case.out.exists()


@pytest.mark.parametrize('bad',[np.inf,-np.inf])
def test_infinite_values_do_not_pass_a_zero_mask(cohort,case,bad):
    value=case.arrays['mocap_left_wrist_yaw'];value[100]=bad;rewrite(case,'mocap_left_wrist_yaw',value)
    mask=case.arrays['generated_label_valid'];mask[100]=False;rewrite(case,'generated_label_valid',mask)
    with pytest.raises(ValueError,match='finite'):admit(cohort,case)
    assert not (case.out/'receipt.json').exists()


def test_empty_label_samples_are_retained_when_role_has_other_supervision(cohort,case):
    mask=case.arrays['generated_label_valid'];mask[:,0]=False;rewrite(case,'generated_label_valid',mask)
    result=admit(cohort,case)
    assert result['footprints']['train']['hand_samples']==40
    assert all(result['supervision'][r]['left']==0 and result['supervision'][r]['right']>0 for r in ('train','val','test'))


def test_source_guard_is_stat_only_and_checks_after_read(cohort,tmp_path,monkeypatch):
    path=tmp_path/'source.py';path.write_text('frozen')
    guard=cohort._StatGuard([path])
    monkeypatch.setattr(cohort,'file_sha256',lambda _:(_ for _ in ()).throw(AssertionError('hash during cheap guard')))
    guard.check()
    with pytest.raises(ValueError,match='changed'):
        with guard.read():path.write_text('changed')


def test_all_actual_scan_arrays_obey_requested_byte_bound(cohort,case,monkeypatch):
    seen=[];original=cohort._consume
    def consume(cell,values):
        seen.append(values.nbytes)
        assert values.nbytes<=65536
        return original(cell,values)
    monkeypatch.setattr(cohort,'_consume',consume)
    admit(cohort,case,chunk_bytes=65536)
    assert seen and max(seen)<=65536


def test_late_directory_fsync_error_cannot_leave_owned_complete(cohort,case,monkeypatch):
    original=cohort._fsync_dir;failed=False
    def fsync(path):
        nonlocal failed
        if (case.out/'receipt.json').exists() and not failed:
            failed=True
            raise OSError('fixture final directory fsync failure')
        return original(path)
    monkeypatch.setattr(cohort,'_fsync_dir',fsync)
    with pytest.raises(OSError,match='fsync'):admit(cohort,case)
    assert failed and not (case.out/'receipt.json').exists()
    failure=json.loads((case.out/'failure.json').read_text())
    assert not failure['complete'] and not failure['production_ready']


def test_concurrent_receipt_target_is_never_overwritten_or_deleted(cohort,case,monkeypatch):
    original=cohort.os.link
    def link(src,dst,**kwargs):
        if Path(dst).name=='receipt.json':Path(dst).write_text('concurrent unrelated bytes')
        return original(src,dst,**kwargs)
    monkeypatch.setattr(cohort.os,'link',link)
    with pytest.raises(FileExistsError):admit(cohort,case)
    assert (case.out/'receipt.json').read_text()=='concurrent unrelated bytes'
    assert json.loads((case.out/'failure.json').read_text())['complete'] is False


def test_capacity_falling_before_publication_leaves_no_complete(cohort,case,monkeypatch):
    def free(path):
        return SimpleNamespace(free=15*1024**3 if case.out.exists() else 64*1024**3)
    monkeypatch.setattr(cohort.shutil,'disk_usage',free)
    with pytest.raises(ValueError,match='capacity|reserve'):admit(cohort,case)
    assert not (case.out/'receipt.json').exists()


def test_partial_or_self_asserted_probe_is_never_authority(cohort,case,tmp_path):
    probe=tmp_path/'probe.json';probe.write_text(json.dumps({'passed':True,'all_consumed_raw_float_values_finite':True}))
    with pytest.raises(ValueError,match='probe'):
        admit(cohort,case,finite_probe=probe)
    assert not (case.out/'receipt.json').exists()


@pytest.mark.parametrize('reserve',[0,15*1024**3-1])
def test_protected_fifteen_gib_reserve_cannot_be_lowered(cohort,case,reserve):
    with pytest.raises(ValueError,match='reserve'):
        admit(cohort,case,reserve_bytes=reserve)
    assert not case.out.exists()


def test_receipt_file_changed_during_verify_cannot_survive(cohort,case,monkeypatch):
    admit(cohort,case)
    path=case.out/'receipt.json';original=data.classic_data_identity
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        path.write_text(path.read_text()+'\n')
        return result
    monkeypatch.setattr(data,'classic_data_identity',changed)
    with pytest.raises(ValueError,match='changed|stamp'):
        verify(cohort,case,path)


def test_owned_receipt_changed_after_atomic_write_is_revoked(cohort,case,monkeypatch):
    original=cohort._atomic_json
    def changed(path,value,transaction):
        result=original(path,value,transaction)
        if path.name=='receipt.json':path.write_text(path.read_text()+'\n')
        return result
    monkeypatch.setattr(cohort,'_atomic_json',changed)
    with pytest.raises(ValueError,match='published|receipt|changed|hash'):
        admit(cohort,case)
    assert not (case.out/'receipt.json').exists()


def test_revoke_directory_rebind_never_unlinks_neighbor(cohort,tmp_path,monkeypatch):
    output=tmp_path/'owned';output.mkdir();receipt=output/'receipt.json';receipt.write_text('owned')
    transaction=cohort._Transaction(output,8*1024**2,cohort.RESERVE_BYTES)
    value=receipt.stat();transaction.owned[receipt]=(value.st_dev,value.st_ino)
    neighbor=tmp_path/'neighbor';neighbor.mkdir();(neighbor/'receipt.json').write_text('unrelated')
    (neighbor/'status.json').write_text('status');(neighbor/'checkpoint').write_text('checkpoint')
    original=Path.lstat;changed=[]
    def lstat(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        if path==receipt and not changed:
            changed.append(True);output.rename(tmp_path/'old-owned');output.symlink_to(neighbor,target_is_directory=True)
        return result
    monkeypatch.setattr(Path,'lstat',lstat)
    try:
        with pytest.raises(ValueError,match='symlink|changed'):transaction.revoke(receipt)
    finally:
        transaction.close()
    assert (neighbor/'receipt.json').read_text()=='unrelated'
    assert (neighbor/'status.json').read_text()=='status' and (neighbor/'checkpoint').read_text()=='checkpoint'


def test_input_change_inside_published_hash_is_caught_by_last_guard(cohort,case,monkeypatch):
    original=cohort.file_sha256;changed=[]
    def digest(path):
        value=original(path)
        if Path(path)==case.out/'receipt.json' and not changed:
            changed.append(True);v=case.arrays['emg_left_filtered'];v[0,0]+=1;rewrite(case,'emg_left_filtered',v)
        return value
    monkeypatch.setattr(cohort,'file_sha256',digest)
    with pytest.raises(ValueError,match='changed|stamp'):admit(cohort,case)
    assert changed and not (case.out/'receipt.json').exists()


def test_receipt_change_inside_final_comparison_is_caught(cohort,case,monkeypatch):
    admit(cohort,case);path=case.out/'receipt.json';original=cohort._hash;changed=[]
    def digest(value):
        result=original(value)
        if isinstance(value,dict) and value.get('format')==cohort._FORMAT and 'sha256' in value and not changed:
            changed.append(True);path.write_text(path.read_text()+'\n')
        return result
    monkeypatch.setattr(cohort,'_hash',digest)
    with pytest.raises(ValueError,match='changed|stamp'):verify(cohort,case,path)
    assert changed

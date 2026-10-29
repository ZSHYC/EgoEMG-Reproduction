"""Actual parent ownership from recorder JSON lacking a UID field."""
import importlib
import os
import pytest


def api():
    try:return importlib.import_module('egoemg_repro.owned_recorder')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.owned_recorder':pytest.fail('recorded parent identity binder missing')
        raise


def parent_snapshot():
    from egoemg_repro.emg2pose_memory_process import process_identity
    p=process_identity(os.getppid());return {k:p[k]for k in('pid','start_ticks')}


def test_recorded_two_field_parent_resolves_to_live_same_user_owner():
    from egoemg_repro.emg2pose_memory_process import same_process
    snapshot=parent_snapshot();owner=api().recorder_owner(snapshot)
    assert owner['pid']==os.getppid()and owner['uid']==os.getuid()and same_process(owner)
    assert set(snapshot)=={'pid','start_ticks'}


@pytest.mark.parametrize('damage',['pid','ticks','bool_pid','bad_ticks','extra_uid','not_dict'])
def test_wrong_parent_or_malformed_recorder_cannot_be_bound(damage):
    value=parent_snapshot()
    if damage=='pid':value['pid']=os.getpid()
    elif damage=='ticks':value['start_ticks']=str(int(value['start_ticks'])+1)
    elif damage=='bool_pid':value['pid']=True
    elif damage=='bad_ticks':value['start_ticks']='not-a-start-tick'
    elif damage=='extra_uid':value['uid']=os.getuid()
    else:value=None
    with pytest.raises(ValueError):api().recorder_owner(value)


def test_foreign_uid_cannot_be_accepted_even_when_pid_and_start_match(monkeypatch):
    from egoemg_repro.emg2pose_memory_process import process_identity
    value=parent_snapshot();actual=process_identity(os.getppid());actual['uid']=os.getuid()+1
    module=api();monkeypatch.setattr(module,'process_identity',lambda pid:actual)
    with pytest.raises(ValueError):module.recorder_owner(value)


def test_compact_external_training_owner_is_live_without_being_our_parent():
    from egoemg_repro.emg2pose_memory_process import process_identity
    me=process_identity();snapshot={k:me[k]for k in('pid','start_ticks')}
    assert api().recorded_process_live(snapshot)is True
    snapshot['start_ticks']=str(int(snapshot['start_ticks'])+1)
    assert api().recorded_process_live(snapshot)is False


def test_exited_recorded_child_is_closed_instead_of_being_inferred_from_missing_fields():
    import subprocess,sys
    from egoemg_repro.emg2pose_memory_process import process_identity
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        identity=process_identity(child.pid);snapshot={k:identity[k]for k in('pid','start_ticks')}
        assert api().recorded_process_live(snapshot)is True
    finally:child.terminate();child.wait(timeout=5)
    assert api().recorded_process_live(snapshot)is False


def test_malformed_record_cannot_be_misinterpreted_as_a_closed_training_owner():
    with pytest.raises(ValueError):api().recorded_process_live({'pid':True,'start_ticks':'1'})

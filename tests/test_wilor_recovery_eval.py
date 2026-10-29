"""Recovered evaluation boundaries; small tensors never qualify a WiLoR model."""
import copy
import importlib
import json
from pathlib import Path

import pytest


def api():
    try:return importlib.import_module('egoemg_repro.wilor_recovery_eval')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.wilor_recovery_eval':pytest.fail('recovered WiLoR evaluation entry is missing')
        raise


@pytest.mark.parametrize('role,kind,reference,profile',[
    ('train','native',None,None),('test','native',None,None),('val','replay',None,None),
    ('val','native','reference',None),('val','native',None,'profile')])
def test_recovered_test_requires_frozen_validation_before_fit_or_output(tmp_path,role,kind,reference,profile):
    with pytest.raises(ValueError):
        api().run_wilor_recovery_pass(None,tmp_path/'out',role=role,kind=kind,reference=reference,profile=profile,require_full=False)
    assert not(tmp_path/'out').exists()


@pytest.fixture
def selected_recovery(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from egoemg_repro.classic_emg_cohort import _StatGuard
    from egoemg_repro.vision_literal_runtime import vision_state_sha256
    from egoemg_repro import wilor_recovery
    state={'weight':torch.tensor([1.,2.])};directory=tmp_path/'fit';(directory/'checkpoints').mkdir(parents=True)
    torch.save({'state_dict':state},directory/'checkpoints/best.ckpt')
    historical=dict(format='historical-format',sha256='a'*64,runtime_device=dict(accelerator='gpu',visible_device='GPU-original'),
        policy={'scope':'full'},counts={'train':202020,'val':3152,'test':8612})
    continuation=dict(sha256='b'*64,actual_runtime_device=dict(accelerator='gpu',visible_device='GPU-actual'),
        prefix_epochs=6,prefix_optimizer_executions=67344,parent_admission=dict(parent_checkpoint_sha256='c'*64),
        device_admission={'cuda_uuid_changed':True},new_process={'pid':123,'start_ticks':'456','uid':1005})
    result=dict(sha256='d'*64,continuation_identity_sha256='b'*64,identity_sha256='a'*64,prefix_epochs=6,
        prefix_optimizer_executions=67344,segment_optimizer_executions=269376,uninterrupted_execution=False,
        selected_best=dict(file='best.ckpt',model_sha256=vision_state_sha256(state)))
    (directory/'recovery-identity.json').write_text(json.dumps(continuation))
    guard=_StatGuard([directory/'recovery-identity.json',directory/'checkpoints/best.ckpt'])
    # Only replace external full-corpus/raw-631M admission. Selected-state loading,
    # source/file generation guards and the new evaluation view remain real.
    monkeypatch.setattr(wilor_recovery,'_read_recovery',lambda path,require_full:(result,historical,None,object(),object(),guard))
    return directory,historical,continuation,result


def test_evaluation_uses_actual_continuation_device_without_resealing_historical_identity(selected_recovery):
    directory,historical,continuation,_=selected_recovery;before=copy.deepcopy(historical)
    context=api()._Evidence();_,view,_,_,_=context.fit(directory)
    assert historical==before and historical['runtime_device']['visible_device']=='GPU-original'
    assert view['runtime_device']['visible_device']=='GPU-actual'
    assert view['historical_identity_sha256']=='a'*64 and 'sha256'not in view
    assert view['format']=='explicit_wilor_recovery_evaluation_runtime_view_v1'
    assert context.recoveries[directory]['continuation_identity_sha256']=='b'*64
    context.guard.check()


def test_selected_raw_checkpoint_change_cannot_be_hidden_by_recovery_summary(selected_recovery):
    torch=pytest.importorskip('torch')
    directory,_,_,_=selected_recovery
    torch.save({'state_dict':{'weight':torch.tensor([9.,2.])}},directory/'checkpoints/best.ckpt')
    with pytest.raises(ValueError):api()._Evidence().fit(directory)


def test_continuation_generation_is_guarded_after_evaluation_context_admission(selected_recovery):
    directory,_,_,_=selected_recovery;context=api()._Evidence();context.fit(directory)
    path=directory/'recovery-identity.json';path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError):context.guard.check()


def test_new_pass_format_is_rejected_by_closed_uninterrupted_reader(tmp_path):
    pytest.importorskip('torch')
    from egoemg_repro import wilor_native_eval
    m=api();(tmp_path/'pass.json').write_text(json.dumps(m._sealed(dict(format=m.FORMAT,scientific_eligible=False,full_paper_complete=False))))
    for n in('predictions.pt','native-metrics.json'):(tmp_path/n).write_text('{}')
    with pytest.raises(ValueError,match='format|seal'):wilor_native_eval.read_wilor_pass(tmp_path)


def test_recovered_curve_marks_real_prefix_boundary_and_keeps_native_epoch_values(tmp_path):
    from test_wilor_native_eval import curve_fixture
    rows,result,_=curve_fixture(tmp_path)
    result.update(prefix_epochs=1,prefix_optimizer_executions=2,segment_optimizer_executions=2,continuation_identity_sha256='b'*64)
    curves=api()._curve_data(tmp_path,result)
    assert curves['epochs']==rows and curves['resume_boundary']==dict(after_complete_epoch=1,parent_optimizer_executions=2,new_optimizer_executions=2)
    assert curves['continuation_identity_sha256']=='b'*64
    images=api()._curve_images(curves)
    assert images['training-curves.png'].startswith(b'\x89PNG')
    assert b'Resume boundary'in images['training-curves.svg'] and b'ENGINEERING ONLY'in images['training-curves.svg']


@pytest.mark.parametrize('field,value',[('prefix_epochs',0),('prefix_epochs',2),('prefix_optimizer_executions',0),('segment_optimizer_executions',3)])
def test_recovered_curves_reject_false_interruption_or_update_partition(tmp_path,field,value):
    from test_wilor_native_eval import curve_fixture
    _,result,_=curve_fixture(tmp_path)
    result.update(prefix_epochs=1,prefix_optimizer_executions=2,segment_optimizer_executions=2,continuation_identity_sha256='b'*64)
    result[field]=value
    with pytest.raises(ValueError):api()._curve_data(tmp_path,result)


def test_recovered_profile_cannot_freeze_missing_native_validation(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):
        api().freeze_wilor_recovery_validation(tmp_path/'absent-native',tmp_path/'absent-replay',tmp_path/'out')
    assert not(tmp_path/'out').exists()


def test_report_plot_receipt_names_actual_recovered_svg_runtime():
    receipt=api()._plot_runtime()
    assert receipt['svg_hashsalt']=='egoemg-recovered-wilor-v1'
    assert receipt['backend']=='Agg' and len(receipt['font_sha256'])==64


def test_report_completion_distinguishes_recovery_and_hashes_actual_public_files(tmp_path):
    from egoemg_repro import wilor_native_eval
    for n in wilor_native_eval.PUBLIC_FILES:(tmp_path/n).write_bytes(n.encode())
    receipt=api()._completion(tmp_path,dict(sha256='a'*64,scientific_eligible=False))
    assert receipt['format']=='explicit_recovered_wilor_report_completion_v1'
    assert receipt['report_sha256']=='a'*64 and receipt['scientific_eligible']is False
    import hashlib
    assert receipt['files_sha256']['metrics.md']==hashlib.sha256(b'metrics.md').hexdigest()


@pytest.mark.parametrize('function,args',[
    ('publish_wilor_recovery_report',(None,None,None)),('run_wilor_recovery_pass',(None,))])
def test_recovered_outputs_are_exclusive_before_any_fit_is_opened(tmp_path,function,args):
    output=tmp_path/'retained';output.mkdir();(output/'keep').write_bytes(b'keep')
    kwargs=dict(require_full=False)
    if function.startswith('run'):kwargs.update(role='val',kind='native')
    with pytest.raises(FileExistsError):getattr(api(),function)(*args,output,**kwargs)
    assert (output/'keep').read_bytes()==b'keep'

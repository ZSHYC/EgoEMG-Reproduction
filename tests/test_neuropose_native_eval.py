"""Selected-best native NeuroPose replay and transaction integration."""
from pathlib import Path
import importlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from test_neuropose_native_train import native_fit as cpu_fit

PROJECT=Path(__file__).parents[1]


def api():
    try:return importlib.import_module('egoemg_repro.neuropose_native_eval')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.neuropose_native_eval':pytest.fail('native NeuroPose evaluation entry is missing')
        raise


def child(action,*args):
    result=subprocess.run([sys.executable,'-B','-m','egoemg_repro.neuropose_native_eval',action,*map(str,args)],
        cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope='module')
def passes(cpu_fit,tmp_path_factory):
    root,fit=cpu_fit
    retained=os.environ.get('EGOEMG_NEURO_NATIVE_RETAINED_PASSES')
    if retained:
        out=Path(retained);context=api()._Evidence()
        values=[api()._read_pass(out/name,context) for name in ('val-native','val-replay','test-native','test-replay')]
        profile=api()._read_profile(out/'profile',context);context.guard.check()
        return root,out,fit,values[0],values[1],profile,values[2],values[3]
    out=tmp_path_factory.mktemp('native-neuro-passes')
    native=api().run_neuropose_pass(root/'fit',out/'val-native',role='val',kind='native',require_full=False)
    replay=child('pass','--fit',root/'fit','--output',out/'val-replay','--role','val','--kind','replay',
        '--reference',out/'val-native','--engineering')
    profile=api().freeze_neuropose_validation(out/'val-native',out/'val-replay',out/'profile')
    test=api().run_neuropose_pass(root/'fit',out/'test-native',role='test',kind='native',profile=out/'profile',require_full=False)
    test_replay=child('pass','--fit',root/'fit','--output',out/'test-replay','--role','test','--kind','replay',
        '--reference',out/'test-native','--profile',out/'profile','--engineering')
    return root,out,fit,native,replay,profile,test,test_replay


def test_native_selected_best_and_fresh_process_full_validation_agree(passes):
    root,out,fit,native,replay,profile,_,_=passes
    assert native['selected_best']==fit['selected_best']
    assert native['coverage']==replay['coverage'] and native['coverage']['samples']==2
    assert native['coverage']['predictions'][0]['shape']==[2,22,7790]
    assert native['coverage']['predictions'][0]['dtype']=='torch.float32'
    assert native['model_before_sha256']==native['model_after_sha256']==fit['selected_best']['model_sha256']
    assert native['process']!=replay['process']
    assert native['native_metrics']['val_loss']==pytest.approx(fit['selected_best']['score'],rel=2e-6,abs=1e-7)
    assert profile['agreement']['prediction_hashes_equal'] is True
    assert api().read_neuropose_pass(out/'val-replay')==replay


def test_cpu_pass_records_actual_device_and_never_initializes_cuda(passes):
    for value in (passes[3],passes[4],passes[6],passes[7]):
        assert value.get('device_type')=='cpu'
        assert value.get('cuda_initialized') is False


def test_test_pass_keeps_original_generated_mask_center_and_pooled_degrees(passes):
    _,_,_,_,_,profile,test,replay=passes
    assert test['coverage']==replay['coverage'] and test['statistics']==replay['statistics']
    assert test['profile_sha256']==profile['sha256'] and test['started_ns']>profile['finished_ns']
    stats=test['statistics']
    assert stats['samples']==2 and stats['alignment']['output_center']==3895
    assert len(stats['center']['by_joint'])==22
    assert set(stats['trajectory']['by_hand'])=={'left','right'}
    assert 'test_loss' in test['native_metrics'] and replay['native_metrics']=={}
    assert test['scientific_eligible'] is False


def test_same_process_replay_and_test_before_validation_profile_refuse(cpu_fit,tmp_path):
    root,_=cpu_fit
    with pytest.raises(ValueError,match='profile|validation'):
        api().run_neuropose_pass(root/'fit',tmp_path/'early',role='test',kind='native',require_full=False)
    assert not (tmp_path/'early').exists()


def test_engineering_fit_cannot_enter_default_full_pass(cpu_fit,tmp_path):
    root,_=cpu_fit
    with pytest.raises(ValueError,match='full|engineering'):
        api().run_neuropose_pass(root/'fit',tmp_path/'full',role='val',kind='native')
    assert not (tmp_path/'full').exists()


def test_replay_cannot_run_in_its_native_reference_process(passes,tmp_path):
    root,out,*_=passes
    with pytest.raises(ValueError,match='process'):
        api().run_neuropose_pass(root/'fit',tmp_path/'same',role='val',kind='replay',reference=out/'val-native',require_full=False)
    assert not (tmp_path/'same').exists()


@pytest.mark.parametrize('change',['later_prediction','dtype','sample_count','best','scope','native_loss','process','scaler','before_fit'])
def test_resealed_pass_cannot_replace_raw_receipts_selected_best_or_native_metrics(passes,tmp_path,change):
    _,out,*_=passes
    path=tmp_path/'copy';shutil.copytree(out/'val-replay' if change=='process' else out/'val-native',path)
    from egoemg_repro.neuropose_native_train import _sealed
    v=json.loads((path/'pass.json').read_text());v.pop('sha256')
    if change=='later_prediction':v['coverage']['predictions'][-1]['prediction_sha256']='0'*64
    elif change=='dtype':v['coverage']['predictions'][-1]['dtype']='torch.float64'
    elif change=='sample_count':v['coverage']['samples']+=1
    elif change=='best':v['selected_best']['epoch']+=1
    elif change=='scope':v['scope']='full'
    elif change=='native_loss':v['native_metrics']['val_loss']+=1
    elif change=='process':v['process']=api().read_neuropose_pass(out/'val-native')['process']
    elif change=='scaler':v['scaler_before_sha256']=v['scaler_after_sha256']='0'*64
    else:v['started_ns']=1
    (path/'coverage.json').write_text(json.dumps(v['coverage']))
    (path/'native-metrics.json').write_text(json.dumps(v['native_metrics']))
    (path/'pass.json').write_text(json.dumps(_sealed(v)))
    with pytest.raises(ValueError):api().read_neuropose_pass(path)


def test_cpu_engineering_and_pass_booleans_cannot_qualify_cuda(passes,tmp_path):
    root,out,*_=passes
    with pytest.raises(ValueError,match='CUDA|cuda|engineering'):
        api().qualify_neuropose(root/'fit',out/'val-native',out/'val-replay',tmp_path/'qualification')
    assert not (tmp_path/'qualification').exists()
    with pytest.raises((ValueError,TypeError)):
        api().read_neuropose_qualification({'passed':True})


def test_qualification_reopens_numeric_runtime_instead_of_a_pass_boolean(tmp_path):
    p=tmp_path/'numeric';p.mkdir();(p/'identity.json').write_text(json.dumps({'runtime':{'torch_build':'wrong'}}))
    with pytest.raises(ValueError,match='runtime'):
        api()._qualified_runtime({'runtime':{'torch_build':'2.2.2+cu118'},'preconditions':{'numeric_prefix':str(p)}})


def test_fit_retains_actual_training_process_for_post_exit_qualification(cpu_fit):
    _,result=cpu_fit
    assert 'execution' in result,'qualification must know which actual fitter has exited'
    assert result['execution']['process']['pid']>0
    assert int(result['execution']['process']['start_ticks'])>0
    assert result['finished_ns']>result['execution']['started_ns']


def test_public_report_reopens_native_test_chain_and_actual_best_curve(passes,tmp_path):
    root,out,fit,_,_,_,test,replay=passes
    report=api().publish_neuropose_report(root/'fit',out/'test-native',out/'test-replay',tmp_path/'report',require_full=False)
    assert report['scientific_eligible'] is False and report['scope']=='cpu_engineering'
    assert report['statistics']['center']['overall']==test['statistics']['center']['overall']
    assert report['curve_data']['selected_epoch_zero_based']==fit['selected_best']['epoch']
    assert report['curve_data']['epochs'][-1]['epoch']==1
    assert 'composite' in report['curve_data']['validation_unit']
    assert api().verify_neuropose_report(out/'test-replay',tmp_path/'report',require_full=False)==report
    raw=(tmp_path/'report/metrics.json').read_text()
    assert str(root) not in raw
    private_users=test['statistics']['center']['by_user']
    assert set(report['statistics']['center']['by_user']).isdisjoint(private_users)
    assert (tmp_path/'report/completion.json').exists()
    assert (tmp_path/'report/training-curves.png').read_bytes().startswith(b'\x89PNG')
    with pytest.raises(FileExistsError):
        api().publish_neuropose_report(root/'fit',out/'test-native',out/'test-replay',tmp_path/'report',require_full=False)
    with pytest.raises(ValueError,match='full|engineering'):
        api().publish_neuropose_report(root/'fit',out/'test-native',out/'test-replay',tmp_path/'full')
    image=tmp_path/'report/training-curves.png';image.write_bytes(image.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='report|curve|completion'):
        api().verify_neuropose_report(out/'test-replay',tmp_path/'report',require_full=False)


def test_report_cannot_bless_a_modified_actual_training_journal(passes,tmp_path):
    root,out,*_=passes;path=root/'fit/epochs.jsonl';original=path.read_bytes()
    try:
        rows=[json.loads(line) for line in original.splitlines()];rows[-1]['train_loss']+=1
        path.write_text(''.join(json.dumps(v)+'\n' for v in rows))
        with pytest.raises(ValueError):
            api().publish_neuropose_report(root/'fit',out/'test-native',out/'test-replay',tmp_path/'report',require_full=False)
        assert not (tmp_path/'report').exists()
    finally:path.write_bytes(original)


def test_report_checks_admitted_journal_after_curve_rendering(passes,tmp_path,monkeypatch):
    root,out,*_=passes;path=root/'fit/epochs.jsonl';original=path.read_bytes();render=api()._curve_images
    def render_then_change(curves):
        images=render(curves);path.write_bytes(original+b'changed-after-render\n');return images
    monkeypatch.setattr(api(),'_curve_images',render_then_change)
    try:
        with pytest.raises(ValueError,match='changed|generation|differ'):
            api().publish_neuropose_report(root/'fit',out/'test-native',out/'test-replay',tmp_path/'report',require_full=False)
        assert not (tmp_path/'report/completion.json').exists()
    finally:path.write_bytes(original)


def test_fit_transaction_guard_keeps_actual_admitted_inputs_guarded(cpu_fit):
    import numpy as np
    from egoemg_repro.neuropose_native_train import _read_fit
    root,_=cpu_fit
    _,_,_,guard=_read_fit(root/'fit',require_full=False)
    path=root/'data/emg_right_filtered.dat';before=path.read_bytes()
    try:
        values=np.memmap(path,mode='r+',dtype=np.float32,shape=(path.stat().st_size//32,8))
        values[7790+7000,3]+=1;values.flush();values._mmap.close()
        with pytest.raises(ValueError,match='changed|differ|identity'):guard.check()
    finally:path.write_bytes(before)

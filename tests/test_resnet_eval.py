"""Selected original ResNet, frozen validation and complete point replay."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_resnet_train import native_fit
from test_vision_literal_data import UPSTREAM

ROOT=Path(__file__).resolve().parents[1]
_EVALUATED={}


def api():return importlib.import_module('egoemg_repro.resnet_eval')


def evaluated(request):
    m=api();fit=request.getfixturevalue('native_fit');root=fit['root'];output=root/'evaluation'
    if str(output)in _EVALUATED:return output
    output.mkdir()
    base=[sys.executable,'-B','-m','egoemg_repro.resnet_eval']
    shared=['--data',str(root/'data'),'--crops',str(root/'crops'),'--upstream',str(UPSTREAM),
        '--fit',str(fit['output']),'--variant','resnet18','--engineering']
    phases=[('val-native',['pass',*shared,'--role','val','--kind','native','--output',str(output/'val-native')]),
        ('val-replay',['pass',*shared,'--role','val','--kind','replay','--reference',str(output/'val-native'),'--output',str(output/'val-replay')]),
        ('freeze',['freeze','--reference',str(output/'val-native'),'--replay',str(output/'val-replay'),'--output',str(output/'profile')]),
        ('test-native',['pass',*shared,'--role','test','--kind','native','--profile',str(output/'profile'),'--output',str(output/'test-native')]),
        ('test-replay',['pass',*shared,'--role','test','--kind','replay','--profile',str(output/'profile'),
            '--reference',str(output/'test-native'),'--output',str(output/'test-replay')])]
    for name,args in phases:
        run=subprocess.run([*base,*args],cwd=ROOT,capture_output=True,text=True,timeout=180)
        (output/(name+'.stdout')).write_text(run.stdout);(output/(name+'.stderr')).write_text(run.stderr)
        assert run.returncode==0,run.stderr[-5000:]
    _EVALUATED[str(output)]=True;return output


def test_actual_resnet_selected_state_saved_validation_and_complete_test_replay(request):
    m=api();out=evaluated(request)
    val=m.read_resnet_pass(out/'val-native');native=m.read_resnet_pass(out/'test-native')
    replay=m.read_resnet_pass(out/'test-replay');profile=m.read_resnet_profile(out/'profile')
    assert val['samples']==4 and val['saved_fit_validation']['pose_exact_match'] is True
    assert abs(val['saved_fit_validation']['native_mae_difference_rad'])<2e-6
    assert native['model_class']=='egoemg.models.modules.resnet_vision.ResNetVisionPose'
    assert native['variant']=='resnet18' and native['selected_best']['completed_epochs']in(2,4)
    assert native['samples']==12 and native['statistics']['unique_samples']==6
    assert native['statistics']['duplicate_consumptions']==6
    assert native['statistics']['consumed']['samples']==dict(all=12,vision_valid=6,missing_crop=6)
    assert native['prediction_sha256']==replay['prediction_sha256']
    assert native['process']!=replay['process'] and native['model_before_sha256']==native['model_after_sha256']
    assert replay['reference_sha256']==native['sha256'] and native['profile_sha256']==profile['sha256']
    assert profile['finished_ns']<native['started_ns']
    assert native['scientific_eligible'] is False and native['training_full_terminated'] is False


def test_engineering_report_reopens_raw_chain_and_never_becomes_a_scientific_result(request,tmp_path):
    m=api();out=evaluated(request);report=m.build_resnet_report(out/'test-replay')
    assert report['variant']=='resnet18' and report['training']['completed_epochs']==4
    assert report['training']['execution_policy']['check_val_every_n_epoch']==2
    assert report['test']['consumed_samples']==12 and report['test']['unique_samples']==6
    assert report['scientific_eligible'] is False and report['full_paper_complete'] is False
    assert list(report['test']['consumed']['all']['by_user'])==['user_000']
    assert str(ROOT)not in json.dumps(report)and '/tmp/'not in json.dumps(report)
    destination=tmp_path/'report';m.publish_resnet_report(out/'test-replay',destination)
    assert m.verify_resnet_report(out/'test-replay',destination)==report
    (destination/'metrics.md').write_text('altered report')
    with pytest.raises(ValueError):m.verify_resnet_report(out/'test-replay',destination)


@pytest.mark.parametrize('role,kind',[(None,'native'),('train','native'),('val',None),('test','unknown')])
def test_unknown_pass_roles_or_kinds_reject_before_data_or_model_io(role,kind):
    with pytest.raises(ValueError):api().run_resnet_pass(None,None,None,role=role,kind=kind)


@pytest.mark.parametrize('damage',['payload_digest','statistics','native_metric','model_class',
    'variant','saved_validation','process','pretest_time'])
def test_resealed_pass_or_profile_metadata_cannot_override_raw_resnet_chain(request,monkeypatch,damage):
    from egoemg_repro.classic_emg import _hash
    m=api();out=evaluated(request);original=m._json
    def read(path):
        value=original(path);path=Path(path)
        if damage=='pretest_time'and path==out/'profile/profile.json':
            value=copy.deepcopy(value);value['finished_ns']+=10**18
        elif path==out/('val-native/pass.json'if damage=='saved_validation'else'test-replay/pass.json'):
            value=copy.deepcopy(value)
            if damage=='payload_digest':value['payload_sha256']='0'*64
            elif damage=='statistics':value['statistics']['consumed_samples']=8
            elif damage=='native_metric':value['native_metrics']={'test_mae':0.}
            elif damage=='model_class':value['model_class']='VisionViTPose'
            elif damage=='variant':value['variant']='resnet50'
            elif damage=='saved_validation':value['saved_fit_validation']['pose_exact_match']=False
            elif damage=='process':value['process']=original(out/'test-native/pass.json')['process']
            else:return value
        else:return value
        value['sha256']=_hash({k:v for k,v in value.items()if k!='sha256'});return value
    monkeypatch.setattr(m,'_json',read)
    path=out/('val-native'if damage=='saved_validation'else'test-replay')
    with pytest.raises(ValueError):m.read_resnet_pass(path)


def test_report_audit_verifies_each_distinct_fit_once_and_reopens_on_next_transaction(request,monkeypatch):
    m=api();audit=m._Audit();out=evaluated(request);original=m.read_resnet_fit;calls=[]
    def read(*args,**kwargs):
        calls.append(str(args[0]));return original(*args,**kwargs)
    monkeypatch.setattr(m,'read_resnet_fit',read)
    report=audit.report(out/'test-replay')
    assert report['test']['consumed_samples']==12 and len(calls)==1
    assert m.build_resnet_report(out/'test-replay')==report and len(calls)==2


def test_cached_audit_guards_all_raw_inputs_and_rejects_generation_changes(request,tmp_path):
    m=api();audit=m._Audit();out=evaluated(request);fit=request.getfixturevalue('native_fit')['output']
    marker=tmp_path/'registered-input';marker.write_bytes(b'original');audit.bind([marker])
    audit.pass_(out/'test-replay');audit.check()
    result=m._json(fit/'fit-result.json')
    required={fit/'identity.json',fit/'fit-result.json',fit/'epochs.jsonl',fit/'checkpoints/last.ckpt',
        fit/'checkpoints'/result['selected_best']['file'],fit/result['resume_evidence']['parent_file'],
        fit/result['resume_evidence']['observation_file'],out/'profile/profile.json'}
    for role in ('val-native','val-replay','test-native','test-replay'):
        required.update({out/role/'pass.json',out/role/'predictions.pt'})
        if role.endswith('native'):required.add(out/role/'native-metrics.json')
    assert required<=set(audit.guard.paths)
    marker.write_bytes(b'changed generation')
    with pytest.raises(ValueError):audit.pass_(out/'test-replay')


def test_artifact_reference_cycle_rejects_without_recursive_checkpoint_reloads(request,monkeypatch):
    from egoemg_repro.classic_emg import _hash
    m=api();audit=m._Audit();out=evaluated(request);original=m._json
    def read(path):
        value=original(path)
        if Path(path)==out/'test-replay/pass.json':
            value=copy.deepcopy(value);value['reference_path']=str(out/'test-replay')
            value['sha256']=_hash({k:v for k,v in value.items()if k!='sha256'})
        return value
    monkeypatch.setattr(m,'_json',read)
    with pytest.raises(ValueError,match='cyclic|cycle'):audit.pass_(out/'test-replay')
